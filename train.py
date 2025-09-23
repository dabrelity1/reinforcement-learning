from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional
import json
import tempfile
import threading
import queue

import numpy as np
import tyro
from collections import deque

from agents.dqn import DQNAgent, DQNConfig
import torch
from agents.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils.logger import create_writer, ensure_dir
from utils.schedules import LinearSchedule


# Lazy imports to avoid requiring pygame/mss until needed
def make_env(env: str, capture_rect: Optional[list[int]] = None, sim_speed_scale: float = 1.0, frame_skip: int = 4, stack_frames: int = 4):
    if env == "sim":
        from env.sim_env import FishingSimEnv, SimConfig
        cfg = SimConfig()
        cfg.fish_speed *= sim_speed_scale
        cfg.frame_skip = frame_skip
        cfg.stack_frames = stack_frames
        return FishingSimEnv(cfg)
    elif env == "chet-sim":
        from env.chet_sim_env import ChetMiniGameEnv, ChetSimConfig
        cfg = ChetSimConfig(frame_skip=frame_skip, stack_frames=stack_frames)
        return ChetMiniGameEnv(cfg)
    elif env == "real":
        from env.fishing_env import FishingEnv, RealEnvConfig
        from utils.screen import CaptureRect
        assert capture_rect is not None and len(capture_rect) == 4, "Provide --capture-rect x y w h"
        rect = CaptureRect(capture_rect[0], capture_rect[1], capture_rect[2], capture_rect[3])
        cfg = RealEnvConfig(capture_rect=rect, frame_skip=frame_skip, stack_frames=stack_frames)
        return FishingEnv(cfg)
    else:
        raise ValueError("env must be 'sim', 'chet-sim', or 'real'")


# Top-level, picklable factory for AsyncVectorEnv (Windows spawn-safe)
def _make_chet_sim_env(frame_skip: int, stack_frames: int):
    from env.chet_sim_env import ChetMiniGameEnv, ChetSimConfig
    return ChetMiniGameEnv(ChetSimConfig(frame_skip=frame_skip, stack_frames=stack_frames))


@dataclass
class Args:
    env: str = "sim"  # sim | chet-sim | real
    capture_rect: Optional[list[int]] = None  # x y w h for real env
    total_steps: int = 200_000
    buffer_size: int = 100_000
    batch_size: int = 64
    start_learning: int = 10_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 150_000
    gamma: float = 0.99
    lr: float = 1e-4
    target_update_every: int = 1000
    tau: float = 0.0  # Polyak average factor; >0 enables soft target updates (e.g., 0.005 for stability)
    eval_every: int = 25_000
    eval_episodes: int = 5
    save_every: int = 50_000
    run_name: Optional[str] = None
    resume_from: Optional[str] = None  # resume checkpoint path
    seed: int = 42
    deterministic: bool = False
    render_every: int = 0  # only for sim env; 0 to disable
    sim_speed_scale: float = 1.0  # curriculum hook: 0.6..1.5 typical
    frame_skip: int = 4
    stack_frames: int = 4
    curriculum: bool = True
    num_envs: int = 1  # run multiple parallel simulator envs (chet-sim only for now)
    warmup_steps: int = 5000  # prefill buffer with heuristic policy in chet-sim
    updates_per_step: int = 1  # number of gradient updates per environment step
    dueling: bool = True
    noisy: bool = True
    amp: bool = True
    # Optional DrQ augmentation
    drq: bool = False
    drq_pad: int = 4
    # Memory format / backend
    channels_last: bool = False
    allow_tf32: bool = True
    # Optimizer options
    optimizer: str = "adamw"  # adam | adamw
    weight_decay: float = 0.0
    fused_optimizer: bool = True
    foreach_optimizer: bool = True
    # Rainbow-style options
    prioritized: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_decay: int = 200_000
    n_step: int = 1
    # Performance toggles
    compile: bool = False
    pin_memory: bool = False
    nvtx: bool = False  # annotate phases for Nsight profiling
    # Memory-efficient replay
    replay_memmap_dir: Optional[str] = None  # e.g., 'replay_memmap' to store frames on disk
    # Distributional C51
    c51: bool = False
    num_atoms: int = 51
    vmin: float = -10.0
    vmax: float = 10.0
    # Quantile Regression DQN (recommended for better distributional approximation)
    qr_dqn: bool = False  # Set to True for improved performance over C51
    num_quantiles: int = 51
    huber_kappa: float = 1.0
    # LR scheduler options
    lr_schedule: Optional[str] = None  # 'cosine' or None
    lr_warmup_steps: int = 0
    # Noisy annealing
    noisy_sigma_init: float = 0.5
    noisy_sigma_final: float = 0.5
    noisy_sigma_decay_steps: int = 200_000
    # Munchausen DQN
    munchausen: bool = False
    munchausen_alpha: float = 0.9
    munchausen_tau: float = 0.03
    munchausen_clip: float = -1.0
    # Video logging
    log_video_every: int = 0  # steps; 0 disables
    video_length: int = 600   # max steps per video episode
    # Async actor-learner
    async_learner: bool = False
    # Config presets
    config: Optional[str] = None
    # Curriculum v2 (chet-sim)
    curriculum_v2: bool = False
    curv2_reward_threshold: float = 0.7
    curv2_scale: float = 1.10
    # (video logging fields already defined above)


def main(args: Args):
    # Optional: load config file and override args
    def _apply_config_overrides(a: Args):
        try:
            if a.config:
                with open(a.config, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Respect CLI: only override values that are still at their dataclass defaults
                try:
                    defaults = Args()  # baseline defaults
                except Exception:
                    defaults = None
                for k, v in data.items():
                    if hasattr(a, k):
                        if defaults is None:
                            # fallback: overlay unconditionally
                            setattr(a, k, v)
                        else:
                            cur = getattr(a, k)
                            dfl = getattr(defaults, k)
                            if cur == dfl:
                                setattr(a, k, v)
        except Exception as e:
            print(f"[config] failed to load {a.config}: {e}")

    _apply_config_overrides(args)
    # Let PyTorch use multiple CPU threads but leave one for the system
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    except Exception:
        pass
    np.random.seed(args.seed)
    # Optional deterministic behavior
    if args.deterministic:
        try:
            import random as pyrand
            pyrand.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    writer = create_writer(args.run_name)
    # LR scheduler helpers (cosine with optional warmup)
    def compute_lr(step: int) -> float:
        if args.lr_schedule == 'cosine':
            import math
            T = max(1, args.total_steps)
            warm = max(0, int(args.lr_warmup_steps))
            if step < warm:
                return args.lr * (step / max(1, warm))
            t = (step - warm) / max(1, T - warm)
            return args.lr * 0.5 * (1 + math.cos(math.pi * t))
        return args.lr
    def maybe_step_lr(step: int):
        if args.lr_schedule:
            lr = compute_lr(step)
            for pg in dqn.optimizer.param_groups:
                pg['lr'] = lr
            writer.add_scalar('train/lr', lr, step)
    # Build env(s)
    if args.num_envs > 1 and args.env == 'chet-sim':
        # True parallelism via AsyncVectorEnv
        import gymnasium as gym
        from functools import partial
        venv = gym.vector.AsyncVectorEnv([partial(_make_chet_sim_env, args.frame_skip, args.stack_frames) for _ in range(args.num_envs)])
        obs, _ = venv.reset()
        obs_shape = obs.shape[1:]
        try:
            num_actions = int(getattr(venv, 'single_action_space', None).n)
        except Exception:
            num_actions = 3
    else:
        env = make_env(args.env, args.capture_rect, args.sim_speed_scale, args.frame_skip, args.stack_frames)
        obs, _ = env.reset()
        obs_shape = obs.shape  # (C,84,84)
        try:
            num_actions = int(getattr(env, 'action_space', None).n)
        except Exception:
            num_actions = 3

    # guard mutually exclusive distributional modes
    if args.c51 and getattr(args, 'qr_dqn', False):
        raise ValueError("Choose only one: --c51 or --qr-dqn")
    dqn = DQNAgent(DQNConfig(
        in_channels=obs_shape[0], num_actions=num_actions, gamma=args.gamma, lr=args.lr, target_update_every=args.target_update_every,
        dueling=args.dueling, noisy=args.noisy, amp=args.amp, compile=args.compile, pin_memory=args.pin_memory,
        c51=args.c51, num_atoms=args.num_atoms, vmin=args.vmin, vmax=args.vmax, n_step=args.n_step, tau=args.tau,
        qr_dqn=getattr(args, 'qr_dqn', False), num_quantiles=getattr(args, 'num_quantiles', 51), huber_kappa=getattr(args, 'huber_kappa', 1.0),
        drq=args.drq, drq_pad=args.drq_pad,
        noisy_sigma_init=args.noisy_sigma_init, noisy_sigma_final=args.noisy_sigma_final, noisy_sigma_decay_steps=args.noisy_sigma_decay_steps,
        munchausen=args.munchausen, munchausen_alpha=args.munchausen_alpha, munchausen_tau=args.munchausen_tau, munchausen_clip=args.munchausen_clip,
        channels_last=args.channels_last, allow_tf32=args.allow_tf32,
        optimizer=args.optimizer, weight_decay=args.weight_decay, fused_optimizer=args.fused_optimizer, foreach_optimizer=args.foreach_optimizer
    ))
    # Choose replay buffer implementation
    if args.prioritized:
        if args.replay_memmap_dir:
            from agents.replay_buffer import PrioritizedMemmapReplayBuffer
            buffer = PrioritizedMemmapReplayBuffer(args.buffer_size, obs_shape, args.replay_memmap_dir, alpha=args.per_alpha)
        else:
            buffer = PrioritizedReplayBuffer(args.buffer_size, obs_shape, alpha=args.per_alpha)
    else:
        if args.replay_memmap_dir:
            from agents.replay_buffer import MemmapReplayBuffer
            buffer = MemmapReplayBuffer(args.buffer_size, obs_shape, args.replay_memmap_dir)
        else:
            buffer = ReplayBuffer(args.buffer_size, obs_shape)
    eps_schedule = LinearSchedule(args.epsilon_start, args.epsilon_end, args.epsilon_decay)

    state = obs
    episode_reward = 0.0
    episode_len = 0
    global_step = 0
    episodes = 0
    base_speed = args.sim_speed_scale
    # moving average of episode returns for best-model saving
    best_reward_ma: Optional[float] = None
    ma_beta = 0.98

    # optional separate eval env (single env) created lazily
    eval_env = None
    # Dump startup config to run directory
    try:
        run_dir = getattr(writer, 'log_dir', None)
        if run_dir:
            with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(vars(args), f, indent=2)
    except Exception:
        pass
    # Optional resume from checkpoint
    if args.resume_from:
        try:
            dqn.load(args.resume_from)
            # Try read trainer state
            state_path = os.path.join('models', 'trainer_state.json')
            gs = None
            if os.path.exists(state_path):
                with open(state_path, 'r', encoding='utf-8') as f:
                    st = json.load(f)
                    gs = int(st.get('global_step', 0))
            if gs is None:
                # Parse from filename dqn_stepXXXXX.pt if possible
                base = os.path.basename(args.resume_from)
                import re
                m = re.search(r"step(\d+)", base)
                if m:
                    gs = int(m.group(1))
            if gs is not None:
                global_step = gs
            # keep latest pointer
            try:
                import shutil
                shutil.copy2(args.resume_from, os.path.join('models', 'dqn_latest.pt'))
            except Exception:
                pass
            print(f"[resume] resumed from {args.resume_from} @ global_step={global_step}, train_steps={dqn.train_steps}")
            # align LR scheduler state on resume
            try:
                if args.lr_schedule:
                    lr_now = compute_lr(global_step)
                    for pg in dqn.optimizer.param_groups:
                        pg['lr'] = lr_now
                    writer.add_scalar('train/lr', lr_now, global_step)
                else:
                    # If no schedule, try use saved LR if present
                    if gs is not None and os.path.exists(state_path):
                        with open(state_path, 'r', encoding='utf-8') as f:
                            st = json.load(f)
                        lr_saved = st.get('lr', None)
                        if lr_saved is not None:
                            for pg in dqn.optimizer.param_groups:
                                pg['lr'] = float(lr_saved)
                            writer.add_scalar('train/lr', float(lr_saved), global_step)
            except Exception:
                pass
        except Exception as e:
            print(f"[resume] failed to load {args.resume_from}: {e}")

    # (moved compute_lr/maybe_step_lr above to be available for resume path)

    ensure_dir("models")
    # n-step helpers
    use_n = max(1, int(args.n_step))
    gamma_pows = np.array([args.gamma ** k for k in range(use_n)], dtype=np.float32)
    def make_nstep_buffers(n_envs: int):
        return [deque(maxlen=use_n) for _ in range(n_envs)]
    def nstep_append(buf: deque, trans: tuple):
        """Append (s, a, r, ns, done) and emit 1 or 0 transitions depending on length/done."""
        buf.append(trans)
        if len(buf) < use_n:
            return None
        # compute n-step from leftmost
        R = 0.0
        done_out = False
        next_state_out = None
        for k, (s_k, a_k, r_k, ns_k, d_k) in enumerate(buf):
            R += (gamma_pows[k] * float(r_k)) * (0.0 if done_out else 1.0)
            if d_k and not done_out:
                done_out = True
                next_state_out = ns_k
                # we still continue loop to sum nothing due to done_out mask
        if next_state_out is None:
            # if no early done, next state after n-1 steps
            next_state_out = buf[use_n - 1][3]
        s0, a0 = buf[0][0], buf[0][1]
        # pop leftmost after emitting one
        buf.popleft()
        return (s0, a0, R, next_state_out, done_out)
    def nstep_flush(buf: deque):
        outs = []
        while len(buf) > 0:
            R = 0.0
            done_out = False
            next_state_out = None
            for k, (s_k, a_k, r_k, ns_k, d_k) in enumerate(buf):
                R += (gamma_pows[k] * float(r_k)) * (0.0 if done_out else 1.0)
                if d_k and not done_out:
                    done_out = True
                    next_state_out = ns_k
            if next_state_out is None:
                next_state_out = buf[-1][3]
            s0, a0 = buf[0][0], buf[0][1]
            outs.append((s0, a0, R, next_state_out, done_out))
            buf.popleft()
        return outs
    # Startup prints and small perf tweaks
    try:
        import sys
        import torch
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        # Enable TF32 for matmul and cudnn when requested
        try:
            allow_tf32 = bool(getattr(args, 'allow_tf32', True))
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        print(f"[startup] python={sys.executable}")
        print(f"[startup] device={dqn.cfg.device}, cuda={torch.cuda.is_available()}, threads={torch.get_num_threads()}, num_envs={args.num_envs}")
        print(f"[startup] batch_size={args.batch_size}, updates_per_step={args.updates_per_step}, dueling={args.dueling}, noisy={args.noisy}, amp={args.amp}")
        print(f"[startup] prioritized={args.prioritized}, per_alpha={args.per_alpha}, per_beta={args.per_beta_start}->{args.per_beta_end}/{args.per_beta_decay}, n_step={args.n_step}")
        try:
            compiled_flag = getattr(dqn, 'compiled', False)
        except Exception:
            compiled_flag = False
        print(f"[startup] compile={args.compile} (active={compiled_flag}), pin_memory={args.pin_memory}")
        print(f"[startup] channels_last={args.channels_last}, allow_tf32={args.allow_tf32}")
        print(f"[startup] optimizer={args.optimizer}, weight_decay={args.weight_decay}, fused_opt={args.fused_optimizer}, foreach_opt={args.foreach_optimizer}")
        print(f"[startup] c51={args.c51}, atoms={args.num_atoms}, vmin={args.vmin}, vmax={args.vmax}")
    except Exception:
        pass

    # Save an initial checkpoint so evaluators can start immediately
    init_path = os.path.join('models', f'dqn_step{0:04d}.pt')
    dqn.save(init_path)
    try:
        import shutil
        shutil.copy2(init_path, os.path.join('models', 'dqn_latest.pt'))
    except Exception:
        pass

    def heuristic_action(obs_batch: np.ndarray) -> np.ndarray:
        # Simple image-based heuristic for chet-sim: estimate centers and push toward target
        # Works on preprocessed (C,H,W) grayscale stack; use last frame ([-1])
        last = obs_batch[:, -1, :, :] if obs_batch.ndim == 4 else obs_batch[-1]
        # Horizontal projection to detect line (darker/gray) vs white bar region; we know the bar stripe is centered vertically.
        # Simplify: compute center of mass along width of bright (white bar) and medium gray (target line) using crude thresholds.
        import numpy as _np
        N = last.shape[0] if obs_batch.ndim == 4 else 1
        def centers(img):
            H, W = img.shape
            xs = _np.arange(W)
            white_mask = img > 0.85
            gray_mask = (img > 0.45) & (img < 0.75)
            w_sum = white_mask.sum(axis=0) + 1e-6
            g_sum = gray_mask.sum(axis=0) + 1e-6
            white_cx = (xs * w_sum).sum() / w_sum.sum()
            gray_cx = (xs * g_sum).sum() / g_sum.sum()
            return white_cx, gray_cx
        if N == 1:
            img2d = last[0] if last.ndim == 3 else last
            wc, gc = centers(img2d)
            return _np.array([0 if wc > gc else 1], dtype=_np.int64)  # left if bar right of target, else right
        actions = _np.zeros((N,), dtype=_np.int64)
        for i in range(N):
            wc, gc = centers(last[i])
            actions[i] = 0 if wc > gc else 1
        return actions

    try:
        # Optional warm-up for chet-sim to teach control mapping
        if args.env == 'chet-sim' and args.warmup_steps > 0:
            steps_left = args.warmup_steps
            if args.num_envs > 1:
                nstep_bufs = make_nstep_buffers(args.num_envs) if use_n > 1 else None
                while steps_left > 0:
                    acts = heuristic_action(state)
                    ns, r, term, trunc, _ = venv.step(acts)
                    done = np.logical_or(term, trunc)
                    for i in range(args.num_envs):
                        if use_n > 1:
                            out = nstep_append(nstep_bufs[i], (state[i], int(acts[i]), float(r[i]), ns[i], bool(done[i])))
                            if out is not None:
                                s0, a0, R, ns_o, d_o = out
                                buffer.add(s0, a0, R, ns_o, d_o)
                        else:
                            buffer.add(state[i], int(acts[i]), float(r[i]), ns[i], bool(done[i]))
                    # selective resets handled by vector env auto-reset off by default; we trigger resets
                    if done.any():
                        if use_n > 1:
                            # flush all to keep alignment after global reset
                            for i in range(args.num_envs):
                                for s0, a0, R, ns_o, d_o in nstep_flush(nstep_bufs[i]):
                                    buffer.add(s0, a0, R, ns_o, d_o)
                        reset_idx = np.where(done)[0]
                        if reset_idx.size > 0:
                            # gym AsyncVectorEnv reset resets all; workaround: step with noops until all done are reset
                            # simpler: perform full reset when any done to keep logic simple
                            ns, _ = venv.reset()
                    state = ns
                    global_step += args.num_envs
                    steps_left -= args.num_envs
            else:
                nstep_buf = deque(maxlen=use_n) if use_n > 1 else None
                while steps_left > 0:
                    act = int(heuristic_action(state[None, ...])[0])
                    ns, r, term, trunc, _ = env.step(act)
                    d = term or trunc
                    if use_n > 1:
                        out = nstep_append(nstep_buf, (state, act, float(r), ns, d))
                        if out is not None:
                            s0, a0, R, ns_o, d_o = out
                            buffer.add(s0, a0, R, ns_o, d_o)
                    else:
                        buffer.add(state, act, float(r), ns, d)
                    if d:
                        if use_n > 1:
                            for s0, a0, R, ns_o, d_o in nstep_flush(nstep_buf):
                                buffer.add(s0, a0, R, ns_o, d_o)
                        ns, _ = env.reset()
                    state = ns
                    global_step += 1
                    steps_left -= 1

        # Per-env episode trackers for vectorized mode
        if args.num_envs > 1 and args.env == 'chet-sim':
            ep_reward_vec = np.zeros((args.num_envs,), dtype=np.float64)
            ep_len_vec = np.zeros((args.num_envs,), dtype=np.int64)
            nstep_bufs = make_nstep_buffers(args.num_envs) if use_n > 1 else None

        # perf stats
        last_log_step = 0
        last_log_time = time.time()
        # Optional async learner setup
        buffer_lock = threading.Lock()
        stats_q: "queue.Queue[tuple[int,float]]" = queue.Queue(maxsize=256)
        upd_q: "queue.Queue[tuple[int,float]]" = queue.Queue(maxsize=256)  # (global_step, updates_per_sec)
        stop_event = threading.Event()

        def learner_loop():
            nonlocal global_step
            try:
                last_t = time.time()
                upd_count = 0
                while not stop_event.is_set():
                    if global_step <= args.start_learning or not buffer.can_sample(args.batch_size):
                        time.sleep(0.005)
                        # flush updates/sec periodically even if idle
                        now_t = time.time()
                        if now_t - last_t >= 1.0:
                            try:
                                upd_q.put_nowait((global_step, float(upd_count) / max(1e-6, now_t - last_t)))
                            except Exception:
                                pass
                            upd_count = 0
                            last_t = now_t
                        continue
                    # compute PER beta locally from shared global_step
                    if args.prioritized:
                        frac = min(1.0, global_step / max(1, args.per_beta_decay))
                        per_beta_local = args.per_beta_start + frac * (args.per_beta_end - args.per_beta_start)
                    else:
                        per_beta_local = None
                    # perform a burst of updates
                    updates = max(1, int(args.updates_per_step))
                    for _ in range(updates):
                        if not buffer.can_sample(args.batch_size):
                            break
                        if args.prioritized:
                            with buffer_lock:
                                batch, idxs, weights = buffer.sample(args.batch_size, beta=(per_beta_local if per_beta_local is not None else 1.0))
                            last_loss, td_err = dqn.train_step(batch, weights=weights)
                            with buffer_lock:
                                buffer.update_priorities(idxs, td_err)
                        else:
                            with buffer_lock:
                                batch = buffer.sample(args.batch_size)
                            last_loss, td_err = dqn.train_step(batch)
                        dqn.maybe_update_target()
                        upd_count += 1
                    # push stat
                    try:
                        if last_loss is not None:
                            stats_q.put_nowait((global_step, float(last_loss)))
                    except Exception:
                        pass
                    # push updates/sec once per second
                    now_t = time.time()
                    if now_t - last_t >= 1.0:
                        try:
                            upd_q.put_nowait((global_step, float(upd_count) / max(1e-6, now_t - last_t)))
                        except Exception:
                            pass
                        upd_count = 0
                        last_t = now_t
            except Exception as e:
                print(f"[async] learner error: {e}")

        learner_thread = None
        if args.async_learner:
            learner_thread = threading.Thread(target=learner_loop, name="learner", daemon=True)
            learner_thread.start()

        while global_step < args.total_steps:
                epsilon = eps_schedule(global_step)
                # PER beta annealing
                if args.prioritized:
                    frac = min(1.0, global_step / max(1, args.per_beta_decay))
                    per_beta = args.per_beta_start + frac * (args.per_beta_end - args.per_beta_start)
                else:
                    per_beta = None
                if args.num_envs > 1 and args.env == 'chet-sim':
                    actions = dqn.act_batch(state, epsilon)  # (N,)
                    ns, r, term, trunc, _ = venv.step(actions)
                    done = np.logical_or(term, trunc)
                    if global_step % max(2000, args.num_envs * 100) == 0:
                        # action distribution across vector envs
                        try:
                            import numpy as _np
                            counts = _np.bincount(actions, minlength=num_actions)
                            for a_i, c in enumerate(counts):
                                writer.add_scalar(f'action/freq_{a_i}', float(c) / float(len(actions)), global_step)
                        except Exception:
                            pass
                    for i in range(args.num_envs):
                        if use_n > 1:
                            out = nstep_append(nstep_bufs[i], (state[i], int(actions[i]), float(r[i]), ns[i], bool(done[i])))
                            if out is not None:
                                s0, a0, R, ns_o, d_o = out
                                buffer.add(s0, a0, R, ns_o, d_o)
                        else:
                            buffer.add(state[i], int(actions[i]), float(r[i]), ns[i], bool(done[i]))
                        ep_reward_vec[i] += float(r[i])
                        ep_len_vec[i] += 1
                    # log and reset any finished episodes
                    if done.any():
                        for i in np.where(done)[0]:
                            writer.add_scalar('episode/reward', ep_reward_vec[i], global_step)
                            writer.add_scalar('episode/length', ep_len_vec[i], global_step)
                            # update moving average and log
                            if best_reward_ma is None:
                                best_reward_ma = float(ep_reward_vec[i])
                            else:
                                best_reward_ma = ma_beta * best_reward_ma + (1 - ma_beta) * float(ep_reward_vec[i])
                            writer.add_scalar('episode/reward_ma', best_reward_ma, global_step)
                            ep_reward_vec[i] = 0.0
                            ep_len_vec[i] = 0
                        if use_n > 1:
                            # flush all before reset to avoid misalignment
                            for i in range(args.num_envs):
                                for s0, a0, R, ns_o, d_o in nstep_flush(nstep_bufs[i]):
                                    buffer.add(s0, a0, R, ns_o, d_o)
                        # Simpler approach: reset all envs to keep observations aligned
                        ns, _ = venv.reset()
                    state = ns
                    global_step += args.num_envs
                    # Rendering for vector envs is disabled by default; optionally render one env by manual call
                else:
                    action = dqn.act(state, epsilon)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    if global_step % 2000 == 0:
                        try:
                            writer.add_scalar(f'action/freq_{int(action)}', 1.0, global_step)
                        except Exception:
                            pass
                    if use_n > 1:
                        if 'nstep_buf' not in locals():
                            nstep_buf = deque(maxlen=use_n)
                        out = nstep_append(nstep_buf, (state, action, float(reward), next_state, bool(done)))
                        if out is not None:
                            s0, a0, R, ns_o, d_o = out
                            buffer.add(s0, a0, R, ns_o, d_o)
                    else:
                        buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    episode_len += 1
                    global_step += 1

                # Learn: sync mode only when async_learner is False
                if (not args.async_learner) and global_step > args.start_learning and buffer.can_sample(args.batch_size):
                    last_loss = None
                    if args.nvtx and torch.cuda.is_available():
                        try:
                            torch.cuda.nvtx.range_push("update_loop")
                        except Exception:
                            pass
                    td_err = None
                    for _ in range(max(1, int(args.updates_per_step))):
                        if not buffer.can_sample(args.batch_size):
                            break
                        if args.prioritized:
                            with buffer_lock:
                                batch, idxs, weights = buffer.sample(args.batch_size, beta=(per_beta if per_beta is not None else 1.0))
                            last_loss, td_err = dqn.train_step(batch, weights=weights)
                            with buffer_lock:
                                buffer.update_priorities(idxs, td_err)
                        else:
                            with buffer_lock:
                                batch = buffer.sample(args.batch_size)
                            last_loss, td_err = dqn.train_step(batch)
                        # Target update logging
                        if args.tau and args.tau > 0.0:
                            if (global_step % max(5000, args.num_envs * 200)) == 0:
                                writer.add_scalar('target/tau', float(args.tau), global_step)
                        else:
                            if (dqn.train_steps % dqn.cfg.target_update_every) == 0:
                                writer.add_scalar('target/hard_update', 1.0, global_step)
                        dqn.maybe_update_target()
                    if args.nvtx and torch.cuda.is_available():
                        try:
                            torch.cuda.nvtx.range_pop()
                        except Exception:
                            pass
                    if (global_step % 1000 == 0) and (last_loss is not None):
                        writer.add_scalar('train/loss', last_loss, global_step)
                        writer.add_scalar('train/epsilon', epsilon, global_step)
                        # gradient and AMP diagnostics
                        try:
                            if getattr(dqn, 'last_grad_norm', None) is not None:
                                writer.add_scalar('train/grad_norm', float(dqn.last_grad_norm), global_step)
                            if getattr(dqn, 'last_amp_scale', None) is not None:
                                writer.add_scalar('train/amp_scale', float(dqn.last_amp_scale), global_step)
                            if getattr(dqn, 'last_q_mean', None) is not None:
                                writer.add_scalar('train/q_mean', float(dqn.last_q_mean), global_step)
                            if getattr(dqn, 'last_q_max', None) is not None:
                                writer.add_scalar('train/q_max', float(dqn.last_q_max), global_step)
                        except Exception:
                            pass
                    # TD-error histogram occasionally
                    try:
                        if (global_step % 5000 == 0) and (td_err is not None):
                            import numpy as _np
                            writer.add_histogram('train/td_error', _np.asarray(td_err), global_step)
                    except Exception:
                        pass

                # perf: log steps/sec every ~5k steps
                if global_step - last_log_step >= max(5000, args.num_envs * 200):
                    now = time.time()
                    dt = max(1e-6, now - last_log_time)
                    sps = (global_step - last_log_step) / dt
                    writer.add_scalar('perf/steps_per_sec', sps, global_step)
                    # GPU memory metrics
                    try:
                        if torch.cuda.is_available():
                            mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                            mem_reserved = torch.cuda.memory_reserved() / (1024**2)
                            writer.add_scalar('perf/gpu_mem_alloc_mb', mem_alloc, global_step)
                            writer.add_scalar('perf/gpu_mem_reserved_mb', mem_reserved, global_step)
                            # Optional: GPU utilization via NVML
                            try:
                                import importlib
                                _p = importlib.util.find_spec('pynvml')
                                if _p is not None:
                                    pynvml = importlib.import_module('pynvml')
                                    pynvml.nvmlInit()
                                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                                    writer.add_scalar('perf/gpu_util_percent', float(util.gpu), global_step)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    last_log_step = global_step
                    last_log_time = now

                # Flush async learner loss stats
                if args.async_learner:
                    try:
                        while not stats_q.empty():
                            gs_s, loss_s = stats_q.get_nowait()
                            writer.add_scalar('train/loss_async', loss_s, gs_s)
                        while not upd_q.empty():
                            gs_u, ups = upd_q.get_nowait()
                            writer.add_scalar('perf/async_updates_per_sec', ups, gs_u)
                    except Exception:
                        pass

                # Episode end (single-env mode only)
                if args.num_envs == 1 and done:
                    writer.add_scalar('episode/reward', episode_reward, global_step)
                    writer.add_scalar('episode/length', episode_len, global_step)
                    episodes += 1
                    # update MA and log
                    if best_reward_ma is None:
                        best_reward_ma = float(episode_reward)
                    else:
                        best_reward_ma = ma_beta * best_reward_ma + (1 - ma_beta) * float(episode_reward)
                    writer.add_scalar('episode/reward_ma', best_reward_ma, global_step)
                    # Simple curriculum: progressively increase sim fish speed
                    if args.curriculum and args.env == 'sim':
                        # increase every 100 episodes up to 1.6x
                        new_scale = min(1.6, base_speed * (1.0 + 0.1 * (episodes // 100)))
                        try:
                            env.set_fish_speed(env.cfg.fish_speed * (new_scale / base_speed))
                        except Exception:
                            pass
                    # Curriculum v2 for chet-sim: scale difficulty when reward_ma crosses threshold
                    if args.curriculum_v2 and args.env == 'chet-sim':
                        try:
                            if best_reward_ma is not None and best_reward_ma >= float(args.curv2_reward_threshold):
                                # increase line speed cap and accel modestly
                                env.cfg.line_speed_cap *= float(args.curv2_scale)
                                env.cfg.line_accel_fast *= float(args.curv2_scale)
                                env.cfg.line_accel_slow *= float(args.curv2_scale)
                                # avoid repeated rescaling by bumping threshold high
                                args.curv2_reward_threshold += 0.1
                        except Exception:
                            pass
                    if use_n > 1 and 'nstep_buf' in locals():
                        for s0, a0, R, ns_o, d_o in nstep_flush(nstep_buf):
                            buffer.add(s0, a0, R, ns_o, d_o)
                    state, _ = env.reset()
                    episode_reward = 0.0
                    episode_len = 0

                # Periodic save/eval hooks
                if args.save_every and global_step % args.save_every == 0:
                    path = os.path.join('models', f'dqn_step{global_step}.pt')
                    # Atomic model save
                    try:
                        os.makedirs('models', exist_ok=True)
                        with tempfile.NamedTemporaryFile(delete=False, dir='models', suffix='.pt') as tf:
                            tmp_path = tf.name
                        dqn.save(tmp_path)
                        os.replace(tmp_path, path)
                    except Exception:
                        dqn.save(path)
                    # Save trainer state atomically
                    try:
                        # Persist basic trainer state (step, lr, seed)
                        cur_lr = None
                        try:
                            cur_lr = float(dqn.optimizer.param_groups[0]['lr'])
                        except Exception:
                            cur_lr = None
                        st = {'global_step': int(global_step), 'lr': cur_lr, 'seed': int(args.seed)}
                        with tempfile.NamedTemporaryFile(delete=False, dir='models', suffix='.json', mode='w', encoding='utf-8') as tf:
                            json.dump(st, tf)
                            tmpj = tf.name
                        os.replace(tmpj, os.path.join('models', 'trainer_state.json'))
                    except Exception:
                        pass
                    # Also update a symlink/copy to latest for quick eval
                    try:
                        latest = os.path.join('models', 'dqn_latest.pt')
                        import shutil
                        shutil.copy2(path, latest)
                    except Exception:
                        pass
                    # Save best model snapshot by moving-average reward
                    try:
                        if best_reward_ma is not None:
                            best_path = os.path.join('models', 'dqn_best.pt')
                            sentinel = os.path.join('models', '.best_reward_ma')
                            prev = None
                            if os.path.exists(sentinel):
                                with open(sentinel, 'r') as f:
                                    try:
                                        prev = float(f.read().strip())
                                    except Exception:
                                        prev = None
                            if prev is None or best_reward_ma > prev:
                                dqn.save(best_path)
                                with open(sentinel, 'w') as f:
                                    f.write(str(best_reward_ma))
                    except Exception:
                        pass

                # periodic evaluation
                if args.eval_every and global_step % args.eval_every == 0:
                    try:
                        if eval_env is None:
                            eval_env = make_env(args.env, args.capture_rect, args.sim_speed_scale, args.frame_skip, args.stack_frames)
                        ep_returns = []
                        for _ in range(max(1, int(args.eval_episodes))):
                            es, _ = eval_env.reset()
                            done_eval = False
                            er = 0.0
                            steps_guard = 0
                            while not done_eval and steps_guard < 5000:
                                a = dqn.act(es, epsilon=0.0)
                                es, rr, term_e, trunc_e, _ = eval_env.step(a)
                                er += float(rr)
                                done_eval = bool(term_e or trunc_e)
                                steps_guard += 1
                            ep_returns.append(er)
                        if len(ep_returns) > 0:
                            import numpy as _np
                            writer.add_scalar('eval/return_mean', float(_np.mean(ep_returns)), global_step)
                            writer.add_scalar('eval/return_std', float(_np.std(ep_returns)), global_step)
                    except Exception:
                        pass

                # periodic video logging (sim/chet-sim only, single env for simplicity)
                if args.log_video_every and (global_step % max(1, int(args.log_video_every)) == 0) and args.env in ('sim', 'chet-sim'):
                    try:
                        # Create a fresh env to record a short episode
                        venv_rec = make_env(args.env, args.capture_rect, args.sim_speed_scale, args.frame_skip, args.stack_frames)
                        frames = []
                        es, _ = venv_rec.reset()
                        done_v = False
                        steps_guard = 0
                        while not done_v and steps_guard < int(args.video_length):
                            a = dqn.act(es, epsilon=0.0)
                            es, rr, term_v, trunc_v, _ = venv_rec.step(a)
                            # Render frame from sim env
                            try:
                                if hasattr(venv_rec, '_last_render_frame') and venv_rec._last_render_frame is not None:
                                    frm = venv_rec._last_render_frame
                                else:
                                    # fallback: generate from observation
                                    import cv2
                                    frm = (np.repeat(es[-1] * 255.0, 3, axis=0).transpose(1, 2, 0)).astype(np.uint8)
                                frames.append(frm)
                            except Exception:
                                pass
                            done_v = bool(term_v or trunc_v)
                            steps_guard += 1
                        # stack and log video (T,H,W,C) with uint8
                        if len(frames) > 0:
                            import numpy as _np
                            vid = _np.stack(frames, axis=0)
                            # expand to (N,T,C,H,W) for TensorBoard (N batch=1)
                            vid_nchw = vid.transpose(0, 3, 1, 2)[None, ...]
                            writer.add_video('eval/video', vid_nchw, global_step, fps=30)
                    except Exception:
                        pass

                # LR schedule step
                maybe_step_lr(global_step)

                if args.render_every and args.env in ('sim', 'chet-sim') and args.num_envs == 1 and (global_step % args.render_every == 0):
                    env.render()

    except KeyboardInterrupt:
        pass
    finally:
        try:
            # Stop async learner thread
            if 'stop_event' in locals():
                stop_event.set()
            if 'learner_thread' in locals() and learner_thread is not None:
                learner_thread.join(timeout=5.0)
        except Exception:
            pass
        # Final save on normal end or Ctrl+C
        final_path = os.path.join('models', 'dqn_final.pt')
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir='models', suffix='.pt') as tf:
                tmpf = tf.name
            dqn.save(tmpf)
            os.replace(tmpf, final_path)
        except Exception:
            dqn.save(final_path)
        try:
            import shutil
            shutil.copy2(final_path, os.path.join('models', 'dqn_latest.pt'))
        except Exception:
            pass
        writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
