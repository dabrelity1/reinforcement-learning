from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized NoisyNet layer (Fortunato et al.)."""
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('eps_in', torch.zeros(1, in_features))
        self.register_buffer('eps_out', torch.zeros(out_features, 1))
        # store base sigma (per-layer scalar) for annealing
        self._sigma_base = 0.0
        self.reset_parameters(sigma0)

    def reset_parameters(self, sigma0: float):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # Factorized sigma
        sigma_init = float(sigma0) / float(np.sqrt(self.in_features))
        self._sigma_base = sigma_init
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    def set_noise_scale(self, scale: float):
        """Anneal noise by scaling sigma parameters relative to the initial base value."""
        s = max(0.0, float(scale))
        val = self._sigma_base * s
        self.weight_sigma.data.fill_(val)
        self.bias_sigma.data.fill_(val)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.sqrt(torch.clamp(x.abs(), min=1e-12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps_in = torch.randn_like(self.eps_in)
            eps_out = torch.randn_like(self.eps_out)
            fin = self._f(eps_in)
            fout = self._f(eps_out)
            w_eps = fout @ fin
            weight = self.weight_mu + self.weight_sigma * w_eps
            bias = self.bias_mu + self.bias_sigma * fout.squeeze(1)
        else:
            # In eval mode, use deterministic (mean) weights
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DQNNet(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, dueling: bool = True, noisy: bool = True, c51: bool = False, num_atoms: int = 51, noisy_sigma0: float = 0.5, qr_dqn: bool = False, num_quantiles: int = 51):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # compute conv output size for 84x84
        # formula: ((W-K)/S + 1)
        def out_size(w, k, s):
            return (w - k) // s + 1
        convw = out_size(out_size(out_size(84, 8, 4), 4, 2), 3, 1)
        convh = convw
        linear_input = 64 * convw * convh
        self.noisy = noisy
        self.dueling = dueling
        self.c51 = c51
        self.qr_dqn = qr_dqn
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.num_quantiles = num_quantiles
        # allow passing initial sigma0 for Noisy layers
        Linear = (lambda in_f, out_f: NoisyLinear(in_f, out_f, sigma0=noisy_sigma0)) if noisy else nn.Linear
        if self.c51:
            if dueling:
                self.fc_adv = Linear(linear_input, 512)
                self.out_adv = Linear(512, num_actions * num_atoms)
                self.fc_val = Linear(linear_input, 512)
                self.out_val = Linear(512, num_atoms)
            else:
                self.fc1 = Linear(linear_input, 512)
                self.out = Linear(512, num_actions * num_atoms)
        elif self.qr_dqn:
            if dueling:
                self.fc_adv = Linear(linear_input, 512)
                self.out_adv = Linear(512, num_actions * num_quantiles)
                self.fc_val = Linear(linear_input, 512)
                self.out_val = Linear(512, num_quantiles)
            else:
                self.fc1 = Linear(linear_input, 512)
                self.out = Linear(512, num_actions * num_quantiles)
        else:
            if dueling:
                self.fc_adv = Linear(linear_input, 512)
                self.out_adv = Linear(512, num_actions)
                self.fc_val = Linear(linear_input, 512)
                self.out_val = Linear(512, 1)
            else:
                self.fc1 = Linear(linear_input, 512)
                self.out = Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        if self.c51:
            if self.dueling:
                adv = F.relu(self.fc_adv(x))
                adv = self.out_adv(adv)  # (N, A*num_atoms)
                val = F.relu(self.fc_val(x))
                val = self.out_val(val)  # (N, num_atoms)
                adv = adv.view(-1, self.num_actions, self.num_atoms)
                val = val.view(-1, 1, self.num_atoms)
                adv_mean = adv.mean(dim=1, keepdim=True)
                logits = val + adv - adv_mean  # (N, A, num_atoms)
            else:
                x = F.relu(self.fc1(x))
                logits = self.out(x).view(-1, self.num_actions, self.num_atoms)
            return logits
        if self.qr_dqn:
            if self.dueling:
                adv = F.relu(self.fc_adv(x))
                adv = self.out_adv(adv)  # (N, A*num_quantiles)
                val = F.relu(self.fc_val(x))
                val = self.out_val(val)  # (N, num_quantiles)
                adv = adv.view(-1, self.num_actions, self.num_quantiles)
                val = val.view(-1, 1, self.num_quantiles)
                adv_mean = adv.mean(dim=1, keepdim=True)
                qtls = val + adv - adv_mean  # (N, A, num_quantiles)
            else:
                x = F.relu(self.fc1(x))
                qtls = self.out(x).view(-1, self.num_actions, self.num_quantiles)
            return qtls
        if self.dueling:
            adv = F.relu(self.fc_adv(x))
            adv = self.out_adv(adv)
            val = F.relu(self.fc_val(x))
            val = self.out_val(val)
            # Combine advantage and value: Q = V + A - mean(A)
            return val + adv - adv.mean(dim=1, keepdim=True)
        else:
            x = F.relu(self.fc1(x))
            return self.out(x)


@dataclass
class DQNConfig:
    in_channels: int = 4
    num_actions: int = 3
    gamma: float = 0.99
    lr: float = 1e-4
    target_update_every: int = 1000
    # Polyak averaging factor for soft target updates (0.0 disables soft updates)
    tau: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dueling: bool = True
    noisy: bool = True
    amp: bool = True
    compile: bool = False
    pin_memory: bool = False
    # Memory format / backend speed-ups
    channels_last: bool = False  # set True to enable NHWC for convs
    allow_tf32: bool = True
    # Optimizer
    optimizer: str = "adamw"  # 'adam' | 'adamw'
    weight_decay: float = 0.0
    fused_optimizer: bool = True   # try fused CUDA kernels when available
    foreach_optimizer: bool = True # prefer foreach multi-tensor kernels
    # DrQ random shift augmentation (applied in train_step only)
    drq: bool = False
    drq_pad: int = 4
    # Distributional C51
    c51: bool = False
    num_atoms: int = 51
    vmin: float = -10.0
    vmax: float = 10.0
    # Quantile Regression DQN
    qr_dqn: bool = False
    num_quantiles: int = 51
    huber_kappa: float = 1.0
    # N-step for target bootstrapping
    n_step: int = 1
    # NoisyNet annealing
    noisy_sigma_init: float = 0.5
    noisy_sigma_final: float = 0.5
    noisy_sigma_decay_steps: int = 200_000
    # Munchausen DQN options
    munchausen: bool = False
    munchausen_alpha: float = 0.9
    munchausen_tau: float = 0.03
    munchausen_clip: float = -1.0


class DQNAgent:
    def __init__(self, config: Optional[DQNConfig] = None):
        self.cfg = config or DQNConfig()
        policy = DQNNet(
            self.cfg.in_channels,
            self.cfg.num_actions,
            dueling=self.cfg.dueling,
            noisy=self.cfg.noisy,
            c51=self.cfg.c51,
            num_atoms=self.cfg.num_atoms,
            noisy_sigma0=self.cfg.noisy_sigma_init,
            qr_dqn=self.cfg.qr_dqn,
            num_quantiles=self.cfg.num_quantiles,
        )
        # Target must mirror policy architecture (including noisy/dueling/C51/QR) for state_dict compatibility
        target = DQNNet(
            self.cfg.in_channels,
            self.cfg.num_actions,
            dueling=self.cfg.dueling,
            noisy=self.cfg.noisy,
            c51=self.cfg.c51,
            num_atoms=self.cfg.num_atoms,
            noisy_sigma0=self.cfg.noisy_sigma_init,
            qr_dqn=self.cfg.qr_dqn,
            num_quantiles=self.cfg.num_quantiles,
        )
        # Optional torch.compile for speed (PyTorch 2.x)
        # Only try torch.compile on CUDA and when Triton is available; enable fallback to eager if compilation fails
        self.compiled = False
        triton_available = importlib.util.find_spec('triton') is not None
        can_compile = (
            self.cfg.compile
            and hasattr(torch, 'compile')
            and (self.cfg.device.startswith('cuda') or self.cfg.device == 'cuda')
            and triton_available
        )
        if self.cfg.compile:
            try:
                dynamo = importlib.import_module('torch._dynamo')  # type: ignore
                dynamo.config.suppress_errors = True
            except Exception:
                pass
        if can_compile:
            try:
                policy = torch.compile(policy, mode='max-autotune')
                target = torch.compile(target, mode='max-autotune')
                self.compiled = True
            except Exception:
                self.compiled = False
        # Move to device with optional channels_last memory format for 4D params
        try:
            if self.cfg.channels_last:
                self.policy = policy.to(device=self.cfg.device, memory_format=torch.channels_last)
                self.target = target.to(device=self.cfg.device, memory_format=torch.channels_last)
            else:
                self.policy = policy.to(self.cfg.device)
                self.target = target.to(self.cfg.device)
        except Exception:
            # Fallback in case memory_format kw is unsupported
            self.policy = policy.to(self.cfg.device)
            self.target = target.to(self.cfg.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        # Optimizer setup with safe fused/foreach toggles
        self.optimizer = None  # type: ignore
        opt_name = str(getattr(self.cfg, 'optimizer', 'adamw')).lower()
        fused_ok = bool(self.cfg.fused_optimizer)
        foreach_ok = bool(self.cfg.foreach_optimizer)
        opt_lr = float(self.cfg.lr)
        try:
            if opt_name == 'adamw' and hasattr(torch.optim, 'AdamW'):
                opt_kwargs: dict = { 'lr': opt_lr, 'weight_decay': float(self.cfg.weight_decay) }
                # Attach optional impl hints if supported by this torch version
                if foreach_ok:
                    opt_kwargs['foreach'] = True
                if fused_ok:
                    opt_kwargs['fused'] = True  # type: ignore[arg-type]
                self.optimizer = torch.optim.AdamW(self.policy.parameters(), **opt_kwargs)  # type: ignore[assignment]
            else:
                opt_kwargs2: dict = { 'lr': opt_lr }
                if foreach_ok:
                    opt_kwargs2['foreach'] = True
                if fused_ok:
                    opt_kwargs2['fused'] = True  # type: ignore[arg-type]
                self.optimizer = torch.optim.Adam(self.policy.parameters(), **opt_kwargs2)  # type: ignore[assignment]
        except TypeError:
            # Older torch: retry without fused/foreach
            if opt_name == 'adamw' and hasattr(torch.optim, 'AdamW'):
                self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=opt_lr, weight_decay=float(self.cfg.weight_decay))  # type: ignore[assignment]
            else:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=opt_lr)  # type: ignore[assignment]
        except Exception:
            # Absolute fallback
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=opt_lr)  # type: ignore[assignment]
        self.train_steps = 0
        # AMP scaler
        self.use_amp = self.cfg.amp and (self.cfg.device.startswith('cuda') or self.cfg.device == 'cuda')
        try:
            # New API (PyTorch 2.4+): pass device as keyword
            self.scaler = torch.amp.GradScaler(device='cuda', enabled=self.use_amp)
        except Exception:
            # Fallback for older versions
            try:
                self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            except Exception:
                class _NoopScaler:
                    def scale(self, x):
                        return x
                    def step(self, opt):
                        opt.step()
                    def update(self):
                        pass
                self.scaler = _NoopScaler()
        # Diagnostics
        self.last_grad_norm: Optional[float] = None
        self.last_amp_scale: Optional[float] = None
        self.last_q_mean: Optional[float] = None
        self.last_q_max: Optional[float] = None

    def _anneal_noisy(self):
        """Linearly anneal NoisyNet sigma toward final value across decay steps."""
        if not self.cfg.noisy:
            return
        steps = max(1, int(self.cfg.noisy_sigma_decay_steps))
        alpha = min(1.0, float(self.train_steps) / float(steps))
        # scale relative to init base
        target_ratio = self.cfg.noisy_sigma_final / max(1e-8, self.cfg.noisy_sigma_init)
        scale = (1.0 - alpha) + alpha * target_ratio
        def set_scale(module: nn.Module):
            if isinstance(module, NoisyLinear):
                module.set_noise_scale(scale)
        self.policy.apply(set_scale)
        # keep target noise matched to policy to maintain symmetry
        self.target.apply(set_scale)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.cfg.num_actions)
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0)
            if self.cfg.channels_last and s.ndim == 4:
                try:
                    s = s.to(device=self.cfg.device, memory_format=torch.channels_last)
                except Exception:
                    s = s.to(self.cfg.device)
            else:
                s = s.to(self.cfg.device)
            if self.cfg.c51:
                logits = self.policy(s)  # (1, A, atoms)
                probs = torch.softmax(logits, dim=-1)
                z = torch.linspace(self.cfg.vmin, self.cfg.vmax, self.cfg.num_atoms, device=self.cfg.device)
                q = torch.sum(probs * z.view(1, 1, -1), dim=-1)
                return int(torch.argmax(q, dim=1).item())
            elif self.cfg.qr_dqn:
                qtls = self.policy(s)  # (1, A, Q)
                q = qtls.mean(dim=-1)  # expectation across quantiles
                return int(torch.argmax(q, dim=1).item())
            else:
                q = self.policy(s)
                return int(torch.argmax(q, dim=1).item())

    def act_batch(self, states: np.ndarray, epsilon: float) -> np.ndarray:
        """States shape: (N, C, H, W). Returns actions shape: (N,). Epsilon applied per env."""
        N = states.shape[0]
        actions = np.zeros((N,), dtype=np.int64)
        with torch.no_grad():
            s = torch.from_numpy(states)
            if self.cfg.channels_last and s.ndim == 4:
                try:
                    s = s.to(device=self.cfg.device, memory_format=torch.channels_last)
                except Exception:
                    s = s.to(self.cfg.device)
            else:
                s = s.to(self.cfg.device)
            if self.cfg.c51:
                logits = self.policy(s)  # (N, A, atoms)
                probs = torch.softmax(logits, dim=-1)
                z = torch.linspace(self.cfg.vmin, self.cfg.vmax, self.cfg.num_atoms, device=self.cfg.device)
                q = torch.sum(probs * z.view(1, 1, -1), dim=-1)
            elif self.cfg.qr_dqn:
                qtls = self.policy(s)
                q = qtls.mean(dim=-1)
            else:
                q = self.policy(s)  # (N, A)
            greedy = torch.argmax(q, dim=1).cpu().numpy()
        # Epsilon randomization per env
        rand_mask = np.random.rand(N) < epsilon
        rand_actions = np.random.randint(0, self.cfg.num_actions, size=(N,))
        actions = np.where(rand_mask, rand_actions, greedy)
        return actions.astype(np.int64)

    def train_step(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], weights: Optional[np.ndarray] = None):
        states, actions, rewards, next_states, dones = batch
        device = self.cfg.device
        use_pinned = bool(self.cfg.pin_memory and device.startswith('cuda'))
        def to_dev(np_arr: np.ndarray, dtype=None):
            arr = np_arr if dtype is None else np_arr.astype(dtype)
            t = torch.from_numpy(arr)
            if use_pinned:
                t = t.pin_memory()
                return t.to(device, non_blocking=True)
            return t.to(device)
        states_t = to_dev(states)
        actions_t = to_dev(actions, np.int64)
        rewards_t = to_dev(rewards, np.float32)
        next_states_t = to_dev(next_states)
        dones_t = to_dev(dones, np.float32)
        w_t = None
        if weights is not None:
            w_t = to_dev(weights, np.float32)
        # Apply channels_last memory format for image-like tensors if requested
        if self.cfg.channels_last:
            try:
                if states_t.dim() == 4:
                    states_t = states_t.to(memory_format=torch.channels_last)
                if next_states_t.dim() == 4:
                    next_states_t = next_states_t.to(memory_format=torch.channels_last)
            except Exception:
                pass

        # DrQ random shift augmentation (image-based)
        def rand_shift(imgs: torch.Tensor, pad: int) -> torch.Tensor:
            """DrQ random shift (vectorized).
            Pads by `pad` with replicate, then extracts an HxW crop shifted by a random (dy, dx) in [0, 2*pad].
            Keeps original spatial size; fully vectorized via grid_sample with nearest sampling.
            """
            if pad <= 0:
                return imgs
            # imgs: (N,C,H,W) -> pad to (N,C,Hp,Wp)
            imgs_pad = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')
            N, C, Hp, Wp = imgs_pad.shape
            H = Hp - 2 * pad
            W = Wp - 2 * pad
            # integer shifts per sample in [0, 2*pad]
            dy = torch.randint(0, 2 * pad + 1, size=(N,), device=imgs_pad.device)
            dx = torch.randint(0, 2 * pad + 1, size=(N,), device=imgs_pad.device)
            # Build sampling grid in normalized coords for grid_sample (align_corners=True)
            ys = torch.arange(H, device=imgs_pad.device).view(1, H, 1) + dy.view(N, 1, 1)
            xs = torch.arange(W, device=imgs_pad.device).view(1, 1, W) + dx.view(N, 1, 1)
            ys = ys.expand(N, H, W)
            xs = xs.expand(N, H, W)
            # normalize to [-1,1]; align_corners=True -> divides by (size-1)
            ys_norm = (ys.float() / max(1, Hp - 1)) * 2 - 1
            xs_norm = (xs.float() / max(1, Wp - 1)) * 2 - 1
            grid = torch.stack((xs_norm, ys_norm), dim=-1)  # (N,H,W,2)
            out = F.grid_sample(
                imgs_pad, grid, mode='nearest', padding_mode='border', align_corners=True
            )
            return out

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            gamma_n = self.cfg.gamma ** max(1, int(self.cfg.n_step))
            # apply DrQ augmentation for training-time features
            use_states = states_t
            use_next_states = next_states_t
            if self.cfg.drq:
                p = int(self.cfg.drq_pad)
                use_states = rand_shift(states_t, p)
                use_next_states = rand_shift(next_states_t, p)
            # Munchausen reward augmentation precompute
            rewards_used = rewards_t
            if self.cfg.munchausen:
                if self.cfg.c51:
                    # derive Q-values from distributions for policy logits
                    logits_all = self.policy(use_states)
                    probs_all = torch.softmax(logits_all, dim=-1)
                    z = torch.linspace(self.cfg.vmin, self.cfg.vmax, self.cfg.num_atoms, device=states_t.device)
                    q_all = torch.sum(probs_all * z.view(1, 1, -1), dim=-1)  # (N, A)
                elif self.cfg.qr_dqn:
                    qtls_all = self.policy(use_states)
                    q_all = qtls_all.mean(dim=-1)
                else:
                    q_all = self.policy(use_states)
                # temperature softmax policy
                tau = max(1e-6, float(self.cfg.munchausen_tau))
                logsum = torch.logsumexp(q_all / tau, dim=1, keepdim=True)
                log_pi_all = (q_all / tau) - logsum
                log_pi_a = log_pi_all.gather(1, actions_t.view(-1, 1)).squeeze(1)
                log_pi_a = torch.clamp(log_pi_a, min=self.cfg.munchausen_clip, max=0.0)
                rewards_used = rewards_t + self.cfg.munchausen_alpha * log_pi_a
            if self.cfg.c51:
                # Current logits and probabilities for chosen actions
                logits = self.policy(use_states)  # (N, A, atoms)
                logits_a = logits.gather(1, actions_t.view(-1, 1, 1).expand(-1, 1, self.cfg.num_atoms)).squeeze(1)  # (N, atoms)
                log_probs_a = torch.log_softmax(logits_a, dim=-1)
                with torch.no_grad():
                    # Q stats from expected values
                    probs_all = torch.softmax(logits, dim=-1)
                    z_all = torch.linspace(self.cfg.vmin, self.cfg.vmax, self.cfg.num_atoms, device=states_t.device)
                    q_all_now = torch.sum(probs_all * z_all.view(1, 1, -1), dim=-1)  # (N, A)
                    self.last_q_mean = float(q_all_now.mean().item())
                    self.last_q_max = float(q_all_now.max().item())
                # Next action by online net using expected value over atoms
                with torch.no_grad():
                    next_logits_online = self.policy(use_next_states)
                    next_probs_online = torch.softmax(next_logits_online, dim=-1)
                    z = torch.linspace(self.cfg.vmin, self.cfg.vmax, self.cfg.num_atoms, device=states_t.device)
                    next_q_online = torch.sum(next_probs_online * z.view(1, 1, -1), dim=-1)  # (N, A)
                    next_acts = next_q_online.argmax(dim=1)  # (N,)
                    # Target distribution from target net for chosen next action
                    next_logits_target = self.target(use_next_states)
                    next_probs_target = torch.softmax(next_logits_target, dim=-1)
                    next_probs_a = next_probs_target.gather(1, next_acts.view(-1, 1, 1).expand(-1, 1, self.cfg.num_atoms)).squeeze(1)  # (N, atoms)
                    # Distributional projection onto support
                    tz = rewards_used.view(-1, 1) + (1.0 - dones_t.view(-1, 1)) * gamma_n * z.view(1, -1)
                    tz = torch.clamp(tz, self.cfg.vmin, self.cfg.vmax)
                    b = (tz - self.cfg.vmin) / ((self.cfg.vmax - self.cfg.vmin) / (self.cfg.num_atoms - 1))
                    l = b.floor().clamp(0, self.cfg.num_atoms - 1)
                    u = b.ceil().clamp(0, self.cfg.num_atoms - 1)
                    m = torch.zeros_like(next_probs_a)
                    l_idx = l.long()
                    u_idx = u.long()
                    m.scatter_add_(1, l_idx, next_probs_a * (u - b))
                    m.scatter_add_(1, u_idx, next_probs_a * (b - l))
                # Cross-entropy loss (forward KL): - sum m * log p
                per_sample_loss = -(m * log_probs_a).sum(dim=1)
                loss = (w_t * per_sample_loss).mean() if w_t is not None else per_sample_loss.mean()
                # TD error for PER: use expected value difference
                with torch.no_grad():
                    probs_a = torch.softmax(logits_a, dim=-1)
                    q_exp = torch.sum(probs_a * z.view(1, -1), dim=-1)
                    q_tgt = torch.sum(m * z.view(1, -1), dim=-1)
                    td_error = q_exp - q_tgt
            elif self.cfg.qr_dqn:
                # Current quantiles for chosen actions
                qtls_all = self.policy(use_states)  # (N, A, Q)
                qtls_a = qtls_all.gather(1, actions_t.view(-1, 1, 1).expand(-1, 1, self.cfg.num_quantiles)).squeeze(1)  # (N, Q)
                with torch.no_grad():
                    q_all_now = qtls_all.mean(dim=-1)  # (N, A)
                    self.last_q_mean = float(q_all_now.mean().item())
                    self.last_q_max = float(q_all_now.max().item())
                with torch.no_grad():
                    # Next action by mean of quantiles from online
                    next_qtls_online = self.policy(use_next_states)  # (N, A, Q)
                    next_q_online_mean = next_qtls_online.mean(dim=-1)  # (N, A)
                    next_acts = next_q_online_mean.argmax(dim=1)  # (N,)
                    next_qtls_target = self.target(use_next_states)
                    next_qtls_a = next_qtls_target.gather(1, next_acts.view(-1, 1, 1).expand(-1, 1, self.cfg.num_quantiles)).squeeze(1)  # (N, Q)
                    target_qtls = rewards_used.view(-1, 1) + (1.0 - dones_t.view(-1, 1)) * gamma_n * next_qtls_a  # (N, Q)
                # Quantile Huber loss
                kappa = max(1e-6, float(self.cfg.huber_kappa))
                taus = (torch.arange(self.cfg.num_quantiles, device=qtls_a.device, dtype=qtls_a.dtype) + 0.5) / self.cfg.num_quantiles  # (Q,)
                td = target_qtls.unsqueeze(1) - qtls_a.unsqueeze(2)  # (N, Q, Q)
                huber = torch.where(td.abs() <= kappa, 0.5 * td.pow(2), kappa * (td.abs() - 0.5 * kappa))  # (N,Q,Q)
                # indicator: td < 0
                inv = (td.detach() < 0.0).float()
                loss_q = (torch.abs(taus.view(1, -1, 1) - inv) * huber).mean(dim=2).sum(dim=1)  # sum over target quantiles, mean over current quantiles
                loss = (w_t * loss_q).mean() if w_t is not None else loss_q.mean()
                # TD error proxy for PER using mean difference
                with torch.no_grad():
                    q_mean = qtls_a.mean(dim=1)
                    tgt_mean = target_qtls.mean(dim=1)
                    td_error = q_mean - tgt_mean
            else:
                q_curr_all = self.policy(use_states)
                q_values = q_curr_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    self.last_q_mean = float(q_curr_all.mean().item())
                    self.last_q_max = float(q_curr_all.max().item())
                with torch.no_grad():
                    # Double DQN: online picks action, target evaluates
                    next_online = self.policy(use_next_states).argmax(1)
                    next_q = self.target(use_next_states).gather(1, next_online.unsqueeze(1)).squeeze(1)
                    target = rewards_used + (1.0 - dones_t) * gamma_n * next_q
                td_error = q_values - target
                if w_t is not None:
                    loss = (w_t * F.smooth_l1_loss(q_values, target, reduction='none')).mean()
                else:
                    loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            # clip_grad_norm_ returns the total norm of the parameters (before clipping)
            try:
                total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
                self.last_grad_norm = float(total_norm if isinstance(total_norm, float) else total_norm.item())
            except Exception:
                self.last_grad_norm = None
            self.scaler.step(self.optimizer)
            self.scaler.update()
            try:
                get_scale = getattr(self.scaler, 'get_scale', None)
                if callable(get_scale):
                    self.last_amp_scale = float(get_scale())
            except Exception:
                self.last_amp_scale = None
        else:
            loss.backward()
            try:
                total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
                self.last_grad_norm = float(total_norm if isinstance(total_norm, float) else total_norm.item())
            except Exception:
                self.last_grad_norm = None
            self.optimizer.step()
        self.train_steps += 1
        # update noisy annealing each step
        self._anneal_noisy()
        return float(loss.item()), td_error.detach().cpu().numpy()

    def maybe_update_target(self):
        # Soft update if tau>0, else periodic hard copy
        if getattr(self.cfg, 'tau', 0.0) and self.cfg.tau > 0.0:
            with torch.no_grad():
                for t_param, p_param in zip(self.target.parameters(), self.policy.parameters()):
                    t_param.data.mul_(1.0 - self.cfg.tau).add_(p_param.data, alpha=self.cfg.tau)
        elif self.train_steps % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy': self.policy.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.train_steps,
            'config': self.cfg.__dict__,
        }, path)

    def load(self, path: str, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location or self.cfg.device)
        self.policy.load_state_dict(ckpt['policy'])
        self.target.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.train_steps = int(ckpt.get('steps', 0))
