"""Local visual minigame for testing FishBot detection & control.

Updated version implements requested mechanics:
1. Grey target line that moves unpredictably across a main bar (no teleporting) using stochastic motion regimes
     (very fast, fast, slow, very slow, pause) with random accelerations inside each regime.
2. White player bar (with inertia, drag, directional acceleration on mouse hold) must stay overlapping the grey line.
3. Dynamic bounce physics: bounce restitution scales with impact speed (higher speed -> stronger rebound within cap).
4. Progress bar: fills only while the white bar overlaps the grey line. When full, win state + Restart button appear.
5. Restart regenerates a new random motion pattern (fresh RNG seed) and resets progress.
6. Motion remains continuous (no teleport), acceleration can spike in fast regimes (unpredictable).

Usage:
    python -m fishbot.minigame --width 520 --height 140 --show-roi

Controls:
    Hold Left Mouse  : Accelerate right (release accelerates left) – intended for AI simulation of press/release.
    ESC or q         : Quit
    R (after win)    : Restart
    Click RESTART    : Restart when button visible

Calibration: Select the window ROI or use mouse-based calibration. The moving grey line + white bar are mid-window.
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Tuple, Dict
import cv2
import numpy as np


@dataclass
class MiniGameConfig:
    width: int = 520
    height: int = 140
    rect_w: int = 80                     # widened default white bar
    rect_h: int = 22
    rect_accel: float = 520.0            # Increased acceleration magnitude for snappier response
    rect_drag: float = 0.95              # Higher multiplier = less decay per frame (was 0.90)
    wall_restitution_min: float = 0.20   # Min bounce factor
    wall_restitution_max: float = 0.65   # Max bounce factor (applied at high speed)
    wall_restitution_speed_ref: float = 420.0  # Speed that maps to max restitution

    # Target grey line motion (stochastic regimes)
    line_speed_cap: float = 760.0        # Increased absolute maximum speed (px/s) for snappier motion
    regime_min_time: float = 0.18
    regime_max_time: float = 1.25
    regime_pause_prob: float = 0.12
    regime_slow_prob: float = 0.22
    regime_fast_prob: float = 0.28       # Remaining probability distributed to very fast / very slow automatically
    line_accel_fast: float = 1250.0      # Increased acceleration values for higher agility
    line_accel_slow: float = 220.0
    line_accel_pause_jitter: float = 40.0
    line_width: int = 3

    # Progress logic
    progress_fill_rate: float = 0.23     # Progress per second while overlapping
    progress_required: float = 1.0       # Completion threshold

    # Visual / noise config (kept for realism)
    occlusion_prob: float = 0.00         # Disabled for clarity with line (can set >0 for stress tests)
    occlusion_duration: float = 0.4
    noise_prob: float = 0.004
    fps: int = 90
    theme_dark: bool = True
    seed: int | None = None

    def choose_seed(self):
        if self.seed is None:
            self.seed = random.randrange(1, 10_000_000)
        random.seed(self.seed)


class TargetLine:
    """Grey vertical line that moves horizontally with stochastic acceleration regimes."""

    def __init__(self, cfg: MiniGameConfig):
        self.cfg = cfg
        self.x = cfg.width * 0.3
        self.v = random.uniform(-cfg.line_speed_cap * 0.3, cfg.line_speed_cap * 0.3)
        self.regime_t = 0.0
        self.regime_kind = "init"
        self._pick_regime()

    def _pick_regime(self):
        cfg = self.cfg
        self.regime_t = random.uniform(cfg.regime_min_time, cfg.regime_max_time)
        r = random.random()
        # Determine regime category probabilities
        pause_p = cfg.regime_pause_prob
        slow_p = pause_p + cfg.regime_slow_prob
        fast_p = slow_p + cfg.regime_fast_prob
        if r < pause_p:
            self.regime_kind = "pause"
        elif r < slow_p:
            self.regime_kind = random.choice(["very_slow", "slow"])  # add nuance
        elif r < fast_p:
            self.regime_kind = random.choice(["fast", "fast", "very_fast"])  # bias toward fast
        else:
            # remaining probability space -> extreme ends
            self.regime_kind = random.choice(["very_fast", "very_slow"])

    def update(self, dt: float):
        cfg = self.cfg
        self.regime_t -= dt
        if self.regime_t <= 0:
            self._pick_regime()

        # Apply acceleration depending on regime
        kind = self.regime_kind
        ax = 0.0
        if kind == "pause":
            # Gentle jitter around stillness
            ax = random.uniform(-cfg.line_accel_pause_jitter, cfg.line_accel_pause_jitter)
            # Dampen velocity toward zero
            self.v *= 0.85
        elif kind in ("very_slow", "slow"):
            mag = cfg.line_accel_slow * (0.35 if kind == "very_slow" else 1.0)
            ax = random.uniform(-mag, mag)
        elif kind in ("fast", "very_fast"):
            mag = cfg.line_accel_fast * (1.0 if kind == "fast" else 1.8)
            ax = random.uniform(-mag, mag)
        # integrate
        self.v += ax * dt
        # speed clamp (soft) -> scale if exceeding cap
        cap = cfg.line_speed_cap * (1.0 if kind != "very_fast" else 1.15)
        spd = abs(self.v)
        if spd > cap:
            self.v *= cap / (spd + 1e-6)
        self.x += self.v * dt

        # Boundary bounce (reflect) - keep continuous, no teleport
        if self.x < 0:
            self.x = 0
            self.v = abs(self.v) * (0.65 + random.random() * 0.25)
        elif self.x > cfg.width:
            self.x = cfg.width
            self.v = -abs(self.v) * (0.65 + random.random() * 0.25)


class PlayerRect:
    def __init__(self, cfg: MiniGameConfig):
        self.cfg = cfg
        self.x = cfg.width * 0.55
        self.v = 0.0
        self.holding = False

    def update(self, dt: float):
        # Direction: holding -> accelerate right, else left
        accel = self.cfg.rect_accel if self.holding else -self.cfg.rect_accel
        self.v += accel * dt
        self.v *= self.cfg.rect_drag
        self.x += self.v * dt
        # Clamp extreme speeds to avoid instability
        max_speed = self.cfg.wall_restitution_speed_ref * 1.4
        if abs(self.v) > max_speed:
            self.v = max_speed if self.v > 0 else -max_speed
        # Boundary bounce with dynamic restitution
        if self.x < 0:
            overshoot = -self.x
            self.x = 0
            speed = abs(self.v)
            lerp = min(1.0, speed / self.cfg.wall_restitution_speed_ref)
            restitution = (self.cfg.wall_restitution_min + (self.cfg.wall_restitution_max - self.cfg.wall_restitution_min) * lerp)
            self.v = abs(self.v) * restitution
            # small positional push if large overshoot
            if overshoot > 2:
                self.x += min(2, overshoot * 0.2)
        elif self.x > self.cfg.width:
            overshoot = self.x - self.cfg.width
            self.x = self.cfg.width
            speed = abs(self.v)
            lerp = min(1.0, speed / self.cfg.wall_restitution_speed_ref)
            restitution = (self.cfg.wall_restitution_min + (self.cfg.wall_restitution_max - self.cfg.wall_restitution_min) * lerp)
            self.v = -abs(self.v) * restitution
            if overshoot > 2:
                self.x -= min(2, overshoot * 0.2)


def overlap_progress(rect_x: float, rect_w: int, line_x: float) -> bool:
    half = rect_w / 2.0
    return (rect_x - half) <= line_x <= (rect_x + half)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=520)
    p.add_argument("--height", type=int, default=140)
    p.add_argument("--fps", type=int, default=90)
    p.add_argument("--seed", type=int, default=None, help="Deterministic seed for reproducible motion pattern")
    p.add_argument("--theme-light", action="store_true", help="Use light background instead of dark")
    p.add_argument("--rect-accel", type=float, default=None, help="Override player rect acceleration (px/s^2)")
    p.add_argument("--rect-drag", type=float, default=None, help="Override player rect drag multiplier (0-1, higher=less decay)")
    p.add_argument("--rect-width", type=int, default=None, help="Override player rect width (px)")
    p.add_argument("--line-speed-mult", type=float, default=1.0, help="Multiply all line motion speeds/accels by this factor")
    p.add_argument("--bar-width-mult", type=float, default=1.0, help="Multiply base white bar width after other overrides")
    p.add_argument("--auto-scale", action="store_true", help="Auto scale speeds / bar size based on window width")
    p.add_argument("--show-roi", action="store_true", help="Print window ROI coords for calibration")
    p.add_argument("--fullscreen", action="store_true", help="Fullscreen 1920x1080 (uses actual screen size if available)")
    # Integrated bot spawning options
    p.add_argument("--spawn-bot", action="store_true", help="Spawn fishbot bot process (line-mode auto-minigame) after launching minigame")
    p.add_argument("--bot-delay", type=float, default=0.8, help="Delay before spawning bot (s)")
    p.add_argument("--bot-headless", action="store_true", help="Run bot headless (no overlay window)")
    p.add_argument("--bot-dry-run", action="store_true", help="Run bot in dry-run (no real mouse input)")
    p.add_argument("--bot-debug", action="store_true", help="Enable bot debug overlay (overrides headless if both set)")
    p.add_argument("--bot-trace-detect", action="store_true", help="Enable verbose per-frame trace detection in bot")
    p.add_argument("--bot-log-level", type=str, default=None, help="Override bot log level (e.g. DEBUG, INFO)")
    # Global mouse polling (useful when bot generates system clicks outside window callback path)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--poll-mouse", action="store_true", help="Poll global left mouse each frame (GetAsyncKeyState)")
    g.add_argument("--no-poll-mouse", action="store_true", help="Disable global polling even if spawn-bot active")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fullscreen:
        # Try to get real screen size via tkinter
        try:
            import tkinter as tk  # type: ignore
            r = tk.Tk(); r.withdraw()
            args.width, args.height = r.winfo_screenwidth(), r.winfo_screenheight()
            r.destroy()
        except Exception:
            args.width, args.height = 1920, 1080
    cfg = MiniGameConfig(width=args.width, height=args.height, fps=args.fps, theme_dark=not args.theme_light, seed=args.seed)
    # Apply CLI overrides for player physics if provided
    if args.rect_accel is not None:
        cfg.rect_accel = args.rect_accel
    if args.rect_drag is not None:
        cfg.rect_drag = max(0.0, min(0.9999, args.rect_drag))
    if args.rect_width is not None and args.rect_width > 4:
        cfg.rect_w = int(args.rect_width)

    # Auto scaling for large fullscreen resolutions
    if args.auto_scale:
        scale_w = cfg.width / 520.0  # baseline width
        # Increase speed and acceleration roughly proportional (clamped)
        width_factor = min(4.0, max(1.0, scale_w))
        cfg.line_speed_cap *= width_factor * 1.1
        cfg.line_accel_fast *= width_factor * 1.15
        cfg.line_accel_slow *= max(1.0, width_factor * 0.9)
        cfg.rect_accel *= width_factor * 1.05
        cfg.rect_w = int(cfg.rect_w * min(width_factor, 3.0))
        # Slightly reduce drag for larger widths (more responsive)
        cfg.rect_drag = 1.0 - (1.0 - cfg.rect_drag) / width_factor

    # Apply line speed multiplier uniformly
    if args.line_speed_mult != 1.0:
        m = args.line_speed_mult
        cfg.line_speed_cap *= m
        cfg.line_accel_fast *= m
        cfg.line_accel_slow *= m
        cfg.line_accel_pause_jitter *= m
    if args.bar_width_mult != 1.0:
        cfg.rect_w = int(cfg.rect_w * args.bar_width_mult)
    cfg.choose_seed()

    rect = PlayerRect(cfg)
    line = TargetLine(cfg)

    bot_proc = None
    bot_spawned = False
    import subprocess, sys, threading

    def spawn_bot_later():
        nonlocal bot_proc, bot_spawned
        delay = max(0.0, args.bot_delay)
        import time as _t
        _t.sleep(delay)
        if not args.spawn_bot:
            return
        # Build bot command
        bot_cmd = [sys.executable or "python", "-m", "fishbot.main", "--line-mode", "--auto-minigame"]
        if args.bot_headless and not args.bot_debug:
            bot_cmd.append("--headless-debug")
        if args.bot_debug:
            bot_cmd.append("--debug")
        if args.bot_dry_run:
            bot_cmd.append("--dry-run")
        if args.bot_trace_detect:
            bot_cmd.append("--trace-detect")
        # Allow forcing log level by writing a small env var or argument (simpler: adjust config log level via env)
        if args.bot_log_level:
            # Provide a generic environment variable consumed by logging_setup (not implemented yet) -> fallback: print instruction
            import os
            os.environ["FISHBOT_LOG_LEVEL_OVERRIDE"] = args.bot_log_level.upper()
        print(f"[Minigame] Spawning bot: {' '.join(bot_cmd)}")
        try:
            bot_proc = subprocess.Popen(bot_cmd)
            bot_spawned = True
        except Exception as e:
            print(f"[Minigame] Failed to spawn bot: {e}")

    if args.spawn_bot:
        threading.Thread(target=spawn_bot_later, daemon=True).start()

    # Auto-enable polling if spawning bot unless explicitly disabled
    if args.spawn_bot and not args.no_poll_mouse:
        args.poll_mouse = True

    # Windows global mouse state polling setup
    poll_mouse = args.poll_mouse
    get_async = None
    if poll_mouse:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            get_async = user32.GetAsyncKeyState
            print("[Miniggame] Global mouse polling enabled")
        except Exception as e:
            print(f"[Minigame] Failed to enable global mouse polling: {e}")
            poll_mouse = False

    win_name = "FishBot Test Minigame"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, cfg.width, cfg.height)
    if args.fullscreen:
        try:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
    if args.show_roi:
        print("Minigame ROI (window client area). Calibrate by selecting the full window content.")

    # Mouse handling state
    state: Dict[str, bool] = {"holding": False, "click": False}
    mouse_pos: Tuple[int, int] = (0, 0)

    def on_mouse(event, x, y, flags, param):  # noqa: N803 (OpenCV callback signature)
        nonlocal mouse_pos
        mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["holding"] = True
            state["click"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            state["holding"] = False

    cv2.setMouseCallback(win_name, on_mouse)

    last_time = time.perf_counter()
    target_frame_time = 1.0 / cfg.fps
    frame_accum = 0
    t0 = last_time
    font = cv2.FONT_HERSHEY_SIMPLEX

    progress = 0.0
    win = False
    restart_button_rect = None  # (x1,y1,x2,y2)

    # For potential occlusion logic (optional off by default now)
    occl_timer = 0.0
    occl_active = False

    while True:
        # Heartbeat check: if bot was spawned and exited early, notify once
        if bot_spawned and bot_proc and bot_proc.poll() is not None and not win:
            print("[Minigame] Notice: Bot process exited (code", bot_proc.returncode, ")")
            bot_spawned = False
        now = time.perf_counter()
        dt = now - last_time
        if dt < target_frame_time:
            time.sleep(max(0, target_frame_time - dt) * 0.55)
            now = time.perf_counter()
            dt = now - last_time
        last_time = now
        # Clamp dt (avoid physics explode on window drag)
        if dt > 0.05:
            dt = 0.05

        # Update input state
        if poll_mouse and get_async is not None:
            try:
                # 0x01 VK_LBUTTON
                down = bool(get_async(0x01) & 0x8000)
                state["holding"] = down
            except Exception:
                pass
        rect.holding = state["holding"] and not win

        if not win:
            rect.update(dt)
            line.update(dt)

        # Occlusion (if enabled)
        if occl_active:
            occl_timer -= dt
            if occl_timer <= 0:
                occl_active = False
        else:
            if not win and random.random() < cfg.occlusion_prob:
                occl_active = True
                occl_timer = cfg.occlusion_duration

        # Progress accumulation
        if not win and overlap_progress(rect.x, cfg.rect_w, line.x):
            progress += cfg.progress_fill_rate * dt
            if progress >= cfg.progress_required:
                progress = cfg.progress_required
                win = True

        bg_color = (10, 10, 14) if cfg.theme_dark else (240, 240, 240)
        # Gradient background for higher resolution aesthetics
        frame = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
        top_col = np.array(bg_color, dtype=np.float32)
        bottom_col = np.array((bg_color[0]+26, bg_color[1]+26, bg_color[2]+30), dtype=np.float32)
        alphas = np.linspace(0, 1, cfg.height, dtype=np.float32)[:, None]
        grad = (top_col * (1 - alphas) + bottom_col * alphas).astype(np.uint8)  # shape (H,3)
        grad = grad[:, None, :]  # shape (H,1,3)
        frame[:] = grad  # broadcast across width

        # Noise speckles (low density for realism)
        if random.random() < cfg.noise_prob:
            for _ in range(random.randint(2, 12)):
                px = random.randint(0, cfg.width - 1)
                py = random.randint(0, cfg.height - 1)
                frame[py, px] = (255, 255, 255) if random.random() < 0.5 else (80, 80, 80)

        mid_y = cfg.height // 2
        bar_h = max(28, int(cfg.rect_h * 1.3))
        bar_top = mid_y - bar_h // 2
        bar_bottom = bar_top + bar_h
        # Draw main bar background (darker stripe)
        stripe_color = (bg_color[0] + 8, bg_color[1] + 8, bg_color[2] + 8)
        cv2.rectangle(frame, (0, bar_top), (cfg.width, bar_bottom), stripe_color, -1)

        # Player white rectangle with subtle outline/glow effect first (so target line can render over it)
        rx = int(rect.x - cfg.rect_w / 2)
        ry = mid_y - cfg.rect_h // 2
        cv2.rectangle(frame, (rx-2, ry-2), (rx + cfg.rect_w+2, ry + cfg.rect_h+2), (255,255,255), 1)
        cv2.rectangle(frame, (rx, ry), (rx + cfg.rect_w, ry + cfg.rect_h), (255, 255, 255), -1)

        # Draw grey target line AFTER player so it is always visible; if overlapping, draw darker border
        if not occl_active:
            lx = int(round(line.x))
            line_left = lx - cfg.line_width // 2
            line_right = lx + cfg.line_width // 2
            # Base fill
            cv2.rectangle(frame, (line_left, bar_top), (line_right, bar_bottom), (128, 128, 128), -1)
            # If overlapping white rect, add a contrasting slim border to emphasize center
            if not (line_right < rx or line_left > rx + cfg.rect_w):
                cv2.rectangle(frame, (line_left, bar_top), (line_right, bar_bottom), (90, 90, 90), 1)
        else:
            cv2.rectangle(frame, (0, bar_top), (cfg.width, bar_bottom), (bg_color[0] + 4,) * 3, -1)

        # Progress bar (top)
        p_outer_h = 12
        p_margin = 4
        prog_w = cfg.width - p_margin * 2
        filled_w = int(prog_w * (progress / cfg.progress_required))
        outer_tl = (p_margin, p_margin)
        outer_br = (p_margin + prog_w, p_margin + p_outer_h)
        cv2.rectangle(frame, outer_tl, outer_br, (70, 70, 70), 1)
        if filled_w > 0:
            cv2.rectangle(frame, (p_margin + 1, p_margin + 1), (p_margin + filled_w - 1, p_margin + p_outer_h - 1), (255, 255, 255), -1)

        # HUD metrics
        frame_accum += 1
        elapsed = now - t0
        if elapsed >= 1.0:
            fps = frame_accum / elapsed
            frame_accum = 0
            t0 = now
        else:
            fps = 0
        hud = f"Hold=LMB  RectX={rect.x:.1f} LineX={line.x:.1f} v={rect.v:.1f} prog={progress*100:.0f}% fps~{fps:.0f}"
        cv2.putText(frame, hud, (12, cfg.height - 14), font, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        # Win state & restart button
        if win:
            msg = "COMPLETED!" if progress >= cfg.progress_required else "WIN!"
            cv2.putText(frame, msg, (cfg.width // 2 - 70, bar_top - 10), font, 0.7, (0, 240, 255), 2, cv2.LINE_AA)
            btn_w, btn_h = 140, 30
            btn_x = cfg.width // 2 - btn_w // 2
            btn_y = bar_bottom + 8
            restart_button_rect = (btn_x, btn_y, btn_x + btn_w, btn_y + btn_h)
            cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 255), 2)
            cv2.putText(frame, "RESTART (R)", (btn_x + 10, btn_y + 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if win and key in (ord('r'), ord('R')):
            # Restart
            cfg.seed = random.randrange(1, 10_000_000)
            random.seed(cfg.seed)
            line = TargetLine(cfg)
            rect = PlayerRect(cfg)
            progress = 0.0
            win = False
            restart_button_rect = None
            continue

        # Mouse click restart
        if win and state["click"] and restart_button_rect is not None:
            x1, y1, x2, y2 = restart_button_rect
            mx, my = mouse_pos
            if x1 <= mx <= x2 and y1 <= my <= y2:
                cfg.seed = random.randrange(1, 10_000_000)
                random.seed(cfg.seed)
                line = TargetLine(cfg)
                rect = PlayerRect(cfg)
                progress = 0.0
                win = False
                restart_button_rect = None
        # Reset click flag
        state["click"] = False

    cv2.destroyAllWindows()
    # Terminate bot if spawned
    if bot_proc and bot_proc.poll() is None:
        try:
            bot_proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
