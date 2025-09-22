"""Main CLI entrypoint for FishBot."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import sys

import cv2
import numpy as np

from .config import load_config, save_config, BotConfig
from .capture import ScreenCapture
from .detection import FishDetector, RectDetector, LineDetector
from .controller import Controller
from .input_control import MouseController
from .utils import EWMAVelocity, FPSCounter
from .metrics import LoopMetrics
from .calibration import live_select_roi_with_freeze, select_roi, capture_template, mouse_coordinate_calibration
from .logging_setup import setup_logging
from .simulator import Simulator


log = logging.getLogger("fishbot")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FishBot visual controller")
    p.add_argument("--config", type=str, help="Path to config JSON", default=None)
    p.add_argument("--calibrate", action="store_true", help="Run calibration workflow")
    p.add_argument("--mouse-calib", action="store_true", help="Use mouse position based calibration (no UI windows)")
    p.add_argument("--debug", action="store_true", help="Enable debug overlay & verbose logging")
    p.add_argument("--dry-run", action="store_true", help="Do not send real inputs")
    p.add_argument("--simulate", action="store_true", help="Run simulator instead of live capture (WIP)")
    p.add_argument("--use-dxcam", action="store_true", help="Prefer dxcam backend")
    p.add_argument("--monitor", type=int, default=None, help="Monitor index for calibration (live mode)")
    p.add_argument("--line-mode", action="store_true", help="Use vertical grey line detection instead of fish template")
    p.add_argument("--auto-minigame", action="store_true", help="Auto-detect minigame window / fullscreen and ignore saved ROI (no recalibration)")
    p.add_argument("--headless-debug", action="store_true", help="Do not open any OpenCV window; print state info to log/console")
    p.add_argument("--trace-detect", action="store_true", help="Verbose per-frame detection logging (headless recommended)")
    return p.parse_args(argv)


def run_calibration(cfg: BotConfig, args):
    monitor_index = args.monitor if args.monitor is not None else 1
    log.info(f"Starting live calibration (monitor {monitor_index}) ...")
    if args.mouse_calib:
        log.info("Using mouse coordinate calibration mode.")
        try:
            roi = mouse_coordinate_calibration()
        except RuntimeError as e:
            log.error(f"Calibration aborted: {e}")
            return
    else:
        try:
            roi = live_select_roi_with_freeze(monitor_index)
        except RuntimeError as e:
            log.error(f"Calibration aborted: {e}")
            return
    cfg.capture.roi = roi
    log.info(f"ROI selected: {roi}")
    from mss import mss  # type: ignore
    import numpy as np
    x,y,w,h = roi
    with mss() as sct:
        mon_roi = {"left": x, "top": y, "width": w, "height": h}
        frame = np.array(sct.grab(mon_roi))[:, :, :3]
    template = capture_template(frame)
    if template is not None:
        templ_path = Path("fish_template.png")
        cv2.imwrite(str(templ_path), template)
        cfg.detection.template_path = str(templ_path)
        log.info(f"Saved template to {templ_path}")
    save_config(cfg, args.config)
    log.info("Calibration complete.")


def load_template(cfg: BotConfig):
    if not cfg.detection.template_path:
        return None
    import cv2
    p = Path(cfg.detection.template_path)
    if not p.exists():
        return None
    t = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return t


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.debug:
        cfg.debug.debug = True
        cfg.debug.show_overlay = True
        cfg.debug.log_level = "DEBUG"
    if args.headless_debug:
        cfg.debug.debug = True
        cfg.debug.show_overlay = False  # force off
    if args.dry_run:
        cfg.debug.dry_run = True
    if args.use_dxcam:
        cfg.capture.use_dxcam = True
    setup_logging(cfg.debug)
    log.info("FishBot starting")
    if args.calibrate:
        run_calibration(cfg, args)
        return
    simulate = args.simulate and cfg.simulator.enabled or args.simulate
    if simulate:
        cfg.simulator.enabled = True
        # Override ROI to simulator dimensions
        cfg.capture.roi = (0, 0, cfg.simulator.width, 120)  # fixed small height for synthetic frame
        log.info("Running in simulator mode.")

    # Auto-minigame ROI override: locate the minigame window dynamically so user doesn't need calibration.
    if args.auto_minigame and args.line_mode and not simulate:
        try:
            # Attempt to find the OpenCV window by title used in minigame
            win_title = "FishBot Test Minigame"
            import ctypes
            import ctypes.wintypes as wt
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            FindWindowW = user32.FindWindowW
            FindWindowW.argtypes = [wt.LPCWSTR, wt.LPCWSTR]
            FindWindowW.restype = wt.HWND
            hwnd = FindWindowW(None, win_title)
            if hwnd:
                rect = wt.RECT()
                if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    x, y = rect.left, rect.top
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    # Heuristic: shrink a little to exclude title bar if present
                    if h > 80:
                        # Assume title bar ~30px; OpenCV adds borders sometimes, be conservative
                        adj = 30
                        y += adj
                        h -= adj
                    cfg.capture.roi = (x, y, w, h)
                    log.info(f"Auto-minigame ROI set to {cfg.capture.roi}")
                else:
                    raise RuntimeError("GetWindowRect failed")
            else:
                # fallback: full primary screen via tkinter
                try:
                    import tkinter as tk
                    r = tk.Tk(); r.withdraw()
                    sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
                    r.destroy()
                except Exception:
                    sw, sh = 1920, 1080
                cfg.capture.roi = (0, 0, sw, sh)
                log.warning("Minigame window not found; using full screen as ROI")
        except Exception as e:
            log.error(f"Auto-minigame ROI detection failed: {e}. Keeping existing ROI {cfg.capture.roi}")

    roi = cfg.capture.roi
    if not simulate:
        cap = ScreenCapture(roi, cfg.capture.use_dxcam, cfg.capture.downscale)
    else:
        cap = None
        simulator = Simulator(cfg.simulator)
    template = load_template(cfg)
    if args.line_mode:
        cfg.detection.line_mode = True
    fish_det = LineDetector(cfg.detection) if cfg.detection.line_mode else FishDetector(template, cfg.detection)
    rect_det = RectDetector(cfg.detection)
    # Boost grace frames in line mode for smoother continuity
    if cfg.detection.line_mode and cfg.detection.lost_fish_grace_frames < 15:
        cfg.detection.lost_fish_grace_frames = 15
    mouse = MouseController(dry_run=cfg.debug.dry_run or simulate)  # simulator shouldn't press real inputs
    controller = Controller(cfg.control, mouse, roi_width=int(roi[2]*cfg.capture.downscale))
    fish_tracker = EWMAVelocity(cfg.control.ewma_alpha_pos, cfg.control.ewma_alpha_vel)
    rect_tracker = EWMAVelocity(cfg.control.ewma_alpha_pos, cfg.control.ewma_alpha_vel)
    fps_counter = FPSCounter()
    metrics = LoopMetrics()
    last_metrics_log = time.perf_counter()
    prev_time = time.perf_counter()
    headless_frame_counter = 0
    while True:
        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if simulate:
            state = simulator.step(mouse.is_down(), dt if 0 < dt < 0.2 else 1/60)
            # Build synthetic frame (grayscale layering)
            w = cfg.simulator.width
            h = 120
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # fish: gray circle
            cv2.circle(frame, (int(state.fish_x), h//2), 6, (128,128,128), -1)
            # rect: white rectangle
            rect_w, rect_h = 30, 18
            rx = int(max(0, min(w-rect_w, state.rect_x - rect_w/2)))
            ry = h//2 - rect_h//2
            cv2.rectangle(frame, (rx, ry), (rx+rect_w, ry+rect_h), (255,255,255), -1)
        else:
            frame = cap.grab()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fish_x, fish_conf = fish_det.detect(gray)
        rect_x, rect_box = rect_det.detect(gray)
        if fish_x is None or rect_x is None:
            # Attempt grace reuse: if line mode & fish recently seen, reuse last known position for limited frames
            reuse_ok = False
            if cfg.detection.line_mode and fish_det.last_x is not None:
                # Accept reuse if consecutive misses below grace threshold
                if metrics.lost_fish_frames < cfg.detection.lost_fish_grace_frames:
                    fish_x = fish_det.last_x
                    fish_conf = (int(fish_x), (int(fish_x)-1, 0, 3, gray.shape[0]))  # synthetic box
                    reuse_ok = True
            if not reuse_ok:
                metrics.lost_fish_frames += 1
                if args.trace_detect and metrics.lost_fish_frames % 5 == 0:
                    log.warning(f"DETECT MISS fish={fish_x} rect={rect_x} lost_frames={metrics.lost_fish_frames}")
                if cfg.debug.show_overlay:
                    cv2.imshow("FishBot", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                continue
        # Normalize fish_conf only after confirming detection
        if isinstance(fish_conf, tuple):  # line mode bounding box
            fish_box = fish_conf
            fish_conf_val = 1.0
        else:
            fish_box = None
            fish_conf_val = float(fish_conf) if fish_conf is not None else 0.0
        fish_pos, fish_vel = fish_tracker.update(fish_x)
        # Guard: rect_x could be None if detector missed (even when fish reused); reuse last known or skip
        if rect_x is None:
            # If we have a previous position, keep it with zero vel; else skip
            if rect_tracker.last_pos is not None:
                rect_pos, rect_vel = rect_tracker.last_pos, 0.0
            else:
                if args.trace_detect:
                    log.debug("RECT MISS no previous pos; skipping frame")
                continue
        else:
            rect_pos, rect_vel = rect_tracker.update(rect_x)
        ctrl_info = controller.step(fish_pos, fish_vel, rect_pos, rect_vel)
        metrics.frames += 1
        metrics.update_error(ctrl_info["error"])  # using raw error for now
        if ctrl_info.get("toggled"):
            metrics.toggles += 1
            if args.headless_debug:
                log.info(f"TOGGLE action -> {'DOWN' if mouse.is_down() else 'UP'} err={ctrl_info['error']:.2f} state={ctrl_info['state']}")
        fps_counter.tick()
        headless_frame_counter += 1

        if cfg.debug.show_overlay or simulate:
            if not args.headless_debug:
                overlay = frame.copy()
                cx = int(fish_pos)
                cv2.circle(overlay, (cx, overlay.shape[0]//2), 6, (0,255,255), 2)
                if rect_box:
                    x,y,w_box,h_box = rect_box
                    cv2.rectangle(overlay, (x,y), (x+w_box,y+h_box), (255,255,255), 1)
                cv2.line(overlay, (int(ctrl_info["fish_pred"]), 0), (int(ctrl_info["fish_pred"]), overlay.shape[0]), (0,255,0), 1)
                text = f"FPS:{fps_counter.fps():.1f} State:{ctrl_info['state']} Err:{ctrl_info['error']:.1f} FishConf:{fish_conf_val:.2f} Tog:{metrics.toggles}"
                cv2.putText(overlay, text, (8,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)
                cv2.imshow("FishBot", overlay)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        if args.headless_debug:
            # Print concise state line every other frame (simple throttle)
            if headless_frame_counter % 2 == 0:
                log.info(f"STATE pos_f={fish_pos:.1f} pos_r={rect_pos:.1f} err={ctrl_info['error']:.1f} state={ctrl_info['state']} toggles={metrics.toggles} pred={ctrl_info['fish_pred']:.1f}")
            if args.trace_detect and headless_frame_counter % 5 == 0:
                log.debug(f"TRACE fish_x={fish_x} rect_x={rect_x} err={ctrl_info['error']:.2f} pred={ctrl_info['fish_pred']:.2f}")

        if now - last_metrics_log >= cfg.debug.metrics_interval_sec:
            log.info(f"Metrics: {metrics.summary()}")
            last_metrics_log = now
    if not args.headless_debug:
        cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
    main()
