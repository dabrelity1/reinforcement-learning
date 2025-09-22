FishBot Automation (Visual Reactive Controller)
================================================

Overview
--------
This tool visually tracks a grey fish marker and a white player rectangle inside a selected game window region (ROI) and automatically simulates mouse hold/release to keep the rectangle aligned with the fish. It relies only on screen capture and synthetic mouse input (SendInput) – no game memory access or internal APIs.

Features (Implemented / Planned)
--------------------------------
- [x] Modular architecture (capture, detection, controller, input, calibration, simulator placeholder)
- [x] JSON config + CLI flags
- [ ] ROI & fish template calibration UI
- [ ] Real-time detection (template + fallback mask)
- [ ] Predictive hysteresis controller with collision mitigation
- [ ] Debug overlay with FPS & metrics
- [ ] Offline simulator harness
- [ ] Logging & metrics aggregation

Quick Start (Work in Progress)
------------------------------
1. Install dependencies:
   ````
   pip install -r requirements.txt
   ````
2. Run (dry-run, no inputs sent):
   ````
   python -m fishbot.main --dry-run --debug
   ````
3. Calibrate (will create config + prompt for ROI/template once implemented):
   ````
   python -m fishbot.main --calibrate
   ````

CLI Flags (initial)
-------------------
--config path/to/config.json  Use / save configuration
--calibrate                   Launch calibration workflow (ROI + template capture)
--debug                       Show debug overlay window
--dry-run                     Do not send real mouse input
--simulate                    Use internal simulator (offline test) (planned)
--use-dxcam                   Prefer dxcam over mss for capture

Safety & Ethics Notice
----------------------
This tool automates input by simulating a user. Usage may violate Terms of Service of some games. It is provided strictly for educational, single-player, or offline experimental purposes. You are solely responsible for ensuring legal and ethical use. Do not use in competitive or multiplayer environments.

Tuning (Preview)
----------------
Key parameters live in the JSON config and can be overridden by flags later:
- capture_fps_target (default 60)
- template_match_threshold (default 0.72)
- hysteresis_low / hysteresis_high (6 / 12 px)
- prediction_factor (0.07 sec)
- latency_compensation_default (0.02 sec)

Planned Modules
---------------
fishbot/
  config.py        (load/save config dataclasses)
  capture.py       (dxcam / mss capture pipeline)
  detection.py     (fish + rectangle detection)
  controller.py    (FSM predictive control)
  input_control.py (SendInput + fallbacks)
  calibration.py   (ROI & template selection UI)
   minigame.py      (Local visual test harness window)
  metrics.py       (rolling stats, logging integration)
  simulator.py     (physics & fish motion offline test)
  utils.py         (timing, smoothing, geometry helpers)
  main.py          (CLI entrypoint & loop orchestration)

Local Minigame Test
-------------------
An embedded standalone window simulates the mechanic so you can iterate on detection & control without a real game client.

Launch:
```
python -m fishbot.minigame --width 520 --height 140 --fps 90
```

Mechanics Implemented:
- Grey vertical target line moves across the central bar using stochastic motion regimes (pause, very slow, slow, fast, very fast) with random accelerations. Movement is continuous (no teleporting) and bounces at edges.
- White player bar has inertia: holding LMB accelerates right, releasing accelerates left. Velocity is damped by drag and bounces with restitution scaled by impact speed.
- Progress bar (top) fills ONLY while the white bar horizontally overlaps the grey line.
- Win State: When progress reaches 100%, a RESTART button appears (and 'R' key works). Restart regenerates a brand-new random motion pattern (new seed) and resets progress.
- Optional low-level noise speckles. Occlusion currently disabled by default for clarity (can be re-enabled in code/config if you want harder detection).

Controls:
- Hold Left Mouse: accelerate right
- Release: accelerate left (auto)
- R (after completion) or click RESTART button: start new pattern
- ESC / q: Quit

Reproducible Motion:
```
python -m fishbot.minigame --seed 12345
```

Calibration Workflow with Minigame:
1. Position minigame window where you want it to stay.
2. Run (optional ROI print): `python -m fishbot.minigame --show-roi`
3. Calibrate bot (choose one):
   - Visual ROI selection: `python -m fishbot.main --calibrate`
   - Mouse coordinate calibration: `python -m fishbot.main --calibrate --mouse-calib`
4. Test control overlay: `python -m fishbot.main --debug --dry-run`
5. Run active control (careful: sends input): `python -m fishbot.main --debug`

Detection Hint:
You can target either the white bar and grey line contrast or adapt template matching around the grey line segment. The line is only a few pixels wide; consider expanding search bands or using edge detection + vertical aggregation.

Tuning Ideas:
- Increase stochastic difficulty: raise `line_accel_fast` / `line_speed_cap` inside `minigame.py`.
- Harder tracking: re-enable occlusion probability in config dataclass.
- Narrow tolerance: reduce player bar width (`rect_w`).

Example Combined Run (two terminals):
1. `python -m fishbot.minigame --fps 90 --seed 999`
2. `python -m fishbot.main --debug --dry-run`

Restart cycles will change motion seed automatically; your controller robustness can be evaluated over multiple regenerations.

Launcher GUI (One-Click Flow)
-----------------------------
A simple tkinter launcher is provided to start both the fullscreen minigame and the bot (line detection mode) with chosen parameters.

Start the launcher:
```
python -m fishbot.launcher
```
Options:
- Seed (blank = random each run)
- Rect Accel / Rect Drag (player bar physics overrides)
- Fullscreen toggle
- Debug Overlay (passes --debug to bot)
- Dry Run (prevents real input events)
- Bot Start Delay (seconds between spawning minigame and bot)

Workflow:
1. Click Launch: minigame appears (fullscreen if selected).
2. After delay, bot starts in --line-mode reading the moving grey line.
3. Calibrate once beforehand if needed (mouse-calib or ROI). Line mode ignores template path.
4. Stop terminates both processes; Restart by pressing Launch again.

Note: The bot in line mode uses the vertical grey line; ensure the capture ROI covers the play bar region or run calibration with the fullscreen window visible.

License
-------
No license specified yet (default: All rights reserved). Add a license if you plan to distribute.
