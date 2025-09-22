"""Calibration UI (ROI & template capture).

Provides two approaches:
1. live_select_roi_with_freeze: live monitor feed with 'f' to freeze frame, hides calibration window during capture to avoid recursive window-in-window effect.
2. select_roi: simple static screenshot fallback.
"""
from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np
from mss import mss  # type: ignore

try:  # optional window hiding for recursion avoidance (Windows)
    import win32gui  # type: ignore
    import win32con  # type: ignore
except Exception:  # pragma: no cover
    win32gui = None  # type: ignore
    win32con = None  # type: ignore


def _grab_monitor(monitor_index: int):
    with mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            monitor_index = 1
        mon = monitors[monitor_index]
        frame = np.array(sct.grab(mon))[:, :, :3]
    return frame, mon


def live_select_roi_with_freeze(monitor_index: int = 1, window_name: str = "Calibration Live") -> Tuple[int,int,int,int]:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    hwnd = None
    if win32gui:
        # Show initial blank to create window
        cv2.imshow(window_name, np.zeros((200,400,3), dtype=np.uint8))
        cv2.waitKey(1)
        hwnd = win32gui.FindWindow(None, window_name)
    frozen = None
    print("[Calibration] Press f to freeze, q to cancel, ESC to cancel.")
    while True:
        if win32gui and hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
        frame, mon = _grab_monitor(monitor_index)
        if win32gui and hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        disp = frame.copy()
        h, w = disp.shape[:2]
        cv2.putText(disp, "f=freeze  q=quit", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, disp)
        k = cv2.waitKey(15) & 0xFF
        if k in (27, ord('q')):  # ESC or q
            cv2.destroyWindow(window_name)
            raise RuntimeError("Calibration cancelled")
        if k == ord('f'):
            frozen = frame
            break
    cv2.destroyWindow(window_name)
    # ROI selection on frozen image (avoid self view)
    roi = cv2.selectROI("Select ROI", frozen, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    if roi[2] <= 0 or roi[3] <= 0:
        raise RuntimeError("Empty ROI selection")
    x_abs = mon['left'] + int(roi[0])
    y_abs = mon['top'] + int(roi[1])
    return x_abs, y_abs, int(roi[2]), int(roi[3])


def select_roi(initial_frame) -> Tuple[int,int,int,int]:  # fallback
    r = cv2.selectROI("Select ROI", initial_frame, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return int(r[0]), int(r[1]), int(r[2]), int(r[3])


def capture_template(frame, window_name: str = "Template"):
    r = cv2.selectROI(window_name, frame, showCrosshair=True)
    cv2.destroyWindow(window_name)
    if r[2] <= 0 or r[3] <= 0:
        return None
    x,y,w,h = map(int, r)
    return cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)


# --- Mouse coordinate based calibration ---
def _get_mouse_position() -> Tuple[int,int]:
    try:
        from pynput import mouse  # type: ignore
        ctrl = mouse.Controller()
        pos = ctrl.position
        return int(pos[0]), int(pos[1])
    except Exception:
        try:
            import win32gui  # type: ignore
            pt = win32gui.GetCursorPos()
            return int(pt[0]), int(pt[1])
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Unable to read mouse position; install pynput or pywin32") from e


def mouse_coordinate_calibration() -> Tuple[int,int,int,int]:
    print("Mouse calibration: you'll set TOP-LEFT then BOTTOM-RIGHT corners.")
    input("Place mouse at TOP-LEFT corner of ROI then press Enter here...")
    x1, y1 = _get_mouse_position()
    print(f"Captured top-left: ({x1},{y1})")
    input("Place mouse at BOTTOM-RIGHT corner of ROI then press Enter here...")
    x2, y2 = _get_mouse_position()
    print(f"Captured bottom-right: ({x2},{y2})")
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    if w < 5 or h < 5:
        raise RuntimeError("ROI too small; aborting.")
    print(f"Final ROI: (x={x}, y={y}, w={w}, h={h})")
    return x, y, w, h

