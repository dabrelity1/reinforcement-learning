from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import time

import numpy as np

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover
    mss = None

try:
    import pyautogui  # type: ignore
except Exception:  # pragma: no cover
    pyautogui = None


@dataclass
class CaptureRect:
    left: int
    top: int
    width: int
    height: int


class ScreenController:
    def __init__(self, capture_rect: CaptureRect, safe_mode: bool = True):
        self.capture_rect = capture_rect
        self.safe_mode = safe_mode
        self._sct = mss.mss() if mss is not None else None

    def grab(self) -> np.ndarray:
        if self._sct is None:
            raise RuntimeError("mss is not available. Install dependencies.")
        rect = {
            "left": self.capture_rect.left,
            "top": self.capture_rect.top,
            "width": self.capture_rect.width,
            "height": self.capture_rect.height,
        }
        img = self._sct.grab(rect)
        frame = np.array(img)  # BGRA
        frame = frame[:, :, :3]  # BGR
        return frame

    def move_mouse_vertical(self, delta_y: int, speed: float = 1.0):
        if pyautogui is None:
            raise RuntimeError("pyautogui is not available. Install dependencies.")
        x, y = pyautogui.position()
        # Limit delta in safe mode
        if self.safe_mode:
            delta_y = int(np.clip(delta_y, -25, 25))
        pyautogui.moveTo(x, y + int(delta_y), duration=max(0.0, 0.0 if speed >= 1 else (1 - speed) * 0.05))

    def center_mouse(self):
        if pyautogui is None:
            return
        x = self.capture_rect.left + self.capture_rect.width // 2
        y = self.capture_rect.top + self.capture_rect.height // 2
        pyautogui.moveTo(x, y)
        time.sleep(0.05)
