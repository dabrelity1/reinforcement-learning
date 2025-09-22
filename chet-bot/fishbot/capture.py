"""Screen capture abstraction using dxcam (preferred on Windows) or mss."""
from __future__ import annotations

import time
from typing import Optional, Tuple
import numpy as np


class ScreenCapture:
    def __init__(self, roi: Tuple[int, int, int, int], use_dxcam: bool, downscale: float = 1.0):
        self.roi = roi
        self.use_dxcam = use_dxcam
        self.downscale = downscale
        self._init_backend()

    def _init_backend(self):
        self._dx = None
        self._mss = None
        if self.use_dxcam:
            try:
                import dxcam  # type: ignore
                self._dx = dxcam.create(output_color="BGR")
            except Exception:
                self._dx = None
        if self._dx is None:
            from mss import mss  # type: ignore
            self._mss = mss()

    def grab(self) -> np.ndarray:
        x, y, w, h = self.roi
        if self._dx is not None:
            frame = self._dx.grab(region=(x, y, x + w, y + h))
        else:
            mon = {"top": y, "left": x, "width": w, "height": h}
            frame = np.array(self._mss.grab(mon))[:, :, :3]  # BGR
        if self.downscale != 1.0:
            import cv2
            frame = cv2.resize(frame, (int(frame.shape[1] * self.downscale), int(frame.shape[0] * self.downscale)), interpolation=cv2.INTER_AREA)
        return frame
