"""Metrics & logging helpers."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass


log = logging.getLogger("fishbot")


@dataclass
class LoopMetrics:
    frames: int = 0
    lost_fish_frames: int = 0
    toggles: int = 0
    cumulative_error: float = 0.0
    start_time: float = time.perf_counter()

    def update_error(self, e: float):
        self.cumulative_error += abs(e)

    def avg_error(self) -> float:
        return self.cumulative_error / max(1, self.frames)

    def summary(self) -> dict:
        elapsed = time.perf_counter() - self.start_time
        return {
            "fps": self.frames / elapsed if elapsed > 0 else 0.0,
            "avg_error": self.avg_error(),
            "lost_ratio": self.lost_fish_frames / max(1, self.frames),
            "toggles": self.toggles,
        }
