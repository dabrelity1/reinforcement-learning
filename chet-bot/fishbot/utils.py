"""Utility helpers: timing, smoothing, rolling stats."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Deque, Optional


def ewma(prev: float, new: float, alpha: float) -> float:
    if prev is None:
        return new
    return alpha * new + (1 - alpha) * prev


@dataclass
class EWMAVelocity:
    alpha_pos: float
    alpha_vel: float
    last_pos: Optional[float] = None
    vel: float = 0.0
    last_time: Optional[float] = None

    def update(self, pos: float, t: Optional[float] = None) -> tuple[float, float]:
        now = t if t is not None else time.perf_counter()
        if self.last_time is None or self.last_pos is None:
            self.last_pos = pos
            self.last_time = now
            return pos, 0.0
        dt = max(1e-6, now - self.last_time)
        raw_vel = (pos - self.last_pos) / dt
        self.last_pos = ewma(self.last_pos, pos, self.alpha_pos)
        self.vel = ewma(self.vel, raw_vel, self.alpha_vel)
        self.last_time = now
        return self.last_pos, self.vel


class RollingStat:
    def __init__(self, size: int = 120):
        self.q: Deque[float] = deque(maxlen=size)

    def add(self, v: float):
        self.q.append(v)

    def mean(self) -> float:
        return sum(self.q) / len(self.q) if self.q else 0.0

    def max(self) -> float:
        return max(self.q) if self.q else 0.0

    def min(self) -> float:
        return min(self.q) if self.q else 0.0


class FPSCounter:
    def __init__(self, window: int = 60):
        self.times: Deque[float] = deque(maxlen=window)

    def tick(self):
        now = time.perf_counter()
        self.times.append(now)

    def fps(self) -> float:
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0
