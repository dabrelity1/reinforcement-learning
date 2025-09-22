from __future__ import annotations

from dataclasses import dataclass


class LinearSchedule:
    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = max(1, duration)

    def __call__(self, t: int) -> float:
        if t >= self.duration:
            return self.end
        frac = t / self.duration
        return self.start + frac * (self.end - self.start)


@dataclass
class Curriculum:
    # Simple curriculum: speed scales with phase; phase advances each N episodes
    base_speed: float = 1.0
    phases: int = 3
    advance_every: int = 200

    def speed_for_episode(self, episode: int) -> float:
        phase = min(self.phases - 1, episode // self.advance_every)
        return self.base_speed * (0.6 + 0.4 * (phase + 1))
