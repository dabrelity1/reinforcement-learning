"""Offline simulator placeholder (physics + fish trajectory) - to be implemented."""
from __future__ import annotations

from dataclasses import dataclass
import random
import math


@dataclass
class SimState:
    rect_x: float
    rect_v: float
    fish_x: float
    fish_v: float


class Simulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = SimState(
            rect_x=cfg.width/2,
            rect_v=0.0,
            fish_x=cfg.width/2,
            fish_v=random.uniform(-cfg.fish_speed_max, cfg.fish_speed_max)
        )

    def step(self, hold: bool, dt: float):
        c = self.cfg
        s = self.state
        accel = c.rect_accel if hold else -c.rect_accel
        s.rect_v += accel * dt
        s.rect_v *= c.rect_drag
        s.rect_x += s.rect_v * dt
        if s.rect_x < 0:
            s.rect_x = 0
            s.rect_v = -s.rect_v * c.wall_restitution
        elif s.rect_x > c.width:
            s.rect_x = c.width
            s.rect_v = -s.rect_v * c.wall_restitution
        # fish
        s.fish_x += s.fish_v * dt
        if s.fish_x < 0 or s.fish_x > c.width:
            s.fish_v = -s.fish_v
            s.fish_x = max(0, min(c.width, s.fish_x))
        # occasional speed change
        if random.random() < 0.01:
            s.fish_v = random.uniform(-c.fish_speed_max, c.fish_speed_max)
        return s
