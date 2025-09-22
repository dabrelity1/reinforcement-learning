"""Predictive hysteresis finite-state controller."""
from __future__ import annotations

from enum import Enum
import time


class ActionState(str, Enum):
    HOLD = "HOLD"
    RELEASE = "RELEASE"


class Controller:
    def __init__(self, cfg, mouse, roi_width: int):
        self.cfg = cfg
        self.mouse = mouse
        self.roi_width = roi_width
        self.state = ActionState.RELEASE
        self.last_collision_time = 0.0
        # History variables
        self._last_rect_vel = None
        self._last_rect_pos = None

    def step(self, fish_x, fish_vel, rect_x, rect_vel) -> dict:
        prediction_time = self.cfg.prediction_factor + self.cfg.latency_compensation_default
        fish_pred = fish_x + fish_vel * prediction_time
        fish_pred = max(0, min(self.roi_width, fish_pred))
        error = fish_pred - rect_x
        hi = self.cfg.hysteresis_high
        lo = self.cfg.hysteresis_low
        if error > hi:
            desired = ActionState.HOLD
        elif error < -hi:
            desired = ActionState.RELEASE
        elif abs(error) < lo:
            desired = self.state
        else:
            desired = self.state
        # Wall bias
        wall_zone = self.roi_width * self.cfg.wall_bias_zone_fraction
        if rect_x < wall_zone:
            error += self.cfg.wall_bias_magnitude
        elif rect_x > self.roi_width - wall_zone:
            error -= self.cfg.wall_bias_magnitude
        toggled = False
        if desired != self.state and self.mouse.last_toggle_age() >= self.cfg.min_toggle_interval:
            if desired == ActionState.HOLD:
                self.mouse.set_down(True)
            else:
                self.mouse.set_down(False)
            self.state = desired
            toggled = True
        self._last_rect_vel = rect_vel
        self._last_rect_pos = rect_x
        return {
            "fish_pred": fish_pred,
            "error": error,
            "state": self.state.value,
            "toggled": toggled,
        }
