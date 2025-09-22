"""Configuration management for FishBot.

Defines dataclasses representing runtime parameters and provides
JSON (de)serialization helpers. All tunable parameters are centralized here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
from typing import Tuple, Optional, Any, Dict


DEFAULT_CONFIG_FILENAME = "fishbot_config.json"


@dataclass
class CaptureConfig:
    roi: Tuple[int, int, int, int] = (0, 0, 800, 600)  # x, y, w, h
    downscale: float = 1.0
    capture_fps_target: int = 60
    use_dxcam: bool = False
    frame_buffer_size: int = 2  # small for low latency
    monitor_index: int = 1  # mss style (1 = primary)


@dataclass
class DetectionConfig:
    template_path: Optional[str] = None
    template_match_threshold: float = 0.72
    template_multi_scale: bool = True
    template_scales: Tuple[float, float, float] = (0.9, 1.0, 1.1)  # start, mid, end
    max_template_candidates: int = 5
    fallback_enabled: bool = True
    fish_grey_lower: int = 70
    fish_grey_upper: int = 170
    rect_white_threshold: int = 230
    morphological_kernel: int = 3
    lost_fish_grace_frames: int = 6
    # New adaptive / band parameters
    use_adaptive_rect_threshold: bool = True
    rect_adaptive_block_size: int = 35  # must be odd
    rect_adaptive_C: int = -5
    rect_search_band_mid_fraction: float = 0.5  # vertical center of expected rect (0-1)
    rect_search_band_height_fraction: float = 0.5  # portion of height to search (0-1)
    min_rect_area: int = 80
    max_rect_area_fraction: float = 0.6  # ignore giant white areas
    rect_aspect_min: float = 0.2
    rect_aspect_max: float = 18.0  # allow very wide bars for fullscreen scaling
    debug_masks: bool = False
    # Line mode (minigame grey vertical line) vs fish template/mask
    line_mode: bool = False
    line_search_band_height_fraction: float = 0.5  # fraction of height to scan for line
    line_min_contrast: int = 18  # minimal contrast difference line vs surroundings
    line_expected_grey: int = 128  # approximate grey value of line center
    line_grey_tolerance: int = 50  # tolerance around expected grey
    line_min_column_hits: int = 14  # minimal contiguous vertical pixels to qualify
    # Adaptive line tolerance (used if repeated misses)
    line_adaptive_enable: bool = True
    line_adaptive_expand_step: int = 10  # how much to widen tolerance per adaptation
    line_adaptive_max_tolerance: int = 90  # cap on widened tolerance
    line_adaptive_miss_threshold: int = 12  # misses in a row before widening


@dataclass
class ControlConfig:
    hysteresis_low: int = 6
    hysteresis_high: int = 12
    prediction_factor: float = 0.07
    latency_compensation_default: float = 0.02
    min_toggle_interval: float = 0.020
    collision_damp_window: float = 0.08
    wall_bias_zone_fraction: float = 0.08  # fraction of width near walls
    wall_bias_magnitude: float = 6.0
    ewma_alpha_pos: float = 0.3
    ewma_alpha_vel: float = 0.2
    max_frame_skip: int = 1


@dataclass
class DebugConfig:
    debug: bool = False
    dry_run: bool = False
    show_overlay: bool = False
    log_level: str = "INFO"
    metrics_interval_sec: float = 5.0
    overlay_font_scale: float = 0.6
    overlay_thickness: int = 1
    log_file: str | None = "fishbot.log"
    log_max_bytes: int = 2_000_000
    log_backup_count: int = 3


@dataclass
class SimulatorConfig:
    enabled: bool = False
    width: int = 800
    height: int = 400
    rect_accel: float = 140.0
    rect_drag: float = 0.85
    wall_restitution: float = 0.25
    fish_speed_min: float = 30.0
    fish_speed_max: float = 380.0
    occlusion_probability: float = 0.02


@dataclass
class BotConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BotConfig":
        def sub(cls, key):
            data = d.get(key, {})
            return cls(**data) if isinstance(data, dict) else cls()

        return BotConfig(
            capture=sub(CaptureConfig, "capture"),
            detection=sub(DetectionConfig, "detection"),
            control=sub(ControlConfig, "control"),
            debug=sub(DebugConfig, "debug"),
            simulator=sub(SimulatorConfig, "simulator"),
            version=d.get("version", 1),
        )


def get_default_config_path() -> Path:
    return Path.cwd() / DEFAULT_CONFIG_FILENAME


def load_config(path: Optional[str | Path] = None) -> BotConfig:
    path = Path(path) if path else get_default_config_path()
    if not path.exists():
        return BotConfig()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return BotConfig.from_dict(data)


def save_config(cfg: BotConfig, path: Optional[str | Path] = None) -> Path:
    path = Path(path) if path else get_default_config_path()
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return path

