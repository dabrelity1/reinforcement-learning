"""FishBot package initializer.

Provides high-level exports for convenience.
"""

from .config import BotConfig, load_config, save_config, get_default_config_path

__all__ = [
    "BotConfig",
    "load_config",
    "save_config",
    "get_default_config_path",
]
