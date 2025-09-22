"""Central logging configuration with console + rotating file handler."""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .config import DebugConfig


def setup_logging(debug_cfg: DebugConfig):
    root = logging.getLogger()
    # Allow runtime override via environment (used by minigame bot spawn)
    lvl_override = os.environ.get("FISHBOT_LOG_LEVEL_OVERRIDE")
    level_name = lvl_override if lvl_override else debug_cfg.log_level
    root.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Clear existing handlers (avoid duplicates on reload)
    for h in list(root.handlers):
        root.removeHandler(h)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if debug_cfg.log_file:
        path = Path(debug_cfg.log_file)
        fh = RotatingFileHandler(path, maxBytes=debug_cfg.log_max_bytes, backupCount=debug_cfg.log_backup_count)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    logging.getLogger("dxcam").setLevel(logging.WARNING)
    logging.getLogger("mss").setLevel(logging.WARNING)
