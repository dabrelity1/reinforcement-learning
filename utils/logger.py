from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def create_writer(run_name: Optional[str] = None) -> SummaryWriter:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or f"run-{ts}"
    path = Path("runs") / name
    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(path))


def ensure_dir(path: str | Path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
