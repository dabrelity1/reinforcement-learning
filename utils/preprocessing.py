from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def resize84(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)


def normalize(frame: np.ndarray) -> np.ndarray:
    # 0..255 -> 0..1 float32
    return (frame.astype(np.float32) / 255.0)


def preprocess_observation(frame: np.ndarray) -> np.ndarray:
    gray = to_grayscale(frame)
    small = resize84(gray)
    norm = normalize(small)
    return norm  # shape (84,84)


def stack_frames(frames: list[np.ndarray], num_stack: int = 4) -> np.ndarray:
    # frames are (84,84); stack along channel axis
    if len(frames) < num_stack:
        # pad with first frame
        first = frames[0]
        frames = [first] * (num_stack - len(frames)) + frames
    arr = np.stack(frames[-num_stack:], axis=0)
    return arr  # (C,H,W)
