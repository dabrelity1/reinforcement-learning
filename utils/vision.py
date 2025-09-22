from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2


def find_bar_y_positions(frame_bgr: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """Detect approximate vertical center positions of white bar and gray bar.

    Heuristic approach: convert to HSV/GRAY, threshold for bright (white) and medium gray, then
    compute row with max sum for each mask.
    Returns (white_y, gray_y) as integer indices in image coordinates, or None if not found.
    """
    h, w, _ = frame_bgr.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # White bar: very bright
    _, white_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    # Gray bar: medium intensity, not too bright
    gray_mask = cv2.inRange(gray, 120, 200)

    def row_peak(mask: np.ndarray) -> Optional[int]:
        scores = mask.sum(axis=1)
        peak = int(np.argmax(scores))
        if scores[peak] < w * 10:  # require minimum pixels
            return None
        return peak

    white_y = row_peak(white_mask)
    gray_y = row_peak(gray_mask)
    return white_y, gray_y


def overlap_ratio_from_positions(white_y: Optional[int], gray_y: Optional[int], tol: int = 12) -> float:
    """Return overlap ratio [0,1] based on distance; simplistic proxy when only centers are known.

    tol is a tolerance in pixels considered as full overlap; decays linearly beyond that.
    """
    if white_y is None or gray_y is None:
        return 0.0
    d = abs(int(white_y) - int(gray_y))
    if d >= 3 * tol:
        return 0.0
    return float(max(0.0, 1.0 - d / (3.0 * tol)))
