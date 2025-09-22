"""Detection routines for fish (template + fallback) and player rectangle."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2


class FishDetector:
    def __init__(self, template: Optional[np.ndarray], cfg):
        self.template = template
        self.cfg = cfg
        self.last_x: Optional[float] = None
        self._scaled_templates: list[tuple[float,np.ndarray]] = []
        if template is not None and cfg.template_multi_scale:
            self._prepare_scales()

    def _prepare_scales(self):
        start, mid, end = self.cfg.template_scales
        scales = np.linspace(start, end, 5)
        base = self.template
        for s in scales:
            if abs(s-1.0) < 1e-3:
                self._scaled_templates.append((s, base))
            else:
                h,w = base.shape
                resized = cv2.resize(base, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
                if resized.shape[0] >= 4 and resized.shape[1] >= 4:
                    self._scaled_templates.append((s, resized))

    def detect(self, gray: np.ndarray) -> Tuple[Optional[float], float]:
        conf = 0.0
        x = None
        # Multi-scale template matching
        if self.template is not None:
            templates = self._scaled_templates if self._scaled_templates else [(1.0, self.template)]
            best = None
            for scale, templ in templates:
                try:
                    res = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
                    min_v, max_v, min_l, max_l = cv2.minMaxLoc(res)
                    if best is None or max_v > best[0]:
                        th, tw = templ.shape
                        cx = max_l[0] + tw / 2
                        best = (max_v, cx)
                except Exception:
                    continue
            if best and best[0] >= self.cfg.template_match_threshold:
                conf, x = best
        if x is None and self.cfg.fallback_enabled:
            x_fb = self._fallback(gray)
            if x_fb is not None:
                x = x_fb
                conf = max(conf, 0.5)
        if x is not None:
            self.last_x = x
        return x, conf

    def _fallback(self, gray: np.ndarray) -> Optional[float]:
        g1, g2 = self.cfg.fish_grey_lower, self.cfg.fish_grey_upper
        mask = cv2.inRange(gray, g1, g2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.morphological_kernel,)*2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        last_x = self.last_x
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w < 4 or h < 4:
                continue
            aspect = w / max(1, h)
            if not (0.6 <= aspect <= 3.0):
                continue
            cx = x + w/2
            score = -abs(cx - last_x) if last_x is not None else w*h
            if best is None or score > best[0]:
                best = (score, cx)
        return None if best is None else best[1]


class RectDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_x: Optional[float] = None

    def detect(self, gray: np.ndarray):
        h, w = gray.shape
        band_h = int(h * self.cfg.rect_search_band_height_fraction)
        mid_y = int(h * self.cfg.rect_search_band_mid_fraction)
        y1 = max(0, mid_y - band_h//2)
        y2 = min(h, y1 + band_h)
        roi_band = gray[y1:y2]
        if self.cfg.use_adaptive_rect_threshold:
            block = self.cfg.rect_adaptive_block_size
            if block % 2 == 0:
                block += 1
            thresh_band = cv2.adaptiveThreshold(roi_band, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, block, self.cfg.rect_adaptive_C)
        else:
            _, thresh_band = cv2.threshold(roi_band, self.cfg.rect_white_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.cfg.morphological_kernel,)*2)
        thresh_band = cv2.morphologyEx(thresh_band, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(thresh_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        max_area_allowed = w * h * self.cfg.max_rect_area_fraction
        for c in cnts:
            x,y,wc,hc = cv2.boundingRect(c)
            area = wc * hc
            if area < self.cfg.min_rect_area or area > max_area_allowed:
                continue
            cx = x + wc/2
            aspect = wc / max(1,hc)
            if not (self.cfg.rect_aspect_min <= aspect <= self.cfg.rect_aspect_max):
                continue
            # Adjust y to full-frame coordinates
            y_abs = y1 + y
            proximity = -abs(cx - self.last_x) if self.last_x is not None else area
            score = area * 0.8 + proximity
            if best is None or score > best[0]:
                best = (score, (cx, (x, y_abs, wc, hc)))
        if best:
            self.last_x = best[1][0]
            return best[1]
        return None, None


class LineDetector:
    """Detect a thin vertical grey line inside a horizontal band.

    Strategy:
      1. Crop vertical band similar to rectangle band.
      2. Compute column intensity mean and std; optionally mask by proximity to expected grey.
      3. Use a simple 1D edge/contrast response (difference of gaussian-like via convolution with [-1,2,-1]).
      4. Score columns by negative absolute difference from expected grey plus contrast response.
      5. Validate by counting contiguous vertical pixels near expected grey within tolerance.
    Returns center x and (x, (x0,y0,w,h)) bounding box analog for overlay.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_x: Optional[float] = None
        self._miss_streak: int = 0
        self._dynamic_tol: Optional[int] = None

    def detect(self, gray: np.ndarray):
        h, w = gray.shape
        band_h = int(h * self.cfg.line_search_band_height_fraction)
        mid_y = int(h * self.cfg.rect_search_band_mid_fraction)
        y1 = max(0, mid_y - band_h//2)
        y2 = min(h, y1 + band_h)
        band = gray[y1:y2]
        # Adaptive expected grey: compute robust median of mid band region to adapt to lighting
        # Use central horizontal slice to avoid edges
        central_slice = band[:, max(0, w//4): min(w, 3*w//4)]
        median_grey = int(np.median(central_slice)) if central_slice.size else self.cfg.line_expected_grey
        expected = median_grey if abs(median_grey - self.cfg.line_expected_grey) < 40 else self.cfg.line_expected_grey
        # Start with base tolerance (may be dynamically widened)
        base_tol = self.cfg.line_grey_tolerance
        if self._dynamic_tol is None:
            self._dynamic_tol = base_tol
        tol = self._dynamic_tol
        lower = max(0, expected - tol)
        upper = min(255, expected + tol)
        # Build hit mask within tolerance
        mask = (band >= lower) & (band <= upper)
        # Column hit counts
        hit_counts = mask.sum(axis=0).astype(np.float32)
        # Emphasize columns with vertical continuity: compute simple derivative along y and penalize flicker
        # (optional future improvement)
        # Add gentle smoothing
        hit_counts_sm = cv2.GaussianBlur(hit_counts.reshape(1, -1), (1,0), 0).ravel()
        # Gradient/contrast enhancement: columns where intensity is close to expected and local std is low
        col_mean = band.mean(axis=0)
        local_dev = np.abs(col_mean - expected)
        contrast_bonus = np.maximum(0, (tol - local_dev))
        score = hit_counts_sm + 0.3 * contrast_bonus
        if self.last_x is not None:
            px = np.arange(w, dtype=np.float32)
            score -= 0.0008 * (px - self.last_x)**2  # soft spatial prior
        best_x = int(np.argmax(score))
        # Validate: contiguous run of tolerance pixels near best_x
        column = band[:, best_x]
        good = (column >= lower) & (column <= upper)
        max_run = 0; cur = 0
        for g in good:
            if g:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        # Relax: allow detection if hit count reasonably high even if contiguous run short
        min_hits = self.cfg.line_min_column_hits
        if max_run < min_hits and hit_counts[best_x] < max(min_hits, band.shape[0]*0.25):
            # Miss path: update streak & maybe widen tolerance
            self._miss_streak += 1
            if (self.cfg.line_adaptive_enable and
                self._miss_streak >= self.cfg.line_adaptive_miss_threshold and
                self._dynamic_tol < self.cfg.line_adaptive_max_tolerance):
                self._dynamic_tol = min(self.cfg.line_adaptive_max_tolerance,
                                        self._dynamic_tol + self.cfg.line_adaptive_expand_step)
            return None, None
        # Success path: reset miss streak & slowly relax tolerance toward base
        self._miss_streak = 0
        if self._dynamic_tol is not None and self._dynamic_tol > base_tol:
            # decay dynamic tolerance back toward base
            self._dynamic_tol = max(base_tol, self._dynamic_tol - self.cfg.line_adaptive_expand_step//2)
        cx = float(best_x)
        # Sub-pixel refine: weighted average of neighbors by score window
        left = max(0, best_x - 2)
        right = min(w, best_x + 3)
        win = score[left:right]
        if win.sum() > 0:
            sub = np.arange(left, right, dtype=np.float32)
            cx = float((sub * win).sum() / win.sum())
        self.last_x = cx
        box = (int(cx)-1, y1, 3, y2 - y1)
        return cx, box
