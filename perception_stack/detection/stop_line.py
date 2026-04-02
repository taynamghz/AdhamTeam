"""
PSU Eco Racing — Perception Stack
detection/stop_line.py  |  Orange horizontal stop-stripe detection.

Requirements for a positive detection (all must pass):
  1. Pixel is on the floor (floor_mask)
  2. HSV hue in orange range, sufficient saturation/value
  3. Row density: ≥ STOP_ROW_THRESH of frame width is orange
  4. Perpendicularity: cluster principal axis ≤ STOP_PERP_MAX_DEG from horizontal
  5. Lane coverage: orange spans ≥ STOP_COVERAGE_MIN of lane width
  6. Located in bottom 60% of frame
"""

import math
import numpy as np
import cv2
from typing import Optional, Tuple

from perception_stack.config import (
    STOP_ORANGE_H_MIN, STOP_ORANGE_H_MAX, STOP_ORANGE_S_MIN, STOP_ORANGE_V_MIN,
    STOP_ROW_THRESH, STOP_COVERAGE_MIN, STOP_PERP_MAX_DEG,
    STOP_DIST_MIN_M, STOP_DIST_MAX_M, STOP_DIST_N_PTS, STOP_DIST_MIN_VALID,
    STOP_WIDTH_MIN_FRAC,
)
from perception_stack.lane.fitting import eval_x


def _stop_perp_ok(orange: np.ndarray, row: int, H: int, W: int) -> bool:
    """
    Check that the orange cluster around `row` is nearly horizontal
    (within ±STOP_PERP_MAX_DEG of the horizontal axis).
    Uses PCA on the pixel cluster — rejects angled shadows and diagonal markings.
    """
    r0, r1 = max(0, row - 10), min(H, row + 10)
    ys_l, xs_l = np.where(orange[r0:r1, :] > 0)
    if xs_l.size < 10:
        return False
    pts = np.stack([xs_l.astype(float), (ys_l + r0).astype(float)], axis=1)
    pts -= pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    ang = abs(math.degrees(math.atan2(Vt[0, 1], Vt[0, 0])))
    if ang > 90:
        ang = 180 - ang
    return ang <= STOP_PERP_MAX_DEG


def _stop_median_dist(pc: np.ndarray, row: int,
                      lx: int, rx: int, H: int, W: int) -> float:
    """Sample STOP_DIST_N_PTS evenly across lane, return median Z distance."""
    sample_row = min(row + 20, H - 1)
    xs = np.clip(np.linspace(lx, rx, STOP_DIST_N_PTS, dtype=int), 0, W - 1)
    dists = []
    for sx in xs:
        pt = pc[sample_row, sx, :3]
        if np.isfinite(pt).all():
            d = abs(float(pt[2]))
            if STOP_DIST_MIN_M <= d <= STOP_DIST_MAX_M:
                dists.append(d)
    return float(np.median(dists)) if len(dists) >= STOP_DIST_MIN_VALID else 0.0


def detect_stop_line(
    frame: np.ndarray,
    floor_mask: np.ndarray,
    lf, rf,
    pc: np.ndarray,
    H: int, W: int,
    hls: np.ndarray,
    hsv: np.ndarray,
) -> Tuple[bool, Optional[int], float]:
    """
    Detect an orange horizontal stop stripe painted on the road.
    Returns (detected, row_y, distance_m).
    """
    fp    = floor_mask == 255
    H_ch  = hsv[:, :, 0]
    S_hsv = hsv[:, :, 1]
    V_hsv = hsv[:, :, 2]

    orange = np.zeros((H, W), np.uint8)
    orange[fp &
           (H_ch >= STOP_ORANGE_H_MIN) & (H_ch <= STOP_ORANGE_H_MAX) &
           (S_hsv >= STOP_ORANGE_S_MIN) & (V_hsv >= STOP_ORANGE_V_MIN)] = 255

    row_counts = orange.sum(axis=1)
    candidates = np.where(row_counts > W * STOP_ROW_THRESH)[0]
    candidates = candidates[candidates > int(H * 0.40)]   # bottom 60% only

    for row in candidates:
        if not _stop_perp_ok(orange, row, H, W):
            continue

        if lf is not None and rf is not None:
            lx     = int(np.clip(eval_x(lf, row), 0, W - 1))
            rx     = int(np.clip(eval_x(rf, row), 0, W - 1))
            lane_w = rx - lx
            if lane_w < 20:
                continue
            coverage = int(orange[row, lx:rx].sum()) / lane_w
            if coverage < STOP_COVERAGE_MIN:
                continue

            # ── Physical stripe-width gate ────────────────────────────────────
            # SEM stop line spans full track width (≥ 2 m).
            # Measure lane width and stripe width from ZED X-coordinates.
            # Rejects narrow orange fragments (cones, shadows, debris).
            lp = pc[row, lx, :3]
            rp = pc[row, rx, :3]
            if np.isfinite(lp).all() and np.isfinite(rp).all():
                lane_width_m = abs(float(rp[0] - lp[0]))
                if lane_width_m > 0.3:
                    stripe_width_m = _orange_stripe_width_m(
                        orange, pc, row, lx, rx)
                    if (stripe_width_m > 0 and
                            stripe_width_m < STOP_WIDTH_MIN_FRAC * lane_width_m):
                        continue

            dist = _stop_median_dist(pc, row, lx, rx, H, W)
            return True, row, dist
        else:
            if row_counts[row] > W * 0.15:
                mid  = W // 2
                dist = _stop_median_dist(pc, row, mid - 50, mid + 50, H, W)
                return True, row, dist

    return False, None, 0.0


def _orange_stripe_width_m(
    orange: np.ndarray,
    pc: np.ndarray,
    row: int,
    lx: int,
    rx: int,
) -> float:
    """
    Measure the physical width (m) of the orange stripe at this row using
    ZED X-coordinates of orange pixels between lx and rx.
    Returns 0.0 if insufficient valid samples.
    """
    orange_cols = np.where(orange[row, lx:rx] > 0)[0] + lx
    if orange_cols.size < 6:
        return 0.0
    # Sample every 4 px to stay fast
    x_world = []
    for col in orange_cols[::4]:
        pt = pc[row, int(col), :3]
        if np.isfinite(pt).all():
            x_world.append(float(pt[0]))
    if len(x_world) < 4:
        return 0.0
    return float(np.percentile(x_world, 90) - np.percentile(x_world, 10))
