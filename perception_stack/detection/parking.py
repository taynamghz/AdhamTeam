"""
PSU Eco Racing — Perception Stack
detection/parking.py  |  Blue-line parking bay detection.

SEM spec:
  Bay dimensions : 4 m × 2 m
  Boundary       : 0.15 m-wide blue lines, white borders at sides and back
  Obstacle inside: grey crates 1.2 × 1.0 × 1.0 m — must detect and skip bay

Key insight
-----------
  parking pixels = blue AND floor_mask == 255  (ON the floor plane)
  obstacle pixels = blue AND floor_mask == 0   (above the floor)

floor_mask is the discriminator between these two blue targets.

Detection pipeline
------------------
  1. Blue HSV mask restricted to floor-plane pixels
  2. Morphological cleanup
  3. Minimum pixel count gate (PARK_MIN_PIXELS)
  4. Centroid of the blue pixel cluster
  5. ZED depth at centroid for metric (X_m, Z_m) position
  6. PCA orientation angle of the blue line cluster (bay heading)
  7. Empty-space check via ZED point cloud:
       Strategy A (floor_y known): count points > floor_y + 0.2 m in bay volume
       Strategy B (fallback)     : count points much shallower than back wall

Returns
-------
  (detected: bool, is_empty: bool,
   center_m: tuple[float, float] | None,  # (X_m, Z_m)
   angle_deg: float)
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from perception_stack.config import (
    PARK_BLUE_H_MIN, PARK_BLUE_H_MAX,
    PARK_BLUE_S_MIN, PARK_BLUE_V_MIN,
    PARK_MIN_PIXELS, PARK_DIST_MIN_M, PARK_DIST_MAX_M,
    PARK_OBS_THRESHOLD,
    ROI_TOP_FRACTION,
)


def detect_parking_bay(
    frame: np.ndarray,
    floor_mask: np.ndarray,
    lf,
    rf,
    pc: np.ndarray,
    H: int,
    W: int,
    hsv: np.ndarray,
    floor_y: Optional[float] = None,
) -> Tuple[bool, bool, Optional[Tuple[float, float]], float]:
    """
    Detect a blue-marked parking bay on the floor plane.

    Returns (detected, is_empty, center_m, angle_deg).
      center_m   = (X_m, Z_m) in ZED world frame.
      is_empty   = True when no grey crates detected in the bay volume.
      angle_deg  = principal orientation of the blue marking cluster.
    """
    fp = floor_mask == 255

    roi_top = int(H * ROI_TOP_FRACTION)

    Hh = hsv[:, :, 0]
    S  = hsv[:, :, 1]
    V  = hsv[:, :, 2]

    # Blue floor pixels only
    blue_floor = np.zeros((H, W), np.uint8)
    blue_floor[fp &
               (Hh >= PARK_BLUE_H_MIN) & (Hh <= PARK_BLUE_H_MAX) &
               (S  >= PARK_BLUE_S_MIN) &
               (V  >= PARK_BLUE_V_MIN)] = 255
    blue_floor[:roi_top, :] = 0    # ignore sky region

    k          = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blue_floor = cv2.morphologyEx(blue_floor, cv2.MORPH_OPEN,  k)
    blue_floor = cv2.morphologyEx(blue_floor, cv2.MORPH_CLOSE, k)

    pixel_count = int(blue_floor.sum() // 255)
    if pixel_count < PARK_MIN_PIXELS:
        return False, False, None, 0.0

    # ── Centroid of the blue pixel cluster ────────────────────────────────────
    nz_ys, nz_xs = np.where(blue_floor > 0)
    if nz_ys.size < 20:
        return False, False, None, 0.0

    cy_px = int(np.mean(nz_ys))
    cx_px = int(np.mean(nz_xs))

    # ── ZED 3D position at centroid ───────────────────────────────────────────
    pt = _sample_valid_point(pc, cy_px, cx_px, H, W)
    if pt is None:
        return False, False, None, 0.0

    dist_m = abs(float(pt[2]))
    if not (PARK_DIST_MIN_M <= dist_m <= PARK_DIST_MAX_M):
        return False, False, None, 0.0

    lateral_m = float(pt[0])
    center_m  = (lateral_m, dist_m)

    # ── PCA orientation of the marking cluster ────────────────────────────────
    pts_f  = np.stack([nz_xs.astype(float), nz_ys.astype(float)], axis=1)
    pts_f -= pts_f.mean(axis=0)
    _, _, Vt  = np.linalg.svd(pts_f, full_matrices=False)
    angle_deg = float(np.degrees(np.arctan2(Vt[0, 1], Vt[0, 0])))

    # ── Empty-space check (grey crate occupancy) ──────────────────────────────
    is_empty = _check_empty(pc, cy_px, cx_px, H, W, dist_m, floor_y)

    return True, is_empty, center_m, angle_deg


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sample_valid_point(
    pc: np.ndarray, cy: int, cx: int, H: int, W: int
) -> Optional[np.ndarray]:
    """Return the first finite, forward-facing point near (cx, cy)."""
    for dy in range(-8, 9, 4):
        for dx in range(-8, 9, 4):
            ry = max(0, min(H - 1, cy + dy))
            rx = max(0, min(W - 1, cx + dx))
            p  = pc[ry, rx, :3]
            if np.isfinite(p).all() and float(p[2]) < 0:
                return p
    return None


def _check_empty(
    pc: np.ndarray,
    cy: int, cx: int,
    H: int, W: int,
    bay_dist_m: float,
    floor_y: Optional[float],
) -> bool:
    """
    Estimate whether the parking bay is free of obstacles (grey crates).

    Grey crates are 1.2 × 1.0 × 1.0 m — they generate a dense cluster of
    point-cloud returns between 0.2 m and 1.1 m above the floor.

    Strategy A (floor_y known):
        Count points with world-Y > floor_y + 0.20 m inside the search window.
        More than PARK_OBS_THRESHOLD → bay occupied.

    Strategy B (floor_y unknown / fallback):
        Count ZED returns that are significantly shallower than the bay's
        measured back-wall depth.  A crate would appear much closer.
    """
    SEARCH_PY = 25
    SEARCH_PX = 35

    py0 = max(0, cy - SEARCH_PY);  py1 = min(H, cy + SEARCH_PY)
    px0 = max(0, cx - SEARCH_PX);  px1 = min(W, cx + SEARCH_PX)

    sub_pc = pc[py0:py1, px0:px1, :]
    Y_ch   = sub_pc[:, :, 1]     # world Y (height)
    Z_ch   = sub_pc[:, :, 2]     # world Z (depth, negative = forward)

    if floor_y is not None:
        # A — preferred path
        above = (np.isfinite(Y_ch) &
                 (Y_ch > floor_y + 0.20) &
                 (Y_ch < floor_y + 1.20))
        count = int(above.sum())
    else:
        # B — fallback: points much shallower than bay back wall
        bay_z = -bay_dist_m   # negative in ZED convention
        close = (np.isfinite(Z_ch) &
                 (Z_ch < 0) &
                 (Z_ch > bay_z + 0.50))
        count = int(close.sum())

    return count < PARK_OBS_THRESHOLD
