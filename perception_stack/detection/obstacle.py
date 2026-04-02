"""
PSU Eco Racing — Perception Stack
detection/obstacle.py  |  Blue inflatable obstacle pin detection.

SEM spec: ~1.10 m tall × 0.45 m diameter, blue inflatable.

Key insight shared with parking.py
-----------------------------------
  obstacle pixels = blue AND floor_mask == 0   (ABOVE the floor)
  parking pixels  = blue AND floor_mask == 255 (ON the floor)

Both targets share the same blue HSV range. floor_mask is the discriminator.

Detection pipeline
------------------
  1. Blue HSV mask restricted to above-floor pixels and ROI rows
  2. Morphological open+close to remove noise and fill the inflatable surface
  3. Connected-component analysis — keep largest blob above OBS_MIN_PIXELS
  4. Aspect-ratio gate: pin is taller than wide (height/width > OBS_ASPECT_MIN)
  5. ZED depth at centroid → dist_m
  6. 3D height check: blob top–bottom Y span in world coords must be 0.3–1.5 m
     (eliminates short objects, spectators' blue clothing at distance, sky)
  7. Lateral position from ZED X coordinate at centroid

Returns
-------
  (detected: bool, dist_m: float, lateral_m: float)
    lateral_m > 0  →  obstacle to the right of camera centre
    lateral_m < 0  →  obstacle to the left
    lateral_m == 0 →  lateral unknown (no valid ZED return at centroid)
"""

import cv2
import numpy as np
from typing import Tuple

from perception_stack.config import (
    OBS_BLUE_H_MIN, OBS_BLUE_H_MAX,
    OBS_BLUE_S_MIN, OBS_BLUE_V_MIN,
    OBS_MIN_PIXELS, OBS_ASPECT_MIN,
    OBS_HEIGHT_3D_MIN, OBS_HEIGHT_3D_MAX,
    OBS_DIST_MIN_M, OBS_DIST_MAX_M,
    ROI_TOP_FRACTION,
)


def detect_obstacle(
    frame: np.ndarray,
    floor_mask: np.ndarray,
    lf,
    rf,
    pc: np.ndarray,
    H: int,
    W: int,
    hsv: np.ndarray,
) -> Tuple[bool, float, float]:
    """
    Detect the SEM blue inflatable obstacle pin.
    Returns (detected, dist_m, lateral_m).
    """
    above_floor = floor_mask < 128

    roi_top    = int(H * ROI_TOP_FRACTION)
    roi_bottom = int(H * 0.90)          # exclude vehicle hood at the very bottom

    Hh = hsv[:, :, 0]
    S  = hsv[:, :, 1]
    V  = hsv[:, :, 2]

    blue = np.zeros((H, W), np.uint8)
    blue[above_floor &
         (Hh >= OBS_BLUE_H_MIN) & (Hh <= OBS_BLUE_H_MAX) &
         (S  >= OBS_BLUE_S_MIN) &
         (V  >= OBS_BLUE_V_MIN)] = 255
    # Enforce ROI
    blue[:roi_top, :]    = 0
    blue[roi_bottom:, :] = 0

    # Morphological cleanup: open removes speckle noise, close fills holes in
    # the inflatable's curved surface
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN,  k)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k)

    # ── Connected-component analysis ──────────────────────────────────────────
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        blue, connectivity=8)
    if n_labels < 2:
        return False, 0.0, 0.0

    # Select largest blob that passes the aspect-ratio gate
    best_label = -1
    best_area  = 0
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < OBS_MIN_PIXELS:
            continue
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        if bw < 1:
            continue
        # Pin is always taller than wide from front/side view
        if bh / bw < OBS_ASPECT_MIN:
            continue
        if area > best_area:
            best_area  = area
            best_label = lbl

    if best_label < 0:
        return False, 0.0, 0.0

    cx = int(centroids[best_label, 0])
    cy = int(centroids[best_label, 1])

    # ── ZED depth at centroid ─────────────────────────────────────────────────
    patch_r = 6
    py0 = max(0, cy - patch_r);  py1 = min(H, cy + patch_r)
    px0 = max(0, cx - patch_r);  px1 = min(W, cx + patch_r)
    patch_z = pc[py0:py1, px0:px1, 2]
    finite  = patch_z[np.isfinite(patch_z) & (patch_z < 0)]
    if finite.size < 3:
        return False, 0.0, 0.0

    dist_m = float(np.median(np.abs(finite)))
    if not (OBS_DIST_MIN_M <= dist_m <= OBS_DIST_MAX_M):
        return False, 0.0, 0.0

    # ── 3D height check — blob top row vs bottom row in world Y ──────────────
    y_top    = int(stats[best_label, cv2.CC_STAT_TOP])
    y_bottom = min(y_top + int(stats[best_label, cv2.CC_STAT_HEIGHT]) - 1, H - 1)
    pt_top    = pc[y_top,    cx, :3]
    pt_bottom = pc[y_bottom, cx, :3]

    if np.isfinite(pt_top).all() and np.isfinite(pt_bottom).all():
        # ZED RIGHT_HANDED_Y_UP: Y is world height
        height_3d = abs(float(pt_top[1] - pt_bottom[1]))
        if not (OBS_HEIGHT_3D_MIN <= height_3d <= OBS_HEIGHT_3D_MAX):
            return False, 0.0, 0.0

    # ── Lateral position from ZED X at centroid ───────────────────────────────
    pt_center = pc[cy, cx, :3]
    if not np.isfinite(pt_center).all():
        # Scan nearby pixels for a valid return
        lateral_m = _find_lateral_fallback(pc, cy, cx, H, W)
        return True, dist_m, lateral_m

    return True, dist_m, float(pt_center[0])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_lateral_fallback(
    pc: np.ndarray, cy: int, cx: int, H: int, W: int
) -> float:
    """Scan a small neighbourhood for a valid ZED point when centroid is invalid."""
    for dy in range(-4, 5, 2):
        for dx in range(-4, 5, 2):
            ry = max(0, min(H - 1, cy + dy))
            rx = max(0, min(W - 1, cx + dx))
            p  = pc[ry, rx, :3]
            if np.isfinite(p).all():
                return float(p[0])
    return 0.0
