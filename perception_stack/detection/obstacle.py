"""
PSU Eco Racing — Perception Stack
detection/obstacle.py  |  Point-cloud obstacle detection above the floor plane.

Scans the ZED XYZ point cloud for objects physically above the floor and
within the lane corridor.  Returns the largest connected blob's distance,
lateral offset, and image bounding box.
"""

import numpy as np
import cv2
from typing import Optional, Tuple

from perception_stack.config import (
    OBS_MIN_HEIGHT_M, OBS_MAX_HEIGHT_M,
    OBS_MIN_DIST_M, OBS_MAX_DIST_M,
    OBS_LANE_MARGIN_M, OBS_MIN_CLUSTER_PX,
    ROI_TOP_FRACTION,
)


def detect_obstacle_pc(
    pc: np.ndarray,
    floor_y: float,
    lf, rf,
    H: int, W: int, fx: float,
) -> Tuple[bool, float, float, Optional[Tuple[int,int,int,int]]]:
    """
    Scan the point cloud for objects physically above the floor and within
    the lane corridor (+ OBS_LANE_MARGIN_M on each side).

    Returns (detected, dist_m, lateral_m, bbox_image_px).
      dist_m      — forward distance to obstacle centroid (|Z|)
      lateral_m   — signed lateral offset from image centre (+left / -right)
      bbox        — (x, y, w, h) tight bounding box in image pixels
    """
    if floor_y is None:
        return False, 0.0, 0.0, None

    X = pc[:, :, 0]   # lateral  (right = +)
    Y = pc[:, :, 1]   # vertical (up    = +)
    Z = pc[:, :, 2]   # depth    (forward = negative in RIGHT_HANDED_Y_UP)

    valid = (np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z))

    # Height gate: above floor surface, below ceiling
    above = (Y > floor_y + OBS_MIN_HEIGHT_M) & (Y < floor_y + OBS_MAX_HEIGHT_M)

    # Forward gate
    fwd = (Z < -OBS_MIN_DIST_M) & (Z > -OBS_MAX_DIST_M)

    # Lateral gate: inside lane + margin (fallback to full frame if no fits)
    if lf is not None or rf is not None:
        row_idx = np.arange(H, dtype=np.float32)
        if lf is not None and rf is not None:
            lx_px = np.clip(np.polyval(lf, row_idx).astype(int), 0, W - 1)
            rx_px = np.clip(np.polyval(rf, row_idx).astype(int), 0, W - 1)
        elif lf is not None:
            lx_px = np.clip(np.polyval(lf, row_idx).astype(int), 0, W - 1)
            rx_px = np.full(H, W - 1, dtype=int)
        else:
            lx_px = np.zeros(H, dtype=int)
            rx_px = np.clip(np.polyval(rf, row_idx).astype(int), 0, W - 1)

        col_idx = np.tile(np.arange(W, dtype=int), (H, 1))
        lane_l  = lx_px[:, None] - int(OBS_LANE_MARGIN_M * fx / 5.0)
        lane_r  = rx_px[:, None] + int(OBS_LANE_MARGIN_M * fx / 5.0)
        in_lane = (col_idx >= lane_l) & (col_idx <= lane_r)
    else:
        in_lane = np.ones((H, W), bool)

    obs_mask = (valid & above & fwd & in_lane).astype(np.uint8) * 255
    obs_mask[:int(H * ROI_TOP_FRACTION), :] = 0

    k = np.ones((5, 5), np.uint8)
    obs_mask = cv2.morphologyEx(obs_mask, cv2.MORPH_OPEN,  k)
    obs_mask = cv2.morphologyEx(obs_mask, cv2.MORPH_CLOSE, k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        obs_mask, connectivity=8)

    best_label = -1
    best_area  = OBS_MIN_CLUSTER_PX - 1
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area  = area
            best_label = i

    if best_label < 0:
        return False, 0.0, 0.0, None

    bx = stats[best_label, cv2.CC_STAT_LEFT]
    by = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]

    blob   = labels == best_label
    Zb     = Z[blob]
    Xb     = X[blob]
    finite = np.isfinite(Zb) & np.isfinite(Xb)
    if finite.sum() < 4:
        return False, 0.0, 0.0, None

    dist_m    = float(np.median(np.abs(Zb[finite])))
    lateral_m = float(np.median(Xb[finite]))

    return True, dist_m, lateral_m, (int(bx), int(by), int(bw), int(bh))
