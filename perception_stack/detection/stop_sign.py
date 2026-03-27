"""
PSU Eco Racing — Perception Stack
detection/stop_sign.py  |  Red octagon stop-sign detector.

Algorithm:
  1. Build a red HSV mask (two hue bands — red wraps around 0/180 in HSV).
  2. Find external contours on the mask.
  3. For each large-enough contour:
       a. Approximate polygon (Ramer–Douglas–Peucker).
       b. Accept 6–10 sides (octagon looks 6-8 sided under perspective/distance).
       c. Aspect-ratio check (bounding rect W/H must be 0.5–1.8).
  4. Pick the largest qualifying contour.
  5. Estimate distance from the ZED point cloud at the sign's centroid.

Returns (detected, dist_m, bbox_px) where bbox_px = (x, y, w, h).
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from perception_stack.config import (
    SIGN_RED_S_MIN, SIGN_RED_V_MIN,
    SIGN_RED_H_LOW_MAX, SIGN_RED_H_HIGH_MIN,
    SIGN_MIN_AREA_PX, SIGN_MAX_AREA_PX,
    SIGN_POLY_SIDES_MIN, SIGN_POLY_SIDES_MAX,
    SIGN_ASPECT_MIN, SIGN_ASPECT_MAX,
    SIGN_DIST_MIN_M, SIGN_DIST_MAX_M,
    ROI_TOP_FRACTION,
)


def _red_mask(frame: np.ndarray, H: int, W: int) -> np.ndarray:
    """Two-range HSV red mask; red wraps at hue=0/180."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo1 = np.array([0,                   SIGN_RED_S_MIN, SIGN_RED_V_MIN])
    hi1 = np.array([SIGN_RED_H_LOW_MAX,  255,            255           ])
    lo2 = np.array([SIGN_RED_H_HIGH_MIN, SIGN_RED_S_MIN, SIGN_RED_V_MIN])
    hi2 = np.array([180,                 255,            255           ])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1),
                          cv2.inRange(hsv, lo2, hi2))
    # Exclude sky / hood ROI
    mask[:int(H * ROI_TOP_FRACTION), :] = 0
    # Morphological clean-up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_stop_sign(
    frame: np.ndarray,
    pc:    np.ndarray,
    H: int, W: int,
) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
    """
    Detect a red octagonal stop sign in the frame.

    Returns:
        detected  — True if a qualifying sign was found
        dist_m    — forward distance to sign centroid via ZED point cloud
        bbox      — (x, y, w, h) bounding box in image pixels, or None
    """
    mask = _red_mask(frame, H, W)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, 0.0, None

    best_cnt  = None
    best_area = SIGN_MIN_AREA_PX - 1

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (SIGN_MIN_AREA_PX <= area <= SIGN_MAX_AREA_PX):
            continue

        # Polygon approximation — epsilon ~3% of perimeter
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        sides  = len(approx)
        if not (SIGN_POLY_SIDES_MIN <= sides <= SIGN_POLY_SIDES_MAX):
            continue

        # Aspect ratio — octagon bounding rect is nearly square
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        if not (SIGN_ASPECT_MIN <= aspect <= SIGN_ASPECT_MAX):
            continue

        if area > best_area:
            best_area = area
            best_cnt  = cnt

    if best_cnt is None:
        return False, 0.0, None

    x, y, w, h = cv2.boundingRect(best_cnt)
    cx = int(np.clip(x + w // 2, 0, W - 1))
    cy = int(np.clip(y + h // 2, 0, H - 1))

    # Distance from point cloud — sample a small patch around centroid
    r = max(4, h // 8)
    y0, y1 = max(0, cy - r), min(H, cy + r)
    x0, x1 = max(0, cx - r), min(W, cx + r)
    patch_z = pc[y0:y1, x0:x1, 2]          # Z channel (forward = negative)
    finite  = patch_z[np.isfinite(patch_z)]
    finite  = finite[finite < 0]            # must be in front of camera
    if finite.size >= 4:
        dist_m = float(np.median(np.abs(finite)))
    else:
        dist_m = 0.0

    if dist_m > 0 and not (SIGN_DIST_MIN_M <= dist_m <= SIGN_DIST_MAX_M):
        return False, 0.0, None

    return True, dist_m, (int(x), int(y), int(w), int(h))
