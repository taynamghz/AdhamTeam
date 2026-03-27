"""
PSU Eco Racing — Perception Stack
perception/warp.py  |  Bird's-eye perspective warp transform.

Caches perspective matrices.  Fits are performed in BEV space then
re-projected back to image space so deviation + drawing remain
coordinate-agnostic.  Enable by setting WARP_ENABLED=True in config.py
and calibrating WARP_SRC/DST for your camera mount.
"""

import numpy as np
import cv2
from typing import Optional

from perception_stack.config import (
    WARP_SRC, WARP_DST, ROI_TOP_FRACTION, POLY_DEG,
)


class WarpTransform:

    def __init__(self):
        self.M    = None
        self.Minv = None

    def build(self, W: int, H: int):
        src = WARP_SRC * np.array([W, H], dtype=np.float32)
        dst = WARP_DST * np.array([W, H], dtype=np.float32)
        self.M    = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        return cv2.warpPerspective(img, self.M, (W, H), flags=cv2.INTER_LINEAR)

    def bev_poly_to_img(
        self, fit_bev: np.ndarray, H: int, W: int
    ) -> Optional[np.ndarray]:
        """Sample a BEV polynomial and unproject the curve to image space."""
        ys  = np.linspace(int(H * ROI_TOP_FRACTION), H - 1, 40)
        xs  = np.polyval(fit_bev, ys)
        pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
        try:
            pts_img = cv2.perspectiveTransform(pts, self.Minv).reshape(-1, 2)
        except cv2.error:
            return None
        ix, iy = pts_img[:, 0], pts_img[:, 1]
        valid  = (np.isfinite(ix) & np.isfinite(iy) &
                  (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H))
        if valid.sum() < POLY_DEG + 1:
            return None
        try:
            return np.polyfit(iy[valid], ix[valid], POLY_DEG)
        except np.linalg.LinAlgError:
            return None
