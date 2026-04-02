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
    PITCH_BASELINE_DEG, PITCH_PX_PER_DEG, ROLL_PX_PER_DEG,
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

    def update_tilt(self, pitch_deg: float, roll_deg: float, W: int, H: int) -> None:
        """
        Dynamically adjust the perspective warp to compensate for IMU pitch/roll.

        Called each frame from pipeline.py when WARP_ENABLED is True.
        Only recomputes matrices when deviation from the calibration baseline
        exceeds 0.5° to avoid unnecessary work.

        Sign conventions (ZED IMU, RIGHT_HANDED_Y_UP):
          pitch > 0 → nose tilts up  → road features appear lower in the frame
                       → shift bottom warp source points upward (smaller y)
          roll  > 0 → car leans right → shift asymmetrically left/right
        """
        delta_pitch = pitch_deg - PITCH_BASELINE_DEG
        delta_roll  = roll_deg           # roll baseline assumed 0°

        if abs(delta_pitch) < 0.5 and abs(delta_roll) < 0.5:
            return    # within dead-band — keep existing matrices

        src = WARP_SRC.copy() * np.array([W, H], dtype=np.float32)
        dst = WARP_DST        * np.array([W, H], dtype=np.float32)

        # Pitch: shift the two bottom source corners vertically
        pitch_shift = delta_pitch * PITCH_PX_PER_DEG
        src[2, 1] -= pitch_shift    # bottom-right corner
        src[3, 1] -= pitch_shift    # bottom-left corner

        # Roll: shift bottom corners horizontally (smaller effect)
        roll_shift = delta_roll * ROLL_PX_PER_DEG
        src[2, 0] += roll_shift     # bottom-right shifts right on positive roll
        src[3, 0] -= roll_shift     # bottom-left shifts left

        # Clamp to valid image coordinates
        src[:, 0] = np.clip(src[:, 0], 0, W - 1)
        src[:, 1] = np.clip(src[:, 1], 0, H - 1)

        try:
            self.M    = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
            self.Minv = cv2.getPerspectiveTransform(dst.astype(np.float32), src)
        except cv2.error:
            pass    # degenerate points — keep previous matrices

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
