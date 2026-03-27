"""
PSU Eco Racing — Perception Stack
lane/memory.py  |  Rolling lane-width history + virtual boundary projection.

When only one lane boundary is visible (tight turn, partial occlusion),
LaneMemory reconstructs the missing side using stored width and ZED depth.
"""

import numpy as np
from typing import Optional

from perception_stack.config import (
    ROI_TOP_FRACTION, POLY_DEG, LANE_MEM_MAX,
    LANE_WIDTH_MIN, LANE_WIDTH_MAX,
)
from perception_stack.lane.fitting import eval_x


class LaneMemory:
    """
    Accumulates real-world lane width observations over a rolling window.
    When only one boundary is visible (tight turn, occlusion), projects the
    missing side using stored width + depth from the point cloud so the
    virtual line stays geometrically correct under perspective.
    """

    def __init__(self):
        self._px_samples: list = []
        self._m_samples:  list = []
        self.mean_px: Optional[float] = None
        self.mean_m:  Optional[float] = None

    def update(self, lf, rf, pc: np.ndarray, H: int, W: int):
        if lf is None or rf is None:
            return
        for row in [int(H * f) for f in [0.95, 0.85, 0.75, 0.65, 0.55]]:
            lx = int(np.clip(eval_x(lf, row), 0, W - 1))
            rx = int(np.clip(eval_x(rf, row), 0, W - 1))
            if rx <= lx:
                continue
            self._px_samples.append(float(rx - lx))
            lp, rp = pc[row, lx, :3], pc[row, rx, :3]
            if np.isfinite(lp).all() and np.isfinite(rp).all():
                wm = abs(float(rp[0] - lp[0]))
                if LANE_WIDTH_MIN < wm < LANE_WIDTH_MAX:
                    self._m_samples.append(wm)
        if len(self._px_samples) > LANE_MEM_MAX:
            self._px_samples = self._px_samples[-LANE_MEM_MAX:]
        if len(self._m_samples) > LANE_MEM_MAX:
            self._m_samples = self._m_samples[-LANE_MEM_MAX:]
        if self._px_samples:
            self.mean_px = float(np.median(self._px_samples))
        if self._m_samples:
            self.mean_m = float(np.median(self._m_samples))

    def virtual_right(self, lf: np.ndarray, pc: np.ndarray,
                      H: int, W: int, fx: float) -> Optional[np.ndarray]:
        """Project right boundary from known left + real-world width."""
        if lf is None or self.mean_m is None:
            return None
        ys, vxs = [], []
        for row in np.arange(int(H * ROI_TOP_FRACTION), H, 8, dtype=int):
            lx = int(np.clip(np.polyval(lf, row), 0, W - 1))
            pt = pc[row, lx, :3]
            if np.isfinite(pt).all() and float(pt[2]) < 0:
                depth = abs(float(pt[2]))
                ys.append(float(row))
                vxs.append(float(np.clip(lx + self.mean_m * fx / depth, 0, W - 1)))
        if len(ys) >= POLY_DEG + 2:
            try:
                return np.polyfit(np.array(ys), np.array(vxs), POLY_DEG)
            except np.linalg.LinAlgError:
                pass
        if self.mean_px is None:
            return None
        vrf = lf.copy(); vrf[-1] += self.mean_px
        return vrf

    def virtual_left(self, rf: np.ndarray, pc: np.ndarray,
                     H: int, W: int, fx: float) -> Optional[np.ndarray]:
        """Project left boundary from known right − real-world width."""
        if rf is None or self.mean_m is None:
            return None
        ys, vxs = [], []
        for row in np.arange(int(H * ROI_TOP_FRACTION), H, 8, dtype=int):
            rx = int(np.clip(np.polyval(rf, row), 0, W - 1))
            pt = pc[row, rx, :3]
            if np.isfinite(pt).all() and float(pt[2]) < 0:
                depth = abs(float(pt[2]))
                ys.append(float(row))
                vxs.append(float(np.clip(rx - self.mean_m * fx / depth, 0, W - 1)))
        if len(ys) >= POLY_DEG + 2:
            try:
                return np.polyfit(np.array(ys), np.array(vxs), POLY_DEG)
            except np.linalg.LinAlgError:
                pass
        if self.mean_px is None:
            return None
        vlf = rf.copy(); vlf[-1] -= self.mean_px
        return vlf
