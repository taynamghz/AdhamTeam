"""
PSU Eco Racing — Perception Stack
lane/control.py  |  Control-level outputs from polynomial lane fits.

Computes heading angle, curvature, and lookahead point reusing the
existing smoothed polynomial fits — no new expensive steps.

Coordinate conventions (ZED RIGHT_HANDED_Y_UP):
  X  : lateral (positive = right)
  Z  : forward distance = abs(pt[2])  (ZED Z is negative-forward)
  y  : image row — large y = near vehicle, small y = far ahead
"""

import numpy as np
from typing import Optional, Tuple

from perception_stack.config import (
    CTRL_LOOKAHEAD_M,
    CTRL_HEADING_ALPHA,
    CTRL_CURVATURE_ALPHA,
    CTRL_EVAL_Y_FRAC,
    ROI_TOP_FRACTION,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _center_fit(
    lf: Optional[np.ndarray],
    rf: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Average left and right fits into a centerline poly.  Uses whichever side
    is available when the other is None."""
    if lf is not None and rf is not None:
        return (lf + rf) / 2.0
    return lf if lf is not None else rf


# ── Public computation functions ───────────────────────────────────────────────

def compute_heading(
    lf: Optional[np.ndarray],
    rf: Optional[np.ndarray],
    y_eval: float,
) -> float:
    """
    Lane heading angle θ at image row y_eval.

    For the polynomial  x = a·y² + b·y + c  the tangent slope is
        dx/dy = 2a·y + b
    so θ = arctan(2a·y + b).

    Returns radians.  Positive θ means the lane direction tilts toward
    increasing x (rightward in image), which indicates a left road curve
    ahead (standard perspective geometry).
    """
    cf = _center_fit(lf, rf)
    if cf is None:
        return 0.0
    a, b = float(cf[0]), float(cf[1])
    return float(np.arctan(2.0 * a * y_eval + b))


def compute_curvature(
    lf: Optional[np.ndarray],
    rf: Optional[np.ndarray],
    y_eval: float,
    lane_width_px: float,
    lane_width_m: float,
) -> float:
    """
    Signed metric curvature  κ = 1/R  (m⁻¹).

    Derivation:
        κ_px = f''(y) / (1 + f'(y)²)^(3/2)   [image-space, units 1/px]
             = 2a / (1 + (2ay+b)²)^(3/2)

    Metric conversion via lane-width calibration:
        px_per_m ≈ lane_width_px / lane_width_m
        κ_metric = κ_px * px_per_m

    Sign convention: positive κ = curving left (road bends left ahead).
    Returns 0.0 when lane width calibration is unavailable.
    """
    cf = _center_fit(lf, rf)
    if cf is None or lane_width_px < 1.0 or lane_width_m < 0.1:
        return 0.0
    a, b = float(cf[0]), float(cf[1])
    slope = 2.0 * a * y_eval + b
    kappa_px = (2.0 * a) / (1.0 + slope ** 2) ** 1.5
    px_per_m = lane_width_px / lane_width_m
    return float(kappa_px * px_per_m)


def compute_lookahead(
    lf: Optional[np.ndarray],
    rf: Optional[np.ndarray],
    pc: np.ndarray,
    H: int,
    W: int,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]]]:
    """
    Lookahead point on the lane centreline closest to CTRL_LOOKAHEAD_M ahead.

    Scans image rows from the ROI top down to the image bottom and finds
    the centreline pixel whose ZED-measured forward distance (abs Z) is
    nearest to CTRL_LOOKAHEAD_M.

    Returns
    -------
    world_pt : (X_m, Z_m) | None
        X_m  — lateral offset in metres (positive = right of camera)
        Z_m  — forward distance in metres
    pixel_pt : (x, y) | None
        Pixel coordinates of the lookahead point in the camera image.
    """
    cf = _center_fit(lf, rf)
    if cf is None:
        return None, None

    best_world: Optional[Tuple[float, float]] = None
    best_pixel: Optional[Tuple[int, int]]     = None
    best_dz = float('inf')

    for y in range(int(H * ROI_TOP_FRACTION), H, 4):
        cx = int(np.clip(np.polyval(cf, y), 0, W - 1))
        pt = pc[y, cx, :3]
        if not np.isfinite(pt).all():
            continue
        z_fwd = abs(float(pt[2]))   # ZED RIGHT_HANDED_Y_UP: forward = −Z → abs
        if z_fwd <= 0.0:
            continue
        dz = abs(z_fwd - CTRL_LOOKAHEAD_M)
        if dz < best_dz:
            best_dz  = dz
            best_world = (float(pt[0]), z_fwd)
            best_pixel = (cx, int(y))

    return best_world, best_pixel


# ── Temporal smoother for scalar control outputs ───────────────────────────────

class ControlSmoother:
    """
    Separate EMA for heading angle and curvature.

    Uses lower alpha values than the polynomial smoother so that these
    derived quantities (which amplify high-frequency noise) are extra stable.
    """

    def __init__(self) -> None:
        self._heading:   Optional[float] = None
        self._curvature: Optional[float] = None

    def update(self, heading: float, curvature: float) -> Tuple[float, float]:
        if self._heading is None:
            self._heading = heading
        else:
            self._heading = (CTRL_HEADING_ALPHA * heading
                             + (1.0 - CTRL_HEADING_ALPHA) * self._heading)

        if self._curvature is None:
            self._curvature = curvature
        else:
            self._curvature = (CTRL_CURVATURE_ALPHA * curvature
                               + (1.0 - CTRL_CURVATURE_ALPHA) * self._curvature)

        return self._heading, self._curvature
