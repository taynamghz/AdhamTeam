"""
PSU Eco Racing — Perception Stack
lane/deviation.py  |  Metric deviation from lane centre using ZED point cloud.
"""

import numpy as np
from typing import Optional

from perception_stack.config import LANE_WIDTH_MIN, LANE_WIDTH_MAX
from perception_stack.lane.fitting import eval_x


def compute_deviation(lf, rf, pc: np.ndarray, H: int, W: int):
    """
    Compute signed lateral deviation from lane centre (metres) and lane width.

    Samples multiple rows, weights closer rows more heavily, and uses the
    ZED point cloud X-coordinate difference as the metric width reference.

    Returns (deviation_m, lane_width_m).
      deviation_m > 0  →  vehicle is left of centre
      deviation_m < 0  →  vehicle is right of centre
    """
    devs, widths = [], []
    sample_ys = [int(H * f) for f in [0.95, 0.85, 0.75, 0.65]]
    weights   = [0.40, 0.30, 0.20, 0.10]

    for sy, w in zip(sample_ys, weights):
        lx = int(np.clip(eval_x(lf, sy), 0, W-1)) if lf is not None else None
        rx = int(np.clip(eval_x(rf, sy), 0, W-1)) if rf is not None else None

        if lx is not None and rx is not None and rx > lx:
            lp, rp = pc[sy, lx, :3], pc[sy, rx, :3]
            if np.isfinite(lp).all() and np.isfinite(rp).all():
                wm = abs(float(rp[0] - lp[0]))
                if LANE_WIDTH_MIN < wm < LANE_WIDTH_MAX:
                    dev_m = (W / 2.0 - (lx + rx) / 2.0) / (rx - lx) * wm
                    devs.append((dev_m, w))
                    widths.append(wm)
        elif lx is not None:
            lp = pc[sy, lx, :3]
            if np.isfinite(lp).all() and widths:
                half = float(np.median(widths)) / 2.0
                devs.append(((W // 2 - lx) / (W // 2) * half, w * 0.4))
        elif rx is not None:
            rp = pc[sy, rx, :3]
            if np.isfinite(rp).all() and widths:
                half = float(np.median(widths)) / 2.0
                devs.append(((W // 2 - rx) / (W // 2) * half, w * 0.4))

    if not devs:
        return 0.0, 0.0
    tw  = sum(d[1] for d in devs)
    dev = sum(d[0] * d[1] for d in devs) / tw
    wid = float(np.median(widths)) if widths else 0.0
    return dev, wid
