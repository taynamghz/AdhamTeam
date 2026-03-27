"""
PSU Eco Racing — Perception Stack
lane/smoother.py  |  EMA smoother for polynomial lane coefficients.
"""

import numpy as np
from typing import Optional

from perception_stack.config import SMOOTH_ALPHA


class Smoother:
    """Exponential moving average over left/right lane polynomial coefficients."""

    def __init__(self, alpha: float = SMOOTH_ALPHA):
        self.alpha = alpha
        self.l_ema: Optional[np.ndarray] = None
        self.r_ema: Optional[np.ndarray] = None

    def update(self, lf, rf):
        def _ema(cur, new):
            if new is None:
                return cur
            if cur is None or cur.shape != new.shape:
                return new.copy()
            return self.alpha * new + (1.0 - self.alpha) * cur
        self.l_ema = _ema(self.l_ema, lf)
        self.r_ema = _ema(self.r_ema, rf)
        return self.l_ema, self.r_ema
