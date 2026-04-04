"""
PSU Eco Racing — Perception Stack
lane/fitting.py  |  Polynomial evaluation helper.

The RANSAC/sliding-window fitter was removed when Segformer replaced
colour-threshold lane detection.  eval_x is kept because stop_line.py,
lane/control.py, and visualization.py all use it to query a polynomial.
"""

import numpy as np


def eval_x(coeffs: np.ndarray, y: float) -> float:
    """Evaluate polynomial  x = poly(y)  at image row y."""
    return float(np.polyval(coeffs, y))
