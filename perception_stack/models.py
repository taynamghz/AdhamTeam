"""
PSU Eco Racing — Perception Stack
models.py  |  Shared data structures passed between modules and to the controller.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class PerceptionResult:
    deviation_m:    float = 0.0
    confidence:     float = 0.0
    lane_width_m:   float = 0.0
    source:         str   = "NONE"
    left_fit:       Optional[np.ndarray] = None   # [a,b,c]  x = poly(y)
    right_fit:      Optional[np.ndarray] = None
    left_conf:      float = 0.0
    right_conf:     float = 0.0
    stop_line:          bool  = False
    stop_line_y:        Optional[int] = None
    stop_line_dist:     float = 0.0
    virtual_left:       bool  = False
    virtual_right:      bool  = False
    # Stop sign
    stop_sign:          bool  = False
    stop_sign_dist_m:   float = 0.0
    stop_sign_bbox:     Optional[Tuple[int,int,int,int]] = None  # (x,y,w,h) px
    # Control outputs (for Pure Pursuit / PID downstream)
    heading_angle:      float = 0.0                              # radians; θ = arctan(2ay+b)
    curvature:          float = 0.0                              # κ = 1/R  (m⁻¹); signed
    lookahead_point:    Optional[Tuple[float,float]] = None      # (X_m, Z_m) world space
    lookahead_pixel:    Optional[Tuple[int,int]]     = None      # (x, y) image pixels
