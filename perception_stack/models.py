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
    # Obstacle
    obstacle_detected:  bool  = False
    obstacle_dist_m:    float = 0.0
    obstacle_lateral_m: float = 0.0
    obstacle_bbox:      Optional[Tuple[int,int,int,int]] = None  # (x,y,w,h) px
    # Stop sign
    stop_sign:          bool  = False
    stop_sign_dist_m:   float = 0.0
    stop_sign_bbox:     Optional[Tuple[int,int,int,int]] = None  # (x,y,w,h) px
