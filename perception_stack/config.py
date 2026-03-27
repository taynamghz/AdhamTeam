"""
PSU Eco Racing — Perception Stack
config.py  |  All tunable parameters in one place.
Edit this file to tune thresholds without touching algorithm logic.
"""

import numpy as np
import pyzed.sl as sl

# ── Camera ─────────────────────────────────────────────────────────────────────
CAM_RES = sl.RESOLUTION.HD720
CAM_FPS  = 30

# ── Floor detection ────────────────────────────────────────────────────────────
FLOOR_TOLERANCE      = 0.10     # ±m around floor plane (nominal)
FLOOR_TOLERANCE_WIDE = 0.28     # wider band when uncertain — keeps lane mask alive
FLOOR_STABLE_HZ      = 4        # re-run find_floor_plane every N frames when stable
FLOOR_LOST_CONSEC    = 3        # after N consecutive failures → aggressive mode
ROI_TOP_FRACTION     = 0.35     # ignore top 35% of frame (sky / hood)
# Hit-test fallback grid: (frac_x, frac_y) of frame — bottom-centre first
FLOOR_HIT_POINTS = [(0.50, 0.75), (0.35, 0.80), (0.65, 0.80),
                    (0.50, 0.88), (0.35, 0.88), (0.65, 0.88)]

# ── White lane line  (HLS) ─────────────────────────────────────────────────────
WHITE_L_MIN = 160               # lower slightly for shadowed asphalt
WHITE_S_MAX = 65

# ── Grass / asphalt boundary  (HSV) ───────────────────────────────────────────
GRASS_H_MIN, GRASS_H_MAX = 25, 95
GRASS_S_MIN = 30
GRASS_V_MIN = 35

# ── Stop-line (orange horizontal stripe, floor pixels only) ───────────────────
STOP_ORANGE_H_MIN   = 5         # HSV hue lower  (orange)
STOP_ORANGE_H_MAX   = 20        # HSV hue upper  (orange)
STOP_ORANGE_S_MIN   = 150       # vivid orange only
STOP_ORANGE_V_MIN   = 100       # reject dark/shadowed patches
STOP_ROW_THRESH     = 0.08      # fraction of row width that must be orange
STOP_COVERAGE_MIN   = 0.60      # fraction of lane interior that must be lit
STOP_PERP_MAX_DEG   = 20.0      # cluster must be within ±20° of horizontal
STOP_DIST_MIN_M     = 0.3
STOP_DIST_MAX_M     = 10.0
STOP_DIST_N_PTS     = 10        # sample points for median distance
STOP_DIST_MIN_VALID = 4         # min valid samples needed
STOP_VOTE_NEEDED    = 5         # consecutive frames before triggering

# ── Polynomial lane fitting ────────────────────────────────────────────────────
POLY_DEG         = 2            # quadratic — handles curves
RANSAC_ITER      = 80
RANSAC_THRESH_PX = 6            # pixel residual for inlier test
MIN_INLIERS      = 30
MIN_PIXELS       = 40

# Sliding-window (initial search / lost-lane recovery)
N_WINDOWS  = 12
WIN_MARGIN = 60                 # px half-width per window
WIN_MINPIX = 25                 # min pixels to re-centre window

# Prior-guided search (fast path when previous fit exists)
PRIOR_MARGIN = 50               # px band around previous polynomial

# ── Temporal smoothing ─────────────────────────────────────────────────────────
SMOOTH_ALPHA = 0.30             # lower = smoother, higher = more responsive

# ── Lane geometry sanity ───────────────────────────────────────────────────────
LANE_WIDTH_MIN = 0.20
LANE_WIDTH_MAX = 12.0

# ── Lane memory (virtual boundary on single-side turns) ────────────────────────
LANE_MEM_MAX = 60               # rolling history length (frames × 5 samples)

# ── Confidence gates (per source) ──────────────────────────────────────────────
CONF_WHITE = 0.22
CONF_GRASS = 0.22

# ── Obstacle detection ─────────────────────────────────────────────────────────
OBS_MIN_HEIGHT_M   = 0.15    # m above floor — ignores road surface / small bumps
OBS_MAX_HEIGHT_M   = 2.50    # m above floor — ignores overhead signs
OBS_MIN_DIST_M     = 0.30    # closest distance to report
OBS_MAX_DIST_M     = 4.00    # furthest distance to report
OBS_LANE_MARGIN_M  = 0.40    # lateral margin beyond lane edges still counts
OBS_MIN_CLUSTER_PX = 60      # minimum point-cloud points per blob (noise rejection)
OBS_SDK_ENABLE     = True    # enable ZED object detection (people / vehicles)

# ── Bird's-eye warp ────────────────────────────────────────────────────────────
# Set WARP_ENABLED=True once you have measured SRC corners on your track.
# SRC = road trapezoid in camera image (fractions of W, H).
# DST = rectangle in bird's-eye output  (fractions of W, H).
WARP_ENABLED = False
WARP_SRC = np.float32([[0.40, 0.55], [0.60, 0.55],
                        [0.85, 0.95], [0.15, 0.95]])
WARP_DST = np.float32([[0.20, 0.05], [0.80, 0.05],
                        [0.80, 0.95], [0.20, 0.95]])

# ── Display ────────────────────────────────────────────────────────────────────
DISPLAY = True
