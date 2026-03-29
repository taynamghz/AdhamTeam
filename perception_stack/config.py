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
LANE_WIDTH_MIN   = 0.20
LANE_WIDTH_MAX   = 12.0
# Absolute pixel floor for lf/rf separation (protects the first frames before
# lane memory has built up — prevents degenerate same-feature fits).
MIN_LANE_SEP_PX  = 40
# Memory-relative width gate: if the raw fit separation is less than this
# fraction of the remembered lane width, one fit is contaminated (e.g. by an
# obstacle edge).  The drifting fit is discarded; lane memory fills the gap.
LANE_SEP_MEM_FRAC = 0.70

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
OBS_SDK_ENABLE     = False   # enable ZED object detection (people / vehicles)

# ── Stop-sign detection ────────────────────────────────────────────────────────
# Red HSV ranges (hue wraps: 0-H_LOW_MAX and H_HIGH_MIN-180)
SIGN_RED_H_LOW_MAX   = 10      # upper bound of low-red hue band
SIGN_RED_H_HIGH_MIN  = 160     # lower bound of high-red hue band
SIGN_RED_S_MIN       = 120     # minimum saturation — reject pale/pink
SIGN_RED_V_MIN       = 60      # minimum value — reject very dark red
# Contour / shape filters
SIGN_MIN_AREA_PX     = 800     # ignore tiny blobs (far-away / noise)
SIGN_MAX_AREA_PX     = 80_000  # ignore blobs that fill most of the frame
SIGN_POLY_SIDES_MIN  = 6       # octagon seen at distance may appear 6-sided
SIGN_POLY_SIDES_MAX  = 10      # allow some detection slop
SIGN_ASPECT_MIN      = 0.5     # bounding-rect W/H — rejects thin red banners
SIGN_ASPECT_MAX      = 1.8
# Distance gate
SIGN_DIST_MIN_M      = 0.5
SIGN_DIST_MAX_M      = 15.0
# Temporal vote gate (frames) — avoids single-frame false positives
SIGN_VOTE_NEEDED     = 3

# ── Bird's-eye warp ────────────────────────────────────────────────────────────
# Set WARP_ENABLED=True once you have measured SRC corners on your track.
# SRC = road trapezoid in camera image (fractions of W, H).
# DST = rectangle in bird's-eye output  (fractions of W, H).
WARP_ENABLED = False
WARP_SRC = np.float32([[0.40, 0.55], [0.60, 0.55],
                        [0.85, 0.95], [0.15, 0.95]])
WARP_DST = np.float32([[0.20, 0.05], [0.80, 0.05],
                        [0.80, 0.95], [0.20, 0.95]])

# ── UART / low-level controller ────────────────────────────────────────────────
UART_ENABLED       = True           # set False for vision-only testing (dry run)
UART_PORT          = "/dev/ttyTHS1" # Jetson hardware UART; change to /dev/ttyUSB0 etc.
UART_BAUD          = 115200
UART_TIMEOUT_S     = 0.01           # serial read timeout (s)
UART_ACK_TIMEOUT_S = 0.05           # how long to wait for MCU echo (s)

# ── Vehicle commands ────────────────────────────────────────────────────────────
BRAKE_DIST_M       = 1.5   # m — obstacle closer than this triggers BRAKE
STOP_BRAKE_DIST_M  = 1.0   # m — stop line/sign must be within this to trigger BRAKE
THROTTLE_VALUE     = 20    # 0-255 sent with THROTTLE frame
BRAKE_VALUE        = 255   # 0-255 sent with BRAKE frame

# ── Control outputs ────────────────────────────────────────────────────────────
# Lookahead distance for Pure Pursuit / PID feed-forward (metres)
CTRL_LOOKAHEAD_M     = 2.5
# EMA alpha for heading angle smoothing (lower = smoother, higher = responsive)
CTRL_HEADING_ALPHA   = 0.20
# EMA alpha for curvature (extra-smooth — small errors get amplified in κ)
CTRL_CURVATURE_ALPHA = 0.15
# Image-row fraction used to evaluate heading & curvature (0=top, 1=bottom)
CTRL_EVAL_Y_FRAC     = 0.60

# ── Display ────────────────────────────────────────────────────────────────────
DISPLAY = True
