"""
PSU Eco Racing — Perception Stack
config.py  |  All tunable parameters in one place.
Edit this file to tune thresholds without touching algorithm logic.
"""

import numpy as np
import pyzed.sl as sl

# ── Camera ─────────────────────────────────────────────────────────────────────
CAM_RES        = sl.RESOLUTION.HD720
CAM_FPS        = 30
# NEURAL  = best depth quality, heaviest GPU load (~15-18 FPS real on Nano)
# QUALITY = slightly noisier depth, ~2× faster — recommended for real-time
# ULTRA   = fastest, noisiest — only if QUALITY still too slow
CAM_DEPTH_MODE = sl.DEPTH_MODE.PERFORMANCE

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
RANSAC_ITER      = 40
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

# ── Feature flags ─────────────────────────────────────────────────────────────
# Set LANE_ENABLED = False to skip all floor/lane/obstacle/parking processing.
# Only stop-sign detection + distance runs. Flip to True to re-enable everything.
LANE_ENABLED = False

# ── Stop-sign detection (YOLOv8) ──────────────────────────────────────────────
# Train: python scripts/train_stop_sign.py --api-key YOUR_KEY
# Export to TensorRT (run on Jetson): python scripts/export_trt.py
# Roboflow dataset: universe.roboflow.com/yolo-ifyjn/stop-sign-detection-1
#   class 0 = stop-sign          ← real sign, we want this
#   class 1 = stop-sign-fake
#   class 2 = stop-sign-vandalized  ← still a real stop sign
SIGN_MODEL_PATH      = "perception_stack/weights/stop_sign.pt"   # swap to .engine after TRT export
SIGN_CONF_THRESH     = 0.60                      # YOLO confidence threshold (raised: FP16 quant noise + SEM-specific sign)
SIGN_IMG_SIZE        = 416                       # inference resolution (faster on Nano)
SIGN_ACCEPT_CLASSES  = {0, 2}                    # 0=stop-sign, 2=stop-sign-vandalized
SIGN_SKIP_FRAMES     = 3                         # run YOLO every N frames; cache between
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
UART_HEARTBEAT_S   = 0.080          # force retransmit every 80ms — keeps Nucleo watchdog
                                    # alive (200ms timeout) during sustained THROTTLE

# ── Vehicle commands ────────────────────────────────────────────────────────────
STOP_BRAKE_DIST_M  = 1.0   # m — stop line/sign must be within this to trigger BRAKE
THROTTLE_VALUE     = 189   # 0-255 sent with THROTTLE frame
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

# ── Profiling ──────────────────────────────────────────────────────────────────
# Prints per-step timing table every 30 frames to identify bottlenecks.
PROFILE_ENABLED      = True
PROFILE_PRINT_EVERY  = 30   # frames

# ── CLAHE — lighting normalisation applied before all colour thresholds ─────────
# Equalises the L channel (LAB space) so that auto-exposure shifts, shadows,
# and overcast vs. direct-sun conditions don't collapse fixed HSV/HLS gates.
CLAHE_CLIP_LIMIT  = 2.0
CLAHE_TILE_SIZE   = (8, 8)    # (width, height) grid in tiles

# ── ZED IMU tilt compensation ──────────────────────────────────────────────────
# When the vehicle pitches or rolls, the fixed BEV homography becomes incorrect.
# IMU data shifts the bottom warp source points to compensate.
# Tune PITCH_BASELINE_DEG after mounting the camera in its final position.
PITCH_BASELINE_DEG = 0.0      # camera pitch (deg) recorded during BEV calibration
PITCH_PX_PER_DEG   = 8.0      # BEV source shift (px) per degree of pitch — tune on vehicle
ROLL_PX_PER_DEG    = 3.0      # smaller lateral effect — tune on vehicle

# ── Stop line — physical stripe-width gate ─────────────────────────────────────
# SEM stop line spans full track width (≥ 2 m).
# Reject orange detections that are narrower than this fraction of lane width —
# eliminates orange cones, narrow debris, partial shadows.
STOP_WIDTH_MIN_FRAC = 0.70    # stripe must be ≥ 70% of measured lane width

# ── Stop sign — SEM-specific hardening ────────────────────────────────────────
# The SEM stop sign is a red hexagon on a YELLOW BOARD.
# A secondary yellow-HSV gate around the YOLO bbox rejects generic red objects.
SIGN_YELLOW_H_MIN    = 18     # HSV hue range for SEM yellow board
SIGN_YELLOW_H_MAX    = 38
SIGN_YELLOW_S_MIN    = 120    # vivid yellow only — rejects faded paint
SIGN_YELLOW_V_MIN    = 150
SIGN_YELLOW_ROI_FRAC = 1.3    # expand bbox by this factor when sampling for yellow
SIGN_YELLOW_AREA_FRAC = 0.12  # minimum yellow fraction in expanded roi
# Bbox height sanity check: at distance d, sign should subtend ~(H_m/d)*fy pixels
SIGN_FY_APPROX       = 730    # approx. vertical focal length at 720p (px)
SIGN_HEIGHT_M        = 0.65   # assumed sign height (m) — midpoint of 0.5–1.0m spec
SIGN_BBOX_MIN_FRAC   = 0.35   # bbox height must be ≥ this fraction of expected px height

# ── Obstacle detection — SEM blue inflatable pin (ABOVE floor plane) ──────────
# Pin spec: 1.10 m tall × 0.45 m diameter, blue.
# Key: obstacle pixels are blue AND NOT on floor_mask.
#      Parking pixels are blue AND ON floor_mask.  (same colour, different plane)
OBS_BLUE_H_MIN     = 100      # HSV hue range — blue
OBS_BLUE_H_MAX     = 135
OBS_BLUE_S_MIN     = 80       # reject sky glare and pale blues
OBS_BLUE_V_MIN     = 60
OBS_MIN_PIXELS     = 150      # minimum connected-component area (px)
OBS_ASPECT_MIN     = 1.2      # blob height/width ratio — pin is taller than wide
OBS_HEIGHT_3D_MIN  = 0.3      # minimum 3D height span in world Y (m)
OBS_HEIGHT_3D_MAX  = 1.5      # maximum 3D height span (m)
OBS_DIST_MIN_M     = 0.4
OBS_DIST_MAX_M     = 8.0
OBS_VOTE_NEEDED    = 3        # temporal vote gate (frames)

# ── Parking bay detection — blue floor markings (ON floor plane) ───────────────
# Bay spec: 0.15 m-wide blue lines, white borders, 4 m × 2 m.
# floor_mask separates these from the obstacle pin (same blue HSV range).
PARK_BLUE_H_MIN    = 100
PARK_BLUE_H_MAX    = 135
PARK_BLUE_S_MIN    = 70
PARK_BLUE_V_MIN    = 60
PARK_MIN_PIXELS    = 400      # minimum floor-blue pixel count to consider a bay
PARK_DIST_MIN_M    = 0.5
PARK_DIST_MAX_M    = 8.0
PARK_VOTE_NEEDED   = 4        # temporal vote gate (frames)
PARK_OBS_THRESHOLD = 30       # max ZED points > 0.2 m above floor → bay is occupied

# ── Depth / point-cloud refresh ───────────────────────────────────────────────
# Retrieve full XYZ point cloud at most every N frames.
# Will refresh earlier if any detection vote gate is active.
# Higher = faster pipeline, slightly stale distances between refreshes.
PC_REFRESH_EVERY   = 4

# ── FPS monitoring ─────────────────────────────────────────────────────────────
FPS_WARN_BELOW     = 20.0     # print warning if rolling average FPS drops below this

# ── Telemetry logging ──────────────────────────────────────────────────────────
LOG_TELEMETRY      = True
LOG_DIR            = "logs"
