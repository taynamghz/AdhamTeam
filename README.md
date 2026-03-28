# PSU Eco Racing — Perception Stack: Full Technical Report
**Prince Sultan University | Shell Eco-Marathon 2026**
*Document date: March 2026*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Model — PerceptionResult](#3-data-model--perceptionresult)
4. [Configuration — config.py](#4-configuration--configpy)
5. [Entry Point — main.py](#5-entry-point--mainpy)
6. [Pipeline — perception/pipeline.py](#6-pipeline--perceptionpipelinepy)
7. [Lane Detection Algorithms](#7-lane-detection-algorithms)
   - 7.1 Floor Plane Estimation
   - 7.2 Colour Masking
   - 7.3 Lane Fitting (RANSAC, Sliding Window, Prior-Guided Search)
   - 7.4 Fit Contamination Guard
   - 7.5 Temporal Smoothing (EMA)
   - 7.6 Virtual Boundaries / Lane Memory
   - 7.7 Lateral Deviation
8. [Detection Algorithms](#8-detection-algorithms)
   - 8.1 Stop Line Detection
   - 8.2 Stop Sign Detection
   - 8.3 Obstacle Detection (Point Cloud)
   - 8.4 Temporal Vote Gates
9. [Control Outputs](#9-control-outputs)
   - 9.1 Heading Angle
   - 9.2 Curvature
   - 9.3 Lookahead Point
   - 9.4 Control Smoothing
10. [Commander — control/commander.py](#10-commander--controlcommanderpyy)
11. [UART Transport — control/uart.py](#11-uart-transport--controluartpy)
12. [Visualization — visualization.py](#12-visualization--visualizationpy)
13. [Full Frame Execution Flow](#13-full-frame-execution-flow)
14. [Module Communication Map](#14-module-communication-map)
15. [Tuning Guide](#15-tuning-guide)
16. [Known Gaps and Watchdog Note](#16-known-gaps-and-watchdog-note)

---

## 1. System Overview

The perception stack runs on a Jetson Nano (Brain Layer) and forms the high-level decision side of a two-tier autonomous drive-by-wire system. Each frame it:

1. Reads a stereo frame and XYZ point cloud from a ZED 2i depth camera.
2. Estimates and tracks the road floor plane in 3D.
3. Detects white lane lines (or grass boundary edges as fallback) and fits quadratic polynomials to them.
4. Computes metric lateral deviation, heading angle, curvature, and lookahead point from the fits.
5. Detects orange stop lines, red stop signs, and 3D point-cloud obstacles.
6. Passes all perception outputs through temporal vote gates to suppress false positives.
7. Feeds a `PerceptionResult` dataclass into the Commander, which decides THROTTLE / BRAKE / IDLE.
8. Serialises the command as a 5-byte binary frame and sends it over UART to the STM32 Nucleo H743 (Low Level Controller), which actuates the motor, brake servo, and steering.

```
ZED 2i Camera
    │
    ▼
LanePerception.process()          ← called every frame in main.py
    │   ├─ Floor plane (ZED SDK)
    │   ├─ Colour masks (HLS/HSV)
    │   ├─ Lane fitting (RANSAC + sliding window)
    │   ├─ Contamination guard
    │   ├─ EMA smoother
    │   ├─ Lane memory / virtual boundaries
    │   ├─ Deviation + control outputs
    │   ├─ Stop line detection
    │   ├─ Stop sign detection
    │   └─ Obstacle detection (PC + ZED SDK)
    │
    ▼
PerceptionResult  (dataclass)
    │
    ▼
Commander._decide()
    │   ├─ Obstacle within 1.5 m  → BRAKE
    │   ├─ Stop line confirmed     → BRAKE
    │   ├─ Stop sign confirmed     → BRAKE
    │   ├─ Lane LOST / NO_FLOOR   → IDLE
    │   └─ otherwise              → THROTTLE
    │
    ▼
UARTController.send()
    │
    ▼
[0xAA][LEN=2][CMD][VAL][CRC8]  → /dev/ttyTHS1 @ 115200 baud
    │
    ▼
STM32 Nucleo H743 (LLC)
    ├─ RC PWM → KYDBL4850-1E drive controller → BLDC motor
    ├─ STEP/DIR → CL57T stepper driver → steering motor
    └─ PWM → brake servos x2
```

---

## 2. Repository Structure

```
perception_stack/
│
├── config.py               ALL tunable parameters in one file
├── models.py               PerceptionResult dataclass (shared across all modules)
├── main.py                 Entry point — loop, logging, display
├── visualization.py        OpenCV debug overlay on the live frame
│
├── perception/
│   ├── pipeline.py         LanePerception class — orchestrates every frame
│   └── warp.py             Optional bird's-eye-view (BEV) perspective warp
│
├── lane/
│   ├── fitting.py          RANSAC, sliding window, prior-guided polynomial fitting
│   ├── smoother.py         EMA smoother for polynomial coefficients
│   ├── memory.py           Rolling lane-width history + virtual boundary projection
│   ├── deviation.py        Metric lateral deviation from lane centre
│   └── control.py          Heading angle, curvature, lookahead point + ControlSmoother
│
├── detection/
│   ├── stop_line.py        Orange horizontal stop-stripe detector
│   ├── stop_sign.py        Red octagon stop-sign detector
│   └── obstacle.py         3D point-cloud obstacle detector
│
└── control/
    ├── commander.py        High-level THROTTLE/BRAKE/IDLE decision layer
    └── uart.py             5-byte binary UART framing to STM32 LLC
```

**Key design principle:** `config.py` is the only file that needs editing for tuning. All algorithm files import their thresholds from there — no magic numbers live in algorithm code.

---

## 3. Data Model — PerceptionResult

**File:** `models.py`

Every module in the system communicates through a single shared dataclass:

```python
@dataclass
class PerceptionResult:
    # Lane geometry
    deviation_m:        float = 0.0       # +left / -right of centre (metres)
    confidence:         float = 0.0       # 0.0–0.99, average of left+right
    lane_width_m:       float = 0.0       # physical width of lane (metres)
    source:             str   = "LOST"    # "WHITE_LINE" | "GRASS" | "LOST" | "NO_FLOOR"

    # Polynomial fits  x = a·y² + b·y + c  (image-space, evaluated on rows)
    left_fit:           Optional[np.ndarray] = None
    right_fit:          Optional[np.ndarray] = None
    left_conf:          float = 0.0
    right_conf:         float = 0.0

    # Stop line
    stop_line:          bool  = False     # confirmed (vote gate passed)
    stop_line_y:        Optional[int] = None   # image row of stripe
    stop_line_dist:     float = 0.0       # metres to stripe

    # Virtual boundaries (True when lane memory reconstructed the side)
    virtual_left:       bool = False
    virtual_right:      bool = False

    # Obstacles
    obstacle_detected:  bool  = False
    obstacle_dist_m:    float = 0.0       # forward distance (metres)
    obstacle_lateral_m: float = 0.0       # +left / -right (metres)
    obstacle_bbox:      Optional[Tuple[int,int,int,int]] = None  # (x,y,w,h) px

    # Stop sign
    stop_sign:          bool  = False     # confirmed (vote gate passed)
    stop_sign_dist_m:   float = 0.0
    stop_sign_bbox:     Optional[Tuple[int,int,int,int]] = None

    # Control outputs
    heading_angle:      float = 0.0       # radians, EMA-smoothed
    curvature:          float = 0.0       # m⁻¹, EMA-smoothed, signed
    lookahead_point:    Optional[Tuple[float,float]] = None  # (X_m, Z_m) world
    lookahead_pixel:    Optional[Tuple[int,int]] = None      # (x, y) image
```

`PerceptionResult` is produced by `pipeline.py`, consumed by `commander.py` (for commands) and `visualization.py` (for display). No other module writes to it.

---

## 4. Configuration — config.py

This is the **single source of truth** for all thresholds. Every section is documented below including how changes affect the system.

### 4.1 Camera

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CAM_RES` | HD720 | Resolution passed to ZED SDK at init. Higher = better detection at range, more CPU |
| `CAM_FPS` | 30 | Frame rate. Lower FPS frees GPU for depth processing |

### 4.2 Floor Detection

| Parameter | Default | Effect |
|-----------|---------|--------|
| `FLOOR_TOLERANCE` | 0.10 m | ±band around floor_y — pixels within this Y range are treated as road surface. **Too small** → sparse floor mask, lane detection degrades. **Too large** → mask bleeds onto walls/obstacles |
| `FLOOR_TOLERANCE_WIDE` | 0.28 m | Used when floor detection is uncertain (miss count ≥ 3). Keeps lane alive during brief floor track loss |
| `FLOOR_STABLE_HZ` | 4 | Re-run `find_floor_plane()` every N frames in stable mode. Higher = less ZED SDK load, slower adaptation to slope changes |
| `FLOOR_LOST_CONSEC` | 3 | After N consecutive floor misses, switch to aggressive mode (retry every frame + fallback hit-tests) |
| `ROI_TOP_FRACTION` | 0.35 | Top 35% of frame is masked from all detection — ignores sky, overhead signs, and camera housing |
| `FLOOR_HIT_POINTS` | 6 points | Fallback grid tested when `find_floor_plane` fails. Coordinates are (frac_x, frac_y) of frame. Add more points near track edges if floor detection is fragile |

### 4.3 White Lane Line (HLS)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `WHITE_L_MIN` | 160 | Minimum HLS Lightness for a pixel to count as white lane line. **Lower** → picks up more of the line under shadow. **Too low** → light-coloured road surface generates false positives |
| `WHITE_S_MAX` | 65 | Maximum HLS Saturation. High saturation means a pixel is coloured, not white — raising this allows slightly cream lines |

### 4.4 Grass / Asphalt Boundary (HSV)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `GRASS_H_MIN` | 25 | Lower hue bound for grass (yellow-green) |
| `GRASS_H_MAX` | 95 | Upper hue bound for grass (blue-green). Widening this range captures more diverse ground cover |
| `GRASS_S_MIN` | 30 | Minimum saturation — prevents grey road from being classified as grass |
| `GRASS_V_MIN` | 35 | Minimum value — prevents shadowed dark patches |

### 4.5 Stop Line (Orange)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `STOP_ORANGE_H_MIN/MAX` | 5–20 | HSV hue range for orange paint. Orange is between red and yellow in HSV. **Widen** if detection misses faded lines; **narrow** to suppress red/yellow false detects |
| `STOP_ORANGE_S_MIN` | 150 | High saturation enforces vivid orange only — rejects pale road surface |
| `STOP_ORANGE_V_MIN` | 100 | Minimum brightness — rejects shadowed orange |
| `STOP_ROW_THRESH` | 0.08 | A row must have ≥8% of its width lit orange to be a candidate. **Lower** → detects faded or partially obscured lines; **too low** → false triggers on orange objects |
| `STOP_COVERAGE_MIN` | 0.60 | Orange must span ≥60% of the measured lane width at that row. Prevents partial marks (cone, debris) from triggering |
| `STOP_PERP_MAX_DEG` | 20° | Maximum angle of the orange cluster's principal axis from horizontal. Rejects diagonal shadows and angled markings via PCA |
| `STOP_DIST_MIN/MAX_M` | 0.3–10.0 m | Valid distance range for the measured stop line. Outside = ignored |
| `STOP_DIST_N_PTS` | 10 | Number of point cloud samples across lane to median-distance the stop line |
| `STOP_DIST_MIN_VALID` | 4 | Minimum valid (finite, in-range) samples for a distance to be reported |
| `STOP_VOTE_NEEDED` | 5 | The stop line must be detected in 5 consecutive frames before confirmed. This prevents a single-frame orange artefact from triggering BRAKE |

### 4.6 Polynomial Lane Fitting

| Parameter | Default | Effect |
|-----------|---------|--------|
| `POLY_DEG` | 2 | Quadratic fit — handles curves. Changing to 1 (linear) makes it faster but can't handle bends |
| `RANSAC_ITER` | 80 | Number of RANSAC trials. More = better outlier rejection but slower. Below ~30 the fit becomes unreliable |
| `RANSAC_THRESH_PX` | 6 px | A pixel is an inlier if its residual from the current model is < 6 px. **Lower** = stricter fit (rejects more, fewer false inliers); **higher** = looser (tolerates noisy lines) |
| `MIN_INLIERS` | 30 | Minimum inlier count to accept a RANSAC fit. Below this the lane is too fragmented |
| `MIN_PIXELS` | 40 | Minimum total pixels passed to RANSAC. Below this the fit is not even attempted |
| `N_WINDOWS` | 12 | Number of sliding window rows. More windows = finer resolution of lane position up the frame |
| `WIN_MARGIN` | 60 px | Half-width of each sliding window. **Wider** = tolerates more lateral shift between frames; **narrower** = tighter but misses curves |
| `WIN_MINPIX` | 25 | Minimum pixels inside a window to re-centre it on those pixels. Below this the window stays at the previous centre (anchored search) |
| `PRIOR_MARGIN` | 50 px | Band around the previous frame's polynomial within which to collect inlier candidates in prior-guided search. Controls the fast-path sensitivity |

### 4.7 Temporal Smoothing

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SMOOTH_ALPHA` | 0.30 | EMA weight for new polynomial coefficient observation. **Higher** = more responsive to lane changes, more jitter. **Lower** = smoother trajectory, slower adaptation. Formula: `new_ema = α·new + (1−α)·old` |

### 4.8 Lane Geometry Sanity

| Parameter | Default | Effect |
|-----------|---------|--------|
| `LANE_WIDTH_MIN` | 0.20 m | If measured lane width is below this, it is rejected as a degenerate fit |
| `LANE_WIDTH_MAX` | 12.0 m | Sanity ceiling — above this the depth point cloud measurement is likely garbage |
| `MIN_LANE_SEP_PX` | 40 px | Absolute pixel-space guard: left and right fits must be at least this far apart at 85% of frame height. Prevents degenerate collapses on the first few frames before memory builds up |
| `LANE_SEP_MEM_FRAC` | 0.70 | If the raw lane separation drops below 70% of the remembered average width, the contamination guard fires and discards the outlier fit |

### 4.9 Lane Memory

| Parameter | Default | Effect |
|-----------|---------|--------|
| `LANE_MEM_MAX` | 60 | Rolling window length for lane width history samples. Longer = more stable average, slower to adapt to changing track width |

### 4.10 Confidence Gates

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CONF_WHITE` | 0.22 | Minimum white-line confidence before white fits are used. If both sides are below this, fallback to grass detection |
| `CONF_GRASS` | 0.22 | Minimum grass confidence for grass fits to be used. If both sources are below both gates, source = "LOST" |

### 4.11 Obstacle Detection

| Parameter | Default | Effect |
|-----------|---------|--------|
| `OBS_MIN_HEIGHT_M` | 0.15 m | Minimum height above floor for a point to count as an obstacle. Prevents the road surface, speed bumps, and low debris from triggering |
| `OBS_MAX_HEIGHT_M` | 2.50 m | Ceiling — overhead signs, bridges are ignored |
| `OBS_MIN_DIST_M` | 0.30 m | Closest reportable obstacle. Very close objects can produce noisy depth |
| `OBS_MAX_DIST_M` | 4.00 m | Furthest reportable obstacle. Beyond this the point cloud is too sparse to be reliable |
| `OBS_LANE_MARGIN_M` | 0.40 m | The lateral detection zone extends 0.40 m beyond each lane boundary. Catches obstacles slightly outside the lane that would still be hit |
| `OBS_MIN_CLUSTER_PX` | 60 | Minimum number of point-cloud pixels in a blob to be reported. Rejects isolated noise points |
| `OBS_SDK_ENABLE` | True | Enables the ZED SDK's neural object detector (people + vehicles) as a second detection path. If this detects something closer than the point-cloud path, its result takes precedence |

### 4.12 Stop Sign

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SIGN_RED_H_LOW_MAX` | 10 | Upper bound of the low-red hue band (hue 0–10). Red wraps in HSV so detection uses two ranges |
| `SIGN_RED_H_HIGH_MIN` | 160 | Lower bound of the high-red hue band (hue 160–180) |
| `SIGN_RED_S_MIN` | 120 | Minimum saturation — rejects pale pink or washed-out red |
| `SIGN_RED_V_MIN` | 60 | Minimum brightness — rejects very dark red or deep shadow |
| `SIGN_MIN_AREA_PX` | 800 | Minimum contour area. Below this the sign is too far or too small to trust |
| `SIGN_MAX_AREA_PX` | 80,000 | Maximum area. Prevents a large red wall/vehicle from triggering |
| `SIGN_POLY_SIDES_MIN/MAX` | 6–10 | An octagon seen at distance or under perspective appears 6–8 sided. 10 allows generous detection slop |
| `SIGN_ASPECT_MIN/MAX` | 0.5–1.8 | Bounding rect W/H ratio — an octagon is roughly square (aspect ~1.0). This rejects tall thin banners or wide strips |
| `SIGN_DIST_MIN/MAX_M` | 0.5–15.0 m | Valid detection distance range |
| `SIGN_VOTE_NEEDED` | 3 | Consecutive frames needed to confirm. Lower than stop line because a sign is structurally distinct — 3 frames is sufficient to reject a brief red flash |

### 4.13 Vehicle Commands

| Parameter | Default | Effect |
|-----------|---------|--------|
| `BRAKE_DIST_M` | 1.5 m | Obstacle closer than this triggers BRAKE regardless of other states. Your "panic distance" |
| `THROTTLE_VALUE` | 20 | 0–255 intensity byte sent with THROTTLE frame. Maps to the RC PWM duty cycle on the Nucleo → motor controller. Currently set to 20 for testing |
| `BRAKE_VALUE` | 255 | 0–255 intensity for BRAKE frame. 255 = maximum brake servo throw |

### 4.14 Control Outputs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CTRL_LOOKAHEAD_M` | 2.5 m | The lookahead distance for pure pursuit / feed-forward. Larger = smoother turns but slower response to upcoming features |
| `CTRL_HEADING_ALPHA` | 0.20 | EMA alpha for heading angle. Lower than `SMOOTH_ALPHA` because heading amplifies noise from the quadratic derivative |
| `CTRL_CURVATURE_ALPHA` | 0.15 | EMA alpha for curvature — extra low because curvature is a second derivative and highly noise-sensitive |
| `CTRL_EVAL_Y_FRAC` | 0.60 | Image row fraction at which heading and curvature are evaluated. 0.60 = 60% down from top (ahead of vehicle). Lower = evaluates further ahead |

---

## 5. Entry Point — main.py

`main.py` runs the entire system. It:

1. Creates a `LanePerception` instance and calls `perc.init()` — this opens the ZED camera and initialises all ZED SDK features.
2. Creates a `Commander` instance and calls `cmd.open()` — this opens the UART serial port.
3. Enters the main loop:
   - Calls `perc.process()` to get a `PerceptionResult` plus debug images.
   - Calls `cmd.update(result)` to issue the UART command.
   - If `DISPLAY=True`, renders the debug overlay with `draw()` and shows it with `cv2.imshow`.
   - Every 30 frames prints a one-line telemetry row to stdout: frame number, source, deviation, lane width, confidence, position (LEFT/CENTER/RIGHT), command, heading, curvature, lookahead, stop line, stop sign, obstacle.
4. On exit (Q key or Ctrl+C), calls `cmd.close()` and `perc.close()` to safely shut down UART and ZED.

**The loop does not run in a fixed-rate thread** — it is limited by the ZED camera's frame rate (configured to 30 FPS) and the depth-processing time. On Jetson Nano in MAXN mode this is typically 20–30 FPS.

---

## 6. Pipeline — perception/pipeline.py

`LanePerception` is the central orchestrator. It owns:
- The ZED camera handle and all ZED SDK objects
- The `Smoother` (EMA on polynomial coefficients)
- The `ControlSmoother` (EMA on heading and curvature)
- The `LaneMemory` (virtual boundary reconstruction)
- Vote accumulators for stop line and stop sign
- Last-known values for all hold-last state (deviation, heading, curvature, stop positions)

### init()

Called once at startup:
1. **Camera reboot** — calls `sl.Camera.reboot(0)` then waits 3 seconds. This clears any stale ZED SDK state from a previous run.
2. **InitParameters** — sets resolution (HD720), FPS (30), depth mode (NEURAL — best quality), units (meters), coordinate system (RIGHT_HANDED_Y_UP: X=right, Y=up, Z=negative forward).
3. **Positional tracking** — enables ZED's internal odometry with floor as origin. This makes the Y axis of every point cloud pixel refer to height above the floor.
4. **Camera calibration** — reads `cal.fx` (horizontal focal length in pixels), frame width and height.
5. **Optional BEV warp** — if `WARP_ENABLED=True`, builds the perspective warp matrix.
6. **Optional object detection** — if `OBS_SDK_ENABLE=True`, enables ZED's neural object detector for people and vehicles.

### process()

Called every frame. Returns `(PerceptionResult, frame_bgr, floor_mask, white_mask, grass_mask)` or `None` if the camera grab failed.

The internal steps are documented in full in Section 7 and 8.

### close()

Gracefully disables ZED object detection, positional tracking, and closes the camera.

---

## 7. Lane Detection Algorithms

### 7.1 Floor Plane Estimation

**File:** `perception/pipeline.py` → `_update_floor()` and `_floor_mask()`

**Why it matters:** All colour masks are applied only to floor pixels. Without a correct floor plane, the white-line and grass masks would include walls, the sky, and obstacles — making lane detection impossible.

**How the ZED floor plane works:**
The ZED SDK's `find_floor_plane()` analyses the point cloud to find the dominant horizontal plane in the lower portion of the frame. It returns a plane equation `[A, B, C, D]` where `Ax + By + Cz + D = 0`. For a floor, `B` (the Y component of the normal) will be large. The floor Y height is extracted as:

```
floor_y = -D / B
```

This is the world Y coordinate (in meters, above the ZED's initial calibration position) of the road surface.

**State machine:**

- **STABLE mode** (miss count < 3): Re-runs `find_floor_plane()` every `FLOOR_STABLE_HZ=4` frames. Between re-runs, the cached `floor_y` is reused. When a new value arrives it is blended: `floor_y = 0.80·old + 0.20·new` — a slow EMA that prevents jumping on a rough road.

- **LOST mode** (miss count ≥ 3): The primary method failed too many times. Now retries every frame AND also runs 6 hit-test fallback queries. `find_plane_at_hit([x,y])` finds the plane at a specific pixel — useful when the full-frame analysis fails but the road is still visible at known positions. If recovering from lost mode, the update is more aggressive: `floor_y = 0.60·old + 0.40·new`.

- **Resilience:** If all methods fail and a cached `floor_y` exists, the cached value is kept but `miss_count` increments. The pipeline never blocks — it returns `source="NO_FLOOR"` only if no cached floor exists at all.

**Floor mask generation (`_floor_mask`):**
After floor_y is established, every pixel's Y-coordinate (from the point cloud) is compared to `floor_y ± FLOOR_TOLERANCE`. Pixels within the band become the floor mask. In lost mode, `FLOOR_TOLERANCE_WIDE` is used. The mask is then morphologically cleaned (OPEN removes isolated noise pixels, CLOSE fills small gaps). The top `ROI_TOP_FRACTION` of the frame is always zeroed.

---

### 7.2 Colour Masking

**File:** `perception/pipeline.py` → `_color_masks()`

Two independent colour masks are built, both restricted to floor pixels only.

**White line mask (HLS colour space):**
The frame is converted to HLS (Hue, Lightness, Saturation). A pixel is white lane line if:
- It is on the floor (`fm == 255`)
- Lightness `L ≥ WHITE_L_MIN` (160) — bright
- Saturation `S_hls ≤ WHITE_S_MAX` (65) — not coloured

HLS is used instead of HSV for white detection because HLS separates lightness as an explicit channel, making it more robust to illumination changes than HSV.

**Grass mask (HSV colour space):**
The frame is converted to HSV. A pixel is grass/grass-edge if:
- It is on the floor
- Hue in range `[GRASS_H_MIN, GRASS_H_MAX]` (25–95) — covers yellow-green to blue-green
- Saturation `≥ GRASS_S_MIN` (30) — not grey asphalt
- Value `≥ GRASS_V_MIN` (35) — not dark shadow

**Obstacle region masking:**
Before the lane fitter sees either mask, the previous frame's obstacle bounding box (inflated by 15 px) is zeroed in both masks. This prevents the fitter from treating obstacle edges or markings on the obstacle as lane lines.

---

### 7.3 Lane Fitting Algorithms

**File:** `lane/fitting.py`

Lane boundaries are modelled as **quadratic polynomials** in image space:

```
x = a·y² + b·y + c
```

Note: the polynomial is `x = f(y)`, not `y = f(x)`. This is because lane lines run roughly vertically in the image — fitting x as a function of y avoids infinite slopes and is standard practice in lane detection.

**Three fitting strategies exist:**

#### Strategy 1: Sliding Window (initial / recovery)

Used when no previous fit exists (first good frame or after prolonged lane loss).

1. **Histogram initialisation:** Sum the white/grass mask pixels along each column in the **bottom 40% of the frame**. This gives a 1D histogram of lane pixel density per column. The column with the most pixels in the left half is the left lane base (`lx`). The column with the most pixels in the right half is the right lane base (`rx`).

2. **Window scan:** Divide the frame into `N_WINDOWS=12` horizontal strips. For each strip (from bottom to top):
   - Define a left window centred at `lx ± WIN_MARGIN` pixels wide.
   - Define a right window centred at `rx ± WIN_MARGIN`.
   - Collect all non-zero mask pixels inside each window.
   - If either window has ≥ `WIN_MINPIX=25` pixels, re-centre that window's next iteration on the mean x of those pixels. This lets the windows "follow" a curving lane upward.

3. **RANSAC fit:** All collected left pixels are passed to `ransac_poly()`. All collected right pixels are passed separately.

#### Strategy 2: Prior-Guided Search (fast path)

Used when a valid left **and** right fit exists from the previous frame. This is the dominant path during normal driving.

1. Evaluate the previous polynomial `prev_lf` at every row where there are non-zero pixels.
2. Accept any pixel whose x-coordinate is within `PRIOR_MARGIN=50` pixels of the polynomial's predicted x. These are the inlier candidates for this frame.
3. Run RANSAC on those candidates only.

This is dramatically faster than sliding window because instead of scanning every possible window position, the search space is already narrowed to a 100-pixel-wide band around the expected lane position.

**Fallback:** If prior-guided search fails to find a fit for either side, that side falls back to sliding window independently.

#### RANSAC (Random Sample Consensus)

Used by both strategies to produce a robust polynomial fit from noisy pixel coordinates.

```
For RANSAC_ITER = 80 iterations:
    1. Randomly sample (POLY_DEG+1) = 3 pixels
    2. Fit a quadratic through those 3 points exactly (np.polyfit)
    3. Compute residual |x_actual - x_predicted| for all pixels
    4. Count inliers: pixels with residual < RANSAC_THRESH_PX (6 px)
    5. If this is the best inlier count so far:
       - Refit using ALL inliers (least-squares, more stable than 3-point fit)
       - Record as best model

Accept the best model only if best_n >= MIN_INLIERS (30)
Confidence = min(0.99, best_n / (MIN_INLIERS × 6))
```

RANSAC is critical here because lane masks always contain noise (reflections, shadows, road texture). A pure least-squares fit would be dragged by outliers. RANSAC finds the largest self-consistent subset of pixels and fits to those.

#### Two-Source Selection

After fitting both white and grass sources, the pipeline picks the best fit per side:

```python
raw_l = best of [(wl, wlc × 1.4), (gl, glc)]  # white weighted 1.4×
raw_r = best of [(wr, wrc × 1.4), (gr, grc)]
```

White lines get a 1.4× confidence boost because they are the primary, unambiguous lane marker. Grass edges are the fallback. A side that has a white fit above `CONF_WHITE` will almost always win over grass.

---

### 7.4 Fit Contamination Guard

**File:** `perception/pipeline.py` (inline in `process()`)

**The problem:** An obstacle sitting in the lane can appear as a vertical edge in the white/grass mask. The fitter may latch onto this edge, producing a "lane line" that is actually the obstacle's side. This causes the apparent lane to narrow or shift — the car may steer toward the obstacle.

**Detection:** At row `y = 0.85 × H`, measure the pixel-space separation between raw_l and raw_r:

```
sep = eval_x(raw_r, y_g) - eval_x(raw_l, y_g)
```

Two gates fire if separation is too small:
1. **Gate 1 (absolute):** `sep < MIN_LANE_SEP_PX (40 px)` — catches the first frames before lane memory is built.
2. **Gate 2 (relative):** `sep < LANE_SEP_MEM_FRAC (0.70) × lane_mem.mean_px` — once memory has a stable average width, any separation below 70% of that average is suspicious.

**Resolution:** When either gate fires, the contaminated fit is discarded. Which side to discard? If lane memory has a mean width, it can predict where each fit *should* be:
- Expected right position: `lx + mean_px`
- Expected left position: `rx - mean_px`
- The fit with the larger error relative to this expectation is the outlier.

The EMA smoother for the discarded side is also flushed (`smoother.l_ema = None` or `smoother.r_ema = None`) so the contaminated value does not decay into future frames.

---

### 7.5 Temporal Smoothing (EMA)

**File:** `lane/smoother.py`

Raw polynomial coefficients jump frame-to-frame because pixel positions vary with lighting, noise, and minor camera shake. The `Smoother` class applies **Exponential Moving Average (EMA)** to the polynomial coefficient arrays:

```
new_ema = α·new + (1−α)·old
```

This is applied **element-wise** across the `[a, b, c]` coefficient array. With `SMOOTH_ALPHA=0.30`:
- The new observation contributes 30% to the output.
- The previous smoothed value contributes 70%.
- This gives roughly a 3-frame time constant.

**Why not a simple moving average?** EMA weights recent observations more than old ones — appropriate because a lane that was curved 10 frames ago should influence the current estimate less than 2 frames ago. It is also computationally free (no history buffer needed).

**Note:** Smoothing is applied to the raw fits. The `_prev_lf`/`_prev_rf` priors for the next frame's prior-guided search are updated from the **raw** (unsmoothed) fits — not the EMA output. This is correct: the prior-guided search should track where the lane actually was, not where the smoother thinks it was.

---

### 7.6 Virtual Boundaries / Lane Memory

**File:** `lane/memory.py`

**The problem:** On tight turns, one lane boundary (typically the inner boundary of the curve) disappears from the camera's field of view, or is obscured. The pipeline must still estimate both boundaries to compute deviation and control outputs.

**Width history:** `LaneMemory` maintains rolling lists of lane width observations (in pixels and in metres), sampled at 5 rows per frame when both boundaries are visible. It tracks the median of the last `LANE_MEM_MAX=60` observations. The median is used instead of the mean to be robust against outlier width measurements.

**Virtual right boundary from known left:**
```
For each row from ROI_TOP to H (step 8):
    lx = eval_x(left_fit, row)
    depth = abs(point_cloud[row, lx, Z])  # forward distance to that row
    vx = lx + mean_width_m × (focal_length_px / depth)
```

This is the **perspective projection formula**: an object at forward distance `depth` that is `W_m` metres wide in the real world spans `W_m × fx / depth` pixels in the image. By applying the stored physical lane width and the measured depth at each row, the virtual boundary correctly narrows as it recedes into the distance — maintaining geometric accuracy under perspective projection.

The projected virtual x-coordinates `vxs` at each row `ys` are then refitted with `np.polyfit` to get a smooth virtual polynomial. If the depth projection fails (insufficient finite depth points), a pixel-space fallback shifts the known fit by `mean_px` pixels: `vrf = lf.copy(); vrf[-1] += mean_px`. This is less accurate geometrically but keeps the pipeline running.

---

### 7.7 Lateral Deviation

**File:** `lane/deviation.py`

**Goal:** Compute how far left or right the car is from the lane centre in **metres**.

**Multi-row weighted sampling:**
Sample at 4 image rows: `[0.95H, 0.85H, 0.75H, 0.65H]` with weights `[0.40, 0.30, 0.20, 0.10]`. Rows near the bottom of the frame are weighted more because they correspond to the car's immediate position rather than a projected future position.

**At each row (both boundaries visible):**
1. Evaluate left polynomial → `lx` pixels
2. Evaluate right polynomial → `rx` pixels
3. Read ZED point cloud X-coordinates at `(row, lx)` and `(row, rx)`: `lp[X]` and `rp[X]`.
4. Physical lane width: `wm = |rp[X] - lp[X]|` (metres)
5. Lane centre in image pixels: `(lx + rx) / 2`
6. Car position is the image centre: `W / 2`
7. Pixel offset: `W/2 − (lx + rx)/2` (positive = car is right of centre in image = left in world)
8. Metric deviation: `dev_m = pixel_offset / (rx − lx) × wm`

The signed convention: `deviation_m > 0` means the vehicle is to the **left** of centre (lane centre is to the right of the image midpoint).

The weighted average of all valid rows gives the final `deviation_m`. The median of width observations gives `lane_width_m`.

**When only one boundary is visible:** A degraded estimate is made using the stored median width as a reference, with reduced weight (×0.4). This is less accurate but keeps the deviation estimate live rather than going to zero.

---

## 8. Detection Algorithms

### 8.1 Stop Line Detection

**File:** `detection/stop_line.py`

The stop line is an orange horizontal stripe painted across the road. Detection requires **all five** of the following checks to pass simultaneously:

**Step 1 — Orange HSV mask:**
For every floor pixel, check if it falls in the orange HSV range:
- Hue: `[STOP_ORANGE_H_MIN, STOP_ORANGE_H_MAX]` = [5, 20]
- Saturation ≥ `STOP_ORANGE_S_MIN` = 150
- Value ≥ `STOP_ORANGE_V_MIN` = 100

**Step 2 — Row density:**
Sum the orange pixels per image row. A candidate row must have orange pixels covering more than `STOP_ROW_THRESH=8%` of the frame width. This eliminates small orange specks.

**Step 3 — Vertical position:**
Only rows in the **bottom 60%** of the frame are considered (`row > 0.40 × H`). A stop line is always on the road surface in front of the car, never at the top of the frame.

**Step 4 — Perpendicularity check (PCA):**
For each candidate row, extract orange pixels in a ±10 row band around it. Run **Principal Component Analysis (PCA)** on the pixel coordinates: compute the covariance matrix, take the SVD, and find the principal axis direction. If this axis is more than `STOP_PERP_MAX_DEG=20°` from horizontal, the cluster is angled — reject it. This removes diagonal shadows, kerb paint, and angled road markings.

```python
pts -= pts.mean(axis=0)          # centre the cloud
_, _, Vt = np.linalg.svd(pts)    # Vt[0] is the principal axis
ang = degrees(atan2(Vt[0,1], Vt[0,0]))   # angle of principal axis
if ang > 90: ang = 180 - ang
accept if ang <= STOP_PERP_MAX_DEG
```

**Step 5 — Lane coverage:**
If lane fits exist, evaluate the left and right polynomials at the candidate row to find the lane boundaries `lx` and `rx`. Count orange pixels between `lx` and `rx`. If they cover less than `STOP_COVERAGE_MIN=60%` of the lane width, reject. This prevents a small orange cone or traffic barrel from triggering.

**Distance measurement:**
Sample `STOP_DIST_N_PTS=10` evenly-spaced columns across the lane at `row + 20` (slightly ahead of the detected stripe edge). For each, read the ZED Z-coordinate (forward distance = `|Z|`). Take the median of valid samples. Require at least `STOP_DIST_MIN_VALID=4` valid points to report a distance.

**Output:** `(True, row_y, distance_m)` on the first candidate row that passes all checks.

---

### 8.2 Stop Sign Detection

**File:** `detection/stop_sign.py`

**Step 1 — Red HSV mask:**
Red wraps around the HSV hue circle (0/180 boundary), so two ranges are combined:
- Low-red: hue `[0, SIGN_RED_H_LOW_MAX=10]`
- High-red: hue `[SIGN_RED_H_HIGH_MIN=160, 180]`

Both require `S ≥ SIGN_RED_S_MIN=120` and `V ≥ SIGN_RED_V_MIN=60`. The two binary masks are ORed together.

The top `ROI_TOP_FRACTION=35%` of the frame is zeroed (eliminates sky and overhead lights). Morphological OPEN then CLOSE cleans noise and fills gaps.

**Step 2 — Contour extraction:**
`cv2.findContours` with `RETR_EXTERNAL` finds all outer contours of the red blobs.

**Step 3 — Shape filtering:**
For each contour:
- **Area gate:** `SIGN_MIN_AREA_PX ≤ area ≤ SIGN_MAX_AREA_PX` = [800, 80,000]
- **Polygon approximation:** `cv2.approxPolyDP` with epsilon = 3% of the perimeter. This simplifies the contour to N straight-line segments.
- **Side count gate:** `SIGN_POLY_SIDES_MIN ≤ sides ≤ SIGN_POLY_SIDES_MAX` = [6, 10]. An octagon at distance under perspective projection appears 6–8 sided after approximation. The ε is generous enough to handle pixelation.
- **Aspect ratio:** `SIGN_ASPECT_MIN ≤ W/H ≤ SIGN_ASPECT_MAX` = [0.5, 1.8]. An octagon's bounding rectangle is nearly square.

The largest qualifying contour is selected.

**Step 4 — Distance from point cloud:**
Read the Z-values in a patch of radius `max(4, bbox_h/8)` pixels around the centroid. Take the median of valid (finite, negative Z) values. Convert to absolute distance: `dist_m = |median(Z)|`. Reject if outside `[SIGN_DIST_MIN_M, SIGN_DIST_MAX_M]`.

**Output:** `(True, dist_m, (x, y, w, h))`

---

### 8.3 Obstacle Detection (Point Cloud)

**File:** `detection/obstacle.py`

This detector works entirely in 3D space using the ZED XYZ point cloud. It is not colour-based — it detects any physical object above the floor surface, regardless of colour.

**Step 1 — Build obstacle mask:**
For every pixel in the frame, classify it as an obstacle point if all conditions hold:

```
valid:    point cloud values are finite (not NaN/Inf)
above:    Y > floor_y + OBS_MIN_HEIGHT_M    (above road surface)
          Y < floor_y + OBS_MAX_HEIGHT_M    (below ceiling)
fwd:      Z < -OBS_MIN_DIST_M              (in front, forward = -Z in ZED convention)
          Z > -OBS_MAX_DIST_M              (within range)
in_lane:  x pixel is within lane boundaries ± OBS_LANE_MARGIN_M
```

For the lateral gate, the lane boundaries are evaluated at every row using `np.polyval`. The lane margin in pixels is computed from the physical margin using: `margin_px = OBS_LANE_MARGIN_M × fx / 5.0` (the 5.0 is an approximate depth normalization).

**Step 2 — Morphological cleanup:**
OPEN (5×5 kernel) removes isolated noise. CLOSE fills gaps within a real obstacle cluster.

**Step 3 — Connected components:**
`cv2.connectedComponentsWithStats` finds all blobs. The largest blob with area ≥ `OBS_MIN_CLUSTER_PX=60` pixels is the obstacle.

**Step 4 — Distance and lateral offset:**
For the winning blob, collect all Z and X values. The **median** is used for both (not mean) — median is robust to the depth measurement outliers that ZED point clouds contain at object edges.

```
dist_m    = median(|Z|)    for all blob pixels
lateral_m = median(X)      for all blob pixels
```

Sign convention: `lateral_m > 0` = obstacle is to the **right** of the camera centre. This matches ZED's X axis (right = positive).

**Step 5 — ZED SDK overlay:**
If `OBS_SDK_ENABLE=True`, the ZED SDK's neural detector also runs. It detects people and vehicles as tracked objects with 3D positions. If an SDK object is detected **closer** than the point-cloud result, the SDK result replaces it. This catches cases where the point cloud alone might miss a person (thin silhouette, sparse points).

---

### 8.4 Temporal Vote Gates

**File:** `perception/pipeline.py` (inline)

Both the stop line and stop sign use a **vote accumulator** to suppress transient false detections:

```
On raw detection:     votes = min(MAX_VOTES, votes + 1)
On raw non-detection: votes = max(0, votes - 1)
Confirmed:            votes >= VOTE_NEEDED
MAX_VOTES:            VOTE_NEEDED + 5  (caps the accumulator)
```

For the **stop line** (`STOP_VOTE_NEEDED=5`): requires 5 consecutive positive detections to confirm. A single drop resets by 1 — so 5 hits then 1 miss requires only 1 more hit to re-confirm.

For the **stop sign** (`SIGN_VOTE_NEEDED=3`): slightly lower because a stop sign is a structurally distinctive object with low false-positive rate.

**Hold-last behaviour:** When confirmed, the last detected distance and position are held even if the raw detector temporarily misses. This prevents a confirmed stop event from flickering off-on during approach.

---

## 9. Control Outputs

**File:** `lane/control.py`

### 9.1 Heading Angle

The centreline polynomial is the average of left and right fits:

```
cf = (left_fit + right_fit) / 2
```

For the polynomial `x = a·y² + b·y + c`, the **tangent (slope) at row y** is:

```
dx/dy = 2a·y + b
```

The heading angle is:

```
θ = arctan(2a·y_eval + b)
```

Evaluated at `y_eval = CTRL_EVAL_Y_FRAC × H = 0.60 × H` (60% down the frame, ahead of the vehicle).

**Sign convention:** Positive θ means the lane's tangent points right in the image (toward increasing x). Under standard perspective projection (road receding forward), this indicates the road is bending left ahead. Negative θ = road bending right.

**Units:** Radians. The telemetry prints it in degrees for human readability.

### 9.2 Curvature

Curvature `κ = 1/R` (inverse radius in m⁻¹) is derived from the polynomial's second derivative:

**Image-space curvature:**
```
κ_px = f''(y) / (1 + f'(y)²)^(3/2)
     = 2a / (1 + (2ay + b)²)^(3/2)
```

This is the standard Frenet curvature formula for a parametric curve.

**Metric conversion:**
The image-space curvature is in units of 1/pixel. To convert to 1/metre:
```
px_per_m = lane_width_px / lane_width_m
κ_metric = κ_px × px_per_m
```

This uses the lane width as a real-world calibration reference — the lane's known physical width is what connects pixel distances to metric distances.

**Sign convention:** Positive κ = curving left. Negative κ = curving right.

### 9.3 Lookahead Point

Scans image rows from `ROI_TOP_FRACTION × H` down to `H` in steps of 4. For each row:
1. Evaluate the centreline polynomial → `cx` pixel.
2. Read the ZED point cloud at `(row, cx)`: get `(X_m, Z_m)`.
3. Compute `z_fwd = |Z|` (forward distance in metres).
4. Track the row where `|z_fwd - CTRL_LOOKAHEAD_M|` is minimised.

The selected point `(X_m, Z_m)` in world space is the lookahead target. `X_m` is the lateral position of the centreline at 2.5 m ahead (for steering), `Z_m` ≈ 2.5 m confirms the distance measurement is good.

The pixel coordinates `(cx, row)` are also returned for visualization.

### 9.4 Control Smoothing

`ControlSmoother` applies EMA with **lower alphas** than the polynomial smoother:
- Heading: `CTRL_HEADING_ALPHA = 0.20`
- Curvature: `CTRL_CURVATURE_ALPHA = 0.15`

These are lower because heading and curvature are **derivatives** of the polynomial. Small frame-to-frame noise in the coefficients `a` and `b` gets amplified when computing `2ay+b` and especially `2a`. The stronger low-pass filter keeps steering commands smooth.

The control smoother only updates when a geometrically valid lane is present (`wid > 0`). On LOST frames, the last-known values are held — the car keeps its last steering intent rather than snapping to zero.

---

## 10. Commander — control/commander.py

The Commander is the decision layer between perception and actuation. It reads a `PerceptionResult` every frame and decides one of three states: **THROTTLE**, **BRAKE**, or **IDLE**.

### Priority order (hard-coded in `_decide()`):

```
1. Obstacle detected AND obstacle_dist_m ≤ BRAKE_DIST_M (1.5 m)  → BRAKE
2. Stop line confirmed (stop_line == True)                         → BRAKE
3. Stop sign confirmed (stop_sign == True)                         → BRAKE
4. source in ("LOST", "NO_FLOOR")                                  → IDLE
5. Otherwise                                                       → THROTTLE
```

Priority 1 is the **safety-critical** rule — an obstacle within 1.5 m triggers BRAKE regardless of everything else, even if the lane is LOST. This ensures the car stops for physical objects even in degraded perception states.

Priorities 2 and 3 are **traffic rule** stops.

Priority 4 handles **degenerate vision** — if the lane is completely lost, the car coasts (IDLE = no throttle, no brake) rather than driving blind.

### Command issuance:

```python
if state == "BRAKE":    uart.brake(BRAKE_VALUE)    # CMD=0x02, VAL=255
elif state == "THROTTLE": uart.throttle(THROTTLE_VALUE)  # CMD=0x01, VAL=20
else:                   uart.idle()                # CMD=0x00, VAL=0
```

**State change logging:** Only logs when the state transitions. If the car is cruising at THROTTLE for 300 frames, nothing is printed. This keeps logs readable.

---

## 11. UART Transport — control/uart.py

### Protocol Frame (5 bytes)

```
Byte 0:  0xAA              Start byte (synchronisation marker)
Byte 1:  0x02              LEN — payload length (CMD + DATA = 2 bytes)
Byte 2:  CMD               0x00=IDLE  0x01=THROTTLE  0x02=BRAKE
Byte 3:  VAL               0–255 intensity
Byte 4:  CRC8              CRC-8/SMBUS checksum over bytes [1,2,3]
```

This matches the documented protocol in the technical report Section 2.2.

### CRC-8/SMBUS

Polynomial: 0x07 (x⁸ + x² + x + 1), initial value: 0x00, no reflection.

```python
crc = 0x00
for byte in [LEN, CMD, VAL]:
    crc ^= byte
    for _ in range(8):
        if crc & 0x80:
            crc = (crc << 1) ^ 0x07
        else:
            crc <<= 1
    crc &= 0xFF
```

The Nucleo firmware must implement the **identical algorithm** to validate incoming frames.

### De-duplication

To avoid flooding the Nucleo with identical frames every 33 ms, the UART layer skips sending if `cmd == last_cmd AND val == last_val`. The frame is only sent when the command or value changes.

**⚠ Important:** This means the Nucleo's 200 ms watchdog will trigger during sustained THROTTLE if the command never changes. A heartbeat mechanism is needed to periodically force a resend even without state change. This is a known gap — see Section 16.

### Serial port configuration

`/dev/ttyTHS1` (Jetson hardware UART) @ 115200 baud, 8N1. Thread-safe: all writes are protected by `threading.Lock()` so the UART can safely be called from a future control thread without data corruption.

---

## 12. Visualization — visualization.py

The `draw()` function renders a composite debug frame that includes:

- **Lane fill:** Semi-transparent green polygon between left and right polynomials.
- **Lane boundaries:** Left = blue line, Right = yellow line. Virtual boundaries (from lane memory) are shown dashed.
- **Lane width arrow:** Horizontal double-arrow with metric label at 85% of frame height.
- **Deviation bar:** Horizontal bar at the bottom. Filled proportion indicates deviation relative to half-lane-width. "L" and "R" labels indicate which side.
- **Lookahead point:** Yellow circle on the centreline at the closest point to 2.5 m ahead.
- **Stop line:** Red horizontal line drawn across the frame at the detected row.
- **Stop sign box:** Orange bounding rectangle with distance label.
- **Obstacle box:** Red bounding rectangle with distance and side (L/C/R) label.
- **HUD overlay (top-left):**
  - Source (WHITE_LINE / GRASS / LOST)
  - Lane width (m)
  - Deviation (m, signed)
  - Left/right confidence
  - Stop / sign / obstacle status
  - Heading angle (°) and curvature (m⁻¹)
  - Lookahead distance (m)

---

## 13. Full Frame Execution Flow

The following is the exact sequence of operations within a single call to `perc.process()`:

```
1.  cam.grab()                          Acquire new frame + depth from ZED
2.  retrieve_image()                    Get BGR frame (1280×720)
3.  retrieve_measure(XYZ)              Get point cloud (1280×720×3 float32)
4.  get_position()                      Update ZED odometry pose
5.  _update_floor()                     Estimate floor_y (ZED + fallback hit-tests)
    → if no floor at all: return PerceptionResult(source="NO_FLOOR")
6.  _floor_mask()                       Binary mask of floor pixels
7.  _color_masks()                      White (HLS) + Grass (HSV) masks, floor-gated
8.  Blank prev obstacle bbox            Zero obstacle region in wm + gm
9.  _fit_mask(white_mask)              Fit left/right on white pixels
10. if white conf < CONF_WHITE:
        Canny edge on grass mask
        _fit_mask(grass_canny)         Fit left/right on grass edges
11. Source selection                   Best fit per side (white preferred 1.4×)
12. Contamination guard               Check sep vs MIN_LANE_SEP_PX + LANE_SEP_MEM_FRAC
    → discard contaminated fit + flush EMA
13. Update _prev_lf, _prev_rf          Store raw fits as priors for next frame
14. smoother.update(raw_l, raw_r)      EMA smooth polynomial coefficients
15. lane_mem.update(lf, rf)            Add width samples to rolling history
16. Virtual boundary projection        Fill missing side from lane memory if needed
17. Source label                       WHITE_LINE / GRASS / LOST
18. compute_deviation()                Metric deviation and lane width
19. Hold-last on LOST                  Keep previous deviation if source=LOST
20. detect_stop_line()                 Orange stripe detection (5 checks)
21. Stop vote accumulator update       +1 on detect, -1 on miss; confirm at ≥5
22. detect_obstacle_pc()               3D point cloud obstacle scan
23. ZED SDK object detection           If enabled; replace PC result if closer
24. Persist obs_bbox                   For next-frame mask exclusion
25. detect_stop_sign()                 Red octagon detection
26. Sign vote accumulator update       +1 on detect, -1 on miss; confirm at ≥3
27. Control outputs:
    a. y_ctrl = 0.60 × H              Evaluation row
    b. wid_px                          Lane width in pixels at y_ctrl
    c. compute_heading()               θ = arctan(2ay + b) on centreline
    d. compute_curvature()             κ = 2a / (1 + (2ay+b)²)^1.5 × px_per_m
    e. ctrl_smoother.update()          EMA on heading (α=0.20) + curvature (α=0.15)
    f. compute_lookahead()             Find centreline point nearest 2.5 m ahead
28. Build and return PerceptionResult  All fields filled
```

Back in `main.py`:
```
29. cmd.update(result)                 Commander decides THROTTLE/BRAKE/IDLE
30. uart.send(cmd, val)                5-byte frame → /dev/ttyTHS1
31. draw() + cv2.imshow()              Debug overlay (if DISPLAY=True)
32. Print telemetry every 30 frames
```

---

## 14. Module Communication Map

```
config.py
  └─ imported by: ALL modules (read-only, no two-way communication)

models.py (PerceptionResult)
  └─ produced by: pipeline.py
  └─ consumed by: commander.py, visualization.py, main.py

pipeline.py
  ├─ calls: lane/fitting.py     fit_lanes(), eval_x()
  ├─ calls: lane/smoother.py    Smoother.update()
  ├─ calls: lane/memory.py      LaneMemory.update(), virtual_left(), virtual_right()
  ├─ calls: lane/deviation.py   compute_deviation()
  ├─ calls: lane/control.py     compute_heading(), compute_curvature(),
  │                              compute_lookahead(), ControlSmoother.update()
  ├─ calls: detection/stop_line.py   detect_stop_line()
  ├─ calls: detection/stop_sign.py   detect_stop_sign()
  ├─ calls: detection/obstacle.py    detect_obstacle_pc()
  └─ calls: perception/warp.py       WarpTransform.warp(), bev_poly_to_img()

commander.py
  ├─ reads:  PerceptionResult
  └─ calls:  uart.py  UARTController.throttle(), brake(), idle()

main.py
  ├─ calls:  pipeline.py  LanePerception.process()
  ├─ calls:  commander.py  Commander.update()
  └─ calls:  visualization.py  draw()
```

No module calls back into a higher-level module. The data flow is strictly downward: `main → pipeline → [lane, detection, control] → commander → uart`.

---

## 15. Tuning Guide

### The car drifts off-centre even on straight road

- Verify `FLOOR_TOLERANCE` is tight enough — if the floor mask is too wide, wall reflections contaminate the white mask.
- Check `WHITE_L_MIN`: reduce if the track lines are faint/shadowed; increase if non-line road surface is triggering.
- If `deviation_m` reads correctly but steering doesn't respond, the UART command is reaching the Nucleo but the PWM calibration may be off.

### Lane detection drops out on curves

- `N_WINDOWS` too low — increase to 16 or 20 for tighter curves.
- `WIN_MARGIN` too narrow — the window can't follow a sharp bend. Try 80 px.
- `PRIOR_MARGIN` too narrow — prior-guided search fails on fast curves. Try 70 px.
- `SMOOTH_ALPHA` too low — the EMA is too far behind the actual lane on a fast curve. Try 0.40.

### Too many false BRAKE from stop line

- Increase `STOP_VOTE_NEEDED` to 7 or 8.
- Tighten `STOP_ORANGE_H_MIN/MAX` to a narrower range matching your track's actual paint.
- Increase `STOP_COVERAGE_MIN` to 0.75 if partial orange objects are triggering.

### Stop line not detected

- Lower `STOP_ROW_THRESH` to 0.05.
- Widen `STOP_ORANGE_H_MIN/MAX` (orange paint varies significantly in sun vs shade).
- Lower `STOP_ORANGE_S_MIN` to 120 if paint is faded.
- Reduce `STOP_VOTE_NEEDED` to 3.

### Stop sign not detected at distance

- Lower `SIGN_MIN_AREA_PX` to 400.
- Widen `SIGN_POLY_SIDES_MIN/MAX` to [5, 12] — at distance the octagon may appear fewer-sided.
- Lower `SIGN_RED_S_MIN` to 100 if the sign is in shadow.

### Obstacle triggering when no obstacle exists

- Verify `floor_y` is correct — if the floor is estimated too high, the road surface itself will appear as an obstacle.
- Increase `OBS_MIN_HEIGHT_M` to 0.25 m to ignore road texture / markings.
- Increase `OBS_MIN_CLUSTER_PX` to 100 to filter smaller false blobs.
- Reduce `OBS_LANE_MARGIN_M` if objects beside the lane are detected.

### Obstacle not detected

- Lower `OBS_MIN_CLUSTER_PX` to 30.
- Lower `OBS_MIN_HEIGHT_M` to 0.10 m if the obstacle is low (e.g. a box).
- Increase `OBS_LANE_MARGIN_M` if the obstacle is slightly outside the lane.
- Check that `floor_y` is tracking correctly — wrong floor_y shifts the height gate.

### Control output too jittery

- Lower `CTRL_HEADING_ALPHA` and `CTRL_CURVATURE_ALPHA` (e.g. 0.10 each).
- Lower `SMOOTH_ALPHA` to reduce jitter propagating from the polynomial into control.

### Control output too slow to respond to curves

- Increase `CTRL_HEADING_ALPHA` toward 0.30.
- Decrease `CTRL_LOOKAHEAD_M` to 1.5–2.0 m — evaluates nearer, responds sooner.
- Increase `SMOOTH_ALPHA` toward 0.40.

---

## 16. Known Gaps and Watchdog Note

### Critical: 200 ms UART Watchdog Not Fed

The STM32 Nucleo H743 firmware specification (Section 2.2 of the system report) states:

> "Watchdog: no valid packet in 200ms triggers manual fallback."

The current UART de-duplication logic **skips sending** duplicate `(cmd, val)` pairs. During sustained THROTTLE (the most common state), no new UART frame is sent. At 30 FPS the car would be throttling for 6+ frames before a state change occurs — 200 ms is only 6 frames at 30 FPS. If the car is cruising for even half a second with no perception trigger, the Nucleo will drop to manual.

**Fix required:** The UART layer needs a heartbeat that forces a resend of the current command if no packet has been sent within ~100 ms, regardless of de-duplication. This can be implemented with a timestamp in `UARTController.send()`.

### Steering Not Yet Commanded

The current stack only sends THROTTLE/BRAKE/IDLE. The steering angle computed from `heading_angle` and `curvature` is not yet translated into STEP/DIR stepper motor pulses. This is Phase 2 of the LLC firmware development.

### BEV Warp Not Calibrated

`WARP_ENABLED = False`. The SRC trapezoid corners in `config.py` are placeholders. To enable BEV mode, measure the four road corners in an actual camera frame from the mounted car and update `WARP_SRC`.

### Ultrasonic Sensor (MB7062) Not Integrated

The MB7062 short-range sensor (0–1 m) is specified in the system report for blind-spot coverage at very close range. It is not yet integrated into the perception pipeline.
