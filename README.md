# PSU Eco Racing — Autonomous Perception Stack
**Prince Sultan University | Shell Eco-Marathon Urban Concept 2026**
**Venue:** Silesia Ring, Kamień Śląski, Poland | **Target speed:** 10–15 km/h

---

## Overview

This repository contains the real-time autonomous perception stack for the PSU Eco Racing Shell Eco-Marathon vehicle. The system runs on a **Jetson Nano 4GB** with a **ZED 2i stereo camera** and communicates with an **STM32 Nucleo H743** low-level controller over a binary UART protocol.

The perception layer is the foundation of the autonomous stack. It is responsible for producing a clean, reliable `PerceptionResult` every camera frame — the single data contract that all future control layers (Pure Pursuit, mission state machine, parking maneuver) will consume.

```
ZED 2i Camera (720p, 30 FPS)
        │
        ▼
  CLAHE Normalisation  ←── lighting invariance
        │
        ▼
  Floor Plane (ZED SDK)  ←── ZED find_floor_plane() + IMU tilt compensation
        │
        ├──► Lane Detection  (HLS white + HSV grass → RANSAC poly → smoother → deviation)
        │
        ├──► Stop Line       (orange HSV → PCA perp → width gate → temporal vote)
        │
        ├──► Stop Sign       (YOLOv8n thread → yellow board gate → bbox sanity → vote)
        │
        ├──► Obstacle        (blue above-floor blob → aspect ratio → 3D height → vote)
        │
        └──► Parking Bay     (blue on-floor blob → orientation → empty check → vote)
                │
                ▼
         PerceptionResult  (dataclass — all outputs in one object)
                │
                ▼
         Commander  →  UART  →  STM32 Nucleo H743
```

---

## Repository Structure

```
perception_stack/
├── main.py                   Entry point — main loop, telemetry logger, FPS monitor
├── config.py                 All tunable parameters in one place (no magic numbers in code)
├── models.py                 PerceptionResult dataclass — the output contract
├── visualization.py          OpenCV debug overlay (lanes, stops, obstacle, parking, HUD)
│
├── perception/
│   ├── pipeline.py           LanePerception — orchestrates every frame
│   └── warp.py               Bird's-eye perspective warp with IMU tilt compensation
│
├── lane/
│   ├── fitting.py            RANSAC + sliding window + prior-guided polynomial fitting
│   ├── smoother.py           EMA smoother on lane polynomial coefficients
│   ├── memory.py             Rolling lane-width history + virtual boundary projection
│   ├── deviation.py          Metric lateral deviation from ZED point cloud
│   └── control.py            Heading angle, curvature, lookahead point computation
│
├── detection/
│   ├── stop_line.py          Orange stripe detection with physical width validation
│   ├── stop_sign.py          Threaded YOLOv8 with yellow-board gate (SEM-specific)
│   ├── obstacle.py           Blue inflatable pin — above-floor detection + 3D height check
│   └── parking.py            Blue floor markings + empty-space check via ZED depth
│
├── control/
│   ├── commander.py          High-level decision: THROTTLE / BRAKE / IDLE
│   └── uart.py               5-byte binary framed protocol to STM32 with CRC-8
│
└── scripts/
    ├── train_stop_sign.py    YOLOv8 fine-tuning script (Roboflow API)
    └── export_trt.py         TensorRT FP16 export for Jetson deployment
```

---

## How the Pipeline Works

### 1. CLAHE Normalisation
Every frame is normalised before any colour processing:
```
BGR → LAB → CLAHE on L channel → LAB → BGR (frame_norm)
```
This makes all HSV/HLS thresholds invariant to auto-exposure changes, shadows, and the sun/cloud transitions common at outdoor Polish venues. `frame_norm` feeds every downstream colour gate.

### 2. Floor Plane Estimation
`ZED.find_floor_plane()` locates the road surface each frame and gives its Y-coordinate in world space (`floor_y`). The floor mask is then a depth tolerance band (±10 cm nominal, ±28 cm in lost mode) applied to the ZED point cloud. Everything above this band is off-road or an obstacle.

**IMU tilt compensation** (`perception/warp.py → update_tilt()`): pitch and roll from the ZED 2i IMU dynamically shift the BEV perspective warp source points so that road camber and suspension travel don't skew the homography. Dead-band: ±0.5° (tunable).

### 3. Lane Detection
Two colour sources are tried in parallel:
- **White lane lines** — HLS: L ≥ 160, S ≤ 65 (on floor pixels only)
- **Grass boundary** — HSV: hue 25–95°, S ≥ 30, V ≥ 35 (Canny edge, on floor pixels)

Both go through the same fitting pipeline:
1. **Prior-guided search** (fast, when prior fit exists): search band = 50 px around last polynomial
2. **Sliding window fallback** (12 windows, histogram-initialised base): runs on lane loss
3. **RANSAC** polynomial fit (quadratic, 80 iterations, 6 px inlier threshold)
4. **Contamination guard**: rejects fits that collapse the lane width below 70% of memory
5. **EMA smoother** (α = 0.30) applied to polynomial coefficients
6. **Lane memory**: when one boundary is missing, projects a virtual boundary using stored width + ZED depth (perspective-correct via focal length and Z)

Metric lateral deviation is computed at 4 image rows (95%, 85%, 75%, 65% height), weighted by distance (closest row = 0.40 weight), using ZED X-coordinates for true metric output.

### 4. Stop Line Detection
Six sequential gates (all must pass):
1. Floor mask — only ground pixels
2. Orange HSV — hue 5–20°, S ≥ 150, V ≥ 100
3. Row density — ≥ 8% of frame width
4. PCA perpendicularity — principal axis ≤ 20° from horizontal
5. **Physical stripe width** — orange spans ≥ 70% of lane width (ZED X-coords). Rejects cones, debris, narrow shadows.
6. Temporal voting — 5 consecutive frames (pre-arm: drops to 2 frames when stop sign confirmed at 3–8 m)

### 5. Stop Sign Detection
YOLOv8n runs in its own thread with a `maxsize=1` queue (stale frames dropped). Main thread never blocks.

SEM-specific hardening applied to every YOLO prediction:
- **Bbox height sanity**: at distance d m, expected height ≈ `(0.65 / d) × 730` px. Detections too small for their reported depth are rejected.
- **Yellow board gate**: checks for yellow HSV region (hue 18–38°, S ≥ 120, V ≥ 150) in a 1.3× expanded bbox. Hard-required at confidence < 0.72; advisory above that.
- Confidence threshold: **0.60** (raised from 0.45 for FP16 TensorRT stability)
- 3-frame temporal voting

**⚠ CRITICAL — retraining required before competition:**
The current model weights are trained on generic US stop signs. The SEM sign is a **red hexagon on a yellow rectangular board**. The yellow-board gate provides partial compensation but retraining on SEM-format images is mandatory for reliable recall. See `scripts/train_stop_sign.py`.

### 6. Obstacle Detection (`detection/obstacle.py`)
Target: blue inflatable pin, 1.10 m × 0.45 m.

**Key insight**: the obstacle is blue AND above the floor. Parking markings are blue AND on the floor. `floor_mask` is the discriminator — no colour ambiguity.

Pipeline:
1. Blue HSV mask (hue 100–135°) on `above_floor = floor_mask < 128`
2. ROI: skip top 35% (sky) and bottom 10% (hood)
3. Morphological open+close to fill the inflatable surface
4. Largest connected component ≥ 150 px
5. Aspect ratio gate: blob height/width ≥ 1.2 (pin is taller than wide)
6. ZED depth at centroid → `dist_m`
7. **3D height check**: world Y span of blob top–bottom must be 0.3–1.5 m (eliminates spectator clothing, short debris)
8. ZED X at centroid → `lateral_m` (positive = obstacle to the right)
9. 3-frame temporal voting

Output in `PerceptionResult`: `obstacle_detected`, `obstacle_dist_m`, `obstacle_lateral_m`

### 7. Parking Bay Detection (`detection/parking.py`)
Target: 4 m × 2 m bay with 0.15 m-wide blue boundary lines.

**Key insight**: same blue HSV as the obstacle, but ON the floor (`floor_mask == 255`).

Pipeline:
1. Blue HSV mask on floor-plane pixels only
2. Minimum 400 px gate
3. Centroid of blue cluster → ZED 3D position `(X_m, Z_m)`
4. PCA orientation → `angle_deg` (bay heading relative to camera)
5. **Empty check**: counts ZED point-cloud returns > 0.2 m above `floor_y` in the bay volume. Grey crates (1.2 × 1.0 × 1.0 m) generate a dense above-floor cluster → `parking_empty = False`
6. 4-frame temporal voting

Output in `PerceptionResult`: `parking_detected`, `parking_empty`, `parking_center_m`, `parking_angle_deg`

---

## PerceptionResult — The Output Contract

Every frame the pipeline emits one `PerceptionResult` dataclass. This is the single interface between perception and all control layers. No control module ever touches the camera or raw sensor data.

```python
@dataclass
class PerceptionResult:
    # Lane
    deviation_m:       float          # lateral offset from centre (m); + = left of centre
    confidence:        float          # 0–1 combined lane detection confidence
    lane_width_m:      float          # metric lane width from ZED
    source:            str            # "WHITE_LINE" / "GRASS" / "LOST" / "NO_FLOOR"
    left_fit:          np.ndarray     # polynomial coefficients  x = a·y² + b·y + c
    right_fit:         np.ndarray
    left_conf:         float
    right_conf:        float
    virtual_left:      bool           # True if left boundary is synthesised from memory
    virtual_right:     bool

    # Stop line
    stop_line:         bool
    stop_line_dist:    float          # metres to stop line
    stop_line_y:       int            # image row of stop line

    # Stop sign
    stop_sign:         bool
    stop_sign_dist_m:  float
    stop_sign_bbox:    tuple          # (x, y, w, h) pixels

    # Control geometry (input to Pure Pursuit)
    heading_angle:     float          # radians; θ = arctan(2a·y + b)
    curvature:         float          # κ = 1/R  (m⁻¹); signed
    lookahead_point:   tuple          # (X_m, Z_m) world-space lookahead on centreline
    lookahead_pixel:   tuple          # (x, y) image pixel of lookahead point

    # Obstacle
    obstacle_detected:  bool
    obstacle_dist_m:    float
    obstacle_lateral_m: float         # ZED X (m); + = right, - = left

    # Parking
    parking_detected:   bool
    parking_empty:      bool          # False if grey crates detected inside bay
    parking_center_m:   tuple         # (X_m, Z_m) bay centre in world frame
    parking_angle_deg:  float         # bay heading
```

---

## Configuration (`config.py`)

All parameters in one file. Never hardcode values in algorithm files.

| Section | Key parameters |
|---------|---------------|
| Camera | `CAM_RES`, `CAM_FPS`, `CAM_DEPTH_MODE` |
| CLAHE | `CLAHE_CLIP_LIMIT=2.0`, `CLAHE_TILE_SIZE=(8,8)` |
| Floor | `FLOOR_TOLERANCE=0.10m`, `FLOOR_TOLERANCE_WIDE=0.28m` |
| Lane colour | `WHITE_L_MIN=160`, `GRASS_H_MIN/MAX=25–95` |
| Lane fitting | `RANSAC_ITER=80`, `WIN_MARGIN=60px`, `SMOOTH_ALPHA=0.30` |
| Stop line | `STOP_WIDTH_MIN_FRAC=0.70`, `STOP_VOTE_NEEDED=5` |
| Stop sign | `SIGN_CONF_THRESH=0.60`, `SIGN_YELLOW_AREA_FRAC=0.12` |
| Obstacle | `OBS_BLUE_H_MIN/MAX=100–135`, `OBS_ASPECT_MIN=1.2` |
| Parking | `PARK_MIN_PIXELS=400`, `PARK_OBS_THRESHOLD=30` |
| IMU warp | `PITCH_PX_PER_DEG=8.0` (tune on vehicle) |
| UART | `UART_HEARTBEAT_S=0.080`, `THROTTLE_VALUE=189` |
| BEV | `WARP_ENABLED=False` (enable after calibration) |

---

## What Needs to Be Tested and Tuned Before Competition

### On-Vehicle Hardware Tests (required before any driving)

| # | Task | File / Config key | Notes |
|---|------|-------------------|-------|
| 1 | **BEV calibration** | `config.py → WARP_SRC/DST`, then set `WARP_ENABLED=True` | Place markers at 1m/2m/3m/4m ahead, record pixel coords, compute homography. Required before lane metrics are reliable. |
| 2 | **IMU pitch baseline** | `config.py → PITCH_BASELINE_DEG` | Record the camera's resting pitch angle (ZED Euler Y) with car on flat ground. Set this value. |
| 3 | **PITCH_PX_PER_DEG tuning** | `config.py → PITCH_PX_PER_DEG` | Drive over a small bump, watch BEV output. Adjust until lane lines stay parallel through the bump. |
| 4 | **UART sanity** | `uart_test.py` | Verify STM32 echoes correctly, watchdog triggers at 200ms, all commands arrive intact. |
| 5 | **ZED floor plane stability** | Run `pipeline.py`, watch `source` field | Should read `WHITE_LINE` or `GRASS`, not `NO_FLOOR`. Tune `FLOOR_TOLERANCE` if floor is lost. |

### Colour Threshold Tuning (track conditions)

| # | Task | Config key | Method |
|---|------|------------|--------|
| 6 | White lane threshold | `WHITE_L_MIN` | Lower in shadow (try 140), raise in glare (try 175). Check `wm` mask overlay in visualiser. |
| 7 | Orange stop line HSV | `STOP_ORANGE_H_MIN/MAX`, `STOP_ORANGE_S_MIN` | Photograph the actual SEM orange line. Measure HSV range in that image. |
| 8 | Blue obstacle range | `OBS_BLUE_H_MIN/MAX` | Hold the actual blue pin in front of the car at 3m. Tune until `obstacle_detected = True`. |
| 9 | Blue parking range | `PARK_BLUE_H_MIN/MAX` | Tape blue lines on floor, drive past, tune until `parking_detected = True`. |

### Stop Sign Retraining (critical before competition)

```bash
# Collect 100+ images of SEM red-hexagon-on-yellow-board sign
# Annotate with LabelImg or Roboflow
python scripts/train_stop_sign.py --api-key YOUR_ROBOFLOW_KEY
python scripts/export_trt.py  # export to TensorRT FP16 for Jetson
# Test: SIGN_CONF_THRESH=0.60, drive past sign at 5m → stop_sign=True within 3 frames
```

### Obstacle & Parking Bench Tests

| # | Test | Pass condition |
|---|------|---------------|
| 10 | Blue balloon at 3m on asphalt | `obstacle_detected=True`, `obstacle_dist_m ≈ 3.0` within 3 frames |
| 11 | Blue balloon to left of lane | `obstacle_lateral_m < 0` |
| 12 | Tape blue rectangle on floor (parking mock) | `parking_detected=True`, `parking_empty=True` |
| 13 | Place cardboard box inside blue rectangle | `parking_detected=True`, `parking_empty=False` |
| 14 | Full stop: drive at orange tape line | `stop_line=True` at 3m, car stops before the line |

### Performance Targets on Jetson Nano

| Metric | Target | How to check |
|--------|--------|-------------|
| FPS | ≥ 25 | Profile print every 30 frames. `FPS_WARN_BELOW=20` triggers a console warning. |
| Stop line detection latency | < 5 frames (167ms) at 30Hz | Count frames from first orange pixel to `stop_line=True` |
| Obstacle vote confirmation | 3 frames (100ms) | Confirmed before obstacle is within 6m at 15 km/h |
| UART round-trip | < 10ms | Measure ACK latency in `uart_test.py` |

---

## Infrastructure for Future Control Steps

The perception stack is deliberately decoupled from control. The `PerceptionResult` dataclass is the single handoff point. Every future control module reads from it without touching the camera, ZED SDK, or colour masks.

### Pure Pursuit Steering (Phase 4 — ready to implement)

All inputs are already computed and available in `PerceptionResult`:

```python
# These fields are computed every frame by lane/control.py:
result.lookahead_point   # (X_m, Z_m) — the exact input Pure Pursuit needs
result.heading_angle     # radians — fallback when lookahead is None
result.curvature         # κ m⁻¹ — for speed governor
result.deviation_m       # metres — for Kalman filter input

# Pure Pursuit formula (implement in commander.py):
alpha = atan2(X_m, Z_m)
delta = atan(2 * WHEELBASE_M * sin(alpha) / sqrt(X_m² + Z_m²))
steer_byte = clip(127 + round(delta / radians(30) * 127), 0, 255)
```

The `CMD_STEER` command code is already defined in `control/uart.py` (0x03). The `steer()` method and send-rate fix need to be added to `UARTController`.

### Mission State Machine (Phase 4 — scaffolded)

The commander currently handles THROTTLE/BRAKE/IDLE reactively. The required states map directly to `PerceptionResult` fields:

| State | Trigger condition (from PerceptionResult) |
|-------|------------------------------------------|
| `LANE_FOLLOW` | Default — send STEER + THROTTLE |
| `STOP_APPROACH` | `stop_line=True` AND `stop_line_dist < 3.0m` |
| `FULL_STOP` | `stop_line_dist < 1.0m` → hold BRAKE for ~1.5s |
| `PROCEED` | Timer elapsed → resume THROTTLE + STEER |
| `OBSTACLE_AVOID` | `obstacle_detected=True` → shift `lookahead_point.X` by ±0.55m |
| `SEEK_PARK` | `parking_detected=True AND parking_empty=True` → use `parking_center_m` as lateral target |
| `PARKED` | Distance to `parking_center_m` < 0.5m → full BRAKE + IDLE |

### Kalman Filter on Lateral Deviation (Phase 4)

`deviation_m` already has a holdover mechanism (last known value used on `LOST` source). Replace with a proper 2-state Kalman:

```
State:       [deviation_m, lateral_velocity_mps]
Measurement: [deviation_m]    from PerceptionResult
Control:     [steering_angle]  predictive model
```

This runs in microseconds and handles 2–3 frame detection dropouts (shadow, worn marking) cleanly.

### Speed Feedback via ZED Positional Tracking

ZED positional tracking is already enabled (`enable_positional_tracking()`). Sequential pose differences give forward velocity:

```python
# In pipeline.py — extract speed from ZED VO:
self.cam.get_position(self.pose)
translation = self.pose.get_translation()
# diff(translation.z) / dt → forward speed (m/s)
```

This gives ~30Hz speed estimate without IMU integration drift. Feed directly to Pure Pursuit for the correct `lookahead_distance = v * 0.5` scaling.

### LLC Upgrade (Phase 4)

Two changes needed in `control/uart.py`:

1. **Remove STEER deduplication** — `send()` currently skips identical consecutive commands. STEER must be sent every frame (30 Hz).
2. **Add `steer(value: int)` method** — identical pattern to existing `throttle()` and `brake()`.

STM32-side: add a 5-byte status reply frame `[0xBB][LEN=3][STATUS][ENC_HI][ENC_LO][CRC8]` so the Jetson can monitor watchdog health and clutch state without polling.

---

## Running the Stack

```bash
# On Jetson Nano — requires ZED SDK, ultralytics, pyserial
cd /path/to/AdhamTeam
python -m perception_stack.main

# Dry run (no UART / no STM32 needed):
# Set UART_ENABLED = False in config.py

# Display off (SSH / headless):
# Set DISPLAY = False in config.py

# Telemetry replay:
cat logs/run_*.jsonl | python -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    if r['obs']: print(f\"t={r['t']:.1f}s  OBS {r['obs_dist']}m lat={r['obs_lat']}\")
"
```

---

## Competition Checklist

- [ ] BEV homography calibrated and `WARP_ENABLED = True`
- [ ] `PITCH_BASELINE_DEG` set from vehicle IMU reading
- [ ] Orange stop line HSV verified on actual SEM markings
- [ ] Blue obstacle HSV verified on actual SEM pin
- [ ] YOLO retrained on SEM yellow-board sign → weights in `weights/stop_sign.engine`
- [ ] `SIGN_CONF_THRESH = 0.60` confirmed (do not lower below 0.55)
- [ ] Full stop at orange line: stops before line, within 2 car lengths
- [ ] Obstacle detection: `obstacle_detected=True` at 6m, `lateral_m` correct side
- [ ] Parking bay detection: `parking_empty` correctly reports crate presence
- [ ] Telemetry log created at start of each run (`LOG_TELEMETRY = True`)
- [ ] FPS ≥ 25 on Jetson Nano at QUALITY depth mode

---

## Hardware Reference

| Component | Spec |
|-----------|------|
| Compute | Jetson Nano 4GB |
| Camera | ZED 2i — 720p, 30 FPS, stereo depth, IMU |
| Blind spot | MB7062 ultrasonic (0–1m, ZED min range fill) |
| LLC | STM32 Nucleo H743 |
| Drive | D5BLD750-48A-30S + PLF090-10 gearbox (750W, 300 RPM) |
| Steering | NEMA23 + EG23 gearbox (50:1) + CL57T-V41 closed-loop driver |
| EM Clutch | DLD6-20B 24V 20Nm (fail-safe manual override) |
| Brake | 2× 35kg.cm servo (dual redundant) |
| UART | `/dev/ttyTHS1`, 115200 baud, 5-byte framed, CRC-8/SMBUS |

---

*PSU Eco Racing Autonomous Team — 2026*
