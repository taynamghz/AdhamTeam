# PSU Eco Racing — Autonomous Perception Stack
**Prince Sultan University | Shell Eco-Marathon Urban Concept 2026**
**Venue:** Silesia Ring, Kamień Śląski, Poland | **Target speed:** 10–15 km/h

---

## Overview

Real-time autonomous perception and control stack for the PSU Eco Racing Shell Eco-Marathon vehicle. Runs on a **Jetson Orin Nano 8GB Super** with a **ZED 2i stereo camera**, communicates with an **STM32 Nucleo H743** low-level controller over binary UART.

The stack detects the drivable area using **Segformer semantic segmentation**, computes steering via **Pure Pursuit**, and sends throttle/steer/brake setpoints to the Nucleo. The Nucleo runs its own PID — the Jetson only sends target values.

```
ZED 2i Camera (720p, 30 FPS)
        │
        ▼
  CLAHE Normalisation        ← lighting invariance (LAB L-channel)
        │
        ├──► SegformerLane (background thread)
        │       Segformer-B2 cityscapes → road mask → boundary polynomials
        │       → deviation_m, heading_angle, curvature
        │
        ├──► StopSignDetector (background thread)
        │       YOLOv8 → yellow-board gate → vote → confirmed sign + dist
        │
        ├──► detect_stop_line()
        │       orange HSV → row density → width gate → temporal vote
        │
        └──► PerceptionResult
                │
                ▼
         Commander  (Pure Pursuit + anti-jitter stack)
                │
                ▼
         UART → STM32 Nucleo H743
           CMD_STEER     → steering angle  (0=full-left, 127=centre, 255=full-right)
           CMD_THROTTLE  → target speed    (byte = km/h × 10)
           CMD_BRAKE     → emergency stop  (val = 255)
           ← CMD_SPEED_REPORT ← actual speed from Nucleo (tenths of km/h)
```

---

## Repository Structure

```
perception_stack/
├── main.py                   Entry point — main loop, telemetry logger, display
├── config.py                 All tunable parameters (no magic numbers in code)
├── models.py                 PerceptionResult dataclass — single output contract
├── visualization.py          OpenCV debug overlay (UART commands, lanes, HUD)
│
├── perception/
│   ├── pipeline.py           LanePerception — orchestrates every frame
│   └── segformer_lane.py     Segformer-B2 drivable-area detector (async thread)
│
├── lane/
│   ├── fitting.py            eval_x() — polynomial evaluation helper
│   └── control.py            heading angle, curvature, lookahead, ControlSmoother
│
├── detection/
│   ├── stop_line.py          Orange stripe detection with physical width validation
│   └── stop_sign.py          Threaded YOLOv8 with yellow-board gate (SEM-specific)
│
└── control/
    ├── commander.py          Pure Pursuit + anti-jitter → UART setpoints every frame
    └── uart.py               5-byte binary framed protocol to STM32 with CRC-8
```

---

## How the Pipeline Works

### 1. CLAHE Normalisation
Every frame before any processing:
```
BGR → LAB → CLAHE on L channel → LAB → BGR
```
Makes colour thresholds invariant to auto-exposure, shadows, and outdoor lighting.

### 2. Drivable Area Detection (Segformer)

`SegformerLane` runs **nvidia/segformer-b2-finetuned-cityscapes-1024-1024** in a background thread. The main thread never blocks on inference.

```
frame → Segformer → semantic mask (class 0 = road/drivable)
    → scan left  boundary each row → left polynomial  lf
    → scan right boundary each row → right polynomial rf
    → midpoint of lf + rf          → deviation_m  (lateral offset from centre)
    → polynomial 1st derivative    → heading_angle
    → polynomial 2nd derivative    → curvature κ (m⁻¹)
```

**Adaptive submission rate:**

| Condition | Segformer rate | Reason |
|-----------|---------------|--------|
| Straight `\|κ\| < 0.15` | ~6 Hz (1 in 5 frames) | Road shape barely changes; EMA carries the polynomial |
| Curve `\|κ\| ≥ 0.15` | ~30 Hz (every frame) | Maximum freshness for steering accuracy |

**Thread pattern:**
- `submit()` — drops stale queued frame if worker busy (no inference lag accumulation)
- `get_result()` — instant lock-read, always returns most recent completed result

### 3. Stop Line Detection
Six sequential gates (all must pass):
1. Road mask — drivable-area pixels only
2. Orange HSV — hue 5–20°, S ≥ 150, V ≥ 100
3. Row density — ≥ 8% of frame width orange
4. PCA perpendicularity — cluster within ±20° of horizontal
5. Physical stripe width — orange spans ≥ 70% of lane width (rejects cones, debris)
6. Temporal vote — 5 consecutive frames (drops to 3 if stop sign pre-armed at 3–8 m)

### 4. Stop Sign Detection (YOLOv8)
Runs in its own thread. SEM-specific hardening on every detection:
- **Yellow board gate** — checks for yellow HSV (hue 18–38°, S ≥ 120) in 1.3× expanded bbox
- **Bbox height sanity** — expected height ≈ `(0.65m / dist) × 730px`; rejects too-small detections
- Confidence threshold: 0.60
- 3-frame temporal voting

### 5. Pure Pursuit Steering

Every frame:
```
steer_rad = atan(deviation_m / lookahead_m) − heading_angle
```
`deviation_m > 0` = vehicle left of centre → steer right.
`heading_angle` = road direction feed-forward for curve anticipation.

**Anti-jitter stack:**
```
Pure Pursuit
    → clamp ±25°          (hardware limit — right side mechanically restricted)
    → dead-band 2°        (suppress sub-noise corrections)
    → rate-limit 8°/frame (single bad Segformer frame moves wheel ≤ 8°)
    → EMA α=0.40          (~2–3 frame lag, eliminates high-frequency jitter)
    → encode to 0–255 byte → CMD_STEER every frame
```

On `LOST` detection: steering decays 5%/frame toward straight (no snap-to-centre).

### 6. Speed Control

| Condition | Target | Sent as |
|-----------|--------|---------|
| Straight `\|κ\| < 0.15` | 15.0 km/h | `CMD_THROTTLE byte=150` |
| Curve `\|κ\| ≥ 0.15` | 10.0 km/h | `CMD_THROTTLE byte=100` |
| Stop line/sign ≤ 1.0 m | BRAKE | `CMD_BRAKE byte=255` |

Nucleo PID handles all actuation. Jetson sends target only.
Actual speed is received from Nucleo RX packets (`CMD_SPEED_REPORT`).

---

## UART Protocol

### TX — Jetson → Nucleo (5-byte frame)
```
[0xAA] [LEN=2] [CMD] [DATA] [CRC8/SMBUS]

CMD_IDLE     = 0x00
CMD_THROTTLE = 0x01   DATA = int(km/h × 10)   150 → 15.0 km/h
CMD_BRAKE    = 0x02   DATA = 255
CMD_STEER    = 0x03   DATA = 0–255  (0=full-left, 127=centre, 255=full-right)
```
Heartbeat retransmit every 80 ms keeps Nucleo 200 ms watchdog alive.

### RX — Nucleo → Jetson (5-byte frame, background reader thread)
```
[0xBB] [0x02] [0x10] [DATA] [CRC8]
DATA = speed in tenths of km/h  e.g. 153 → 15.3 km/h
```

---

## PerceptionResult — Output Contract

```python
@dataclass
class PerceptionResult:
    deviation_m:        float       # lateral offset from centre (m); + = left of centre
    confidence:         float       # 0–1 combined detection confidence
    lane_width_m:       float       # metric lane width (m)
    source:             str         # "SEGFORMER" / "SEG_PARTIAL" / "LOST" / "DISABLED"
    left_fit:           np.ndarray  # quadratic poly  x = a·y² + b·y + c
    right_fit:          np.ndarray
    left_conf:          float
    right_conf:         float
    stop_line:          bool
    stop_line_dist:     float       # metres to stop line (ZED point cloud)
    stop_line_y:        int         # image row of stop line
    stop_sign:          bool
    stop_sign_dist_m:   float
    stop_sign_bbox:     tuple       # (x, y, w, h) pixels
    heading_angle:      float       # radians
    curvature:          float       # κ (m⁻¹), signed
    lookahead_point:    tuple       # (X_m, Z_m) world-space Pure Pursuit target
    lookahead_pixel:    tuple       # (x, y) image pixel
    speed_kmh:          float       # actual speed from Nucleo UART (filled by Commander)
```

---

## Configuration (`config.py`)

| Section | Key parameters |
|---------|---------------|
| Camera | `CAM_RES=HD720`, `CAM_FPS=30`, `CAM_DEPTH_MODE=PERFORMANCE` |
| CLAHE | `CLAHE_CLIP_LIMIT=2.0`, `CLAHE_TILE_SIZE=(8,8)` |
| Segformer | `SEG_MODEL_ID`, `SEG_ROAD_CLASSES=[0]`, `SEG_POLY_DEG=2` |
| Adaptive rate | `SEG_SKIP_STRAIGHT=5`, `SEG_SKIP_CURVE=1` |
| Stop line | `STOP_WIDTH_MIN_FRAC=0.70`, `STOP_VOTE_NEEDED=5` |
| Stop sign | `SIGN_CONF_THRESH=0.60`, `SIGN_YELLOW_AREA_FRAC=0.12` |
| Steering | `STEER_MAX_DEG=25.0`, `STEER_DEADBAND_DEG=2.0`, `STEER_RATE_DEG=8.0`, `STEER_EMA_ALPHA=0.40` |
| Speed | `SPEED_TARGET_STRAIGHT_KMH=15.0`, `SPEED_TARGET_CURVE_KMH=10.0`, `SPEED_CURVE_THRESH=0.15` |
| Brake | `STOP_BRAKE_DIST_M=1.0`, `BRAKE_VALUE=255` |
| Pure Pursuit | `CTRL_LOOKAHEAD_M=2.5`, `CTRL_EVAL_Y_FRAC=0.60` |
| UART | `UART_PORT=/dev/ttyTHS1`, `UART_BAUD=115200`, `UART_HEARTBEAT_S=0.080` |

---

## Running

```bash
# On Jetson — requires ZED SDK, transformers, ultralytics, pyserial
cd /path/to/AdhamTeam
python -m perception_stack.main

# Dry run (no UART / no STM32):
# Set UART_ENABLED = False in config.py

# Headless / SSH:
# Set DISPLAY = False in config.py
```

---

## Before First Drive — Checklist

| # | Task | Notes |
|---|------|-------|
| 1 | YOLO weights at `perception_stack/weights/stop_sign.pt` | Or `.engine` after TRT export |
| 2 | UART loopback test | Verify Nucleo sends `CMD_SPEED_REPORT` at 115200 |
| 3 | Orange stop line HSV verified | Photograph actual SEM orange tape, check hue 5–20° |
| 4 | `STOP_BRAKE_DIST_M` tuned | Default 1.0 m — adjust to vehicle braking distance |
| 5 | Steering centre confirmed | Byte 127 → straight driving on actual car |
| 6 | Segformer warmup | First inference ~2–3 s (model load); pipeline returns LOST until ready |

### Stop Sign Retraining (required before competition)
```bash
# Current weights: generic US stop signs.
# SEM sign: red hexagon on yellow rectangular board.
python scripts/train_stop_sign.py --api-key YOUR_ROBOFLOW_KEY
python scripts/export_trt.py
# Then: SIGN_MODEL_PATH = "perception_stack/weights/stop_sign.engine"
```

### TensorRT Export (run on Jetson before competition)
```bash
python scripts/export_trt.py
# YOLO: ~60ms → ~8ms on Orin Nano
# Engine is device-specific — build on the deployment Jetson
```

---

## Hardware Reference

| Component | Spec |
|-----------|------|
| Compute | Jetson Orin Nano 8GB Super |
| Camera | ZED 2i — 720p, 30 FPS, stereo depth, IMU |
| LLC | STM32 Nucleo H743 |
| Drive | D5BLD750-48A-30S + PLF090-10 gearbox (750W, 300 RPM) |
| Steering | NEMA23 + EG23 gearbox (50:1) + CL57T-V41 closed-loop driver |
| Steering range | −25° to +25° (right side mechanically limited; physical centre = −10°) |
| EM Clutch | DLD6-20B 24V 20Nm (fail-safe manual override) |
| Brake | 2× 35kg.cm servo (dual redundant) |
| UART | `/dev/ttyTHS1`, 115200 baud, 5-byte framed, CRC-8/SMBUS |

---

*PSU Eco Racing Autonomous Team — 2026*
