# PSU Eco Racing — Perception Stack: Running Modes

All mode switching is done in **one file only**: `perception_stack/config.py`

---

## Simple Test Mode (current default)

**What it does:** Car drives forward (THROTTLE) and brakes when it sees a stop sign or stop line. No lane detection, no obstacle avoidance, no steering.

**How to enable:**
```python
# perception_stack/config.py
SIMPLE_TEST_MODE = True
CAM_DEPTH_MODE   = sl.DEPTH_MODE.PERFORMANCE
```

**What runs:**
- ZED camera + floor plane estimation
- Stop-line detection (orange stripe)
- Stop-sign detection (red octagon)
- UART commands: THROTTLE → BRAKE on detection

**What is disabled (for speed):**
- Lane fitting (no RANSAC, no sliding windows)
- Grass edge detection
- Obstacle point-cloud scan
- Lane memory / virtual boundaries
- Heading, curvature, lookahead computation

**Live display shows:**
- Red line on road when stop line detected
- Red bounding box when stop sign detected
- HUD: Speed | State (THROTTLE / BRAKE / DISENGAGED) | Detection distances

---

## Full Autonomy Mode

**What it does:** Full lane following with deviation correction, obstacle avoidance, steering, and stop detection.

**How to enable:**
```python
# perception_stack/config.py
SIMPLE_TEST_MODE = False
CAM_DEPTH_MODE   = sl.DEPTH_MODE.NEURAL
```

**What runs (everything):**
- ZED camera + floor plane
- White line detection (HLS) + grass edge fallback (HSV)
- RANSAC quadratic lane polynomial fitting
- Lane memory + virtual boundaries for single-side turns
- Deviation, heading angle, curvature, lookahead point (Pure Pursuit ready)
- Stop-line + stop-sign detection
- Obstacle point-cloud detection
- Full UART command output

**Live display shows:**
- Lane corridor fill + left/right boundary lines
- Center deviation bar
- Lookahead point marker
- Obstacle bounding boxes
- Full HUD: source, confidence, width, deviation, heading, curvature, all detections

---

## Key Thresholds to Tune

All in `perception_stack/config.py`:

| Parameter | Default | What it controls |
|---|---|---|
| `STOP_BRAKE_DIST_M` | `1.0` | How close a stop trigger must be before braking |
| `BRAKE_DIST_M` | `1.5` | Obstacle brake distance (full mode only) |
| `THROTTLE_VALUE` | `20` | Throttle intensity sent to Nucleo (0–255) |
| `BRAKE_VALUE` | `255` | Brake intensity sent to Nucleo (0–255) |
| `STOP_VOTE_NEEDED` | `5` | Frames required to confirm a stop line (~167ms at 30fps) |
| `SIGN_VOTE_NEEDED` | `3` | Frames required to confirm a stop sign (~100ms at 30fps) |
| `CAM_FPS` | `30` | Camera frame rate |

---

## Running

```bash
# From inside the AdhamTeam/ directory:
python -m perception_stack.main
```

Press **Q** in the display window to quit cleanly (sends IDLE to Nucleo before closing).

---

## UART Dry-Run (no Nucleo connected)

To run vision-only without a connected Nucleo (e.g. on a laptop for testing):

```python
# perception_stack/config.py
UART_ENABLED = False
```

The perception and display run normally — commands are decided but not transmitted.
