"""
PSU Eco Racing — Perception Stack
control/commander.py  |  High-level command decision layer.

The Nucleo LLC runs PID control internally.  The Jetson sends only setpoints:

  CMD_THROTTLE  DATA = target speed in tenths of km/h  (150 → 15.0 km/h)
  CMD_BRAKE     DATA = brake intensity (emergency stop at stop-line / stop-sign)
  CMD_STEER     DATA = steering angle byte (0=full-left, 127=centre, 255=full-right)

Command update trigger — lookahead-distance gating:
  Steer + throttle are only sent when the car has physically traveled one
  CTRL_LOOKAHEAD_M since the last transmission.  Distance is integrated from
  the actual speed reported by the Nucleo UART telemetry.

  Why: Pure Pursuit computes a target point CTRL_LOOKAHEAD_M ahead.  Sending
  a new command before the car reaches that point just fights the motor PID
  mid-move.  Waiting one lookahead distance lets each command fully execute
  before the next correction is issued.

  Natural rate:
    15 km/h, L=2.5 m → sends every ~0.6 s  (~1.7 Hz)
    10 km/h, L=2.5 m → sends every ~0.9 s  (~1.1 Hz)

  Fallback: if speed = 0 (car stopped / UART not yet connected), falls back
  to a fixed interval = CTRL_LOOKAHEAD_M / SPEED_TARGET_STRAIGHT_KMH so the
  system doesn't freeze waiting for movement that hasn't started.

  Brake commands are always sent immediately, every frame — never gated.
"""

import math
import logging

from perception_stack.config import (
    UART_ENABLED,
    STOP_BRAKE_DIST_M, BRAKE_VALUE,
    SPEED_TARGET_STRAIGHT_KMH, SPEED_TARGET_CURVE_KMH, SPEED_CURVE_THRESH,
    CTRL_LOOKAHEAD_M,
    STEER_MAX_DEG, STEER_DEADBAND_DEG, STEER_RATE_DEG, STEER_EMA_ALPHA,
    STEER_TX_DEADBAND_DEG,
)
from perception_stack.models import PerceptionResult
from perception_stack.control.uart import UARTController

log = logging.getLogger(__name__)


def _deg_to_steer_byte(deg: float) -> int:
    """Map ±STEER_MAX_DEG → [0, 255] with 127 = straight."""
    return int(max(0, min(255, round(127.0 + deg * 127.0 / STEER_MAX_DEG))))


class Commander:
    """
    Usage:
        cmd = Commander()
        cmd.open()
        cmd.update(result)      # call every frame
        cmd.close()

    Public attributes (updated each frame):
        cmd.target_kmh   — speed setpoint sent to Nucleo (km/h)
        cmd.steer_deg    — steering angle sent to Nucleo (degrees)
        cmd.speed_kmh    — current speed received from Nucleo (km/h)
    """

    def __init__(self):
        self.uart = UARTController()

        self._state:          str   = "RUN"
        self._last_steer:     float = 0.0
        self._steer_ema:      float = 0.0
        self._last_sent_deg:  float = 0.0   # last angle actually transmitted to Nucleo

        # Public state (for display / telemetry)
        self.target_kmh: float = SPEED_TARGET_STRAIGHT_KMH
        self.steer_deg:  float = 0.0
        self.speed_kmh:  float = 0.0

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    def open(self) -> bool:
        if not UART_ENABLED:
            log.info("[Commander] UART disabled — dry-run mode")
            return True
        ok = self.uart.open()
        if ok:
            self.uart.idle()
            self.uart.steer(127)   # centre steering on start-up
        return ok

    def close(self):
        self.uart.set_speed(0.0)   # tell Nucleo to stop
        self.uart.steer(127)       # return to centre
        self.uart.close()

    # ── Main update ─────────────────────────────────────────────────────────────

    def update(self, result: PerceptionResult) -> str:
        """
        Send UART setpoints for this frame.
        Returns "BRAKE" or "RUN".
        """
        # Read current speed from the LLC UART reader thread (non-blocking)
        self.speed_kmh = self.uart.speed_kmh

        brake   = self._should_brake(result)
        target  = self._target_speed(result)
        steer   = self._compute_steer(result)

        self.target_kmh = target
        self.steer_deg  = steer
        state = "BRAKE" if brake else "RUN"

        if UART_ENABLED:
            if brake:
                self.uart.brake(BRAKE_VALUE)
                self._last_sent_deg = 0.0   # reset so next RUN sends immediately
            else:
                self.uart.set_speed(target)
                # TX dead-band: only send CMD_STEER when the angle has changed
                # meaningfully from the last transmitted value.  Suppresses rapid
                # micro-corrections from mask noise — motor only moves for real changes.
                if abs(steer - self._last_sent_deg) >= STEER_TX_DEADBAND_DEG:
                    self.uart.steer(_deg_to_steer_byte(steer))
                    self._last_sent_deg = steer

        if state != self._state:
            log.info("[Commander] %s → %s  src=%s  spd=%.1f km/h  steer=%.1f deg",
                     self._state, state, result.source, self.speed_kmh, steer)
            self._state = state

        return state

    # ── Brake decision ───────────────────────────────────────────────────────────

    @staticmethod
    def _should_brake(result: PerceptionResult) -> bool:
        if result.stop_line and 0 < result.stop_line_dist <= STOP_BRAKE_DIST_M:
            return True
        if result.stop_sign and 0 < result.stop_sign_dist_m <= STOP_BRAKE_DIST_M:
            return True
        return False

    # ── Target speed ─────────────────────────────────────────────────────────────

    @staticmethod
    def _target_speed(result: PerceptionResult) -> float:
        """Slow down on curves; maintain nominal speed on straights."""
        if abs(result.curvature) > SPEED_CURVE_THRESH:
            return SPEED_TARGET_CURVE_KMH
        return SPEED_TARGET_STRAIGHT_KMH

    # ── Pure Pursuit steering ────────────────────────────────────────────────────

    def _compute_steer(self, result: PerceptionResult) -> float:
        """
        steer_rad = atan(deviation_m / lookahead_m) − heading_angle

        Sign conventions (both must agree):
            deviation_m  > 0 → vehicle LEFT  of centre → correct right (+)
            heading_angle> 0 → lane going right (left curve) → subtract → steer left (−)

        Anti-jitter: clamp → dead-band → rate-limit → EMA.
        """
        src = result.source

        # No lane data — reset or decay gently
        if src in ("DISABLED", "NONE"):
            self._steer_ema = self._last_steer = 0.0
            return 0.0

        if src == "LOST" or result.confidence < 0.15:
            self._steer_ema  *= 0.95   # decay ~5 % per frame toward straight
            self._last_steer  = self._steer_ema
            return self._steer_ema

        # Pure Pursuit lateral correction + road-heading feed-forward
        raw_rad = (math.atan(result.deviation_m / max(CTRL_LOOKAHEAD_M, 0.1))
                   - result.heading_angle)
        raw_deg = math.degrees(raw_rad)

        # 1. Clamp to physical range
        raw_deg = max(-STEER_MAX_DEG, min(STEER_MAX_DEG, raw_deg))

        # 2. Dead-band — ignore corrections smaller than ~2° (mask noise)
        if abs(raw_deg) < STEER_DEADBAND_DEG:
            raw_deg = 0.0

        # 3. Rate-limit — one bad Segformer frame can't spin the wheel
        delta    = max(-STEER_RATE_DEG, min(STEER_RATE_DEG, raw_deg - self._last_steer))
        rate_lim = self._last_steer + delta

        # 4. EMA — final temporal smoothing
        self._steer_ema  = (STEER_EMA_ALPHA * rate_lim
                            + (1.0 - STEER_EMA_ALPHA) * self._steer_ema)
        self._last_steer = self._steer_ema
        return self._steer_ema
