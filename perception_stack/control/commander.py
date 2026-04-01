"""
PSU Eco Racing — Perception Stack
control/commander.py  |  High-level command decision layer.

Takes a PerceptionResult every frame and decides what to tell the
low-level controller:

    BRAKE    — stop line or stop sign within STOP_BRAKE_DIST_M
    THROTTLE — no stop trigger active

The commander owns a UARTController and handles open/close.
"""

import logging

from perception_stack.config  import (
    UART_ENABLED, THROTTLE_VALUE, BRAKE_VALUE, STOP_BRAKE_DIST_M,
)
from perception_stack.models  import PerceptionResult
from perception_stack.control.uart import UARTController, CMD_IDLE, CMD_THROTTLE, CMD_BRAKE

log = logging.getLogger(__name__)


class Commander:
    """
    Usage (in main loop):
        cmd = Commander()
        cmd.open()
        ...
        cmd.update(result)      # call every frame
        ...
        cmd.close()
    """

    def __init__(self):
        self.uart   = UARTController()
        self._state = "INIT"    # last commanded state for logging

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    def open(self) -> bool:
        if not UART_ENABLED:
            log.info("[Commander] UART disabled — dry-run mode")
            return True
        ok = self.uart.open()
        if ok:
            self.uart.idle()    # start in safe state
        return ok

    def close(self):
        self.uart.close()

    # ── Decision ────────────────────────────────────────────────────────────────

    def update(self, result: PerceptionResult) -> str:
        """
        Evaluate result and send the appropriate UART command.
        Returns the command name string ("THROTTLE" / "BRAKE" / "IDLE").
        """
        state = self._decide(result)

        if UART_ENABLED:
            if state == "BRAKE":
                self.uart.brake(BRAKE_VALUE)
            elif state == "THROTTLE":
                self.uart.throttle(THROTTLE_VALUE)
            else:
                self.uart.idle()

        if state != self._state:
            log.info("[Commander] %s → %s  (src=%s)", self._state, state, result.source)
            self._state = state

        return state

    # ── Internal ────────────────────────────────────────────────────────────────

    @staticmethod
    def _decide(result: PerceptionResult) -> str:
        # 1. Stop line confirmed and within brake distance
        if result.stop_line and 0 < result.stop_line_dist <= STOP_BRAKE_DIST_M:
            return "BRAKE"

        # 2. Stop sign confirmed and within brake distance
        if result.stop_sign and 0 < result.stop_sign_dist_m <= STOP_BRAKE_DIST_M:
            return "BRAKE"

        # 3. No stop trigger — go forward
        return "THROTTLE"
