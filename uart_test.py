"""
UART Manual Test — PSU Eco Racing
Run from AdhamTeam/:  python3 uart_test.py

Commands (type and press Enter):
  t <value>   → THROTTLE  value = km/h  e.g. "t 6"  sends 6.0 km/h
  s <value>   → STEER     value = degrees  e.g. "s -15"  (- = left, + = right)
  b <value>   → BRAKE     value = 0-255  e.g. "b 255"
  i           → IDLE      (stop everything)
  q           → quit
"""

import sys, os, time, logging

# ── Allow running without ZED SDK ─────────────────────────────────────────────
from unittest.mock import MagicMock
sys.modules.setdefault("pyzed",    MagicMock())
sys.modules.setdefault("pyzed.sl", MagicMock())

sys.path.insert(0, os.path.dirname(__file__))

from perception_stack.control.uart import (
    UARTController, CMD_IDLE, CMD_THROTTLE, CMD_BRAKE, CMD_STEER, _build_frame,
)
from perception_stack.config import UART_PORT, UART_BAUD, STEER_MAX_DEG

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("uart_test")


def show_frame(cmd, val):
    frame = _build_frame(cmd, val)
    hex_str = " ".join(f"{b:02X}" for b in frame)
    log.debug("  Raw bytes: %s", hex_str)


def deg_to_byte(deg: float) -> int:
    """±STEER_MAX_DEG → 0-255, 127 = straight."""
    return int(max(0, min(255, round(127.0 + deg * 127.0 / STEER_MAX_DEG))))


def force_send(uart, cmd, val):
    """Always transmit, bypassing de-duplication cache."""
    uart._last_cmd = -1
    uart._last_val = -1
    ok = uart.send(cmd, val)
    log.info("    send() returned: %s", ok)


def main():
    print(f"\n{'='*52}")
    print(f"  UART Manual Test")
    print(f"  Port : {UART_PORT}   Baud : {UART_BAUD}")
    print(f"{'='*52}")
    print("  t <kmh>    — throttle  e.g.  t 6")
    print("  s <deg>    — steer     e.g.  s -20  (- left / + right)")
    print(f"               range: -{STEER_MAX_DEG:.0f} to +{STEER_MAX_DEG:.0f} deg")
    print("  b <0-255>  — brake     e.g.  b 255")
    print("  i          — idle (stop)")
    print("  q          — quit")
    print(f"{'='*52}\n")

    uart = UARTController()
    log.info("Opening %s @ %d baud ...", UART_PORT, UART_BAUD)

    if not uart.open():
        log.error("FAILED to open UART port.")
        log.error("  → Is %s present?  Run: ls /dev/ttyTHS*", UART_PORT)
        log.error("  → Permission?     Run: sudo usermod -aG dialout $USER")
        sys.exit(1)

    log.info("UART opened OK.  Sending IDLE as safe start...")
    show_frame(CMD_IDLE, 0)
    force_send(uart, CMD_IDLE, 0)
    time.sleep(0.1)

    try:
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                continue

            parts = line.split()
            cmd_char = parts[0].lower()

            if cmd_char == "q":
                break

            elif cmd_char == "i":
                log.info(">>> IDLE")
                show_frame(CMD_IDLE, 0)
                force_send(uart, CMD_IDLE, 0)

            elif cmd_char == "t":
                if len(parts) < 2:
                    print("  usage: t <kmh>  e.g.  t 6")
                    continue
                try:
                    kmh = float(parts[1])
                except ValueError:
                    print("  invalid value")
                    continue
                val = int(max(0, min(255, round(kmh * 10.0))))
                log.info(">>> THROTTLE  %.1f km/h  → byte %d", kmh, val)
                show_frame(CMD_THROTTLE, val)
                force_send(uart, CMD_THROTTLE, val)

            elif cmd_char == "s":
                if len(parts) < 2:
                    print(f"  usage: s <deg>  e.g.  s -20  (range ±{STEER_MAX_DEG:.0f})")
                    continue
                try:
                    deg = float(parts[1])
                except ValueError:
                    print("  invalid value")
                    continue
                if abs(deg) > STEER_MAX_DEG:
                    print(f"  clamping to ±{STEER_MAX_DEG:.0f} deg")
                    deg = max(-STEER_MAX_DEG, min(STEER_MAX_DEG, deg))
                val = deg_to_byte(deg)
                log.info(">>> STEER  %.1f deg  → byte %d  (0=full-left 127=centre 255=full-right)",
                         deg, val)
                show_frame(CMD_STEER, val)
                force_send(uart, CMD_STEER, val)

            elif cmd_char == "b":
                if len(parts) < 2:
                    print("  usage: b <0-255>  e.g.  b 255")
                    continue
                try:
                    val = int(parts[1])
                except ValueError:
                    print("  invalid value")
                    continue
                val = max(0, min(255, val))
                log.info(">>> BRAKE  val=%d", val)
                show_frame(CMD_BRAKE, val)
                force_send(uart, CMD_BRAKE, val)

            else:
                print("  unknown command — use t / s / b / i / q")

    finally:
        log.info("Sending IDLE before exit...")
        force_send(uart, CMD_IDLE, 0)
        time.sleep(0.05)
        uart.close()
        log.info("UART closed. Bye.")


if __name__ == "__main__":
    main()
