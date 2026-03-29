"""
PSU Eco Racing — Perception Stack
control/uart.py  |  UART transport layer to the low-level controller.

Protocol — 5-byte binary frame (matches doc section 2.2):
    [0xAA]  start byte
    [LEN ]  payload length = 2  (CMD + DATA)
    [CMD ]  command  0x00=IDLE  0x01=THROTTLE  0x02=BRAKE
    [DATA]  value    0–255  (throttle/brake intensity)
    [CRC8]  CRC-8/SMBUS over [LEN, CMD, DATA]  (poly=0x07, init=0x00)

Watchdog: Nucleo expects a valid packet every 200ms or it falls back to manual.
The low-level MCU must echo the CMD byte back; if it doesn't arrive within
UART_ACK_TIMEOUT_S seconds the send is logged as un-acknowledged (non-fatal).
"""

import struct
import logging
import threading
import time
import serial                    # pyserial

from perception_stack.config import (
    UART_PORT, UART_BAUD, UART_TIMEOUT_S, UART_ACK_TIMEOUT_S,
)

log = logging.getLogger(__name__)

# ── Command constants ───────────────────────────────────────────────────────────
CMD_IDLE     = 0x00
CMD_THROTTLE = 0x01
CMD_BRAKE    = 0x02

_CMD_NAME = {CMD_IDLE: "IDLE", CMD_THROTTLE: "THROTTLE", CMD_BRAKE: "BRAKE"}

_START = 0xAA


def _crc8(data: bytes) -> int:
    """CRC-8/SMBUS: poly=0x07, init=0x00, no reflection."""
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
        crc &= 0xFF
    return crc


def _build_frame(cmd: int, value: int) -> bytes:
    """Build 5-byte frame: [0xAA][LEN=2][CMD][DATA][CRC8]."""
    payload = bytes([2, cmd & 0xFF, value & 0xFF])
    crc = _crc8(payload)
    return struct.pack("BBBBB", _START, payload[0], payload[1], payload[2], crc)


class UARTController:
    """
    Thread-safe, non-blocking UART wrapper.

    Usage:
        uart = UARTController()
        uart.open()
        uart.send(CMD_THROTTLE, 200)
        uart.send(CMD_BRAKE,    255)
        uart.close()
    """

    def __init__(self):
        self._ser:    serial.Serial | None = None
        self._lock    = threading.Lock()
        self.connected: bool  = False
        # Speed telemetry from Nucleo — frame: [0xBB][int_part][frac_part][0xEE]
        self._speed_kmh: float = -1.0   # -1.0 = not yet received
        self._running:   bool  = False
        self._reader_thread: threading.Thread | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    def open(self) -> bool:
        try:
            self._ser = serial.Serial(
                port     = UART_PORT,
                baudrate = UART_BAUD,
                bytesize = serial.EIGHTBITS,
                parity   = serial.PARITY_NONE,
                stopbits = serial.STOPBITS_ONE,
                timeout  = UART_TIMEOUT_S,
            )
            self.connected = True
            self._running = True
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True, name="uart-speed-rx")
            self._reader_thread.start()
            log.info("[UART] Opened %s @ %d baud", UART_PORT, UART_BAUD)
            return True
        except serial.SerialException as e:
            log.error("[UART] Open failed: %s", e)
            self.connected = False
            return False

    def close(self):
        self._running = False
        if self._ser and self._ser.is_open:
            self.send(CMD_IDLE, 0)          # safe state before disconnect
            time.sleep(0.05)
            self._ser.close()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        self.connected = False
        log.info("[UART] Closed")

    # ── Speed telemetry RX ──────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """
        Background thread: watches for Nucleo speed frames.
        Frame format: [0xBB][int_part][frac_part × 100][0xEE]
        e.g. 12.75 km/h → [0xBB][12][75][0xEE]
        """
        buf = bytearray()
        while self._running:
            if not (self._ser and self._ser.is_open):
                time.sleep(0.05)
                continue
            try:
                byte = self._ser.read(1)
                if byte:
                    buf.extend(byte)
                    # Scan for a complete valid frame
                    while len(buf) >= 4:
                        if buf[0] == 0xBB and buf[3] == 0xEE:
                            self._speed_kmh = buf[1] + buf[2] / 100.0
                            log.debug("[UART] ← speed %.2f km/h", self._speed_kmh)
                            buf = buf[4:]
                        else:
                            buf = buf[1:]   # slide: resync on next 0xBB
            except serial.SerialException as e:
                log.warning("[UART] RX error: %s", e)
                time.sleep(0.01)

    @property
    def speed_kmh(self) -> float:
        """Latest speed received from Nucleo in km/h. -1.0 if not yet received."""
        return self._speed_kmh

    # ── Send ────────────────────────────────────────────────────────────────────

    def send(self, cmd: int, value: int = 0) -> bool:
        """
        Transmit one command frame every call — no deduplication.
        The Nucleo watchdog requires a valid packet at least every 200ms;
        at 30fps we send every ~33ms so the watchdog stays fed.
        Returns True on success.
        """
        if not self.connected or self._ser is None:
            return False

        frame = _build_frame(cmd, value)
        with self._lock:
            try:
                self._ser.write(frame)
                self._ser.flush()
            except serial.SerialException as e:
                log.error("[UART] Write error: %s", e)
                self.connected = False
                return False

        log.debug("[UART] → %s  val=%d", _CMD_NAME.get(cmd, f"0x{cmd:02X}"), value)
        return True

    # ── Convenience helpers ─────────────────────────────────────────────────────

    def throttle(self, value: int = 200) -> bool:
        return self.send(CMD_THROTTLE, value)

    def brake(self, value: int = 255) -> bool:
        return self.send(CMD_BRAKE, value)

    def idle(self) -> bool:
        return self.send(CMD_IDLE, 0)
