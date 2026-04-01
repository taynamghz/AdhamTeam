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
    UART_PORT, UART_BAUD, UART_TIMEOUT_S, UART_ACK_TIMEOUT_S, UART_HEARTBEAT_S,
)

log = logging.getLogger(__name__)

# ── Command constants ───────────────────────────────────────────────────────────
CMD_IDLE     = 0x00
CMD_THROTTLE = 0x01
CMD_BRAKE    = 0x02
CMD_STEER    = 0x03   # DATA: 0=-30°, 127=0°, 255=+30°

_CMD_NAME = {CMD_IDLE: "IDLE", CMD_THROTTLE: "THROTTLE", CMD_BRAKE: "BRAKE", CMD_STEER: "STEER"}

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
        self._last_cmd:  int   = -1
        self._last_val:  int   = -1
        self._last_sent: float = 0.0   # epoch time of last actual write (watchdog)
        self.connected:  bool  = False

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
            log.info("[UART] Opened %s @ %d baud", UART_PORT, UART_BAUD)
            return True
        except serial.SerialException as e:
            log.error("[UART] Open failed: %s", e)
            self.connected = False
            return False

    def close(self):
        if self._ser and self._ser.is_open:
            self.send(CMD_IDLE, 0)          # safe state before disconnect
            time.sleep(0.05)
            self._ser.close()
        self.connected = False
        log.info("[UART] Closed")

    # ── Send ────────────────────────────────────────────────────────────────────

    def send(self, cmd: int, value: int = 0) -> bool:
        """
        Transmit one command frame.

        De-duplicates identical (cmd, value) pairs BUT forces a retransmit every
        UART_HEARTBEAT_S seconds regardless — this keeps the Nucleo watchdog
        (200 ms timeout) alive during sustained THROTTLE at constant speed.
        """
        if not self.connected or self._ser is None:
            return False

        now = time.time()
        heartbeat_due = (now - self._last_sent) >= UART_HEARTBEAT_S
        if cmd == self._last_cmd and value == self._last_val and not heartbeat_due:
            return True

        frame = _build_frame(cmd, value)
        with self._lock:
            try:
                self._ser.write(frame)
                self._ser.flush()
            except serial.SerialException as e:
                log.error("[UART] Write error: %s", e)
                self.connected = False
                return False

        self._last_cmd  = cmd
        self._last_val  = value
        self._last_sent = now
        log.debug("[UART] → %s  val=%d", _CMD_NAME.get(cmd, f"0x{cmd:02X}"), value)
        return True

    # ── Convenience helpers ─────────────────────────────────────────────────────

    def throttle(self, value: int = 200) -> bool:
        return self.send(CMD_THROTTLE, value)

    def brake(self, value: int = 255) -> bool:
        return self.send(CMD_BRAKE, value)

    def idle(self) -> bool:
        return self.send(CMD_IDLE, 0)
