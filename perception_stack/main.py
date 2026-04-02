"""
PSU Eco Racing — Perception Stack
main.py  |  Entry point. Run with:  python -m perception_stack.main
"""

import json
import os
import time
import cv2
from perception_stack.config import DISPLAY, LOG_TELEMETRY, LOG_DIR
from perception_stack.perception.pipeline import LanePerception
from perception_stack.control.commander import Commander
from perception_stack.visualization import draw


class TelemetryLogger:
    """
    Writes one JSON line per frame to a .jsonl file.
    After a run, replay the file to reconstruct exactly what the system
    saw and decided — invaluable for post-run debugging.

    Enable / disable via LOG_TELEMETRY in config.py.
    Files land in LOG_DIR / run_YYYYMMDD_HHMMSS.jsonl.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"run_{ts}.jsonl")
        self._f  = open(path, 'w')
        self._t0 = time.monotonic()
        print(f"[Telemetry] Logging to {path}")

    def log(self, frame_idx: int, result, cmd: str) -> None:
        record = {
            't':         round(time.monotonic() - self._t0, 3),
            'f':         frame_idx,
            'dev':       round(result.deviation_m,    4),
            'conf':      round(result.confidence,     3),
            'src':       result.source,
            'head':      round(result.heading_angle,  4),
            'curv':      round(result.curvature,      4),
            'stop_line': result.stop_line,
            'sl_dist':   round(result.stop_line_dist, 2),
            'stop_sign': result.stop_sign,
            'ss_dist':   round(result.stop_sign_dist_m, 2),
            'obs':       result.obstacle_detected,
            'obs_dist':  round(result.obstacle_dist_m,    2),
            'obs_lat':   round(result.obstacle_lateral_m, 3),
            'park':      result.parking_detected,
            'park_ok':   result.parking_empty,
            'cmd':       cmd,
        }
        self._f.write(json.dumps(record) + '\n')

    def close(self) -> None:
        self._f.flush()
        self._f.close()
        print("[Telemetry] Log closed.")


def main():
    perc = LanePerception()
    if not perc.init():
        exit(1)

    cmd = Commander()
    if not cmd.open():
        print("[main] WARNING: UART unavailable — continuing without control output")

    logger = TelemetryLogger(LOG_DIR) if LOG_TELEMETRY else None

    print("\nPerception v4 — press Q to quit\n")
    print(f"{'Frame':>6} | {'Source':>12} | {'Dev(m)':>8} | "
          f"{'Width':>7} | {'Conf':>5} | {'Status':>7} | {'Cmd':>8} | "
          f"{'Head(°)':>8} | {'Curv(m⁻¹)':>10} | {'LA(m)':>6} | "
          f"Stop       | Sign")
    print("-" * 120)

    fc = 0
    try:
        while True:
            out = perc.process()
            if out is None:
                continue

            result, frame, fm, wm, gm = out
            fc += 1

            # ── Send command to low-level controller ──────────────────────────
            control_cmd = cmd.update(result)

            if logger:
                logger.log(fc, result, control_cmd)

            if DISPLAY:
                vis = draw(frame, result, fm, wm, gm, perc.H, perc.W)
                cv2.imshow("PSU Eco Racing — Perception v4", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if fc % 30 == 0:
                import math as _math
                cs = ("CENTER" if abs(result.deviation_m) < 0.1
                      else "LEFT"   if result.deviation_m > 0 else "RIGHT")
                stop_str = (f"STOP@{result.stop_line_dist:.1f}m"
                            if result.stop_line else "-")
                sign_str = (f"SIGN@{result.stop_sign_dist_m:.1f}m"
                            if result.stop_sign else "-")
                la_str   = (f"{result.lookahead_point[1]:.1f}"
                            if result.lookahead_point is not None else "--")
                obs_str  = (f"OBS@{result.obstacle_dist_m:.1f}m"
                            f"({'R' if result.obstacle_lateral_m > 0 else 'L'})"
                            if result.obstacle_detected else "-")
                park_str = (f"PARK@{result.parking_center_m[1]:.1f}m"
                            f"({'OK' if result.parking_empty else 'BLK'})"
                            if result.parking_detected else "-")
                print(f"{fc:>6} | {result.source:>12} | "
                      f"{result.deviation_m:>+8.3f} | "
                      f"{result.lane_width_m:>7.2f} | "
                      f"{result.confidence:>5.0%} | "
                      f"{cs:>7} | {control_cmd:>8} | "
                      f"{_math.degrees(result.heading_angle):>+8.1f} | "
                      f"{result.curvature:>+10.4f} | "
                      f"{la_str:>6} | "
                      f"{stop_str:<10} | {sign_str:<12} | "
                      f"{obs_str:<14} | {park_str}")

    finally:
        if logger:
            logger.close()
        cmd.close()
        perc.close()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
