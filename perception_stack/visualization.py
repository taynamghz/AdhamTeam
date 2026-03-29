"""
PSU Eco Racing — Perception Stack
visualization.py  |  OpenCV debug overlay — simple test mode.

Displays only stop-line / stop-sign overlays on the image and a clean
3-row HUD showing speed, vehicle state, and detection status.
"""

import cv2
import numpy as np

from perception_stack.models import PerceptionResult

_STATE_COLOUR = {
    "THROTTLE":  (0, 220, 0),
    "BRAKE":     (0, 60, 255),
    "IDLE":      (120, 120, 120),
}


def draw(frame: np.ndarray, result: PerceptionResult,
         cmd_state: str, speed_kmh: float, H: int, W: int) -> np.ndarray:
    vis = frame.copy()

    # ── Stop line — red horizontal stripe on road ──────────────────────────────
    if result.stop_line and result.stop_line_y is not None:
        sy = result.stop_line_y
        cv2.line(vis, (0, sy), (W, sy), (0, 0, 255), 4)
        dist_str = f" {result.stop_line_dist:.1f}m" if result.stop_line_dist > 0 else ""
        label = f"STOP LINE{dist_str}"
        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = W // 2 - lsz[0] // 2
        cv2.rectangle(vis, (tx - 5, sy - lsz[1] - 14), (tx + lsz[0] + 5, sy - 2),
                      (0, 0, 160), -1)
        cv2.putText(vis, label, (tx, sy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ── Stop sign — bounding box ───────────────────────────────────────────────
    if result.stop_sign and result.stop_sign_bbox is not None:
        sx, sy, sw, sh = result.stop_sign_bbox
        cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (0, 0, 200), 3)
        dist_str = f" {result.stop_sign_dist_m:.1f}m" if result.stop_sign_dist_m > 0 else ""
        cv2.putText(vis, f"STOP SIGN{dist_str}", (sx, max(sy - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 200), 2)

    # ── HUD (top black bar) ────────────────────────────────────────────────────
    HUD_H = 115
    cv2.rectangle(vis, (0, 0), (W, HUD_H), (0, 0, 0), -1)

    # Row 1 — Speed
    spd_str = f"{speed_kmh:.2f} km/h" if speed_kmh >= 0 else "-- km/h"
    cv2.putText(vis, f"Speed:  {spd_str}",
                (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    # Row 2 — Vehicle state
    state_display = "DISENGAGED" if cmd_state == "IDLE" else cmd_state
    state_col = _STATE_COLOUR.get(cmd_state, (180, 180, 180))
    cv2.putText(vis, f"State:  {state_display}",
                (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.95, state_col, 2)

    # Row 3 — Detection status (split left/right of frame)
    line_str = (f"Stop Line: {result.stop_line_dist:.1f}m"
                if result.stop_line else "Stop Line: --")
    sign_str = (f"Stop Sign: {result.stop_sign_dist_m:.1f}m"
                if result.stop_sign else "Stop Sign: --")
    line_col = (0, 80, 255) if result.stop_line else (70, 70, 70)
    sign_col = (0, 50, 200) if result.stop_sign else (70, 70, 70)
    cv2.putText(vis, line_str, (14, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_col, 2)
    cv2.putText(vis, sign_str, (W // 2, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, sign_col, 2)

    return vis
