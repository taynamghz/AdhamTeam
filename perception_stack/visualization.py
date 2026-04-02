"""
PSU Eco Racing — Perception Stack
visualization.py  |  OpenCV debug overlay for the perception pipeline.
"""

import numpy as np
import cv2

from perception_stack.config import ROI_TOP_FRACTION, LANE_ENABLED
from perception_stack.models import PerceptionResult
from perception_stack.lane.fitting import eval_x

_STATE_COLOUR = {
    "THROTTLE":  (0, 220, 0),
    "BRAKE":     (0, 60, 255),
    "IDLE":      (120, 120, 120),
}


def draw(frame: np.ndarray, result: PerceptionResult,
         fm, wm, gm, H: int, W: int,
         cmd_state: str = "IDLE", fps: float = 0.0) -> np.ndarray:
    vis = frame.copy()

    if LANE_ENABLED:
        # ── Floor mask contour ────────────────────────────────────────────────
        if fm is not None:
            cnts, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)

        # ── White / grass mask overlay ────────────────────────────────────────
        if wm is not None:
            vis[wm == 255] = [255, 255, 255]
        if gm is not None:
            vis[gm == 255] = [0, 180, 0]
        cv2.addWeighted(vis, 0.45, frame, 0.55, 0, vis)

        # ── Lane polynomial overlays ──────────────────────────────────────────
        lf, rf = result.left_fit, result.right_fit
        if lf is not None or rf is not None:
            ys  = np.arange(int(H * ROI_TOP_FRACTION), H, 4)
            ovl = np.zeros_like(vis)
            if lf is not None and rf is not None:
                lxs = np.clip(np.polyval(lf, ys).astype(int), 0, W - 1)
                rxs = np.clip(np.polyval(rf, ys).astype(int), 0, W - 1)
                pts_l = np.stack([lxs, ys], axis=1)
                pts_r = np.stack([rxs, ys], axis=1)[::-1]
                cv2.fillPoly(ovl, [np.vstack([pts_l, pts_r])], (0, 55, 0))
                mxs = (lxs + rxs) // 2
                for i in range(len(ys) - 1):
                    cv2.line(ovl, (mxs[i], ys[i]), (mxs[i + 1], ys[i + 1]), (0, 255, 200), 2)
            l_col = (180, 60, 255) if result.virtual_left  else (255, 80, 0)
            r_col = (180, 60, 255) if result.virtual_right else (0, 80, 255)
            if lf is not None:
                lxs = np.clip(np.polyval(lf, ys).astype(int), 0, W - 1)
                for i in range(len(ys) - 1):
                    cv2.line(ovl, (lxs[i], ys[i]), (lxs[i + 1], ys[i + 1]), l_col, 3)
            if rf is not None:
                rxs = np.clip(np.polyval(rf, ys).astype(int), 0, W - 1)
                for i in range(len(ys) - 1):
                    cv2.line(ovl, (rxs[i], ys[i]), (rxs[i + 1], ys[i + 1]), r_col, 3)
            cv2.addWeighted(vis, 1.0, ovl, 0.55, 0, vis)

        # ── Lane width arrow ──────────────────────────────────────────────────
        lf, rf = result.left_fit, result.right_fit
        if result.lane_width_m > 0.1 and lf is not None and rf is not None:
            ay = int(H * 0.78)
            lx = int(np.clip(eval_x(lf, ay), 0, W - 1))
            rx = int(np.clip(eval_x(rf, ay), 0, W - 1))
            cv2.arrowedLine(vis, (lx, ay), (rx, ay), (0, 255, 255), 2, tipLength=0.03)
            cv2.arrowedLine(vis, (rx, ay), (lx, ay), (0, 255, 255), 2, tipLength=0.03)
            cv2.putText(vis, f"{result.lane_width_m:.2f}m", ((lx + rx) // 2 - 35, ay - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ── Deviation bar ─────────────────────────────────────────────────────
        thresh = max(result.lane_width_m * 0.08, 0.05)
        bw, by = 500, H - 25
        bx = (W - bw) // 2
        cv2.rectangle(vis, (bx, by - 15), (bx + bw, by + 15), (40, 40, 40), -1)
        mid = bx + bw // 2
        cv2.line(vis, (mid, by - 15), (mid, by + 15), (255, 255, 255), 2)
        half = max(result.lane_width_m / 2.0, 0.3)
        dn   = np.clip(result.deviation_m / half, -1.0, 1.0)
        ind  = int(mid + dn * (bw // 2))
        col  = (0, 255, 0) if abs(result.deviation_m) < thresh else (0, 165, 255)
        cv2.circle(vis, (ind, by), 14, col, -1)
        cv2.putText(vis, "L", (bx - 20, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, "R", (bx + bw + 5, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # ── Mini mask panels (right side) ─────────────────────────────────────
        PW, PH, GAP = 200, 112, 8
        px = W - PW - 6
        for i, (lbl, mask) in enumerate([("FLOOR MASK", fm), ("WHITE LINES", wm), ("GRASS", gm)]):
            py = GAP + i * (PH + GAP)
            thumb_bgr = (cv2.cvtColor(cv2.resize(mask, (PW, PH)), cv2.COLOR_GRAY2BGR)
                         if mask is not None else np.zeros((PH, PW, 3), dtype=np.uint8))
            vis[py:py + PH, px:px + PW] = thumb_bgr
            cv2.rectangle(vis, (px, py), (px + PW, py + PH), (80, 80, 80), 1)
            cv2.putText(vis, lbl, (px + 4, py + PH - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        panel_right = W - PW - 6
    else:
        panel_right = W

    # ── Stop line (shown in both modes) ───────────────────────────────────────
    if result.stop_line and result.stop_line_y is not None:
        sy = result.stop_line_y
        cv2.line(vis, (0, sy), (W, sy), (0, 0, 255), 4)
        dist_str = f" {result.stop_line_dist:.1f}m" if result.stop_line_dist > 0 else ""
        label = f"STOP LINE{dist_str}"
        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = W // 2 - lsz[0] // 2
        cv2.rectangle(vis, (tx - 5, sy - lsz[1] - 14), (tx + lsz[0] + 5, sy - 2), (0, 0, 160), -1)
        cv2.putText(vis, label, (tx, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ── Stop sign bbox (shown in both modes) ──────────────────────────────────
    if result.stop_sign_bbox is not None:
        sx, sy, sw, sh = result.stop_sign_bbox
        box_col = (0, 0, 255) if result.stop_sign else (0, 165, 255)
        cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), box_col, 3)
        dist_str = f" {result.stop_sign_dist_m:.1f}m" if result.stop_sign_dist_m > 0 else ""
        label = f"STOP SIGN{dist_str}"
        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = sx + sw // 2 - lsz[0] // 2
        ty = max(sy - 10, lsz[1] + 4)
        cv2.rectangle(vis, (tx - 4, ty - lsz[1] - 4), (tx + lsz[0] + 4, ty + 4), (0, 0, 160), -1)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ── HUD ───────────────────────────────────────────────────────────────────
    HUD_H = 90
    cv2.rectangle(vis, (0, 0), (panel_right, HUD_H), (0, 0, 0), -1)

    state_col = _STATE_COLOUR.get(cmd_state, (180, 180, 180))
    fps_col   = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 60, 255)
    fps_str   = f"{fps:.1f} FPS"
    fps_sz    = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]

    if LANE_ENABLED:
        thresh = max(result.lane_width_m * 0.08, 0.05)
        cs = ("CENTER" if abs(result.deviation_m) < thresh
              else "LEFT" if result.deviation_m > 0 else "RIGHT")
        sc = {"WHITE_LINE": (0, 255, 0), "GRASS": (0, 200, 0),
              "LOST": (0, 0, 255)}.get(result.source, (180, 180, 180))
        cv2.putText(vis,
            f"Lane: {result.lane_width_m:.2f}m  Dev: {result.deviation_m:+.3f}m  {cs}  "
            f"Src: {result.source}  Conf: {result.confidence:.0%}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
        sign_info = (f"STOP SIGN: {result.stop_sign_dist_m:.2f}m  [CONFIRMED]"
                     if result.stop_sign else "Stop Sign: searching...")
        cv2.putText(vis, sign_info, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255) if result.stop_sign else (100, 100, 100), 2)
    else:
        # Sign-only mode — big clear readout
        sign_info = (f"STOP SIGN DETECTED  {result.stop_sign_dist_m:.2f} m"
                     if result.stop_sign else "Stop Sign: searching...")
        sign_col  = (0, 0, 255) if result.stop_sign else (100, 100, 100)
        cv2.putText(vis, sign_info, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, sign_col, 2)
        cv2.putText(vis, f"State: {cmd_state}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_col, 2)

    cv2.putText(vis, fps_str, (panel_right - fps_sz[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, fps_col, 2)

    return vis
