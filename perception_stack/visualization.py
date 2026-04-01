"""
PSU Eco Racing — Perception Stack
visualization.py  |  OpenCV debug overlay for the perception pipeline.
"""

import numpy as np
import cv2

from perception_stack.config import ROI_TOP_FRACTION
from perception_stack.models import PerceptionResult
from perception_stack.lane.fitting import eval_x


def draw(frame: np.ndarray, result: PerceptionResult,
         fm, wm, gm, H: int, W: int) -> np.ndarray:
    vis = frame.copy()

    if wm is not None:
        vis[wm == 255] = [255, 255, 255]
    if gm is not None:
        vis[gm == 255] = [0, 180, 0]
    cv2.addWeighted(vis, 0.45, frame, 0.55, 0, vis)

    if fm is not None:
        cnts, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)

    lf, rf = result.left_fit, result.right_fit
    if lf is not None or rf is not None:
        ys  = np.arange(int(H * ROI_TOP_FRACTION), H, 4)
        ovl = np.zeros_like(vis)

        if lf is not None and rf is not None:
            lxs = np.clip(np.polyval(lf, ys).astype(int), 0, W-1)
            rxs = np.clip(np.polyval(rf, ys).astype(int), 0, W-1)
            pts_l = np.stack([lxs, ys], axis=1)
            pts_r = np.stack([rxs, ys], axis=1)[::-1]
            cv2.fillPoly(ovl, [np.vstack([pts_l, pts_r])], (0, 55, 0))
            mxs = (lxs + rxs) // 2
            for i in range(len(ys) - 1):
                cv2.line(ovl, (mxs[i], ys[i]), (mxs[i+1], ys[i+1]),
                         (0, 255, 200), 2)

        l_col = (180, 60, 255) if result.virtual_left  else (255, 80, 0)
        r_col = (180, 60, 255) if result.virtual_right else (0, 80, 255)
        if lf is not None:
            lxs = np.clip(np.polyval(lf, ys).astype(int), 0, W-1)
            for i in range(len(ys) - 1):
                cv2.line(ovl, (lxs[i], ys[i]), (lxs[i+1], ys[i+1]), l_col, 3)
        if rf is not None:
            rxs = np.clip(np.polyval(rf, ys).astype(int), 0, W-1)
            for i in range(len(ys) - 1):
                cv2.line(ovl, (rxs[i], ys[i]), (rxs[i+1], ys[i+1]), r_col, 3)

        cv2.addWeighted(vis, 1.0, ovl, 0.55, 0, vis)

    # Lane width arrow
    if result.lane_width_m > 0.1 and lf is not None and rf is not None:
        ay = int(H * 0.78)
        lx = int(np.clip(eval_x(lf, ay), 0, W-1))
        rx = int(np.clip(eval_x(rf, ay), 0, W-1))
        cv2.arrowedLine(vis, (lx, ay), (rx, ay), (0,255,255), 2, tipLength=0.03)
        cv2.arrowedLine(vis, (rx, ay), (lx, ay), (0,255,255), 2, tipLength=0.03)
        cv2.putText(vis, f"{result.lane_width_m:.2f}m",
                    ((lx+rx)//2-35, ay-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # Deviation bar
    bw, by = 500, H - 25
    bx = (W - bw) // 2
    cv2.rectangle(vis, (bx, by-15), (bx+bw, by+15), (40,40,40), -1)
    mid = bx + bw // 2
    cv2.line(vis, (mid, by-15), (mid, by+15), (255,255,255), 2)
    half   = max(result.lane_width_m / 2.0, 0.3)
    dn     = np.clip(result.deviation_m / half, -1.0, 1.0)
    ind    = int(mid + dn * (bw // 2))
    thresh = max(result.lane_width_m * 0.08, 0.05)
    col    = (0,255,0) if abs(result.deviation_m) < thresh else (0,165,255)
    cv2.circle(vis, (ind, by), 14, col, -1)
    cv2.putText(vis, "L", (bx-20, by+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(vis, "R", (bx+bw+5, by+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # Stop line
    if result.stop_line and result.stop_line_y is not None:
        sy = result.stop_line_y
        cv2.line(vis, (0, sy), (W, sy), (0,0,255), 3)
        if lf is not None and rf is not None:
            lx = int(np.clip(eval_x(lf, sy), 0, W-1))
            rx = int(np.clip(eval_x(rf, sy), 0, W-1))
            cv2.line(vis, (lx, sy), (rx, sy), (0,0,255), 6)
        dist_str = f" {result.stop_line_dist:.1f}m" if result.stop_line_dist > 0 else ""
        label    = f"STOP{dist_str}"
        lsz      = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
        tx       = W // 2 - lsz[0] // 2
        cv2.rectangle(vis, (tx-5, sy-lsz[1]-18), (tx+lsz[0]+5, sy-3), (0,0,180), -1)
        cv2.putText(vis, label, (tx, sy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)

    # Stop sign bounding box
    if result.stop_sign and result.stop_sign_bbox is not None:
        sx, sy, sw, sh = result.stop_sign_bbox
        cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (0, 0, 200), 3)
        dist_str = f" {result.stop_sign_dist_m:.1f}m" if result.stop_sign_dist_m > 0 else ""
        sign_label = f"STOP SIGN{dist_str}"
        lsz = cv2.getTextSize(sign_label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        tx  = sx + sw // 2 - lsz[0] // 2
        ty  = max(sy - 8, 16)
        cv2.rectangle(vis, (tx - 4, ty - lsz[1] - 6), (tx + lsz[0] + 4, ty + 2), (0, 0, 160), -1)
        cv2.putText(vis, sign_label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Lookahead point marker
    if result.lookahead_pixel is not None:
        lx_pt, ly_pt = result.lookahead_pixel
        cv2.circle(vis, (lx_pt, ly_pt), 10, (0, 255, 255), -1)
        cv2.circle(vis, (lx_pt, ly_pt), 13, (255, 255, 0),  2)
        if result.lookahead_point is not None:
            z_m = result.lookahead_point[1]
            cv2.putText(vis, f"{z_m:.1f}m", (lx_pt + 16, ly_pt + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # HUD
    sc = {"WHITE_LINE": (0,255,0), "GRASS": (0,200,0),
          "LOST": (0,0,255), "NO_FLOOR": (0,0,255)}.get(result.source, (180,180,180))
    virt_tag = ("" + (" vL" if result.virtual_left else "")
                   + (" vR" if result.virtual_right else ""))
    cs = ("CENTER" if abs(result.deviation_m) < thresh
          else "LEFT" if result.deviation_m > 0 else "RIGHT")

    cv2.rectangle(vis, (0, 0), (W, 140), (0,0,0), -1)
    cv2.putText(vis,
        f"Source:{result.source}{virt_tag}  Conf:{result.confidence:.0%}  "
        f"Width:{result.lane_width_m:.2f}m",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc, 2)
    cv2.putText(vis,
        f"Dev:{result.deviation_m:+.3f}m  {cs}  "
        f"L:{result.left_conf:.0%} R:{result.right_conf:.0%}",
        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
    stop_info = (f"STOP LINE: {result.stop_line_dist:.1f}m ahead"
                 if result.stop_line else "No stop line")
    cv2.putText(vis, stop_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,255) if result.stop_line else (80,80,80), 2)
    sign_info = (f"STOP SIGN: {result.stop_sign_dist_m:.1f}m"
                 if result.stop_sign else "No stop sign")
    cv2.putText(vis, sign_info, (10, 113), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 200) if result.stop_sign else (80, 80, 80), 2)
    # Control outputs row
    lhd_str = (f"  LA:{result.lookahead_point[1]:.1f}m"
               if result.lookahead_point is not None else "  LA:--")
    ctrl_col = (0, 200, 255)
    cv2.putText(vis,
        f"Head:{np.degrees(result.heading_angle):+.1f}deg  "
        f"Curv:{result.curvature:+.3f}m⁻¹{lhd_str}",
        (10, 157), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ctrl_col, 2)
    return vis
