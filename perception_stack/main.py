"""
PSU Eco Racing — Perception Stack
main.py  |  Entry point. Run with:  python -m perception_stack.main
"""

import cv2
from perception_stack.config import DISPLAY, SIMPLE_TEST_MODE
from perception_stack.perception.pipeline import LanePerception
from perception_stack.control.commander import Commander
from perception_stack.visualization import draw


def main():
    perc = LanePerception()
    if not perc.init():
        exit(1)

    cmd = Commander()
    if not cmd.open():
        print("[main] WARNING: UART unavailable — continuing without control output")

    print("\nPSU Eco Racing — Simple Test Mode — press Q to quit\n"
          if SIMPLE_TEST_MODE else "\nPerception v4 — press Q to quit\n")
    if SIMPLE_TEST_MODE:
        print(f"{'Frame':>6} | {'State':>11} | {'Stop Line':>14} | {'Stop Sign':>14} | Speed")
        print("-" * 70)
    else:
        print(f"{'Frame':>6} | {'Source':>12} | {'Dev(m)':>8} | "
              f"{'Width':>7} | {'Conf':>5} | {'Status':>7} | {'Cmd':>8} | "
              f"{'Head(°)':>8} | {'Curv(m⁻¹)':>10} | {'LA(m)':>6} | "
              f"Stop       | Sign       | Obstacle")
        print("-" * 155)

    fc = 0
    try:
        while True:
            out = perc.process()
            if out is None:
                continue

            result, frame, *_ = out
            fc += 1

            # ── Send command to low-level controller ──────────────────────────
            control_cmd = cmd.update(result)

            if DISPLAY:
                vis = draw(frame, result, control_cmd,
                           cmd.uart.speed_kmh, perc.H, perc.W)
                cv2.imshow("PSU Eco Racing — Perception v4", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if fc % 30 == 0:
                if SIMPLE_TEST_MODE:
                    stop_str = (f"STOP@{result.stop_line_dist:.1f}m"
                                if result.stop_line else "-")
                    sign_str = (f"SIGN@{result.stop_sign_dist_m:.1f}m"
                                if result.stop_sign else "-")
                    spd = cmd.uart.speed_kmh
                    spd_str = f"{spd:.2f} km/h" if spd >= 0 else "--"
                    print(f"{fc:>6} | {control_cmd:>11} | "
                          f"{stop_str:<14} | {sign_str:<14} | {spd_str}")
                else:
                    import math as _math
                    cs = ("CENTER" if abs(result.deviation_m) < 0.1
                          else "LEFT"   if result.deviation_m > 0 else "RIGHT")
                    stop_str = (f"STOP@{result.stop_line_dist:.1f}m"
                                if result.stop_line else "-")
                    obs_str  = (f"OBS@{result.obstacle_dist_m:.1f}m "
                                f"lat{result.obstacle_lateral_m:+.2f}m"
                                if result.obstacle_detected else "-")
                    sign_str = (f"SIGN@{result.stop_sign_dist_m:.1f}m"
                                if result.stop_sign else "-")
                    la_str   = (f"{result.lookahead_point[1]:.1f}"
                                if result.lookahead_point is not None else "--")
                    print(f"{fc:>6} | {result.source:>12} | "
                          f"{result.deviation_m:>+8.3f} | "
                          f"{result.lane_width_m:>7.2f} | "
                          f"{result.confidence:>5.0%} | "
                          f"{cs:>7} | {control_cmd:>8} | "
                          f"{_math.degrees(result.heading_angle):>+8.1f} | "
                          f"{result.curvature:>+10.4f} | "
                          f"{la_str:>6} | "
                          f"{stop_str:<10} | {sign_str:<10} | {obs_str}")

    finally:
        cmd.close()
        perc.close()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
