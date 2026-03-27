"""
PSU Eco Racing — Perception Stack
main.py  |  Entry point. Run with:  python -m perception_stack.main
"""

import cv2
from perception_stack.config import DISPLAY
from perception_stack.perception.pipeline import LanePerception
from perception_stack.visualization import draw


def main():
    perc = LanePerception()
    if not perc.init():
        exit(1)

    print("\nPerception v4 — press Q to quit\n")
    print(f"{'Frame':>6} | {'Source':>12} | {'Dev(m)':>8} | "
          f"{'Width':>7} | {'Conf':>5} | {'Status':>7} | Stop       | Obstacle")
    print("-" * 95)

    fc = 0
    while True:
        out = perc.process()
        if out is None:
            continue

        result, frame, fm, wm, gm = out
        fc += 1

        if DISPLAY:
            vis = draw(frame, result, fm, wm, gm, perc.H, perc.W)
            cv2.imshow("PSU Eco Racing — Perception v4", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if fc % 30 == 0:
            cs = ("CENTER" if abs(result.deviation_m) < 0.1
                  else "LEFT"   if result.deviation_m > 0 else "RIGHT")
            stop_str = (f"STOP@{result.stop_line_dist:.1f}m"
                        if result.stop_line else "-")
            obs_str  = (f"OBS@{result.obstacle_dist_m:.1f}m "
                        f"lat{result.obstacle_lateral_m:+.2f}m"
                        if result.obstacle_detected else "-")
            print(f"{fc:>6} | {result.source:>12} | "
                  f"{result.deviation_m:>+8.3f} | "
                  f"{result.lane_width_m:>7.2f} | "
                  f"{result.confidence:>5.0%} | "
                  f"{cs:>7} | {stop_str:<10} | {obs_str}")

    perc.close()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
