"""
PSU Eco Racing — Perception Stack
perception/pipeline.py  |  LanePerception — main camera + processing pipeline.

Threading model:
  Main thread  — camera grab, floor estimation, colour masking, lane fitting,
                 stop-line detection, control output.  Targets 30 FPS.
  Sign thread  — YOLOv8 stop-sign inference, runs inside StopSignDetector.
                 Decoupled via a queue; never blocks the main thread.

Call init() once, then process() every frame.
"""

import collections
import time
import numpy as np
import cv2
import pyzed.sl as sl
from typing import Optional, Tuple

from perception_stack.config import (
    CAM_RES, CAM_FPS, CAM_DEPTH_MODE,
    FLOOR_TOLERANCE, FLOOR_TOLERANCE_WIDE, FLOOR_STABLE_HZ, FLOOR_LOST_CONSEC,
    FLOOR_CALIBRATE_FRAMES,
    ROI_TOP_FRACTION, FLOOR_HIT_POINTS,
    WHITE_L_MIN, WHITE_S_MAX,
    GRASS_H_MIN, GRASS_H_MAX, GRASS_S_MIN, GRASS_V_MIN,
    CONF_WHITE, CONF_GRASS,
    STOP_VOTE_NEEDED,
    WARP_ENABLED,
    CTRL_EVAL_Y_FRAC,
    MIN_LANE_SEP_PX,
    LANE_SEP_MEM_FRAC,
    LANE_SKIP_STRAIGHT, LANE_CURVE_THRESH,
    PROFILE_ENABLED, PROFILE_PRINT_EVERY,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    FPS_WARN_BELOW,
    PC_REFRESH_EVERY,
    LANE_ENABLED,
)
from perception_stack.models import PerceptionResult
from perception_stack.lane.fitting import fit_lanes, eval_x
from perception_stack.lane.smoother import Smoother
from perception_stack.lane.memory import LaneMemory
from perception_stack.lane.deviation import compute_deviation
from perception_stack.detection.stop_line import detect_stop_line
from perception_stack.detection.stop_sign import StopSignDetector
from perception_stack.perception.warp import WarpTransform
from perception_stack.lane.control import (
    compute_heading, compute_curvature, compute_lookahead, ControlSmoother,
)


class LanePerception:

    def __init__(self):
        self.cam       = sl.Camera()
        self.floor_y   = None
        self.frame_cnt = 0
        self.smoother      = Smoother()
        self.ctrl_smoother = ControlSmoother()
        self.warp          = WarpTransform()

        self._prev_lf: Optional[np.ndarray] = None
        self._prev_rf: Optional[np.ndarray] = None
        self._bev_lf:  Optional[np.ndarray] = None
        self._bev_rf:  Optional[np.ndarray] = None

        self.image_mat = sl.Mat()
        self.pc_mat    = sl.Mat()
        self.plane     = sl.Plane()
        self.reset_tf  = sl.Transform()
        self.pose      = sl.Pose()
        self.runtime   = sl.RuntimeParameters()

        self.cal = None
        self.W = self.H = None

        self._floor_miss_count:  int  = 0
        self._floor_calibrated:  bool = False   # True after startup calibration

        self._last_deviation: float = 0.0
        self._last_heading:   float = 0.0
        self._last_curvature: float = 0.0

        # Adaptive lane-fitting rate
        self._lane_skip_counter: int   = 0
        self._last_source:       str   = "LOST"
        self._last_lc:           float = 0.0
        self._last_rc:           float = 0.0

        self.lane_mem = LaneMemory()

        # Stop-line temporal vote gate
        self._stop_votes:     int           = 0
        self._last_stop_dist: float         = 0.0
        self._last_stop_y:    Optional[int] = None

        # Stop-sign detector — owns its own thread and vote gate
        self.sign_detector = StopSignDetector()

        # CLAHE for lighting normalisation (applied once per frame before colour masking)
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)

        # ZED IMU sensor data handle
        self._sensors_data = sl.SensorsData()

        # Point-cloud cache — retrieved conditionally, not every frame
        self._pc_cache: Optional[np.ndarray] = None
        self._pc_age:   int                  = PC_REFRESH_EVERY  # force fetch on frame 1

        # Rolling frame-time buffer for FPS monitoring (last 30 frame timestamps)
        self._frame_times: collections.deque = collections.deque(maxlen=30)

        # Profiling accumulators (ms, reset every PROFILE_PRINT_EVERY frames)
        self._prof: dict = {}

    # ── Init ───────────────────────────────────────────────────────────────────

    def init(self) -> bool:
        print("[Perception] Rebooting camera...")
        sl.Camera.reboot(0)
        time.sleep(3)

        init = sl.InitParameters()
        init.camera_resolution = CAM_RES
        init.camera_fps        = CAM_FPS
        init.depth_mode        = CAM_DEPTH_MODE
        init.coordinate_units  = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        if self.cam.open(init) != sl.ERROR_CODE.SUCCESS:
            print("[Perception] Camera open failed")
            return False

        tp = sl.PositionalTrackingParameters()
        tp.set_floor_as_origin = True
        self.cam.enable_positional_tracking(tp)
        self.runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

        info     = self.cam.get_camera_information()
        self.cal = info.camera_configuration.calibration_parameters.left_cam
        self.W   = info.camera_configuration.resolution.width
        self.H   = info.camera_configuration.resolution.height

        if WARP_ENABLED:
            self.warp.build(self.W, self.H)

        print(f"[Perception] OK  {self.W}×{self.H} @ {CAM_FPS} fps  "
              f"depth={CAM_DEPTH_MODE.name}  BEV={'on' if WARP_ENABLED else 'off'}")
        return True

    # ── Floor ──────────────────────────────────────────────────────────────────

    def _update_floor(self) -> bool:
        """
        Estimate floor_y from ZED plane detection.

        Runs every frame during startup (first FLOOR_CALIBRATE_FRAMES frames)
        to get a stable floor estimate, then freezes.  After calibration, only
        re-runs when consecutive floor misses reach FLOOR_LOST_CONSEC — which
        on a flat track essentially never happens.

        This saves ~8-15ms per call compared to re-running every FLOOR_STABLE_HZ
        frames throughout the run.
        """
        in_lost_mode = self._floor_miss_count >= FLOOR_LOST_CONSEC

        # During calibration phase: run every frame to build a stable estimate
        # After calibration: only run if we've lost the floor
        if self._floor_calibrated and not in_lost_mode and self.floor_y is not None:
            return True

        if self.cam.find_floor_plane(self.plane, self.reset_tf) == \
                sl.ERROR_CODE.SUCCESS:
            eq = self.plane.get_plane_equation()
            if abs(eq[1]) > 0.5:
                ny = -eq[3] / eq[1]
                if in_lost_mode or self.floor_y is None:
                    self.floor_y = ny
                else:
                    self.floor_y = 0.80 * self.floor_y + 0.20 * ny
                self._floor_miss_count = 0
                if self.frame_cnt >= FLOOR_CALIBRATE_FRAMES:
                    self._floor_calibrated = True
                return True

        for fx, fy in FLOOR_HIT_POINTS:
            coord = [int(self.W * fx), int(self.H * fy)]
            if self.cam.find_plane_at_hit(coord, self.plane) == \
                    sl.ERROR_CODE.SUCCESS:
                eq = self.plane.get_plane_equation()
                if abs(eq[1]) > 0.5:
                    ny = -eq[3] / eq[1]
                    if in_lost_mode or self.floor_y is None:
                        self.floor_y = (ny if self.floor_y is None
                                        else 0.60 * self.floor_y + 0.40 * ny)
                    else:
                        self.floor_y = 0.85 * self.floor_y + 0.15 * ny
                    self._floor_miss_count = 0
                    if self.frame_cnt >= FLOOR_CALIBRATE_FRAMES:
                        self._floor_calibrated = True
                    return True

        self._floor_miss_count += 1
        return self.floor_y is not None

    def _floor_mask(self, pc: np.ndarray) -> np.ndarray:
        uncertain = self._floor_miss_count >= FLOOR_LOST_CONSEC
        tol  = FLOOR_TOLERANCE_WIDE if uncertain else FLOOR_TOLERANCE
        Y    = pc[:, :, 1]
        mask = (np.isfinite(Y) &
                (Y > self.floor_y - tol) &
                (Y < self.floor_y + tol)).astype(np.uint8) * 255
        mask[:int(self.H * ROI_TOP_FRACTION), :] = 0
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # ── CLAHE normalisation ────────────────────────────────────────────────────

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L channel of LAB colour space.
        Returns a normalised BGR frame with equalised luminance.
        Invariant to auto-exposure changes, shadows, and sun/cloud transitions.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── ZED IMU ────────────────────────────────────────────────────────────────

    def _read_imu(self):
        """
        Pull pitch and roll from the ZED 2i IMU.
        Returns (pitch_deg, roll_deg).  Falls back to (0.0, 0.0) on any error
        so the rest of the pipeline continues unaffected if IMU is unavailable.
        """
        try:
            self.cam.get_sensors_data(self._sensors_data, sl.TIME_REFERENCE.CURRENT)
            imu    = self._sensors_data.get_imu_data()
            pose   = imu.get_pose()
            euler  = pose.get_euler_angles()   # [roll, pitch, yaw] in degrees
            return float(euler[1]), float(euler[0])   # (pitch, roll)
        except Exception:
            return 0.0, 0.0

    # ── Colour pipeline ────────────────────────────────────────────────────────

    def _color_masks(self, frame: np.ndarray, fm: np.ndarray):
        fp  = fm == 255
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        L,  S_hls        = hls[:, :, 1], hls[:, :, 2]
        Hh, S_hsv, V_hsv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        wm = np.zeros((self.H, self.W), np.uint8)
        wm[fp & (L >= WHITE_L_MIN) & (S_hls <= WHITE_S_MAX)] = 255

        gm = np.zeros((self.H, self.W), np.uint8)
        gm[fp & (Hh >= GRASS_H_MIN) & (Hh <= GRASS_H_MAX) &
           (S_hsv >= GRASS_S_MIN) & (V_hsv >= GRASS_V_MIN)] = 255

        return wm, gm, hls, hsv

    # ── Lane fitting ───────────────────────────────────────────────────────────

    def _fit_mask(
        self, mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
        if WARP_ENABLED and self.warp.M is not None:
            bev  = self.warp.warp(mask)
            lf_b, rf_b, lc, rc = fit_lanes(bev, self.W, self.H,
                                             self._bev_lf, self._bev_rf)
            self._bev_lf, self._bev_rf = lf_b, rf_b
            lf = self.warp.bev_poly_to_img(lf_b, self.H, self.W) \
                 if lf_b is not None else None
            rf = self.warp.bev_poly_to_img(rf_b, self.H, self.W) \
                 if rf_b is not None else None
            return lf, rf, lc, rc
        return fit_lanes(mask, self.W, self.H, self._prev_lf, self._prev_rf)

    # ── Profiling ──────────────────────────────────────────────────────────────

    def _tick(self, key: str, t_start: float) -> float:
        """Record elapsed ms since t_start under key; return current time."""
        now = time.perf_counter()
        self._prof[key] = self._prof.get(key, 0.0) + (now - t_start) * 1000.0
        return now

    def _print_profile(self) -> None:
        n = PROFILE_PRINT_EVERY
        total = sum(self._prof.values()) / n
        lines = [f"\n[Profile] avg over {n} frames  (total={total:.1f} ms → "
                 f"{1000.0/total:.1f} fps est.)"]
        for k, v in sorted(self._prof.items(), key=lambda x: -x[1]):
            lines.append(f"  {k:<22s} {v/n:6.1f} ms")
        print("\n".join(lines))
        self._prof = {}

    # ── Main processing loop ───────────────────────────────────────────────────

    def process(self):
        t = time.perf_counter()

        # ── Grab ──────────────────────────────────────────────────────────────
        if self.cam.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        self.frame_cnt += 1

        self.cam.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        frame = self.image_mat.get_data()[:, :, :3].copy()

        # CLAHE: helps both YOLO yellow-board gate and colour thresholds
        frame_norm = self._apply_clahe(frame)

        # ── Point-cloud retrieval — truly on demand ────────────────────────────
        # Only fetch when something actually needs distance data:
        #   • First frame (cache is empty)
        #   • Sign is actively being tracked (fresh Z = accurate braking dist)
        #   • Stop-line vote is accumulating
        #   • Obstacle vote is accumulating
        # Between those conditions, reuse the cached cloud.  At 15 km/h the
        # car moves ~0.14m per frame at 30fps — stale by ≤4 frames = ≤0.56m,
        # acceptable for vote-gated decisions.
        self._pc_age += 1
        sign_active = self.sign_detector.get_result()[0]
        need_pc = (
            self._pc_cache is None
            or sign_active
            or self._stop_votes > 0
        )
        if need_pc:
            self.cam.retrieve_measure(self.pc_mat, sl.MEASURE.XYZ, sl.MEM.CPU)
            self._pc_cache = self.pc_mat.get_data()[:, :, :3].copy()
            self._pc_age   = 0
        pc = self._pc_cache

        if PROFILE_ENABLED: t = self._tick("grab+retrieve", t)

        # ── Stop-sign detector (non-blocking submit to worker thread) ─────────
        self.sign_detector.submit(frame_norm, pc, self.H, self.W)
        sign_confirmed, sign_dist, sign_bbox = self.sign_detector.get_result()

        if PROFILE_ENABLED: t = self._tick("sign_detect", t)

        # ── Lane / floor / obstacle / parking (disabled until LANE_ENABLED) ───
        fm = wm = gm = None
        lf = rf = None
        lc = self._last_lc
        rc = self._last_rc
        dev = wid = 0.0
        virt_left = virt_right = False
        stop_confirmed = False
        out_y = None
        out_dist = 0.0
        heading_sm = self._last_heading
        curv_sm    = self._last_curvature
        lookahead_world = lookahead_px = None
        source = "DISABLED"

        if LANE_ENABLED:
            # IMU for BEV tilt compensation (no-op when WARP_ENABLED=False)
            if WARP_ENABLED:
                pitch_deg, roll_deg = self._read_imu()

            # Floor calibration: runs every frame until stable, then freezes.
            # Re-runs only on sustained floor loss (essentially never on flat track).
            if not self._update_floor():
                return PerceptionResult(source="NO_FLOOR"), frame, None, None, None

            fm = self._floor_mask(pc)
            if PROFILE_ENABLED: t = self._tick("floor", t)

            if WARP_ENABLED and self.warp.M is not None:
                self.warp.update_tilt(pitch_deg, roll_deg, self.W, self.H)

            wm, gm, hls, hsv = self._color_masks(frame_norm, fm)
            if PROFILE_ENABLED: t = self._tick("color_masks", t)

            # ── Adaptive lane fitting rate ─────────────────────────────────────
            # On straight sections, RANSAC runs every LANE_SKIP_STRAIGHT frames;
            # EMA smoother carries the fit on skipped frames.
            # On curves, run every frame for accuracy.
            on_curve = abs(self._last_curvature) >= LANE_CURVE_THRESH
            self._lane_skip_counter += 1
            run_ransac = on_curve or (self._lane_skip_counter >= LANE_SKIP_STRAIGHT)
            if run_ransac:
                self._lane_skip_counter = 0

            if run_ransac:
                wl, wr, wlc, wrc = self._fit_mask(wm)
                if max(wlc, wrc) < CONF_WHITE:
                    ge = cv2.Canny(gm, 50, 150)
                    gl, gr, glc, grc = self._fit_mask(ge)
                else:
                    gl = gr = None
                    glc = grc = 0.0

                def best(opts):
                    v = [(f, c) for f, c in opts if f is not None and c > 0.05]
                    return max(v, key=lambda x: x[1]) if v else (None, 0.0)

                raw_l, lc = best([(wl, wlc * 1.4), (gl, glc)])
                raw_r, rc = best([(wr, wrc * 1.4), (gr, grc)])
                self._last_lc, self._last_rc = lc, rc

                if raw_l is not None and raw_r is not None:
                    y_g = int(self.H * 0.85)
                    sep = eval_x(raw_r, y_g) - eval_x(raw_l, y_g)
                    too_close = sep < MIN_LANE_SEP_PX
                    if not too_close and self.lane_mem.mean_px is not None:
                        too_close = sep < LANE_SEP_MEM_FRAC * self.lane_mem.mean_px
                    if too_close:
                        lx = eval_x(raw_l, y_g)
                        rx = eval_x(raw_r, y_g)
                        if self.lane_mem.mean_px is not None:
                            r_err = abs(rx - (lx + self.lane_mem.mean_px))
                            l_err = abs(lx - (rx - self.lane_mem.mean_px))
                            if r_err >= l_err:
                                raw_r, rc = None, 0.0
                                self.smoother.r_ema = None
                            else:
                                raw_l, lc = None, 0.0
                                self.smoother.l_ema = None
                        elif lc >= rc:
                            raw_r, rc = None, 0.0
                            self.smoother.r_ema = None
                        else:
                            raw_l, lc = None, 0.0
                            self.smoother.l_ema = None

                if raw_l is not None: self._prev_lf = raw_l
                if raw_r is not None: self._prev_rf = raw_r

                if wlc > CONF_WHITE or wrc > CONF_WHITE:
                    source = "WHITE_LINE"
                elif glc > CONF_GRASS or grc > CONF_GRASS:
                    source = "GRASS"
                else:
                    source = "LOST"
                self._last_source = source

                # Pass actual new raw fits (not _prev) so smoother gets None
                # on a lost-lane frame rather than blending toward a stale fit.
                sm_l, sm_r = raw_l, raw_r
            else:
                # Skipped frame — pass None so smoother holds current EMA unchanged
                sm_l = sm_r = None
                source = self._last_source

            lf, rf = self.smoother.update(sm_l, sm_r)
            virt_left = virt_right = False
            self.lane_mem.update(lf, rf, pc, self.H, self.W)
            fx = self.cal.fx
            if lf is None and rf is not None:
                vlf = self.lane_mem.virtual_left(rf, pc, self.H, self.W, fx)
                if vlf is not None:
                    lf, virt_left = vlf, True
            if rf is None and lf is not None:
                vrf = self.lane_mem.virtual_right(lf, pc, self.H, self.W, fx)
                if vrf is not None:
                    rf, virt_right = vrf, True

            if PROFILE_ENABLED: t = self._tick("lane_fitting", t)

            if lf is not None or rf is not None:
                dev, wid = compute_deviation(lf, rf, pc, self.H, self.W)
            if source != "LOST":
                self._last_deviation = dev
            else:
                dev = self._last_deviation

            raw_stop, raw_y, raw_dist = detect_stop_line(
                frame, fm, lf, rf, pc, self.H, self.W, hls, hsv)
            MAX_VOTES = STOP_VOTE_NEEDED + 5
            self._stop_votes = (min(MAX_VOTES, self._stop_votes + 1) if raw_stop
                                else max(0, self._stop_votes - 1))
            if raw_stop and raw_dist > 0:
                self._last_stop_dist = raw_dist
            if raw_stop and raw_y is not None:
                self._last_stop_y = raw_y

            if PROFILE_ENABLED: t = self._tick("stop_line", t)

            sign_prearmed = sign_confirmed and 3.0 <= sign_dist <= 8.0
            effective_stop_thresh = max(2, STOP_VOTE_NEEDED - (2 if sign_prearmed else 0))
            stop_confirmed = self._stop_votes >= effective_stop_thresh
            out_y    = self._last_stop_y    if stop_confirmed else None
            out_dist = self._last_stop_dist if stop_confirmed else 0.0

            y_ctrl = int(self.H * CTRL_EVAL_Y_FRAC)
            wid_px = 0.0
            if lf is not None and rf is not None:
                wid_px = abs(eval_x(rf, y_ctrl) - eval_x(lf, y_ctrl))
            have_valid_lane = (lf is not None or rf is not None) and wid > 0.0
            if have_valid_lane:
                heading_raw = compute_heading(lf, rf, y_ctrl)
                curv_raw    = compute_curvature(lf, rf, y_ctrl, wid_px, wid)
                h_sm, k_sm  = self.ctrl_smoother.update(heading_raw, curv_raw)
                self._last_heading   = h_sm
                self._last_curvature = k_sm
            heading_sm = self._last_heading
            curv_sm    = self._last_curvature
            lookahead_world, lookahead_px = compute_lookahead(lf, rf, pc, self.H, self.W)

            if PROFILE_ENABLED: t = self._tick("control", t)

        # ── Profiling print ───────────────────────────────────────────────────
        if PROFILE_ENABLED and self.frame_cnt % PROFILE_PRINT_EVERY == 0:
            self._print_profile()

        # ── FPS monitoring ────────────────────────────────────────────────────
        now_t = time.perf_counter()
        self._frame_times.append(now_t)
        if len(self._frame_times) == self._frame_times.maxlen:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                fps = (len(self._frame_times) - 1) / span
                if fps < FPS_WARN_BELOW:
                    print(f"[WARNING] FPS = {fps:.1f}  (target ≥ {FPS_WARN_BELOW:.0f})")

        return PerceptionResult(
            deviation_m    = dev,
            confidence     = min(0.99, (lc + rc) / 2.0),
            lane_width_m   = wid,
            source         = source,
            left_fit       = lf,
            right_fit      = rf,
            left_conf      = min(0.99, lc),
            right_conf     = min(0.99, rc),
            stop_line      = stop_confirmed,
            stop_line_y    = out_y,
            stop_line_dist = out_dist,
            virtual_left   = virt_left,
            virtual_right  = virt_right,
            stop_sign      = sign_confirmed,
            stop_sign_dist_m = sign_dist,
            stop_sign_bbox   = sign_bbox,
            heading_angle  = heading_sm,
            curvature      = curv_sm,
            lookahead_point  = lookahead_world,
            lookahead_pixel  = lookahead_px,
        ), frame, fm, wm, gm

    def close(self):
        self.cam.disable_positional_tracking()
        self.cam.close()
