"""
PSU Eco Racing — Perception Stack
perception/pipeline.py  |  LanePerception — main camera + processing pipeline.

Owns the ZED camera, floor plane estimation, colour masking, lane fitting,
virtual boundary projection, stop-line and obstacle detection.
Call init() once, then process() every frame.
"""

import time
import numpy as np
import cv2
import pyzed.sl as sl
from typing import Optional, Tuple

from perception_stack.config import (
    CAM_RES, CAM_FPS,
    FLOOR_TOLERANCE, FLOOR_TOLERANCE_WIDE, FLOOR_STABLE_HZ, FLOOR_LOST_CONSEC,
    ROI_TOP_FRACTION, FLOOR_HIT_POINTS,
    WHITE_L_MIN, WHITE_S_MAX,
    GRASS_H_MIN, GRASS_H_MAX, GRASS_S_MIN, GRASS_V_MIN,
    CONF_WHITE, CONF_GRASS,
    STOP_VOTE_NEEDED,
    OBS_MIN_DIST_M, OBS_MAX_DIST_M, OBS_SDK_ENABLE,
    WARP_ENABLED,
    SIGN_VOTE_NEEDED,
)
from perception_stack.models import PerceptionResult
from perception_stack.lane.fitting import fit_lanes
from perception_stack.lane.smoother import Smoother
from perception_stack.lane.memory import LaneMemory
from perception_stack.lane.deviation import compute_deviation
from perception_stack.detection.stop_line import detect_stop_line
from perception_stack.detection.obstacle import detect_obstacle_pc
from perception_stack.detection.stop_sign import detect_stop_sign
from perception_stack.perception.warp import WarpTransform


class LanePerception:

    def __init__(self):
        self.cam       = sl.Camera()
        self.floor_y   = None
        self.frame_cnt = 0
        self.smoother  = Smoother()
        self.warp      = WarpTransform()

        # Smoothed image-space fits used as priors
        self._prev_lf: Optional[np.ndarray] = None
        self._prev_rf: Optional[np.ndarray] = None
        # BEV-space raw fits used as priors for BEV search (WARP_ENABLED=True)
        self._bev_lf:  Optional[np.ndarray] = None
        self._bev_rf:  Optional[np.ndarray] = None

        self.image_mat = sl.Mat()
        self.pc_mat    = sl.Mat()
        self.plane     = sl.Plane()
        self.reset_tf  = sl.Transform()
        self.pose      = sl.Pose()
        self.runtime   = sl.RuntimeParameters()
        self.objects   = sl.Objects()
        self._sdk_obj_enabled: bool = False

        self.cal = None
        self.W = self.H = None

        # Floor tracking state
        self._floor_miss_count: int = 0

        # Hold-last deviation for LOST state
        self._last_deviation: float = 0.0

        # Lane memory
        self.lane_mem = LaneMemory()

        # Stop-line temporal vote gate
        self._stop_votes:     int           = 0
        self._last_stop_dist: float         = 0.0
        self._last_stop_y:    Optional[int] = None

        # Stop-sign temporal vote gate
        self._sign_votes:     int                        = 0
        self._last_sign_dist: float                      = 0.0
        self._last_sign_bbox: Optional[tuple]            = None

    # ── Init ───────────────────────────────────────────────────────────────────

    def init(self) -> bool:
        print("[Perception] Rebooting camera...")
        sl.Camera.reboot(0)
        time.sleep(3)

        init = sl.InitParameters()
        init.camera_resolution = CAM_RES
        init.camera_fps        = CAM_FPS
        init.depth_mode        = sl.DEPTH_MODE.NEURAL
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

        if OBS_SDK_ENABLE:
            od_params = sl.ObjectDetectionParameters()
            od_params.detection_model     = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
            od_params.enable_tracking     = True
            od_params.enable_segmentation = False
            if self.cam.enable_object_detection(od_params) == sl.ERROR_CODE.SUCCESS:
                self._sdk_obj_enabled = True
                print("[Perception] Object detection: ON")
            else:
                print("[Perception] Object detection: FAILED (PC-only fallback)")

        print(f"[Perception] OK  {self.W}×{self.H} @ {CAM_FPS} fps  "
              f"BEV={'on' if WARP_ENABLED else 'off'}")
        return True

    # ── Floor ──────────────────────────────────────────────────────────────────

    def _update_floor(self) -> bool:
        """
        Continuous, resilient floor plane estimation.

        STABLE mode  — re-run find_floor_plane every FLOOR_STABLE_HZ frames.
        LOST mode    — retry every frame (primary + 6 hit-test fallback points).

        Never blocks the pipeline: if all methods fail and a cached floor_y
        exists, it is kept and the wider mask tolerance is used.
        """
        in_lost_mode = self._floor_miss_count >= FLOOR_LOST_CONSEC
        on_schedule  = (self.frame_cnt % FLOOR_STABLE_HZ == 0)

        if not in_lost_mode and self.floor_y is not None and not on_schedule:
            return True

        # Primary: ZED floor plane fit
        if self.cam.find_floor_plane(self.plane, self.reset_tf) == \
                sl.ERROR_CODE.SUCCESS:
            eq = self.plane.get_plane_equation()
            if abs(eq[1]) > 0.5:
                ny = -eq[3] / eq[1]
                # Snap immediately when recovering — stale value prolongs errors
                if in_lost_mode or self.floor_y is None:
                    self.floor_y = ny
                else:
                    self.floor_y = 0.80 * self.floor_y + 0.20 * ny
                self._floor_miss_count = 0
                return True

        # Fallback: hit-test at multiple road points
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

    # ── Main processing loop ───────────────────────────────────────────────────

    def process(self):
        if self.cam.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None

        self.frame_cnt += 1
        self.cam.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        frame = self.image_mat.get_data()[:, :, :3].copy()

        self.cam.retrieve_measure(self.pc_mat, sl.MEASURE.XYZ, sl.MEM.CPU)
        pc = self.pc_mat.get_data()[:, :, :3]

        self.cam.get_position(self.pose)

        if not self._update_floor():
            return PerceptionResult(source="NO_FLOOR"), frame, None, None, None

        fm               = self._floor_mask(pc)
        wm, gm, hls, hsv = self._color_masks(frame, fm)

        # Level 1: white lines (primary)
        wl, wr, wlc, wrc = self._fit_mask(wm)

        # Level 2: grass edges (fallback when white is weak)
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

        if raw_l is not None: self._prev_lf = raw_l
        if raw_r is not None: self._prev_rf = raw_r

        lf, rf = self.smoother.update(raw_l, raw_r)

        # Virtual boundaries from lane memory
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

        # Source label
        if wlc > CONF_WHITE or wrc > CONF_WHITE:
            source = "WHITE_LINE"
        elif glc > CONF_GRASS or grc > CONF_GRASS:
            source = "GRASS"
        else:
            source = "LOST"

        dev = wid = 0.0
        if lf is not None or rf is not None:
            dev, wid = compute_deviation(lf, rf, pc, self.H, self.W)

        if source != "LOST":
            self._last_deviation = dev
        else:
            dev = self._last_deviation

        # Stop line
        raw_stop, raw_y, raw_dist = detect_stop_line(
            frame, fm, lf, rf, pc, self.H, self.W, hls, hsv)

        MAX_VOTES = STOP_VOTE_NEEDED + 5
        self._stop_votes = (min(MAX_VOTES, self._stop_votes + 1) if raw_stop
                            else max(0, self._stop_votes - 1))
        if raw_stop and raw_dist > 0:
            self._last_stop_dist = raw_dist
        if raw_stop and raw_y is not None:
            self._last_stop_y = raw_y

        stop_confirmed = self._stop_votes >= STOP_VOTE_NEEDED
        out_y    = self._last_stop_y    if stop_confirmed else None
        out_dist = self._last_stop_dist if stop_confirmed else 0.0

        # Obstacle detection — point cloud
        obs_det, obs_dist, obs_lat, obs_bbox = detect_obstacle_pc(
            pc, self.floor_y, lf, rf, self.H, self.W, self.cal.fx)

        # ZED SDK object detection (people / vehicles)
        if self._sdk_obj_enabled:
            od_rt = sl.ObjectDetectionRuntimeParameters()
            od_rt.detection_confidence_threshold = 40
            self.cam.retrieve_objects(self.objects, od_rt)
            for obj in self.objects.object_list:
                if not obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    continue
                pos = obj.position
                if not (np.isfinite(pos[0]) and np.isfinite(pos[2])):
                    continue
                sdk_dist = abs(float(pos[2]))
                if not (OBS_MIN_DIST_M <= sdk_dist <= OBS_MAX_DIST_M):
                    continue
                if not obs_det or sdk_dist < obs_dist:
                    obs_det  = True
                    obs_dist = sdk_dist
                    obs_lat  = float(pos[0])
                    bb2d = obj.bounding_box_2d
                    if bb2d is not None and len(bb2d) == 4:
                        pts = np.array(bb2d, dtype=int)
                        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
                        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
                        obs_bbox = (int(x0), int(y0), int(x1-x0), int(y1-y0))

        # Stop-sign detection
        raw_sign, raw_sign_dist, raw_sign_bbox = detect_stop_sign(
            frame, pc, self.H, self.W)

        MAX_SIGN_VOTES = SIGN_VOTE_NEEDED + 5
        self._sign_votes = (min(MAX_SIGN_VOTES, self._sign_votes + 1) if raw_sign
                            else max(0, self._sign_votes - 1))
        if raw_sign and raw_sign_dist > 0:
            self._last_sign_dist = raw_sign_dist
        if raw_sign and raw_sign_bbox is not None:
            self._last_sign_bbox = raw_sign_bbox

        sign_confirmed = self._sign_votes >= SIGN_VOTE_NEEDED
        out_sign_dist  = self._last_sign_dist if sign_confirmed else 0.0
        out_sign_bbox  = self._last_sign_bbox if sign_confirmed else None

        return PerceptionResult(
            deviation_m        = dev,
            confidence         = min(0.99, (lc + rc) / 2.0),
            lane_width_m       = wid,
            source             = source,
            left_fit           = lf,
            right_fit          = rf,
            left_conf          = min(0.99, lc),
            right_conf         = min(0.99, rc),
            stop_line          = stop_confirmed,
            stop_line_y        = out_y,
            stop_line_dist     = out_dist,
            virtual_left       = virt_left,
            virtual_right      = virt_right,
            obstacle_detected  = obs_det,
            obstacle_dist_m    = obs_dist,
            obstacle_lateral_m = obs_lat,
            obstacle_bbox      = obs_bbox,
            stop_sign          = sign_confirmed,
            stop_sign_dist_m   = out_sign_dist,
            stop_sign_bbox     = out_sign_bbox,
        ), frame, fm, wm, gm

    def close(self):
        if self._sdk_obj_enabled:
            self.cam.disable_object_detection()
        self.cam.disable_positional_tracking()
        self.cam.close()
