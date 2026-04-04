"""
PSU Eco Racing — Perception Stack
perception/segformer_lane.py  |  Drivable-area lane detection via Segformer-B2.

Replaces the RANSAC + colour-threshold lane detector.
Works without white lane markings — detects asphalt/grass boundaries directly.

Pipeline per frame:
  1. Run Segformer (Cityscapes) → road mask (H×W bool)
  2. Scan SEG_BOUNDARY_ROWS evenly-spaced rows → leftmost/rightmost road pixel
  3. Polyfit degree-2: x = f(y) for each boundary
  4. Convert px offset to metres using ZED point-cloud depth at eval row
  5. Return (lf, rf, road_mask, dev_m, wid_m, lc, rc, source)

Threading model
───────────────
Inference runs in a dedicated background thread so the main camera loop
is never blocked.  The main thread calls:

    seg_lane.submit(frame_bgr, pc, H, W, fx)   # non-blocking — drops stale frames
    lf, rf, ... = seg_lane.get_result()         # instant — returns latest result

The result is 1 camera-frame stale on average (≈33 ms at 30 fps, ≈14 cm at 15 km/h).
That latency is negligible for lane-following at the speeds used in SEM.

On Jetson CUDA without TRT: ~15-20 ms per inference → main thread never waits.
With TensorRT FP16 export:  ~8-10 ms per inference.
"""

import queue
import threading
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from perception_stack.config import (
    SEG_MODEL_ID,
    SEG_ROAD_CLASSES,
    SEG_ROI_TOP_FRAC,
    SEG_MIN_ROAD_FRAC,
    SEG_BOUNDARY_ROWS,
    SEG_POLY_DEG,
    SEG_CONF_THRESHOLD,
    SEG_NEAR_FRAC,
    SEG_FAR_FRAC,
)


def _best_device() -> str:
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# Default result returned before the first inference completes
_NULL_RESULT = (None, None,
                None,   # road_mask — None signals "not ready yet"
                0.0, 0.0, 0.0, 0.0, "LOST")


class SegformerLane:
    """
    Wraps Segformer-B2 (Cityscapes) for drivable-area boundary detection.

    Public async interface (used by pipeline.py):
        init()         — load model, start worker thread
        submit(...)    — non-blocking: post latest frame to worker
        get_result()   — non-blocking: read latest computed result

    The synchronous _infer() method is only called from the worker thread.
    """

    def __init__(self):
        self._model      = None
        self._processor  = None
        self._device     = _best_device()

        # EMA state on polynomial coefficients (lives in worker thread — no lock needed)
        self._smoother_l = None
        self._smoother_r = None
        self._alpha      = 0.30

        # Async infrastructure
        self._queue:  queue.Queue = queue.Queue(maxsize=1)
        self._result              = _NULL_RESULT
        self._lock                = threading.Lock()
        self._thread: threading.Thread | None = None

    # ── Initialisation ─────────────────────────────────────────────────────────

    def init(self) -> bool:
        try:
            from transformers import (SegformerForSemanticSegmentation,
                                      SegformerImageProcessor)
            print(f"[SegformerLane] Loading {SEG_MODEL_ID} on {self._device.upper()} ...")
            self._processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
            self._model     = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_ID)
            self._model     = self._model.to(self._device).eval()
            print("[SegformerLane] Ready.")
        except Exception as e:
            print(f"[SegformerLane] init failed: {e}")
            return False

        # Start background inference thread
        self._thread = threading.Thread(target=self._worker, daemon=True, name="SegformerWorker")
        self._thread.start()
        return True

    # ── Async public interface ─────────────────────────────────────────────────

    def submit(self, frame_bgr: np.ndarray, pc: np.ndarray,
               H: int, W: int, fx: float) -> None:
        """
        Post the latest frame to the worker.  Non-blocking.
        If the worker hasn't consumed the previous frame yet, that stale
        frame is silently replaced — the worker always processes the newest.
        """
        try:
            self._queue.put_nowait((frame_bgr, pc, H, W, fx))
        except queue.Full:
            # Replace stale queued item with newest frame
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((frame_bgr, pc, H, W, fx))
            except queue.Full:
                pass

    def get_result(self):
        """
        Non-blocking read of the latest computed result.
        Returns (lf, rf, road_mask, dev_m, wid_m, lc, rc, source).
        road_mask is None until the first inference completes.
        """
        with self._lock:
            return self._result

    # ── Worker thread ──────────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            frame_bgr, pc, H, W, fx = self._queue.get()   # blocks until submitted
            result = self._infer(frame_bgr, pc, H, W, fx)
            with self._lock:
                self._result = result

    # ── Road mask ──────────────────────────────────────────────────────────────

    def _road_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run Segformer; return bool mask (H,W) — True = road/drivable."""
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp  = self._processor(images=rgb, return_tensors="pt").to(self._device)
        with torch.no_grad():
            logits = self._model(**inp).logits          # (1, C, H/4, W/4)
        logits_up = F.interpolate(logits, size=(h, w),
                                  mode="bilinear", align_corners=False)
        pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()
        mask = np.zeros((h, w), dtype=bool)
        for cls in SEG_ROAD_CLASSES:
            mask |= (pred == cls)
        return mask

    # ── Boundary extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_boundary_pts(mask: np.ndarray, roi_top: int):
        h, w   = mask.shape
        rows   = np.linspace(roi_top, h - 1, SEG_BOUNDARY_ROWS, dtype=int)
        left_pts, right_pts = [], []
        for r in rows:
            cols = np.where(mask[r])[0]
            if len(cols) < int(w * SEG_MIN_ROAD_FRAC):
                continue
            left_pts.append((cols.min(), r))
            right_pts.append((cols.max(), r))
        return (np.array(left_pts)  if len(left_pts)  >= 3 else None,
                np.array(right_pts) if len(right_pts) >= 3 else None)

    # ── Polynomial fit + EMA ──────────────────────────────────────────────────

    @staticmethod
    def _fit(pts: np.ndarray):
        if pts is None or len(pts) < 3:
            return None
        return np.polyfit(pts[:, 1], pts[:, 0], deg=SEG_POLY_DEG)

    def _smooth(self, new_l, new_r):
        a = self._alpha
        if new_l is not None:
            self._smoother_l = (new_l if self._smoother_l is None
                                else a * new_l + (1 - a) * self._smoother_l)
        if new_r is not None:
            self._smoother_r = (new_r if self._smoother_r is None
                                else a * new_r + (1 - a) * self._smoother_r)
        return self._smoother_l, self._smoother_r

    # ── Pixel → metres using ZED point cloud ──────────────────────────────────

    @staticmethod
    def _px_to_m(px_val: float, y_row: int, x_col: int,
                 pc: np.ndarray, fx: float) -> float:
        if pc is None or fx <= 0:
            return px_val / 400.0
        pad  = 5
        h, w = pc.shape[:2]
        r0, r1 = max(0, y_row - pad), min(h, y_row + pad)
        c0, c1 = max(0, x_col - pad), min(w, x_col + pad)
        patch  = pc[r0:r1, c0:c1, 2]
        valid  = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 30.0)]
        Z      = float(np.median(valid)) if len(valid) > 0 else 3.0
        return px_val * Z / fx

    # ── Core inference (called only from worker thread) ────────────────────────

    def _infer(self, frame_bgr: np.ndarray, pc: np.ndarray,
               H: int, W: int, fx: float):
        """
        Run one frame of Segformer lane detection.
        Returns (lf, rf, road_mask, dev_m, wid_m, lc, rc, source).
        """
        roi_top = int(H * SEG_ROI_TOP_FRAC)
        y_near  = int(H * SEG_NEAR_FRAC)

        road_mask = self._road_mask(frame_bgr)

        left_pts, right_pts = self._extract_boundary_pts(road_mask, roi_top)
        raw_l = self._fit(left_pts)
        raw_r = self._fit(right_pts)
        lf, rf = self._smooth(raw_l, raw_r)

        max_rows = SEG_BOUNDARY_ROWS
        lc = (len(left_pts)  / max_rows if left_pts  is not None else 0.0)
        rc = (len(right_pts) / max_rows if right_pts is not None else 0.0)

        if lc < SEG_CONF_THRESHOLD: lf = self._smoother_l
        if rc < SEG_CONF_THRESHOLD: rf = self._smoother_r

        dev_m = wid_m = 0.0
        if lf is not None and rf is not None:
            lx = float(np.polyval(lf, y_near))
            rx = float(np.polyval(rf, y_near))
            cx_road  = (lx + rx) / 2.0
            cx_frame = W / 2.0
            dev_px   = cx_frame - cx_road
            wid_px   = max(rx - lx, 1.0)
            cx_col   = int(np.clip(cx_road, 0, W - 1))
            dev_m  = self._px_to_m(dev_px, y_near, cx_col, pc, fx)
            wid_m  = self._px_to_m(wid_px, y_near, cx_col, pc, fx)

        if lf is not None and rf is not None:
            source = "SEGFORMER"
        elif lf is not None or rf is not None:
            source = "SEG_PARTIAL"
        else:
            source = "LOST"

        return lf, rf, road_mask, dev_m, wid_m, lc, rc, source
