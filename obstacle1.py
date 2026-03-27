"""
PSU Eco Racing — Perception Module v4
========================================
Upgrades over v3:
  • Degree-2 polynomial (RANSAC) — handles curves
  • Sliding-window init + prior-guided fast search (no hard W//2 split)
  • Single HLS/HSV pass per frame  (~30 % less colour work)
  • White + yellow stop-line detection
  • Filled lane overlay with vectorised polyval drawing
  • Optional bird's-eye warp: fit in BEV, unproject back to image space
    → set WARP_ENABLED=True and calibrate WARP_SRC/DST for your camera mount

ZED SDK features:
  find_floor_plane(), REFERENCE_FRAME.WORLD, NEURAL depth mode
"""

import pyzed.sl as sl
import numpy as np
import cv2
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

warnings.filterwarnings('ignore', category=np.RankWarning)

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_RES = sl.RESOLUTION.HD720
CAM_FPS  = 30

# ── Floor detection ───────────────────────────────────────────────────────────
FLOOR_TOLERANCE      = 0.10     # ±m around floor plane (nominal)
FLOOR_TOLERANCE_WIDE = 0.18     # wider band used when floor estimate is uncertain
FLOOR_STABLE_HZ      = 8        # re-run find_floor_plane every N frames when stable
FLOOR_LOST_CONSEC    = 4        # after N consecutive failures → aggressive mode
ROI_TOP_FRACTION     = 0.35     # ignore top 35% (sky / hood)
# Hit-test fallback grid: (frac_x, frac_y) of frame — bottom-centre first
FLOOR_HIT_POINTS = [(0.50, 0.75), (0.35, 0.80), (0.65, 0.80),
                    (0.50, 0.88), (0.35, 0.88), (0.65, 0.88)]

# ── White lane line  (HLS) ────────────────────────────────────────────────────
WHITE_L_MIN = 160               # lower slightly for shadowed asphalt
WHITE_S_MAX = 65

# ── Grass / asphalt boundary  (HSV) ──────────────────────────────────────────
GRASS_H_MIN, GRASS_H_MAX = 25, 95
GRASS_S_MIN = 30
GRASS_V_MIN = 35

# ── Stop-line (orange horizontal stripe, floor pixels only) ──────────────────
STOP_ORANGE_H_MIN   = 5         # HSV hue lower  (orange)
STOP_ORANGE_H_MAX   = 20        # HSV hue upper  (orange)
STOP_ORANGE_S_MIN   = 150       # vivid orange only
STOP_ORANGE_V_MIN   = 100       # reject dark/shadowed patches
STOP_ROW_THRESH     = 0.08      # fraction of row width that must be orange
STOP_COVERAGE_MIN   = 0.60      # fraction of lane interior that must be lit
STOP_PERP_MAX_DEG   = 20.0      # cluster must be within ±20° of horizontal
STOP_DIST_MIN_M     = 0.3
STOP_DIST_MAX_M     = 10.0
STOP_DIST_N_PTS     = 10        # sample points for median distance
STOP_DIST_MIN_VALID = 4         # min valid samples needed
STOP_VOTE_NEEDED    = 5         # consecutive frames before triggering

# ── Polynomial lane fitting ───────────────────────────────────────────────────
POLY_DEG         = 2            # quadratic  →  handles curves
RANSAC_ITER      = 80
RANSAC_THRESH_PX = 6            # pixel residual for inlier test
MIN_INLIERS      = 30
MIN_PIXELS       = 40

# Sliding-window (initial search / lost-lane recovery)
N_WINDOWS  = 12
WIN_MARGIN = 60                 # px half-width per window
WIN_MINPIX = 25                 # min pixels to re-centre window

# Prior-guided search (fast path when previous fit exists)
PRIOR_MARGIN = 50               # px band around previous polynomial

# ── Temporal smoothing ────────────────────────────────────────────────────────
SMOOTH_ALPHA = 0.30             # lower = smoother, higher = more responsive

# ── Lane geometry sanity ──────────────────────────────────────────────────────
LANE_WIDTH_MIN = 0.20
LANE_WIDTH_MAX = 12.0

# ── Lane memory (virtual boundary on single-side turns) ───────────────────────
LANE_MEM_MAX = 60               # rolling history length (frames × 5 samples)

# ── Confidence gates (per source) ─────────────────────────────────────────────
CONF_WHITE = 0.22
CONF_GRASS = 0.22

# ── Obstacle detection ────────────────────────────────────────────────────────
OBS_MIN_HEIGHT_M   = 0.15    # m above floor — ignores road surface / small bumps
OBS_MAX_HEIGHT_M   = 2.50    # m above floor — ignores overhead signs
OBS_MIN_DIST_M     = 0.30    # closest distance to report
OBS_MAX_DIST_M     = 4.00    # furthest distance to report
OBS_LANE_MARGIN_M  = 0.40    # lateral margin beyond lane edges still counts
OBS_MIN_CLUSTER_PX = 60      # minimum point-cloud points per blob (noise rejection)
OBS_SDK_ENABLE     = True    # enable ZED object detection (people / vehicles)

# ── Bird's-eye warp ───────────────────────────────────────────────────────────
# Set WARP_ENABLED=True once you have measured SRC corners on your track.
# SRC = road trapezoid in camera image (fractions of W, H).
# DST = rectangle in bird's-eye output  (fractions of W, H).
WARP_ENABLED = False
WARP_SRC = np.float32([[0.40, 0.55], [0.60, 0.55],
                        [0.85, 0.95], [0.15, 0.95]])
WARP_DST = np.float32([[0.20, 0.05], [0.80, 0.05],
                        [0.80, 0.95], [0.20, 0.95]])

DISPLAY = True


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PerceptionResult:
    deviation_m:    float = 0.0
    confidence:     float = 0.0
    lane_width_m:   float = 0.0
    source:         str   = "NONE"
    left_fit:       Optional[np.ndarray] = None   # [a,b,c]  x = poly(y)
    right_fit:      Optional[np.ndarray] = None
    left_conf:      float = 0.0
    right_conf:     float = 0.0
    stop_line:          bool  = False
    stop_line_y:        Optional[int] = None
    stop_line_dist:     float = 0.0
    virtual_left:       bool  = False
    virtual_right:      bool  = False
    # Obstacle
    obstacle_detected:  bool  = False
    obstacle_dist_m:    float = 0.0
    obstacle_lateral_m: float = 0.0
    obstacle_bbox:      Optional[Tuple[int,int,int,int]] = None  # (x,y,w,h) image px


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial helpers
# ─────────────────────────────────────────────────────────────────────────────

def eval_x(coeffs: np.ndarray, y: float) -> float:
    return float(np.polyval(coeffs, y))


def ransac_poly(
    ys: np.ndarray, xs: np.ndarray
) -> Tuple[Optional[np.ndarray], float]:
    """Fit  x = poly(y)  degree POLY_DEG via RANSAC. Returns (coeffs, conf)."""
    n = len(xs)
    if n < MIN_PIXELS:
        return None, 0.0

    n_sample    = POLY_DEG + 1      # 3 pts for quadratic
    best_coeffs = None
    best_n      = 0

    for _ in range(RANSAC_ITER):
        idx = np.random.choice(n, n_sample, replace=False)
        try:
            c = np.polyfit(ys[idx], xs[idx], POLY_DEG)
        except np.linalg.LinAlgError:
            continue
        residuals = np.abs(xs - np.polyval(c, ys))
        n_in = int((residuals < RANSAC_THRESH_PX).sum())
        if n_in > best_n:
            best_n = n_in
            mask = residuals < RANSAC_THRESH_PX
            if mask.sum() >= n_sample:
                try:
                    best_coeffs = np.polyfit(ys[mask], xs[mask], POLY_DEG)
                except np.linalg.LinAlgError:
                    pass

    if best_n < MIN_INLIERS or best_coeffs is None:
        return None, 0.0
    return best_coeffs, min(0.99, best_n / (MIN_INLIERS * 6))


# ─────────────────────────────────────────────────────────────────────────────
# Lane pixel extraction
# ─────────────────────────────────────────────────────────────────────────────

def sliding_window_fit(
    mask: np.ndarray, W: int, H: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
    """
    Histogram-initialised sliding-window search.
    No hard left/right image split — lane bases come from pixel density peaks.
    """
    hist = np.sum(mask[int(H * 0.60):, :], axis=0).astype(np.int32)
    mid  = W // 2
    lx   = int(np.argmax(hist[:mid]))
    rx   = int(np.argmax(hist[mid:])) + mid

    nz   = mask.nonzero()
    nzy  = np.array(nz[0])
    nzx  = np.array(nz[1])

    win_h  = H // N_WINDOWS
    l_idx, r_idx = [], []

    for w in range(N_WINDOWS):
        y_lo    = H - (w + 1) * win_h
        y_hi    = H - w * win_h
        in_band = (nzy >= y_lo) & (nzy < y_hi)
        gl = np.where(in_band & (nzx >= lx - WIN_MARGIN) &
                      (nzx < lx + WIN_MARGIN))[0]
        gr = np.where(in_band & (nzx >= rx - WIN_MARGIN) &
                      (nzx < rx + WIN_MARGIN))[0]
        l_idx.append(gl)
        r_idx.append(gr)
        if gl.size >= WIN_MINPIX:
            lx = int(np.mean(nzx[gl]))
        if gr.size >= WIN_MINPIX:
            rx = int(np.mean(nzx[gr]))

    l_idx = np.concatenate(l_idx) if l_idx else np.array([], dtype=int)
    r_idx = np.concatenate(r_idx) if r_idx else np.array([], dtype=int)

    lfit = ransac_poly(nzy[l_idx], nzx[l_idx])
    rfit = ransac_poly(nzy[r_idx], nzx[r_idx])
    return lfit[0], rfit[0], lfit[1], rfit[1]


def search_near_poly(
    mask: np.ndarray,
    prev_lf: Optional[np.ndarray],
    prev_rf: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
    """Fast band-search around previous polynomial — used after first good frame."""
    nz   = mask.nonzero()
    nzy  = np.array(nz[0])
    nzx  = np.array(nz[1])

    def _collect(fit):
        if fit is None or nzy.size == 0:
            return np.array([]), np.array([])
        pred  = np.polyval(fit, nzy)
        close = np.abs(nzx - pred) < PRIOR_MARGIN
        return nzx[close], nzy[close]

    lxs, lys = _collect(prev_lf)
    rxs, rys = _collect(prev_rf)
    lfit = ransac_poly(lys, lxs) if lxs.size >= MIN_PIXELS else (None, 0.0)
    rfit = ransac_poly(rys, rxs) if rxs.size >= MIN_PIXELS else (None, 0.0)
    return lfit[0], rfit[0], lfit[1], rfit[1]


def fit_lanes(
    mask: np.ndarray, W: int, H: int,
    prev_lf: Optional[np.ndarray],
    prev_rf: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
    """Use fast prior-guided search when possible; fall back to sliding window."""
    if prev_lf is not None and prev_rf is not None:
        lf, rf, lc, rc = search_near_poly(mask, prev_lf, prev_rf)
        if lf is None or rf is None:
            lf2, rf2, lc2, rc2 = sliding_window_fit(mask, W, H)
            if lf is None: lf, lc = lf2, lc2
            if rf is None: rf, rc = rf2, rc2
        return lf, rf, lc, rc
    return sliding_window_fit(mask, W, H)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal smoothing
# ─────────────────────────────────────────────────────────────────────────────

class Smoother:
    def __init__(self, alpha: float = SMOOTH_ALPHA):
        self.alpha = alpha
        self.l_ema: Optional[np.ndarray] = None
        self.r_ema: Optional[np.ndarray] = None

    def update(self, lf, rf):
        def _ema(cur, new):
            if new is None:
                return cur
            if cur is None or cur.shape != new.shape:
                return new.copy()
            return self.alpha * new + (1.0 - self.alpha) * cur
        self.l_ema = _ema(self.l_ema, lf)
        self.r_ema = _ema(self.r_ema, rf)
        return self.l_ema, self.r_ema


# ─────────────────────────────────────────────────────────────────────────────
# Deviation
# ─────────────────────────────────────────────────────────────────────────────

def compute_deviation(lf, rf, pc, H, W):
    devs, widths = [], []
    sample_ys = [int(H * f) for f in [0.95, 0.85, 0.75, 0.65]]
    weights   = [0.40, 0.30, 0.20, 0.10]

    for sy, w in zip(sample_ys, weights):
        lx = int(np.clip(eval_x(lf, sy), 0, W-1)) if lf is not None else None
        rx = int(np.clip(eval_x(rf, sy), 0, W-1)) if rf is not None else None

        if lx is not None and rx is not None and rx > lx:
            lp, rp = pc[sy, lx, :3], pc[sy, rx, :3]
            if np.isfinite(lp).all() and np.isfinite(rp).all():
                wm = abs(float(rp[0] - lp[0]))
                if LANE_WIDTH_MIN < wm < LANE_WIDTH_MAX:
                    dev_m = (W / 2.0 - (lx + rx) / 2.0) / (rx - lx) * wm
                    devs.append((dev_m, w))
                    widths.append(wm)
        elif lx is not None:
            lp = pc[sy, lx, :3]
            if np.isfinite(lp).all() and widths:
                half = float(np.median(widths)) / 2.0
                devs.append(((W // 2 - lx) / (W // 2) * half, w * 0.4))
        elif rx is not None:
            rp = pc[sy, rx, :3]
            if np.isfinite(rp).all() and widths:
                half = float(np.median(widths)) / 2.0
                devs.append(((W // 2 - rx) / (W // 2) * half, w * 0.4))

    if not devs:
        return 0.0, 0.0
    tw  = sum(d[1] for d in devs)
    dev = sum(d[0] * d[1] for d in devs) / tw
    wid = float(np.median(widths)) if widths else 0.0
    return dev, wid


# ─────────────────────────────────────────────────────────────────────────────
# Lane memory  (virtual boundary when one side is out of frame)
# ─────────────────────────────────────────────────────────────────────────────

class LaneMemory:
    """
    Accumulates real-world lane width observations over a rolling window.
    When only one boundary is visible (tight turn, occlusion), projects the
    missing side using stored width + depth from the point cloud so the
    virtual line stays geometrically correct under perspective.
    """

    def __init__(self):
        self._px_samples: list = []
        self._m_samples:  list = []
        self.mean_px: Optional[float] = None
        self.mean_m:  Optional[float] = None

    def update(self, lf, rf, pc: np.ndarray, H: int, W: int):
        if lf is None or rf is None:
            return
        for row in [int(H * f) for f in [0.95, 0.85, 0.75, 0.65, 0.55]]:
            lx = int(np.clip(eval_x(lf, row), 0, W - 1))
            rx = int(np.clip(eval_x(rf, row), 0, W - 1))
            if rx <= lx:
                continue
            self._px_samples.append(float(rx - lx))
            lp, rp = pc[row, lx, :3], pc[row, rx, :3]
            if np.isfinite(lp).all() and np.isfinite(rp).all():
                wm = abs(float(rp[0] - lp[0]))
                if LANE_WIDTH_MIN < wm < LANE_WIDTH_MAX:
                    self._m_samples.append(wm)
        if len(self._px_samples) > LANE_MEM_MAX:
            self._px_samples = self._px_samples[-LANE_MEM_MAX:]
        if len(self._m_samples) > LANE_MEM_MAX:
            self._m_samples = self._m_samples[-LANE_MEM_MAX:]
        if self._px_samples:
            self.mean_px = float(np.median(self._px_samples))
        if self._m_samples:
            self.mean_m = float(np.median(self._m_samples))

    def virtual_right(self, lf: np.ndarray, pc: np.ndarray,
                      H: int, W: int, fx: float) -> Optional[np.ndarray]:
        """Project right boundary from known left + real-world width."""
        if lf is None or self.mean_m is None:
            return None
        ys, vxs = [], []
        for row in np.arange(int(H * ROI_TOP_FRACTION), H, 8, dtype=int):
            lx = int(np.clip(np.polyval(lf, row), 0, W - 1))
            pt = pc[row, lx, :3]
            if np.isfinite(pt).all() and float(pt[2]) < 0:
                depth = abs(float(pt[2]))
                ys.append(float(row))
                vxs.append(float(np.clip(lx + self.mean_m * fx / depth, 0, W - 1)))
        if len(ys) >= POLY_DEG + 2:
            try:
                return np.polyfit(np.array(ys), np.array(vxs), POLY_DEG)
            except np.linalg.LinAlgError:
                pass
        # Fallback: fixed pixel offset
        if self.mean_px is None:
            return None
        vrf = lf.copy(); vrf[-1] += self.mean_px
        return vrf

    def virtual_left(self, rf: np.ndarray, pc: np.ndarray,
                     H: int, W: int, fx: float) -> Optional[np.ndarray]:
        """Project left boundary from known right − real-world width."""
        if rf is None or self.mean_m is None:
            return None
        ys, vxs = [], []
        for row in np.arange(int(H * ROI_TOP_FRACTION), H, 8, dtype=int):
            rx = int(np.clip(np.polyval(rf, row), 0, W - 1))
            pt = pc[row, rx, :3]
            if np.isfinite(pt).all() and float(pt[2]) < 0:
                depth = abs(float(pt[2]))
                ys.append(float(row))
                vxs.append(float(np.clip(rx - self.mean_m * fx / depth, 0, W - 1)))
        if len(ys) >= POLY_DEG + 2:
            try:
                return np.polyfit(np.array(ys), np.array(vxs), POLY_DEG)
            except np.linalg.LinAlgError:
                pass
        if self.mean_px is None:
            return None
        vlf = rf.copy(); vlf[-1] -= self.mean_px
        return vlf


# ─────────────────────────────────────────────────────────────────────────────
# Stop-line detection  (orange horizontal stripe, floor pixels only)
# ─────────────────────────────────────────────────────────────────────────────

def _stop_perp_ok(orange: np.ndarray, row: int, H: int, W: int) -> bool:
    """
    Check that the orange cluster around `row` is nearly horizontal
    (within ±STOP_PERP_MAX_DEG of the horizontal axis).
    Uses PCA on the pixel cluster — rejects angled shadows and diagonal markings.
    """
    import math
    r0, r1 = max(0, row - 10), min(H, row + 10)
    ys_l, xs_l = np.where(orange[r0:r1, :] > 0)
    if xs_l.size < 10:
        return False
    pts = np.stack([xs_l.astype(float), (ys_l + r0).astype(float)], axis=1)
    pts -= pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    ang = abs(math.degrees(math.atan2(Vt[0, 1], Vt[0, 0])))
    if ang > 90:
        ang = 180 - ang
    return ang <= STOP_PERP_MAX_DEG


def _stop_median_dist(pc, row: int, lx: int, rx: int, H: int, W: int) -> float:
    """Sample STOP_DIST_N_PTS evenly across lane, return median Z distance."""
    sample_row = min(row + 20, H - 1)
    xs = np.clip(np.linspace(lx, rx, STOP_DIST_N_PTS, dtype=int), 0, W - 1)
    dists = []
    for sx in xs:
        pt = pc[sample_row, sx, :3]
        if np.isfinite(pt).all():
            d = abs(float(pt[2]))
            if STOP_DIST_MIN_M <= d <= STOP_DIST_MAX_M:
                dists.append(d)
    return float(np.median(dists)) if len(dists) >= STOP_DIST_MIN_VALID else 0.0


def detect_stop_line(frame, floor_mask, lf, rf, pc, H, W, hls, hsv):
    """
    Detect an orange horizontal stop stripe painted on the road.

    Requirements (all must pass):
      1. Pixel is on the floor (floor_mask)
      2. HSV hue in orange range (H=5-20, S≥150, V≥100)
      3. Row density: ≥8% of frame width is orange
      4. Perpendicularity: cluster principal axis ≤20° from horizontal
      5. Lane coverage: orange spans ≥60% of lane width (when fits available)
      6. Located in bottom 60% of frame
    """
    fp    = floor_mask == 255
    H_ch  = hsv[:, :, 0]
    S_hsv = hsv[:, :, 1]
    V_hsv = hsv[:, :, 2]

    orange = np.zeros((H, W), np.uint8)
    orange[fp &
           (H_ch >= STOP_ORANGE_H_MIN) & (H_ch <= STOP_ORANGE_H_MAX) &
           (S_hsv >= STOP_ORANGE_S_MIN) & (V_hsv >= STOP_ORANGE_V_MIN)] = 255

    row_counts = orange.sum(axis=1)
    candidates = np.where(row_counts > W * STOP_ROW_THRESH)[0]
    candidates = candidates[candidates > int(H * 0.40)]   # bottom 60% only

    for row in candidates:
        # Gate 1: must be horizontal (not a diagonal shadow)
        if not _stop_perp_ok(orange, row, H, W):
            continue

        if lf is not None and rf is not None:
            lx     = int(np.clip(eval_x(lf, row), 0, W - 1))
            rx     = int(np.clip(eval_x(rf, row), 0, W - 1))
            lane_w = rx - lx
            if lane_w < 20:
                continue
            # Gate 2: must span most of the lane (60% coverage)
            coverage = int(orange[row, lx:rx].sum()) / lane_w
            if coverage < STOP_COVERAGE_MIN:
                continue
            dist = _stop_median_dist(pc, row, lx, rx, H, W)
            return True, row, dist
        else:
            # No lane fits available — use coarser width gate
            if row_counts[row] > W * 0.15:
                mid  = W // 2
                dist = _stop_median_dist(pc, row, mid - 50, mid + 50, H, W)
                return True, row, dist

    return False, None, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Perspective warp helper
# ─────────────────────────────────────────────────────────────────────────────

class WarpTransform:
    """
    Caches perspective matrices.  Fits are performed in bird's-eye space then
    re-projected back to image space so that deviation + drawing remain
    coordinate-agnostic.
    """
    def __init__(self):
        self.M    = None
        self.Minv = None

    def build(self, W: int, H: int):
        src = WARP_SRC * np.array([W, H], dtype=np.float32)
        dst = WARP_DST * np.array([W, H], dtype=np.float32)
        self.M    = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        return cv2.warpPerspective(img, self.M, (W, H), flags=cv2.INTER_LINEAR)

    def bev_poly_to_img(
        self, fit_bev: np.ndarray, H: int, W: int
    ) -> Optional[np.ndarray]:
        """Sample a BEV polynomial and unproject the curve to image space."""
        ys  = np.linspace(int(H * ROI_TOP_FRACTION), H - 1, 40)
        xs  = np.polyval(fit_bev, ys)
        pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
        try:
            pts_img = cv2.perspectiveTransform(pts, self.Minv).reshape(-1, 2)
        except cv2.error:
            return None
        ix, iy = pts_img[:, 0], pts_img[:, 1]
        valid  = (np.isfinite(ix) & np.isfinite(iy) &
                  (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H))
        if valid.sum() < POLY_DEG + 1:
            return None
        try:
            return np.polyfit(iy[valid], ix[valid], POLY_DEG)
        except np.linalg.LinAlgError:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle detection — point cloud above-floor
# ─────────────────────────────────────────────────────────────────────────────

def detect_obstacle_pc(
    pc: np.ndarray, floor_y: float,
    lf, rf,
    H: int, W: int, fx: float,
) -> Tuple[bool, float, float, Optional[Tuple[int,int,int,int]]]:
    """
    Scan the point cloud for objects that are physically above the floor and
    within the lane corridor (+ OBS_LANE_MARGIN_M on each side).

    Returns (detected, dist_m, lateral_m, bbox_image_px).
      dist_m      — forward distance to nearest obstacle centroid (|Z|)
      lateral_m   — signed lateral offset from image centre  (+left / -right)
      bbox        — (x, y, w, h) tight bounding box in image pixels

    Algorithm:
      1. Build a boolean volume mask — points above floor in height range,
         within forward distance range, and inside the lane corridor.
      2. Project to image-space binary mask (set pixel if any depth slice fires).
      3. Connected components → keep largest blob above OBS_MIN_CLUSTER_PX.
      4. Report distance from 3-D centroid of that blob's world points.
    """
    if floor_y is None:
        return False, 0.0, 0.0, None

    X = pc[:, :, 0]   # lateral  (right = +)
    Y = pc[:, :, 1]   # vertical (up    = +)
    Z = pc[:, :, 2]   # depth    (forward = negative in RIGHT_HANDED_Y_UP)

    valid = (np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z))

    # Height gate: above floor surface, below ceiling
    above = (Y > floor_y + OBS_MIN_HEIGHT_M) & (Y < floor_y + OBS_MAX_HEIGHT_M)

    # Forward gate: Z is negative forward in RIGHT_HANDED_Y_UP
    fwd = (Z < -OBS_MIN_DIST_M) & (Z > -OBS_MAX_DIST_M)

    # Lateral gate: inside lane + margin (fallback to full frame if no fits)
    if lf is not None or rf is not None:
        # Estimate lane X bounds at each row via 3-D projection
        # Use fx and Z to convert pixel lane position → real-world X metres
        row_idx = np.arange(H, dtype=np.float32)
        if lf is not None and rf is not None:
            lx_px = np.clip(np.polyval(lf, row_idx).astype(int), 0, W - 1)
            rx_px = np.clip(np.polyval(rf, row_idx).astype(int), 0, W - 1)
        elif lf is not None:
            lx_px = np.clip(np.polyval(lf, row_idx).astype(int), 0, W - 1)
            rx_px = np.full(H, W - 1, dtype=int)
        else:
            lx_px = np.zeros(H, dtype=int)
            rx_px = np.clip(np.polyval(rf, row_idx).astype(int), 0, W - 1)

        # Build a per-pixel lateral mask from lane pixel boundaries + margin
        col_idx = np.tile(np.arange(W, dtype=int), (H, 1))   # (H, W)
        lane_l  = lx_px[:, None] - int(OBS_LANE_MARGIN_M * fx / 5.0)  # rough px margin
        lane_r  = rx_px[:, None] + int(OBS_LANE_MARGIN_M * fx / 5.0)
        in_lane = (col_idx >= lane_l) & (col_idx <= lane_r)
    else:
        in_lane = np.ones((H, W), bool)

    obs_mask = (valid & above & fwd & in_lane).astype(np.uint8) * 255

    # Remove sky ROI
    obs_mask[:int(H * ROI_TOP_FRACTION), :] = 0

    # Morphological clean-up — close small gaps, remove isolated pixels
    k = np.ones((5, 5), np.uint8)
    obs_mask = cv2.morphologyEx(obs_mask, cv2.MORPH_OPEN,  k)
    obs_mask = cv2.morphologyEx(obs_mask, cv2.MORPH_CLOSE, k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        obs_mask, connectivity=8)

    best_label = -1
    best_area  = OBS_MIN_CLUSTER_PX - 1
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area  = area
            best_label = i

    if best_label < 0:
        return False, 0.0, 0.0, None

    # Bounding box in image pixels
    bx = stats[best_label, cv2.CC_STAT_LEFT]
    by = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]

    # 3-D centroid of blob's world points
    blob = labels == best_label
    Zb = Z[blob]
    Xb = X[blob]
    finite = np.isfinite(Zb) & np.isfinite(Xb)
    if finite.sum() < 4:
        return False, 0.0, 0.0, None

    dist_m    = float(np.median(np.abs(Zb[finite])))
    lateral_m = float(np.median(Xb[finite]))   # + = left of camera

    return True, dist_m, lateral_m, (int(bx), int(by), int(bw), int(bh))


# ─────────────────────────────────────────────────────────────────────────────
# Main perception class
# ─────────────────────────────────────────────────────────────────────────────

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

        # Geometry + lane memory
        self.lane_mem = LaneMemory()

        # Stop-line temporal vote gate
        self._stop_votes:     int            = 0
        self._last_stop_dist: float          = 0.0
        self._last_stop_y:    Optional[int]  = None

    # ── Init ──────────────────────────────────────────────────────────────────

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

        # ── ZED object detection (people + vehicles) ──────────────────────────
        if OBS_SDK_ENABLE:
            od_params = sl.ObjectDetectionParameters()
            od_params.detection_model       = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
            od_params.enable_tracking       = True
            od_params.enable_segmentation   = False
            if self.cam.enable_object_detection(od_params) == sl.ERROR_CODE.SUCCESS:
                self._sdk_obj_enabled = True
                print("[Perception] Object detection: ON")
            else:
                print("[Perception] Object detection: FAILED (PC-only fallback)")

        print(f"[Perception] OK  {self.W}×{self.H} @ {CAM_FPS} fps  "
              f"BEV={'on' if WARP_ENABLED else 'off'}")
        return True

    # ── Floor ─────────────────────────────────────────────────────────────────

    def _update_floor(self) -> bool:
        """
        Continuous, resilient floor plane estimation for a moving vehicle.

        Two modes:
          STABLE   — floor_y known and recently confirmed:
                     re-run find_floor_plane every FLOOR_STABLE_HZ frames.
          LOST     — more than FLOOR_LOST_CONSEC consecutive failures:
                     retry EVERY frame using find_floor_plane + 6 hit-test
                     fallback points until a valid estimate is recovered.

        Never blocks the pipeline — if all methods fail but a cached floor_y
        exists, it is kept and flagged uncertain (wider mask tolerance).
        """
        in_lost_mode = self._floor_miss_count >= FLOOR_LOST_CONSEC
        on_schedule  = (self.frame_cnt % FLOOR_STABLE_HZ == 0)

        # Fast path: stable floor, not yet time to refresh
        if not in_lost_mode and self.floor_y is not None and not on_schedule:
            return True

        # ── Primary: ZED floor plane fit ─────────────────────────────────────
        if self.cam.find_floor_plane(self.plane, self.reset_tf) == \
                sl.ERROR_CODE.SUCCESS:
            eq = self.plane.get_plane_equation()
            if abs(eq[1]) > 0.5:
                ny = -eq[3] / eq[1]
                # EMA blend — faster (0.3) in lost mode to snap back quickly
                alpha = 0.30 if in_lost_mode else 0.20
                self.floor_y = (ny if self.floor_y is None
                                else (1 - alpha) * self.floor_y + alpha * ny)
                self._floor_miss_count = 0
                return True

        # ── Fallback: hit-test at multiple road points ────────────────────────
        for fx, fy in FLOOR_HIT_POINTS:
            coord = [int(self.W * fx), int(self.H * fy)]
            if self.cam.find_plane_at_hit(coord, self.plane) == \
                    sl.ERROR_CODE.SUCCESS:
                eq = self.plane.get_plane_equation()
                if abs(eq[1]) > 0.5:
                    ny = -eq[3] / eq[1]
                    # Hit-test is less accurate — slower blend
                    self.floor_y = (ny if self.floor_y is None
                                    else 0.90 * self.floor_y + 0.10 * ny)
                    self._floor_miss_count = 0
                    return True

        # ── All methods failed this frame ─────────────────────────────────────
        self._floor_miss_count += 1
        # Keep cached floor_y — stale but prevents full pipeline stall
        return self.floor_y is not None

    def _floor_mask(self, pc: np.ndarray) -> np.ndarray:
        # Use wider tolerance when estimate is uncertain (missed recent updates)
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

    # ── Colour pipeline — single HLS + HSV pass ───────────────────────────────

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

    # ── Fit a binary mask → image-space polynomials ───────────────────────────

    def _fit_mask(
        self, mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
        if WARP_ENABLED and self.warp.M is not None:
            bev  = self.warp.warp(mask)
            lf_b, rf_b, lc, rc = fit_lanes(bev, self.W, self.H,
                                             self._bev_lf, self._bev_rf)
            self._bev_lf, self._bev_rf = lf_b, rf_b
            # Unproject back to image space
            lf = self.warp.bev_poly_to_img(lf_b, self.H, self.W) \
                 if lf_b is not None else None
            rf = self.warp.bev_poly_to_img(rf_b, self.H, self.W) \
                 if rf_b is not None else None
            return lf, rf, lc, rc
        return fit_lanes(mask, self.W, self.H, self._prev_lf, self._prev_rf)

    # ── Main processing loop ──────────────────────────────────────────────────

    def process(self):
        if self.cam.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None

        self.frame_cnt += 1
        self.cam.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        frame = self.image_mat.get_data()[:, :, :3].copy()

        # XYZ only — faster than XYZRGBA, RGBA channels not used
        self.cam.retrieve_measure(self.pc_mat, sl.MEASURE.XYZ, sl.MEM.CPU)
        pc = self.pc_mat.get_data()[:, :, :3]

        # Keep tracker ticking but do NOT gate floor update on tracking state.
        # Curves cause tracker to go SEARCHING briefly — floor must keep working.
        self.cam.get_position(self.pose)

        if not self._update_floor():
            return PerceptionResult(source="NO_FLOOR"), frame, None, None, None, None

        fm               = self._floor_mask(pc)
        wm, gm, hls, hsv = self._color_masks(frame, fm)

        # ── Level 1: White lines (primary) ────────────────────────────────────────
        wl, wr, wlc, wrc = self._fit_mask(wm)

        # ── Level 2: Grass edges (fallback when white is weak) ────────────────
        if max(wlc, wrc) < CONF_WHITE:
            ge = cv2.Canny(gm, 50, 150)
            gl, gr, glc, grc = self._fit_mask(ge)
        else:
            gl = gr = None
            glc = grc = 0.0

        def best(opts):
            v = [(f, c) for f, c in opts if f is not None and c > 0.05]
            return max(v, key=lambda x: x[1]) if v else (None, 0.0)

        raw_l, lc = best([
            (wl, wlc * 1.4),
            (gl, glc),
        ])
        raw_r, rc = best([
            (wr, wrc * 1.4),
            (gr, grc),
        ])

        # Update priors with RAW fits before smoothing — curve-responsive
        if raw_l is not None: self._prev_lf = raw_l
        if raw_r is not None: self._prev_rf = raw_r

        lf, rf = self.smoother.update(raw_l, raw_r)

        # ── Virtual boundary from lane memory ─────────────────────────────────
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

        # ── Source label ──────────────────────────────────────────────────────
        if wlc > CONF_WHITE or wrc > CONF_WHITE:
            source = "WHITE_LINE"
        elif glc > CONF_GRASS or grc > CONF_GRASS:
            source = "GRASS"
        else:
            source = "LOST"

        dev = wid = 0.0
        if lf is not None or rf is not None:
            dev, wid = compute_deviation(lf, rf, pc, self.H, self.W)

        # Hold last valid deviation when LOST — zero on a curve drives off track
        if source != "LOST":
            self._last_deviation = dev
        else:
            dev = self._last_deviation

        raw_stop, raw_y, raw_dist = detect_stop_line(
            frame, fm, lf, rf, pc, self.H, self.W, hls, hsv)

        # Temporal vote gate — require STOP_VOTE_NEEDED consecutive detections
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

        # ── Obstacle detection ─────────────────────────────────────────────────
        # Primary: point cloud above-floor scan
        obs_det, obs_dist, obs_lat, obs_bbox = detect_obstacle_pc(
            pc, self.floor_y, lf, rf, self.H, self.W, self.cal.fx)

        # Secondary: ZED SDK object detection (people / vehicles)
        if self._sdk_obj_enabled:
            od_rt = sl.ObjectDetectionRuntimeParameters()
            od_rt.detection_confidence_threshold = 40
            self.cam.retrieve_objects(self.objects, od_rt)
            for obj in self.objects.object_list:
                if not obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    continue
                pos = obj.position          # (X, Y, Z) metres, world frame
                if not (np.isfinite(pos[0]) and np.isfinite(pos[2])):
                    continue
                sdk_dist = abs(float(pos[2]))
                if not (OBS_MIN_DIST_M <= sdk_dist <= OBS_MAX_DIST_M):
                    continue
                # Accept if closer than point-cloud result or PC found nothing
                if not obs_det or sdk_dist < obs_dist:
                    obs_det  = True
                    obs_dist = sdk_dist
                    obs_lat  = float(pos[0])
                    # Convert SDK 3-D bbox to image pixels via pinhole projection
                    bb2d = obj.bounding_box_2d   # list of 4 (x,y) image corners
                    if bb2d is not None and len(bb2d) == 4:
                        pts = np.array(bb2d, dtype=int)
                        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
                        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
                        obs_bbox = (int(x0), int(y0), int(x1-x0), int(y1-y0))

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
        ), frame, fm, wm, gm

    def close(self):
        if self._sdk_obj_enabled:
            self.cam.disable_object_detection()
        self.cam.disable_positional_tracking()
        self.cam.close()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw(frame, result, fm, wm, gm, H, W):
    vis = frame.copy()

    if wm is not None:
        vis[wm == 255] = [255, 255, 255]
    if gm is not None:
        vis[gm == 255] = [0, 180, 0]
    cv2.addWeighted(vis, 0.45, frame, 0.55, 0, vis)

    if fm is not None:
        cnts, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)

    # Vectorised lane overlay
    lf, rf = result.left_fit, result.right_fit
    if lf is not None or rf is not None:
        ys  = np.arange(int(H * ROI_TOP_FRACTION), H, 4)
        ovl = np.zeros_like(vis)

        if lf is not None and rf is not None:
            lxs = np.clip(np.polyval(lf, ys).astype(int), 0, W-1)
            rxs = np.clip(np.polyval(rf, ys).astype(int), 0, W-1)
            pts_l  = np.stack([lxs, ys], axis=1)
            pts_r  = np.stack([rxs, ys], axis=1)[::-1]
            cv2.fillPoly(ovl, [np.vstack([pts_l, pts_r])], (0, 55, 0))
            mxs = (lxs + rxs) // 2
            for i in range(len(ys) - 1):
                cv2.line(ovl, (mxs[i], ys[i]), (mxs[i+1], ys[i+1]),
                         (0, 255, 200), 2)

        # Virtual boundaries drawn in purple; real ones in orange/blue
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
    half  = max(result.lane_width_m / 2.0, 0.3)
    dn    = np.clip(result.deviation_m / half, -1.0, 1.0)
    ind   = int(mid + dn * (bw // 2))
    thresh = max(result.lane_width_m * 0.08, 0.05)
    col   = (0,255,0) if abs(result.deviation_m) < thresh else (0,165,255)
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

    # Obstacle bounding box
    if result.obstacle_detected and result.obstacle_bbox is not None:
        ox, oy, ow, oh = result.obstacle_bbox
        cv2.rectangle(vis, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 3)
        side = ("L" if result.obstacle_lateral_m > 0.1
                else "R" if result.obstacle_lateral_m < -0.1 else "C")
        obs_label = f"OBS {result.obstacle_dist_m:.1f}m {side}"
        cv2.putText(vis, obs_label, (ox, max(oy - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # HUD
    sc  = {"WHITE_LINE": (0,255,0), "GRASS": (0,200,0),
           "LOST": (0,0,255), "NO_FLOOR": (0,0,255)}.get(result.source, (180,180,180))
    virt_tag = ("" + (" vL" if result.virtual_left else "")
                   + (" vR" if result.virtual_right else ""))
    cs  = ("CENTER" if abs(result.deviation_m) < thresh
           else "LEFT" if result.deviation_m > 0 else "RIGHT")

    cv2.rectangle(vis, (0, 0), (W, 118), (0,0,0), -1)
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
    cv2.putText(vis, stop_info, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,255) if result.stop_line else (80,80,80), 2)
    obs_info = (f"OBSTACLE: {result.obstacle_dist_m:.1f}m  lat:{result.obstacle_lateral_m:+.2f}m"
                if result.obstacle_detected else "No obstacle")
    cv2.putText(vis, obs_info, (10, 113),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if result.obstacle_detected else (80, 80, 80), 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
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
            obs_str  = (f"OBS@{result.obstacle_dist_m:.1f}m lat{result.obstacle_lateral_m:+.2f}m"
                        if result.obstacle_detected else "-")
            print(f"{fc:>6} | {result.source:>12} | "
                  f"{result.deviation_m:>+8.3f} | "
                  f"{result.lane_width_m:>7.2f} | "
                  f"{result.confidence:>5.0%} | "
                  f"{cs:>7} | {stop_str:<10} | {obs_str}")

    perc.close()
    cv2.destroyAllWindows()
    print("Done")
