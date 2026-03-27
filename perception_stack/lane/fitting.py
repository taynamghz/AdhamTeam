"""
PSU Eco Racing — Perception Stack
lane/fitting.py  |  Polynomial lane fitting: RANSAC, sliding-window, prior-guided search.
"""

import warnings
import numpy as np
from typing import Optional, Tuple

from perception_stack.config import (
    POLY_DEG, RANSAC_ITER, RANSAC_THRESH_PX, MIN_INLIERS, MIN_PIXELS,
    N_WINDOWS, WIN_MARGIN, WIN_MINPIX, PRIOR_MARGIN,
)

warnings.filterwarnings('ignore', category=np.RankWarning)


def eval_x(coeffs: np.ndarray, y: float) -> float:
    """Evaluate polynomial x = poly(y) at a given row y."""
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
