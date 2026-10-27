"""Does neural activity track fast partner-distance changes while the focal rat rests at a fixed spot?

A continuous, drift-robust alternative to the territory covariates (see docs/TERRITORY_FEASIBILITY_NOTE.md).
Position and movement are held (approximately) constant by *selecting* rest bins at the focal's home base; the
slow part of everything is removed so chunk-level state and drift cannot masquerade as distance coding.

Pipeline (all on 0.5 s bins, rest bins only, contiguous runs = segments):
    1. target ``d`` = log distance to the nearest other rat (caller supplies it);
    2. **high-pass** ``d``, the rates (sqrt-transformed counts) and the nuisance covariates by subtracting a centred
       running mean (default 60 s; 120 s in the first draft, shortened after probing: on a strong slow-drift scenario the population test false-positive rate was 14% at 120 s and 4-5% at 40-60 s) inside each segment;
    3. nuisance basis = intercept + z + z^2 + z^3 of each high-passed nuisance column (speed, body-part motion, ...);
    4. single cells: partial correlation of the cell with ``d`` given the basis; null = independent circular
       shifts of ``d`` inside each segment (keeps its autocorrelation, breaks its link to the cell);
    5. population: mean squared partial correlation over cells, same shifts.

Assumptions / known limits (the synthetic calibration in tests/test_rest_distance_calibration.py measures them):
    - The null removes the link between ``d`` and *both* the cell and the nuisance, so it is only valid if the
      nuisance basis captures everything the cell shares with ``d`` besides the effect. A cell driven by arousal
      that the nuisance measures with noise will leak into the distance test; the calibration reports how much.
    - Shifts are drawn within segments, so segments shorter than ``min_seg_sec`` are dropped, and the number of
      distinct shifts (null resolution) is limited by total rest time.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ephys._stats_utils import benjamini_hochberg


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def rest_segments(rest: np.ndarray, *, dt: float, min_seg_sec: float = 60.0) -> np.ndarray:
    """Segment id per time bin (``-1`` = not analysed): contiguous runs of ``rest`` at least ``min_seg_sec`` long."""
    rest = np.asarray(rest, dtype=bool)
    seg = np.full(len(rest), -1, dtype=int)
    k = 0
    i = 0
    while i < len(rest):
        if rest[i]:
            j = i
            while j < len(rest) and rest[j]:
                j += 1
            if (j - i) * dt >= min_seg_sec:
                seg[i:j] = k
                k += 1
            i = j
        else:
            i += 1
    return seg


def highpass(v: np.ndarray, seg: np.ndarray, *, dt: float, window_sec: float = 60.0) -> np.ndarray:
    """Subtract a centred running mean inside each segment (NaN outside segments).

    ``v`` may be 1-D or 2-D (time x columns). A segment shorter than the window is simply demeaned.
    """
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, np.nan)
    k = max(int(round(window_sec / dt)), 3)
    for s in np.unique(seg[seg >= 0]):
        idx = np.where(seg == s)[0]
        df = pd.DataFrame(v[idx].reshape(len(idx), -1))
        m = df.rolling(k, center=True, min_periods=max(k // 4, 2)).mean().to_numpy()
        out[idx] = (df.to_numpy() - m).reshape(v[idx].shape)
    return out


def nuisance_basis(nuis: np.ndarray) -> np.ndarray:
    """Intercept + z + z^2 + z^3 per column (columns standardised on the analysed rows)."""
    nuis = np.asarray(nuis, dtype=float)
    if nuis.ndim == 1:
        nuis = nuis[:, None]
    z = (nuis - nuis.mean(0)) / np.where(nuis.std(0) > 0, nuis.std(0), 1.0)
    return np.column_stack([np.ones(len(z)), z, z ** 2 - (z ** 2).mean(0), z ** 3 - (z ** 3).mean(0)])


def prepare(counts: np.ndarray, d: np.ndarray, nuis: np.ndarray, rest: np.ndarray, *, dt: float = 0.5,
            window_sec: float = 60.0, min_seg_sec: float = 60.0) -> Dict[str, object]:
    """High-passed, row-aligned analysis arrays for one chunk."""
    counts = np.asarray(counts, dtype=float)
    nuis = np.asarray(nuis, dtype=float)
    if nuis.ndim == 1:
        nuis = nuis[:, None]
    ok = np.asarray(rest, bool) & np.isfinite(d) & np.all(np.isfinite(nuis), axis=1)
    seg = rest_segments(ok, dt=dt, min_seg_sec=min_seg_sec)
    Y = highpass(np.sqrt(counts + 0.375), seg, dt=dt, window_sec=window_sec)
    dh = highpass(d, seg, dt=dt, window_sec=window_sec)
    Nh = highpass(nuis, seg, dt=dt, window_sec=window_sec)
    use = (seg >= 0) & np.isfinite(dh) & np.all(np.isfinite(Y), axis=1) & np.all(np.isfinite(Nh), axis=1)
    return dict(Y=Y[use], d=dh[use], N=nuisance_basis(Nh[use]), seg=seg[use], dt=dt, n=int(use.sum()),
                rest_min=float(use.sum() * dt / 60.0), n_segments=int(len(np.unique(seg[use]))),
                d_sd=float(np.nanstd(dh[use])) if use.any() else float("nan"))


def _shifted(d: np.ndarray, seg: np.ndarray, n_shifts: int, rng: np.random.Generator,
             min_shift_frac: float = 0.1) -> np.ndarray:
    """``n x n_shifts`` copies of ``d``, each segment circularly shifted independently."""
    out = np.empty((len(d), n_shifts))
    groups = [np.where(seg == s)[0] for s in np.unique(seg)]
    for k in range(n_shifts):
        for g in groups:
            lo = max(1, int(min_shift_frac * len(g)))
            hi = max(lo + 1, len(g) - lo + 1)
            out[g, k] = np.roll(d[g], int(rng.integers(lo, hi)))
    return out


# ---------------------------------------------------------------------------
# Single cells
# ---------------------------------------------------------------------------

def single_cell_tests(P: Dict[str, object], *, n_shifts: int = 999, seed: int = 0,
                      min_spikes_rows: int = 0) -> pd.DataFrame:
    """Per-cell partial correlation with ``d`` given the nuisance basis; shift-null two-sided p; BH q."""
    Y, d, N, seg = P["Y"], P["d"], P["N"], P["seg"]
    n, C = Y.shape
    if n < 50:
        return pd.DataFrame(dict(cell=np.arange(C), r=np.nan, p_value=np.nan, q_value=np.nan))
    Q, _ = np.linalg.qr(N)
    res = lambda M: M - Q @ (Q.T @ M)          # noqa: E731
    Yr = res(Y)
    dr = res(d[:, None])[:, 0]
    D = res(_shifted(d, seg, n_shifts, np.random.default_rng(seed)))
    ny = np.linalg.norm(Yr, axis=0)
    ny[ny == 0] = np.nan
    r_obs = (Yr.T @ dr) / (ny * np.linalg.norm(dr))
    r_null = (Yr.T @ D) / (ny[:, None] * np.linalg.norm(D, axis=0)[None, :])
    p = (1 + np.sum(np.abs(r_null) >= np.abs(r_obs)[:, None], axis=1)) / (1 + n_shifts)
    p = np.where(np.isfinite(r_obs), p, np.nan)
    q = benjamini_hochberg(np.where(np.isfinite(p), p, 1.0))
    return pd.DataFrame(dict(cell=np.arange(C), r=r_obs, p_value=p, q_value=q))


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def population_test(P: Dict[str, object], *, n_shifts: int = 999, seed: int = 0) -> Dict[str, object]:
    """Population-level partial association: mean over cells of the squared partial correlation with ``d``.

    Uses the same shifts for every cell, so cross-cell correlation in the rates is preserved in the null. It is
    sensitive to diffuse effects spread over many cells, which a max-type or per-cell FDR test misses.

    A ridge *decoder* version (out-of-fold R^2 of ``d`` from the rates over nuisance only, shift null on the
    nuisance-conditioned residual) was written first and dropped: on synthetic nulls its p-values were
    conservative (obs gain SD 0.050 vs null SD 0.021; p<=0.2 in 3% of sessions) and it had ~7% power at an effect
    where the per-cell test had 81%.
    """
    Y, d, N, seg = P["Y"], P["d"], P["N"], P["seg"]
    if len(d) < 200:
        return dict(status="insufficient_rest", stat=np.nan, p_value=np.nan, n=len(d))
    Q, _ = np.linalg.qr(N)
    res = lambda M: M - Q @ (Q.T @ M)          # noqa: E731
    Yr = res(Y)
    ny = np.linalg.norm(Yr, axis=0)
    good = ny > 0
    Yr = Yr[:, good] / ny[good]
    dr = res(d[:, None])[:, 0]
    D = res(_shifted(d, seg, n_shifts, np.random.default_rng(seed)))
    D = D / np.linalg.norm(D, axis=0)
    stat = float(np.mean((Yr.T @ (dr / np.linalg.norm(dr))) ** 2))
    null = np.mean((Yr.T @ D) ** 2, axis=0)
    p = (1 + np.sum(null >= stat)) / (1 + n_shifts)
    return dict(status="ok", stat=stat, null_mean=float(null.mean()), p_value=float(p), p_floor=1.0 / (1 + n_shifts),
                n=int(len(d)), n_cells=int(good.sum()), n_shifts=int(n_shifts))
