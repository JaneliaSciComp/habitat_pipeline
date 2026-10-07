"""Population-level tests for territory information beyond place coding.

A static territory is a function of position, so "decode own vs foreign" from all
data would just re-discover place coding. Each test below isolates territory from
position in a different way and gets its null from the same idea: **move the
boundary, keep everything else.**

1. :func:`boundary_straddle_test` - are population vectors on opposite sides of the
   territory boundary more different than vectors the *same distance apart on the
   same side*, restricted to a band around the boundary (so local place-field
   density is matched)? Squared distances are cross-validated across interleaved
   time blocks, which removes the noise-dependent bias that would otherwise make
   rarely-visited bins look "different".
2. :func:`territory_decoder_gain` - within a band around the boundary, does adding
   neural activity to a smooth-position decoder improve own-vs-foreign decoding?
   Place coding alone also improves it (neurons resolve position better than a
   smooth basis), which is why the statistic is compared with the same statistic
   for displaced boundaries rather than with chance.
3. :func:`owner_presence_decoding` - while the focal rat is inside a *neighbour's*
   territory, can neural activity tell whether that neighbour is currently home?
   Position is (nearly) constant within an owner's territory, so place coding
   cannot carry the label; focal-owner distance is a nuisance baseline.

Nulls
-----
Tests 1 and 2: territory masks translated across the arena (no wrap), excluding
translations that still overlap the real mask (Jaccard > ``max_jaccard``) or that
lose too much of it. The p-value is ``(1 + #{null >= obs}) / (n_valid + 1)`` so its
floor is ``1 / (n_valid + 1)``; neighbouring displacements are spatially
correlated, so ``n_valid`` overstates the independent resolution - it is returned
as ``n_displacements`` and ``p_floor`` for that reason, not as a power claim.
Test 3: the label sequence is circularly shifted within each owner's rows, which
keeps its autocorrelation and breaks its link to neural activity *and* to the
distance nuisance.

Known limit (tested, see tests/test_territory_population_calibration.py): tests 1
and 2 cannot tell a territory effect from a place-field density that happens to be
higher at the true boundary. Test 1 controls for it by comparing within the same
band; test 2 does not. Treat a test-2 result without a matching test-1 or
single-cell GLM result as unconfirmed.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from ephys.territory_encoding import blocked_folds
from video.territory import TerritoryMap, signed_distance_from_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def preprocess_rates(rates: np.ndarray, dt: float, detrend_sec: Optional[float] = 120.0) -> np.ndarray:
    """sqrt-transform, remove each cell's slow trend, z-score per cell.

    ``rates`` is (n_bins, n_cells) in Hz (or counts; only relative scale matters).
    Detrending subtracts a centred moving average of ``detrend_sec`` so slow gain
    drift - which is correlated with where the animal was *when* - does not
    masquerade as spatial or territorial structure.
    """
    r = np.sqrt(np.clip(np.asarray(rates, dtype=float), 0, None))
    if detrend_sec:
        w = max(3, int(round(detrend_sec / dt)))
        r = r - ndimage.uniform_filter1d(r, size=w, axis=0, mode="nearest")
    sd = r.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (r - r.mean(axis=0)) / sd


def _shifted(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a boolean grid mask by (dx, dy) bins, no wrap, zero fill."""
    out = np.zeros_like(mask)
    nx, ny = mask.shape
    xs_src = slice(max(0, -dx), min(nx, nx - dx))
    xs_dst = slice(max(0, dx), min(nx, nx + dx))
    ys_src = slice(max(0, -dy), min(ny, ny - dy))
    ys_dst = slice(max(0, dy), min(ny, ny + dy))
    out[xs_dst, ys_dst] = mask[xs_src, ys_src]
    return out


def displaced_masks(mask: np.ndarray, *, max_jaccard: float = 0.5, min_area_kept: float = 0.6,
                    max_n: Optional[int] = None, seed: int = 0):
    """Translations of ``mask`` that no longer coincide with it.

    Yields ``(dx, dy, shifted_mask)``. A translation is dropped if it overlaps the
    original (Jaccard > ``max_jaccard``) or pushes more than ``1-min_area_kept``
    of the territory off the grid (a truncated shape is no longer "the same
    boundary, elsewhere").
    """
    mask = np.asarray(mask, dtype=bool)
    area = mask.sum()
    if area == 0:
        return []
    nx, ny = mask.shape
    out = []
    for dx in range(-nx + 1, nx):
        for dy in range(-ny + 1, ny):
            if dx == 0 and dy == 0:
                continue
            m = _shifted(mask, dx, dy)
            if m.sum() < min_area_kept * area:
                continue
            union = np.sum(m | mask)
            if union and np.sum(m & mask) / union > max_jaccard:
                continue
            out.append((dx, dy, m))
    if max_n is not None and len(out) > max_n:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(out), size=max_n, replace=False))
        out = [out[i] for i in keep]
    return out


def _grid_index(tmap: TerritoryMap, x, y):
    ix = np.searchsorted(tmap.x_edges, x, side="right") - 1
    iy = np.searchsorted(tmap.y_edges, y, side="right") - 1
    ix = np.where(x == tmap.x_edges[-1], len(tmap.x_edges) - 2, ix)
    iy = np.where(y == tmap.y_edges[-1], len(tmap.y_edges) - 2, iy)
    ok = (np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (iy >= 0)
          & (ix < len(tmap.x_edges) - 1) & (iy < len(tmap.y_edges) - 1))
    return np.clip(ix, 0, len(tmap.x_edges) - 2), np.clip(iy, 0, len(tmap.y_edges) - 2), ok


# ---------------------------------------------------------------------------
# 1. Boundary-straddling dissimilarity
# ---------------------------------------------------------------------------

def _coarsen(mask: np.ndarray, k: int) -> np.ndarray:
    nx, ny = (mask.shape[0] // k) * k, (mask.shape[1] // k) * k
    m = mask[:nx, :ny].reshape(nx // k, k, ny // k, k).mean(axis=(1, 3))
    return m >= 0.5


def boundary_straddle_test(
    rates: np.ndarray, x: np.ndarray, y: np.ndarray, tmap: TerritoryMap, animal: str, *,
    dt: float = 0.5,
    coarsen: int = 2,
    band: float = 20.0,
    sep_bins: Sequence[int] = (1, 2, 3),
    block_sec: float = 60.0,
    purge_sec: float = 5.0,
    min_bins_per_half: int = 10,
    detrend_sec: Optional[float] = 120.0,
    max_jaccard: float = 0.5,
    n_displace_max: int = 300,
    min_pairs: int = 3,
    seed: int = 0,
) -> Dict[str, object]:
    """Excess dissimilarity of population vectors across the territory boundary.

    Statistic: for each separation stratum, mean cross-validated squared PV distance
    of straddling pairs minus same-side pairs (both bins within ``band`` of the
    boundary), averaged over strata weighted by ``n_s*n_o/(n_s+n_o)``. Positive =
    straddling pairs are more different than their separation explains.
    """
    rates = np.asarray(rates, dtype=float)
    n_t = rates.shape[0]
    if not (len(x) == len(y) == n_t):
        raise ValueError("rates, x, y must have the same number of rows")

    Z = preprocess_rates(rates, dt, detrend_sec)
    own_fine = tmap.owner_mask(animal)
    own_c = _coarsen(own_fine, coarsen)
    nxc, nyc = own_c.shape
    xe = tmap.x_edges[: nxc * coarsen + 1 : coarsen]
    ye = tmap.y_edges[: nyc * coarsen + 1 : coarsen]
    sampling = (float(np.mean(np.diff(xe))), float(np.mean(np.diff(ye))))

    ix = np.searchsorted(xe, x, side="right") - 1
    iy = np.searchsorted(ye, y, side="right") - 1
    ix = np.where(x == xe[-1], nxc - 1, ix)
    iy = np.where(y == ye[-1], nyc - 1, iy)
    in_grid = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (iy >= 0) & (ix < nxc) & (iy < nyc)

    # interleaved time blocks, with a purge at block edges so the halves are independent
    t = np.arange(n_t) * dt
    blk = np.floor(t / block_sec).astype(int)
    pos_in_blk = t - blk * block_sec
    usable = in_grid & (pos_in_blk >= purge_sec) & (pos_in_blk <= block_sec - purge_sec)
    half = blk % 2

    cell = ix * nyc + iy
    n_cells = Z.shape[1]
    sums = [np.zeros((nxc * nyc, n_cells)), np.zeros((nxc * nyc, n_cells))]
    cnts = [np.zeros(nxc * nyc), np.zeros(nxc * nyc)]
    for h in (0, 1):
        sel = usable & (half == h)
        np.add.at(sums[h], cell[sel], Z[sel])
        np.add.at(cnts[h], cell[sel], 1)
    valid = (cnts[0] >= min_bins_per_half) & (cnts[1] >= min_bins_per_half)
    if valid.sum() < 6:
        return _straddle_fail("insufficient_occupancy", int(valid.sum()))

    V = np.where(valid)[0]
    A = sums[0][V] / cnts[0][V, None]
    B = sums[1][V] / cnts[1][V, None]
    G = A @ B.T
    d = np.diag(G)
    cvd = d[:, None] + d[None, :] - G - G.T                   # unbiased ||mu_i - mu_j||^2

    vx, vy = V // nyc, V % nyc
    cx = (xe[vx] + xe[vx + 1]) / 2
    cy = (ye[vy] + ye[vy + 1]) / 2
    mean_bin = float(np.mean(sampling))
    iu, ju = np.triu_indices(len(V), k=1)
    sep = np.hypot(cx[iu] - cx[ju], cy[iu] - cy[ju]) / mean_bin
    sep_r = np.rint(sep).astype(int)
    keep = np.isin(sep_r, list(sep_bins)) & (np.abs(sep - sep_r) < 0.45)
    iu, ju, sep_r = iu[keep], ju[keep], sep_r[keep]
    pair_cvd = cvd[iu, ju]

    def stat_for(mask_c: np.ndarray) -> Tuple[float, int, int]:
        sd = signed_distance_from_mask(mask_c, sampling)
        in_band = np.abs(sd[vx, vy]) <= band
        own = mask_c[vx, vy]
        ok = in_band[iu] & in_band[ju]
        strad = own[iu] != own[ju]
        num, wsum, n_s, n_o = 0.0, 0.0, 0, 0
        for s in sep_bins:
            m = ok & (sep_r == s)
            a, b = m & strad, m & ~strad
            na, nb = int(a.sum()), int(b.sum())
            if na < min_pairs or nb < min_pairs:
                continue
            w = na * nb / (na + nb)
            num += w * (pair_cvd[a].mean() - pair_cvd[b].mean())
            wsum += w
            n_s += na
            n_o += nb
        return (num / wsum if wsum > 0 else np.nan), n_s, n_o

    obs, n_s, n_o = stat_for(own_c)
    if not np.isfinite(obs):
        return _straddle_fail("insufficient_pairs", int(valid.sum()))

    null = []
    for _, _, m in displaced_masks(own_c, max_jaccard=max_jaccard, max_n=n_displace_max, seed=seed):
        v, _, _ = stat_for(m)
        if np.isfinite(v):
            null.append(v)
    null = np.asarray(null)
    if len(null) < 5:
        return _straddle_fail("insufficient_displacements", int(valid.sum()), obs)
    p = float((1 + np.sum(null >= obs)) / (len(null) + 1))
    return dict(status="ok", statistic=float(obs), p_value=p, p_floor=1.0 / (len(null) + 1),
                n_displacements=int(len(null)), null=null, n_pairs_straddle=n_s,
                n_pairs_same_side=n_o, n_valid_bins=int(valid.sum()),
                null_z=float((obs - null.mean()) / (null.std() + 1e-12)))


def _straddle_fail(status: str, n_valid: int, obs: float = np.nan) -> Dict[str, object]:
    return dict(status=status, statistic=obs, p_value=np.nan, p_floor=np.nan, n_displacements=0,
                null=np.array([]), n_pairs_straddle=0, n_pairs_same_side=0, n_valid_bins=n_valid,
                null_z=np.nan)


# ---------------------------------------------------------------------------
# Decoder machinery shared by tests 2 and 3
# ---------------------------------------------------------------------------

def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accs = [np.mean(y_pred[y_true == c] == c) for c in (0, 1) if np.any(y_true == c)]
    return float(np.mean(accs)) if accs else np.nan


def _cv_balanced_accuracy(X: np.ndarray, y: np.ndarray, *, n_folds: int, purge: int,
                          shrinkage: float = 0.3) -> float:
    """Blocked-CV balanced accuracy of a shrinkage-LDA decoder (z-scored in-fold)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    preds = np.full(len(y), -1)
    for train, test in blocked_folds(len(y), n_folds, purge):
        if len(np.unique(y[train])) < 2 or len(train) < 10:
            return np.nan
        mu, sd = X[train].mean(axis=0), X[train].std(axis=0)
        sd[sd < 1e-9] = 1.0
        clf = LDA(solver="lsqr", shrinkage=shrinkage, priors=[0.5, 0.5])
        clf.fit((X[train] - mu) / sd, y[train])
        preds[test] = clf.predict((X[test] - mu) / sd)
    ok = preds >= 0
    return _balanced_accuracy(y[ok], preds[ok])


def _position_features(x, y, extra: Optional[np.ndarray], grid: int) -> np.ndarray:
    """Smooth-position basis (grid x grid Gaussian RBF, columns with no variance dropped)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x0, x1 = np.nanmin(x), max(np.nanmax(x), np.nanmin(x) + 1.0)
    y0, y1 = np.nanmin(y), max(np.nanmax(y), np.nanmin(y) + 1.0)
    cx, cy = np.linspace(x0, x1, grid), np.linspace(y0, y1, grid)
    gx = np.exp(-0.5 * ((x[:, None] - cx) / ((x1 - x0) / (grid - 1))) ** 2)
    gy = np.exp(-0.5 * ((y[:, None] - cy) / ((y1 - y0) / (grid - 1))) ** 2)
    sp = (gx[:, :, None] * gy[:, None, :]).reshape(len(x), -1)
    sp = sp[:, sp.std(axis=0) > 1e-9]
    return sp if extra is None else np.column_stack([sp, extra])


# ---------------------------------------------------------------------------
# 2. Territory decoder gain beyond smooth position
# ---------------------------------------------------------------------------

def territory_decoder_gain(
    rates: np.ndarray, x: np.ndarray, y: np.ndarray, speed: np.ndarray,
    tmap: TerritoryMap, animal: str, *,
    dt: float = 0.5, band: float = 15.0, n_folds: int = 5, purge_bins: int = 20,
    spatial_grid: int = 6, detrend_sec: Optional[float] = 120.0, min_rows_per_class: int = 40,
    max_jaccard: float = 0.5, n_displace_max: int = 100, seed: int = 0,
) -> Dict[str, object]:
    """Gain in own-vs-foreign balanced accuracy from neural activity, near the boundary.

    gain = BA(position + neural) - BA(position), both blocked-CV, rows restricted to
    ``|signed distance| < band``. The p-value compares the real gain with the gain
    obtained for displaced boundaries (see module docstring for why chance is not
    the right reference).
    """
    rates = np.asarray(rates, dtype=float)
    x, y, speed = (np.asarray(v, dtype=float) for v in (x, y, speed))
    Z = preprocess_rates(rates, dt, detrend_sec)
    ix, iy, in_grid = _grid_index(tmap, x, y)
    sampling = tmap.bin_size
    P = _position_features(x, y, speed[:, None], spatial_grid)

    def gain_for(mask: np.ndarray) -> Tuple[float, float, float, int]:
        sd = signed_distance_from_mask(mask, sampling)
        sd_row = np.where(in_grid, sd[ix, iy], np.nan)
        rows = np.where(in_grid & (np.abs(sd_row) < band))[0]
        lab = mask[ix[rows], iy[rows]].astype(int)
        if min(np.sum(lab == 0), np.sum(lab == 1)) < min_rows_per_class:
            return np.nan, np.nan, np.nan, len(rows)
        ba_pos = _cv_balanced_accuracy(P[rows], lab, n_folds=n_folds, purge=purge_bins)
        ba_all = _cv_balanced_accuracy(np.column_stack([P[rows], Z[rows]]), lab,
                                       n_folds=n_folds, purge=purge_bins)
        return ba_all - ba_pos, ba_pos, ba_all, len(rows)

    own = tmap.owner_mask(animal)
    obs, ba_pos, ba_all, n_rows = gain_for(own)
    if not np.isfinite(obs):
        return dict(status="insufficient_occupancy", gain=np.nan, p_value=np.nan, n_rows=n_rows,
                    ba_position=np.nan, ba_position_plus_neural=np.nan, null=np.array([]),
                    n_displacements=0, p_floor=np.nan)
    null = []
    for _, _, m in displaced_masks(own, max_jaccard=max_jaccard, max_n=n_displace_max, seed=seed):
        g = gain_for(m)[0]
        if np.isfinite(g):
            null.append(g)
    null = np.asarray(null)
    if len(null) < 5:
        return dict(status="insufficient_displacements", gain=obs, p_value=np.nan, n_rows=n_rows,
                    ba_position=ba_pos, ba_position_plus_neural=ba_all, null=null,
                    n_displacements=len(null), p_floor=np.nan)
    return dict(status="ok", gain=float(obs), p_value=float((1 + np.sum(null >= obs)) / (len(null) + 1)),
                p_floor=1.0 / (len(null) + 1), ba_position=float(ba_pos),
                ba_position_plus_neural=float(ba_all), n_rows=int(n_rows), null=null,
                n_displacements=int(len(null)))


# ---------------------------------------------------------------------------
# 3. Owner presence at (nearly) fixed position
# ---------------------------------------------------------------------------

def owner_presence_decoding(
    rates: np.ndarray, x: np.ndarray, y: np.ndarray,
    owner: np.ndarray, owner_present: np.ndarray, owner_dist: np.ndarray, *,
    dt: float = 0.5, n_folds: int = 5, purge_bins: int = 20, spatial_grid: int = 6,
    detrend_sec: Optional[float] = 120.0, min_rows_per_class: int = 40,
    n_shifts: int = 199, min_shift_frac: float = 0.1, seed: int = 0,
) -> Dict[str, object]:
    """Can neural activity tell whether the territory's owner is home?

    Rows are time bins in which the focal rat is inside a neighbour's territory
    (``owner`` is that neighbour's name, ``None`` elsewhere) and ``owner_present``
    (bool) says whether that neighbour is currently inside its own territory.
    Baseline features: smooth position, owner identity, focal-owner distance.
    Statistic: BA(baseline + neural) - BA(baseline). Null: circularly shift the
    presence labels within each owner's rows, which keeps their autocorrelation
    and removes their link to both neural activity and the distance nuisance.
    """
    rates = np.asarray(rates, dtype=float)
    owner = np.asarray(owner, dtype=object)
    present = np.asarray(owner_present, dtype=float)
    dist = np.asarray(owner_dist, dtype=float)
    Z = preprocess_rates(rates, dt, detrend_sec)

    use = np.array([o is not None for o in owner]) & np.isfinite(present) & np.isfinite(dist)
    rows = np.where(use)[0]
    if rows.size == 0:
        return _presence_fail("insufficient_occupancy", 0)
    names = sorted(set(owner[rows]))
    lab = present[rows].astype(int)
    if min(np.sum(lab == 0), np.sum(lab == 1)) < min_rows_per_class:
        return _presence_fail("insufficient_occupancy", len(rows))

    onehot = np.column_stack([(owner[rows] == n).astype(float) for n in names])
    base = np.column_stack([_position_features(x[rows], y[rows], None, spatial_grid), onehot,
                            dist[rows]])
    full = np.column_stack([base, Z[rows]])

    def gain(labels: np.ndarray) -> float:
        return (_cv_balanced_accuracy(full, labels, n_folds=n_folds, purge=purge_bins)
                - _cv_balanced_accuracy(base, labels, n_folds=n_folds, purge=purge_bins))

    obs = gain(lab)
    ba_base = _cv_balanced_accuracy(base, lab, n_folds=n_folds, purge=purge_bins)
    if not np.isfinite(obs):
        return _presence_fail("insufficient_occupancy", len(rows))

    rng = np.random.default_rng(seed)
    groups = [np.where(owner[rows] == n)[0] for n in names]
    null = np.empty(n_shifts)
    for k in range(n_shifts):
        shifted = lab.copy()
        for g in groups:
            if len(g) < 2:
                continue
            lo = max(1, int(min_shift_frac * len(g)))
            shifted[g] = np.roll(lab[g], int(rng.integers(lo, len(g) - lo + 1)))
        null[k] = gain(shifted)
    null = null[np.isfinite(null)]
    p = float((1 + np.sum(null >= obs)) / (len(null) + 1))
    return dict(status="ok", gain=float(obs), p_value=p, p_floor=1.0 / (len(null) + 1),
                ba_baseline=float(ba_base), n_rows=int(len(rows)), owners=names,
                n_present=int(lab.sum()), n_absent=int((1 - lab).sum()), null=null,
                n_shifts=int(len(null)))


def _presence_fail(status: str, n_rows: int) -> Dict[str, object]:
    return dict(status=status, gain=np.nan, p_value=np.nan, p_floor=np.nan, ba_baseline=np.nan,
                n_rows=n_rows, owners=[], n_present=0, n_absent=0, null=np.array([]), n_shifts=0)
