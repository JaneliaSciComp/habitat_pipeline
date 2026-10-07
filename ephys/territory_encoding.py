"""Single-cell test for a territory effect beyond place coding.

For one cell, asks whether adding a territory term to a Poisson GLM that already
holds place tuning, speed, partner proximity and slow rate drift explains the
spikes better.

Model (counts per bin, ``log(dt)`` offset)::

    baseline:  1 + RBF(x, y) + speed + partner_dist + RBF(time)
    full:      baseline + territory term

``effect='step'`` uses the own-territory indicator; ``effect='exclusivity'`` uses the
continuous exclusivity measure (``territory['exclusivity']``; the focal's share of
everyone's occupancy, which does not flip when two animals swap rank - the
winner-takes-all owner map did not replicate between chunks on real data);
``effect='gradient'`` uses the signed distance to the own-territory boundary (clipped at +/-``grad_clip``,
in coordinate units - the default is only meaningful for the simulator's 0-100
arena, so set it for real data, which is in pixels until ``pixels_per_cm`` is
re-measured for APT tracking).

Significance
------------
The territory covariate is spatially smooth and temporally autocorrelated, so a
plain chi-square LRT is anti-conservative for any cell with slow drift, and a
shuffled territory label is an unfair null because it breaks the covariate's
link to position. Instead the p-value is a **parametric bootstrap from the fitted
baseline** (which carries the cell's place tuning and drift but no territory
effect). Every bootstrap draw re-estimates the baseline before the efficient
score statistic for the territory term is computed, so the null accounts for
nuisance-parameter estimation. Overdispersion estimated from the baseline
Pearson residuals is reproduced with negative-binomial draws. The p-value uses
the add-one form ``(1 + k) / (n_boot + 1)``; it is NaN, never 1.0 or 0.0, when
the cell cannot be tested.

The bootstrap re-estimates the baseline with one Newton step from the generating
fit. The information matrix at that fit is the same for every draw, so all
draws are evaluated in a few matrix products instead of ``n_boot`` refits.

``spatial_grid=6`` is a calibration result, not a taste: on synthetic cells with a
narrow place field sitting on the territory boundary, a 5x5 basis cannot follow
the field, the misfit correlates with the own-territory indicator, and the
step test rejected 11-13% of true nulls at alpha=0.05 (bootstrap draws come from
the same misfit baseline, so they cannot see it). 6x6 gave 3.9%, 7x7 about 1%
(over-absorbing). Re-check if the arena aspect or field widths change materially.

A cross-validated held-out log-likelihood gain (blocked folds with a purge gap,
never shuffled) is reported as the effect-size measure; it is not used for the
p-value.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MIN_SPIKES = 50
MIN_SIDE_FRACTION = 0.03     # step: each of own / not-own needs at least this share of bins
ETA_CLIP = 20.0
BOOT_BATCH = 400


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def blocked_folds(n: int, n_folds: int, purge_bins: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Contiguous, unshuffled folds with a purge gap around each test block.

    Neighbouring bins share position, speed and rate, so a training bin next to
    a test block is nearly a copy of it. Training indices within ``purge_bins``
    of the test block (either side) are dropped.
    """
    if n_folds < 2 or n < n_folds:
        raise ValueError(f"need n >= n_folds >= 2, got n={n}, n_folds={n_folds}")
    idx = np.arange(n)
    for test in np.array_split(idx, n_folds):
        lo, hi = test[0], test[-1]
        train = idx[(idx < lo - purge_bins) | (idx > hi + purge_bins)]
        yield train, test


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------

def _rbf(values: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((values[:, None] - centers[None, :]) / width) ** 2)


def _baseline_design(x, y, speed, partner, t, *, grid: int, time_knot_sec: float) -> np.ndarray:
    """Intercept + spatial RBF + speed + partner distance + temporal RBF (columns z-scored)."""
    def span(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        return lo, (hi if hi > lo else lo + 1.0)
    (x0, x1), (y0, y1) = span(x), span(y)
    cx, cy = np.linspace(x0, x1, grid), np.linspace(y0, y1, grid)
    wx, wy = (x1 - x0) / (grid - 1), (y1 - y0) / (grid - 1)
    gx, gy = _rbf(x, cx, wx), _rbf(y, cy, wy)
    spatial = (gx[:, :, None] * gy[:, None, :]).reshape(len(x), -1)

    t0, t1 = float(t.min()), float(t.max())
    n_knots = max(2, int(np.ceil((t1 - t0) / time_knot_sec)) + 1)
    temporal = _rbf(t, np.linspace(t0, t1, n_knots), (t1 - t0) / (n_knots - 1) if n_knots > 1 else 1.0)

    cols = np.column_stack([spatial, speed, partner, temporal])
    sd = cols.std(axis=0)
    keep = sd > 1e-9                                    # drop constant columns (e.g. unvisited RBF)
    cols = (cols[:, keep] - cols[:, keep].mean(axis=0)) / sd[keep]
    return np.column_stack([np.ones(len(x)), cols])


def _irls(X: np.ndarray, y: np.ndarray, offset: np.ndarray, lam: np.ndarray,
          beta0: Optional[np.ndarray] = None, iters: int = 50) -> np.ndarray:
    """Ridge-penalised Poisson regression by Newton with step halving."""
    if beta0 is None:
        beta = np.zeros(X.shape[1])
        beta[0] = np.log(max(y.mean(), 1e-6)) - offset.mean()
    else:
        beta = beta0.copy()

    def obj(b):
        eta = np.clip(X @ b + offset, -ETA_CLIP, ETA_CLIP)
        return float(np.sum(y * eta - np.exp(eta)) - 0.5 * np.sum(lam * b * b))

    cur = obj(beta)
    for _ in range(iters):
        eta = np.clip(X @ beta + offset, -ETA_CLIP, ETA_CLIP)
        mu = np.exp(eta)
        grad = X.T @ (y - mu) - lam * beta
        H = (X * mu[:, None]).T @ X + np.diag(lam)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        t, improved = 1.0, False
        for _ in range(12):
            new = obj(beta + t * step)
            if new >= cur - 1e-10:
                improved = True
                break
            t /= 2
        if not improved:
            break
        beta = beta + t * step
        done = abs(new - cur) < 1e-9 * (1 + abs(cur))
        cur = new
        if done:
            break
    return beta


def _loglik(X, y, offset, beta) -> float:
    eta = np.clip(X @ beta + offset, -ETA_CLIP, ETA_CLIP)
    return float(np.sum(y * eta - np.exp(eta)))


def _heldout_loglik(X_test, y_test, off_test, beta, X_train, off_train, *, margin: float = 1.0) -> float:
    """Held-out Poisson log-likelihood with predictions confined to the range the fit produced.

    Coefficients fitted on training blocks are unconstrained for basis functions the
    training data never visited, so a held-out block in a new region can extrapolate to
    absurd rates (observed: held-out gains of 1e6 per spike). Clipping the held-out linear
    predictor to the training fit's own range +/- ``margin`` removes that failure without
    changing what the model says inside the region it was fitted on.
    """
    eta_train = X_train @ beta + off_train
    lo, hi = float(eta_train.min()) - margin, float(eta_train.max()) + margin
    eta = np.clip(X_test @ beta + off_test, lo, hi)
    return float(np.sum(y_test * eta - np.exp(eta)))


# ---------------------------------------------------------------------------
# Public test
# ---------------------------------------------------------------------------

def _result(status: str, n_spikes: int, n_boot: int, **extra) -> Dict[str, object]:
    out = dict(status=status, p_value=np.nan, lrt_stat=np.nan, delta_ll_cv_per_spike=np.nan,
               effect_estimate=np.nan, n_spikes=int(n_spikes), n_boot=int(n_boot))
    out.update(extra)
    return out


def fit_territory_effect(
    counts: np.ndarray,
    covariates: Dict[str, np.ndarray],
    territory: Dict[str, np.ndarray],
    *,
    effect: str = "step",
    dt: float = 0.5,
    n_boot: int = 199,
    n_folds: int = 5,
    purge_bins: int = 20,
    band: Optional[float] = None,
    grad_clip: float = 30.0,
    spatial_grid: int = 6,
    time_knot_sec: float = 120.0,
    ridge: float = 1.0,
    cv_ridge: float = 30.0,
    min_spikes: int = MIN_SPIKES,
    seed: int = 0,
) -> Dict[str, object]:
    """Test one cell for a territory effect beyond place, speed, proximity and drift.

    Returns a dict with ``status`` (``ok`` / ``insufficient_spikes`` /
    ``insufficient_occupancy``), ``p_value`` (add-one parametric bootstrap),
    ``lrt_stat`` (in-sample, for reference only), ``delta_ll_cv_per_spike``
    (held-out log-likelihood gain of the territory term, per spike),
    ``effect_estimate`` (log rate ratio own vs not-own for ``step``; log-rate per
    coordinate unit for ``gradient``; log-rate per unit exclusivity, i.e. from 0 to 1, for
    ``exclusivity``), ``dispersion``, ``n_spikes``, ``n_boot``.
    """
    if effect not in ("step", "gradient", "exclusivity"):
        raise ValueError(f"effect must be 'step', 'gradient' or 'exclusivity', got {effect!r}")

    y = np.asarray(counts, dtype=float)
    n_all = len(y)
    for k in ("x", "y", "speed", "partner_dist"):
        if k not in covariates or len(covariates[k]) != n_all:
            raise ValueError(f"covariates[{k!r}] missing or wrong length")
    sd = np.asarray(territory["signed_dist"], dtype=float) if "signed_dist" in territory         else np.full(n_all, np.nan)
    own = np.asarray(territory["own"], dtype=bool) if "own" in territory else np.zeros(n_all, bool)
    excl = np.asarray(territory["exclusivity"], dtype=float) if "exclusivity" in territory else None
    if len(sd) != n_all or len(own) != n_all or (excl is not None and len(excl) != n_all):
        raise ValueError("territory arrays must match counts length")
    if effect == "exclusivity" and excl is None:
        raise ValueError("effect='exclusivity' needs territory['exclusivity']")

    t = np.arange(n_all) * dt
    ok = np.ones(n_all, bool)
    for k in ("x", "y", "speed", "partner_dist"):
        ok &= np.isfinite(covariates[k])
    ok &= np.isfinite(excl) if effect == "exclusivity" else np.isfinite(sd)
    if band is not None:
        ok &= np.abs(sd) < band
    if ok.sum() < 4 * n_folds:
        return _result("insufficient_occupancy", y[ok].sum(), n_boot)

    y = y[ok]
    n_spikes = int(y.sum())
    if n_spikes < min_spikes:
        return _result("insufficient_spikes", n_spikes, n_boot)

    if effect == "step":
        cov = own[ok].astype(float)
        side = cov.mean()
        if min(side, 1 - side) < MIN_SIDE_FRACTION:
            return _result("insufficient_occupancy", n_spikes, n_boot)
    else:
        cov = excl[ok] if effect == "exclusivity" else np.clip(sd[ok], -grad_clip, grad_clip)
        if np.std(cov) < 1e-6:
            return _result("insufficient_occupancy", n_spikes, n_boot)
    cov_mean, cov_sd = cov.mean(), cov.std()
    cz = (cov - cov_mean) / cov_sd

    Xb = _baseline_design(covariates["x"][ok], covariates["y"][ok], covariates["speed"][ok],
                          covariates["partner_dist"][ok], t[ok],
                          grid=spatial_grid, time_knot_sec=time_knot_sec)
    Xf = np.column_stack([Xb, cz])
    off = np.full(len(y), np.log(dt))
    lam_b = np.r_[0.0, np.full(Xb.shape[1] - 1, ridge)]
    lam_f = np.r_[lam_b, 0.0]

    # --- fits on all rows --------------------------------------------------
    beta_b = _irls(Xb, y, off, lam_b)
    beta_f = _irls(Xf, y, off, lam_f, beta0=np.r_[beta_b, 0.0])
    lrt = max(2.0 * (_loglik(Xf, y, off, beta_f) - _loglik(Xb, y, off, beta_b)), 0.0)
    effect_est = float(beta_f[-1] / cov_sd)

    eta = np.clip(Xb @ beta_b + off, -ETA_CLIP, ETA_CLIP)
    mu = np.exp(eta)
    p_eff = Xb.shape[1]
    dispersion = float(max(1.0, np.sum((y - mu) ** 2 / np.maximum(mu, 1e-9)) / max(len(y) - p_eff, 1)))

    # --- parametric bootstrap of the efficient score -------------------------
    H = (Xb * mu[:, None]).T @ Xb + np.diag(lam_b)
    H_inv = np.linalg.inv(H)
    wc = Xb.T @ (mu * cz)
    i_eff = float(np.sum(mu * cz * cz) - wc @ H_inv @ wc)
    if i_eff <= 1e-9:
        return _result("insufficient_occupancy", n_spikes, n_boot)

    def score_stat(Y: np.ndarray) -> np.ndarray:
        """Efficient score statistic for each row of Y after one-step baseline refit."""
        g = (Y - mu) @ Xb - lam_b * beta_b                       # (m, p)
        delta = g @ H_inv                                         # (m, p) (H symmetric)
        mu1 = np.exp(np.clip(eta[None, :] + delta @ Xb.T, -ETA_CLIP, ETA_CLIP))
        u = (Y - mu1) @ cz
        return u * u / i_eff

    rng = np.random.default_rng(seed)
    r_nb = mu / (dispersion - 1.0) if dispersion > 1.05 else None
    null_parts = []
    for start in range(0, n_boot, BOOT_BATCH):          # batched to bound memory at large n_boot
        m = min(BOOT_BATCH, n_boot - start)
        if r_nb is not None:
            d = rng.negative_binomial(r_nb[None, :], (r_nb / (r_nb + mu))[None, :], size=(m, len(y)))
        else:
            d = rng.poisson(mu, size=(m, len(y)))
        null_parts.append(score_stat(d.astype(float)))
    null = np.concatenate(null_parts)
    obs = float(score_stat(y[None, :])[0])
    p_value = float((1 + np.sum(null >= obs)) / (n_boot + 1))

    # --- cross-validated effect size ---------------------------------------
    gain, held_spikes = 0.0, 0.0
    lam_bc = np.r_[0.0, np.full(Xb.shape[1] - 1, cv_ridge)]
    lam_fc = np.r_[lam_bc, 0.0]
    for train, test in blocked_folds(len(y), n_folds, purge_bins):
        if len(train) < 4 * Xf.shape[1] or y[train].sum() < 10:
            continue
        bb = _irls(Xb[train], y[train], off[train], lam_bc)
        bf = _irls(Xf[train], y[train], off[train], lam_fc, beta0=np.r_[bb, 0.0])
        gain += (_heldout_loglik(Xf[test], y[test], off[test], bf, Xf[train], off[train])
                 - _heldout_loglik(Xb[test], y[test], off[test], bb, Xb[train], off[train]))
        held_spikes += y[test].sum()
    delta_cv = float(gain / held_spikes) if held_spikes > 0 else np.nan

    return dict(status="ok", p_value=p_value, lrt_stat=float(lrt), delta_ll_cv_per_spike=delta_cv,
                effect_estimate=effect_est, dispersion=dispersion, n_spikes=n_spikes,
                n_boot=int(n_boot), n_bins=int(len(y)))


def position_coding_gain(
    counts: np.ndarray, covariates: Dict[str, np.ndarray], *, dt: float = 0.5,
    n_folds: int = 5, purge_bins: int = 20, spatial_grid: int = 6, time_knot_sec: float = 120.0,
    ridge: float = 30.0, min_spikes: int = MIN_SPIKES,
) -> Dict[str, object]:
    """Held-out gain of place + speed + proximity over a time-only model (positive control).

    A territory analysis is only interpretable where position coding is
    detectable at all. Returns ``delta_ll_cv_per_spike`` (blocked folds, purge
    gap); NaN with a status when the cell has too few spikes.
    """
    y = np.asarray(counts, dtype=float)
    ok = np.ones(len(y), bool)
    for k in ("x", "y", "speed", "partner_dist"):
        ok &= np.isfinite(covariates[k])
    t = np.arange(len(y)) * dt
    y = y[ok]
    if y.sum() < min_spikes or len(y) < 4 * n_folds:
        return dict(status="insufficient_spikes", delta_ll_cv_per_spike=np.nan, n_spikes=int(y.sum()))
    full = _baseline_design(covariates["x"][ok], covariates["y"][ok], covariates["speed"][ok],
                            covariates["partner_dist"][ok], t[ok], grid=spatial_grid,
                            time_knot_sec=time_knot_sec)
    zeros = np.zeros(len(y))
    base = _baseline_design(zeros, zeros, zeros, zeros, t[ok], grid=spatial_grid,
                            time_knot_sec=time_knot_sec)          # constant columns dropped
    off = np.full(len(y), np.log(dt))
    lam_f = np.r_[0.0, np.full(full.shape[1] - 1, ridge)]
    lam_b = np.r_[0.0, np.full(base.shape[1] - 1, ridge)]
    gain, held = 0.0, 0.0
    for train, test in blocked_folds(len(y), n_folds, purge_bins):
        if y[train].sum() < 10:
            continue
        bb = _irls(base[train], y[train], off[train], lam_b)
        bf = _irls(full[train], y[train], off[train], lam_f)
        gain += (_heldout_loglik(full[test], y[test], off[test], bf, full[train], off[train])
                 - _heldout_loglik(base[test], y[test], off[test], bb, base[train], off[train]))
        held += y[test].sum()
    return dict(status="ok", delta_ll_cv_per_spike=float(gain / held) if held > 0 else np.nan,
                n_spikes=int(y.sum()))
