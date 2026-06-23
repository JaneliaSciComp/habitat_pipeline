"""
Partner-Distance Decoding (regression).

Decode the continuous Euclidean distance between a focal (implanted) animal
and a *specific* partner animal from the focal animal's neural activity, at
two levels:

* **single cell** — a 1-D distance tuning curve (mean firing rate vs binned
  distance, the distance-space analogue of a place field) plus a
  cross-validated single-feature ridge regression / Pearson correlation per
  cell, ranked into "distance cells";
* **population** — a multivariate ridge regression predicting distance from
  all cells' binned rates, reported as cross-validated R² and RMSE.

This is the *regression* analogue of :mod:`ephys.decode_location` (which does
continuous 2-D position decoding with a Bayesian decoder). Distance is a
continuous variable defined over the whole session, so the analysis is
**whole-session, time-binned**: firing rates and distance are binned onto the
same ephys-second grid and regressed.

Only the **focal** animal needs ephys; the partner contributes a tracking
trajectory only. The data path therefore takes a plain
:class:`~ingestion.kilosort_data_import.KilosortData` for the focal animal and a
session-level :class:`~video.tracking_import.VideoTrackingData` (which already
contains every animal), rather than a ``MultiAnimalSession`` (which would demand
ephys for the partner too).

Confound control
----------------
Distance to a partner is correlated with the focal animal's *own* speed and
position, so a cell can look "distance-tuned" while really coding self-motion.
Every score is therefore reported both raw and as a **partial R²** — the
neural variance explained *beyond* focal speed and focal (x, y). See
``single_cell_distance_scores`` / ``population_distance_regression``.

Statistical notes
------------------
* Distance and firing rates are strongly autocorrelated, so cross-validation
  uses **contiguous** ``KFold(shuffle=False)`` folds (shuffled folds would
  leak adjacent bins between train and test). This mirrors
  ``ephys.inter_brain_dynamics._fit_r2``.
* The null breaks the rate↔distance pairing by **circular-shifting distance**
  (preserving each stream's autocorrelation), mirroring
  ``ephys.decode_location._null_position_decode`` and
  ``ephys.inter_brain_dynamics.shuffle_null_subspace``.

Units
-----
Distances (and RMSE) are in **cm** when the cohort config sets
``pixels_per_cm``, otherwise in **pixels** (``units`` field). Tracking is
pulled through :func:`video.tracking_import.resolve_tracking_on_ephys_clock`,
the single canonical place where the pixels→cm scaling and tracking↔ephys clock
conversion happen.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

CLASS_LABEL = "partner_distance"
ANALYSIS_TITLE = "Partner-Distance Decoding"


# ---------------------------------------------------------------------------
# Core cross-validated regression (generalizes inter_brain_dynamics._fit_r2)
# ---------------------------------------------------------------------------

def _cv_regress(X: np.ndarray, y: np.ndarray, *,
                alpha: float = 1.0, cv_folds: int = 5) -> Dict:
    """Contiguous-fold ridge regression of ``y`` on ``X``.

    Z-scores ``X`` using *training-fold* statistics only (no leakage),
    fits :class:`sklearn.linear_model.Ridge`, and accumulates out-of-fold
    predictions. ``KFold(shuffle=False)`` keeps folds contiguous — essential
    for autocorrelated time series. Falls back to in-sample fit when there
    are too few samples for CV.

    Returns ``{cv_r2, r2_per_fold, rmse, y_pred}`` where ``cv_r2`` is the
    mean of per-fold R² (matching ``_fit_r2``) and ``rmse`` is computed from
    the concatenated out-of-fold predictions.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    y_pred = np.full(n, np.nan, dtype=np.float64)

    if cv_folds and cv_folds > 1 and n >= cv_folds + 2:
        kf = KFold(n_splits=cv_folds, shuffle=False)
        scores = []
        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            mu = X_tr.mean(axis=0)
            sd = X_tr.std(axis=0, ddof=1)
            sd = np.where(sd > 0, sd, 1.0)
            X_tr_z = (X_tr - mu) / sd
            X_te_z = (X_te - mu) / sd
            try:
                model = Ridge(alpha=alpha).fit(X_tr_z, y[tr_idx])
                y_pred[te_idx] = model.predict(X_te_z)
                scores.append(model.score(X_te_z, y[te_idx]))
            except Exception:  # pragma: no cover — defensive
                scores.append(np.nan)
        cv_r2 = float(np.nanmean(scores)) if scores else float("nan")
        r2_per_fold = [float(s) for s in scores]
    else:
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        Xz = (X - mu) / sd
        model = Ridge(alpha=alpha).fit(Xz, y)
        y_pred = model.predict(Xz)
        cv_r2 = float(model.score(Xz, y))
        r2_per_fold = [cv_r2]

    valid = np.isfinite(y_pred)
    rmse = (float(np.sqrt(np.mean((y[valid] - y_pred[valid]) ** 2)))
            if valid.any() else float("nan"))
    return {"cv_r2": cv_r2, "r2_per_fold": r2_per_fold, "rmse": rmse, "y_pred": y_pred}


def _regression_diagnostics(y_true: np.ndarray, y_pred: np.ndarray,
                            r2_per_fold: Optional[Sequence[float]]) -> Dict:
    """Robust scoring of an out-of-fold prediction (no model fitting).

    ``cv_r2`` elsewhere is the *mean of per-fold* ``model.score()``, which is
    fragile on strongly autocorrelated, non-stationary distance with contiguous
    blocked folds: a test block can occupy a distance regime absent from
    training, and R² there is measured against that block's *own* mean, so one
    fold can plunge far negative and drag the mean down. The **pooled** R² —
    ``r2_score`` computed once on the concatenated out-of-fold predictions
    against the *global* mean — is far more stable, and the per-fold spread
    (``r2_fold_min``/``max``/``std``) is itself the non-stationarity diagnostic.

    Returns ``{pooled_r2, pearson_r, r2_fold_mean, r2_fold_std, r2_fold_min,
    r2_fold_max}``.
    """
    from sklearn.metrics import r2_score

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() >= 2 and np.std(y_true[m]) > 0:
        pooled_r2 = float(r2_score(y_true[m], y_pred[m]))
        pearson_r = (float(np.corrcoef(y_true[m], y_pred[m])[0, 1])
                     if np.std(y_pred[m]) > 0 else float("nan"))
    else:
        pooled_r2 = float("nan")
        pearson_r = float("nan")

    folds = (np.asarray(list(r2_per_fold), dtype=np.float64)
             if r2_per_fold is not None else np.empty(0))
    folds = folds[np.isfinite(folds)]
    return {
        "pooled_r2": pooled_r2,
        "pearson_r": pearson_r,
        "r2_fold_mean": float(np.mean(folds)) if folds.size else float("nan"),
        "r2_fold_std": float(np.std(folds)) if folds.size else float("nan"),
        "r2_fold_min": float(np.min(folds)) if folds.size else float("nan"),
        "r2_fold_max": float(np.max(folds)) if folds.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Single-cell: 1-D distance tuning curves
# ---------------------------------------------------------------------------

def compute_distance_tuning(firing_rates: np.ndarray, distance: np.ndarray, *,
                            n_distance_bins: int = 15,
                            smoothing_sigma: float = 1.0,
                            dist_edges: Optional[np.ndarray] = None) -> Dict:
    """Occupancy-normalized mean firing rate per distance bin, per cell.

    The 1-D, distance-space analogue of
    ``ephys.decode_location._build_tuning_curves``.

    Parameters
    ----------
    firing_rates : (n_bins, n_cells) or (n_bins,)
    distance : (n_bins,)
    n_distance_bins : int
        Number of distance bins (ignored if ``dist_edges`` is given).
    smoothing_sigma : float
        Gaussian smoothing along the distance axis, in distance-bin units
        (0 = none).
    dist_edges : np.ndarray, optional
        Explicit bin edges; otherwise linspace over the observed range.

    Returns
    -------
    dict with ``tuning`` (n_distance_bins, n_cells), ``occupancy``
    (n_distance_bins,), ``dist_edges``, ``dist_centers``.
    """
    from scipy.ndimage import gaussian_filter1d

    fr = np.asarray(firing_rates, dtype=np.float64)
    if fr.ndim == 1:
        fr = fr[:, None]
    d = np.asarray(distance, dtype=np.float64)

    if dist_edges is None:
        finite_d = d[np.isfinite(d)]
        lo = float(np.min(finite_d)) if finite_d.size else 0.0
        hi = float(np.max(finite_d)) if finite_d.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        dist_edges = np.linspace(lo, hi, n_distance_bins + 1)
    else:
        dist_edges = np.asarray(dist_edges, dtype=np.float64)
        n_distance_bins = len(dist_edges) - 1
    dist_centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])

    n_cells = fr.shape[1]
    idx = np.clip(np.digitize(d, dist_edges) - 1, 0, n_distance_bins - 1)
    finite = np.isfinite(d) & np.all(np.isfinite(fr), axis=1)

    occupancy = np.zeros(n_distance_bins, dtype=np.float64)
    rate_sum = np.zeros((n_distance_bins, n_cells), dtype=np.float64)
    np.add.at(occupancy, idx[finite], 1.0)
    np.add.at(rate_sum, idx[finite], fr[finite])

    with np.errstate(divide="ignore", invalid="ignore"):
        tuning = rate_sum / occupancy[:, None]
    tuning[~np.isfinite(tuning)] = 0.0

    if smoothing_sigma and smoothing_sigma > 0:
        for c in range(n_cells):
            tuning[:, c] = gaussian_filter1d(tuning[:, c], sigma=smoothing_sigma)

    return {"tuning": tuning, "occupancy": occupancy,
            "dist_edges": dist_edges, "dist_centers": dist_centers}


def single_cell_distance_scores(firing_rates: np.ndarray, distance: np.ndarray, *,
                                alpha: float = 1.0, cv_folds: int = 5,
                                nuisance: Optional[np.ndarray] = None) -> Dict:
    """Per-cell CV R², Pearson r, and (if ``nuisance`` given) partial R².

    The partial R² for cell ``j`` is ``R²(nuisance + cell) − R²(nuisance)`` —
    the variance-partition idea from
    ``ephys.inter_brain_dynamics.regress_shared_on_behavior``. It quantifies
    distance information in the cell *beyond* what the focal self-motion
    covariates already explain. Cells are ranked by partial R² when nuisance
    is provided, else by raw R².

    Returns ``{r2_per_cell, pearson_r_per_cell, r2_partial_per_cell,
    cell_ranking}`` (``r2_partial_per_cell`` is ``None`` when no nuisance).
    """
    fr = np.asarray(firing_rates, dtype=np.float64)
    if fr.ndim == 1:
        fr = fr[:, None]
    d = np.asarray(distance, dtype=np.float64)
    n_cells = fr.shape[1]

    r2 = np.full(n_cells, np.nan, dtype=np.float64)
    pearson = np.full(n_cells, np.nan, dtype=np.float64)
    partial = None
    r2_nuis = None
    if nuisance is not None:
        nuisance = np.asarray(nuisance, dtype=np.float64)
        if nuisance.ndim == 1:
            nuisance = nuisance[:, None]
        partial = np.full(n_cells, np.nan, dtype=np.float64)
        r2_nuis = _cv_regress(nuisance, d, alpha=alpha, cv_folds=cv_folds)["cv_r2"]

    d_std = np.std(d)
    for j in range(n_cells):
        cell = fr[:, [j]]
        r2[j] = _cv_regress(cell, d, alpha=alpha, cv_folds=cv_folds)["cv_r2"]
        if d_std > 0 and np.std(fr[:, j]) > 0:
            pearson[j] = float(np.corrcoef(fr[:, j], d)[0, 1])
        if nuisance is not None:
            both = np.hstack([nuisance, cell])
            r2_both = _cv_regress(both, d, alpha=alpha, cv_folds=cv_folds)["cv_r2"]
            partial[j] = r2_both - r2_nuis

    rank_key = partial if partial is not None else r2
    cell_ranking = np.argsort(
        np.where(np.isfinite(rank_key), rank_key, -np.inf)
    )[::-1]

    return {"r2_per_cell": r2, "pearson_r_per_cell": pearson,
            "r2_partial_per_cell": partial, "cell_ranking": cell_ranking}


# ---------------------------------------------------------------------------
# Population regression
# ---------------------------------------------------------------------------

def population_distance_regression(firing_rates: np.ndarray, distance: np.ndarray, *,
                                   alpha: float = 1.0, cv_folds: int = 5,
                                   nuisance: Optional[np.ndarray] = None) -> Dict:
    """Multivariate ridge regression of distance on all cells' rates.

    Returns ``{cv_r2, r2_per_fold, rmse, y_pred, cv_r2_partial}``.
    ``cv_r2_partial`` (``None`` without nuisance) is
    ``R²(nuisance + rates) − R²(nuisance)`` — the population's neural
    contribution beyond focal self-motion.
    """
    fr = np.asarray(firing_rates, dtype=np.float64)
    if fr.ndim == 1:
        fr = fr[:, None]
    d = np.asarray(distance, dtype=np.float64)

    res = _cv_regress(fr, d, alpha=alpha, cv_folds=cv_folds)
    out = {"cv_r2": res["cv_r2"], "r2_per_fold": res["r2_per_fold"],
           "rmse": res["rmse"], "y_pred": res["y_pred"], "cv_r2_partial": None,
           "diagnostics": _regression_diagnostics(d, res["y_pred"],
                                                   res["r2_per_fold"])}

    if nuisance is not None:
        nuisance = np.asarray(nuisance, dtype=np.float64)
        if nuisance.ndim == 1:
            nuisance = nuisance[:, None]
        r2_nuis = _cv_regress(nuisance, d, alpha=alpha, cv_folds=cv_folds)["cv_r2"]
        r2_both = _cv_regress(np.hstack([nuisance, fr]), d,
                              alpha=alpha, cv_folds=cv_folds)["cv_r2"]
        out["cv_r2_partial"] = r2_both - r2_nuis
    return out


# ---------------------------------------------------------------------------
# Null baseline (broken rate↔distance pairing)
# ---------------------------------------------------------------------------

def _null_distance_regression(firing_rates: np.ndarray, distance: np.ndarray, *,
                              alpha: float = 1.0, cv_folds: int = 5,
                              null: str = "shuffle", n_shuffles: int = 100,
                              seed: int = 0) -> Dict:
    """Population-R² null from a broken rate↔distance pairing.

    Regression analogue of ``ephys.decode_location._null_position_decode``.
    ``null='reverse'`` time-reverses distance once (deterministic, preserves
    autocorrelation and marginals); ``null='shuffle'`` applies ``n_shuffles``
    random circular shifts (bounds ``[0.1T, 0.9T]``, as in
    ``shuffle_null_subspace``) for a distribution. Distance is rolled, not the
    rates, so rate autocorrelation is preserved.
    """
    fr = np.asarray(firing_rates, dtype=np.float64)
    if fr.ndim == 1:
        fr = fr[:, None]
    d = np.asarray(distance, dtype=np.float64)
    n = len(d)

    if null == "reverse":
        variants = [d[::-1]]
    elif null == "shuffle":
        rng = np.random.default_rng(seed)
        lo, hi = max(1, n // 10), max(2, n - n // 10)
        shifts = rng.integers(lo, hi, size=n_shuffles)
        variants = [np.roll(d, int(s)) for s in shifts]
    else:
        raise ValueError(f"Unknown null method {null!r} (use 'reverse' or 'shuffle').")

    r2s, rmses = [], []
    for d_null in variants:
        r = _cv_regress(fr, d_null, alpha=alpha, cv_folds=cv_folds)
        r2s.append(r["cv_r2"])
        rmses.append(r["rmse"])
    r2s = np.asarray(r2s, dtype=np.float64)
    rmses = np.asarray(rmses, dtype=np.float64)

    return {"null_method": null,
            "null_r2": float(np.nanmean(r2s)),
            "null_r2_std": float(np.nanstd(r2s)),
            "null_r2_dist": r2s.tolist(),
            "null_rmse": float(np.nanmean(rmses)),
            "null_rmse_std": float(np.nanstd(rmses))}


# ---------------------------------------------------------------------------
# Bayesian 1-D distance decoder (the 1-D analogue of decode_location)
# ---------------------------------------------------------------------------

def _bayesian_decode_1d(rates: np.ndarray, tuning: np.ndarray,
                        occupancy: Optional[np.ndarray], time_bin_size: float,
                        dist_centers: np.ndarray, *,
                        use_occupancy_prior: bool = True,
                        estimate: str = "expected") -> tuple:
    """Poisson-likelihood Bayesian decode of distance from population rates.

    The 1-D, distance-space analogue of
    ``ephys.decode_location._bayesian_decode``. Unlike the linear ridge
    decoder, this exploits the *shape* of each cell's distance tuning curve, so
    a cell that fires maximally at an intermediate distance (a "distance field",
    near-zero linear correlation) still contributes.

    In log-space (dropping constant ``n_i!`` terms)::

        log P(d | spikes) ∝ Σ_i [ n_i·log(λ_i(d)·Δt) − λ_i(d)·Δt ] + log P(d)

    Parameters
    ----------
    rates : (n_time, n_cells) or (n_time,) — Hz
    tuning : (n_dist_bins, n_cells) — mean Hz per distance bin (from
        :func:`compute_distance_tuning`).
    occupancy : (n_dist_bins,) or None — time-bin counts, used as the prior.
    time_bin_size : float — seconds per time bin.
    dist_centers : (n_dist_bins,) — distance-bin centres.
    estimate : {'expected', 'map'} — posterior mean (continuous) or MAP bin.

    Returns ``(predicted_distance (n_time,), posterior (n_time, n_dist_bins))``.
    """
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim == 1:
        rates = rates[:, None]
    tuning = np.asarray(tuning, dtype=np.float64)          # (n_dist, n_cells)
    dist_centers = np.asarray(dist_centers, dtype=np.float64)
    n_dist = tuning.shape[0]

    lam_dt = tuning * time_bin_size                        # (n_dist, n_cells)
    log_lam_dt = np.log(np.clip(lam_dt, 1e-10, None))      # avoid log(0)
    neg_sum_lam = -lam_dt.sum(axis=1)                      # (n_dist,)

    if (use_occupancy_prior and occupancy is not None
            and np.sum(occupancy) > 0):
        prior = np.clip(occupancy / np.sum(occupancy), 1e-10, None)
    else:
        prior = np.full(n_dist, 1.0 / n_dist)
    log_prior = np.log(prior)

    spike_counts = rates * time_bin_size                   # (n_time, n_cells)
    log_post = spike_counts @ log_lam_dt.T                 # (n_time, n_dist)
    log_post += neg_sum_lam + log_prior

    log_post -= log_post.max(axis=1, keepdims=True)        # stability
    post = np.exp(log_post)
    post /= post.sum(axis=1, keepdims=True)

    if estimate == "map":
        predicted = dist_centers[post.argmax(axis=1)]
    else:  # posterior mean — continuous, sub-bin resolution
        predicted = post @ dist_centers
    return predicted, post


def _cv_bayesian_distance(firing_rates: np.ndarray, distance: np.ndarray, *,
                          bin_size: float, n_distance_bins: int = 15,
                          tuning_smoothing_sigma: float = 1.0, cv_folds: int = 5,
                          dist_edges: Optional[np.ndarray] = None,
                          use_occupancy_prior: bool = True,
                          estimate: str = "expected",
                          return_posterior: bool = False) -> Dict:
    """Cross-validated Bayesian distance decode over a fixed distance grid.

    Per **contiguous** fold (``KFold(shuffle=False)`` — this module's
    convention for autocorrelated series), build distance tuning curves on the
    training bins with :func:`compute_distance_tuning` and decode the test bins.
    A **single** ``dist_edges`` grid spanning the full session is shared across
    folds (mirroring the shared spatial grid in ``decode_location._cv_decode``).

    ``cv_r2`` is the **pooled** out-of-fold R² (``r2_score`` on concatenated
    predictions) rather than a mean-of-fold score, matching
    :func:`_regression_diagnostics`. Falls back to an in-sample fit when there
    are too few samples for CV.

    Returns ``{y_pred, median_error, mean_error, error_per_fold, cv_r2,
    dist_edges, dist_centers}`` (plus ``posterior`` if requested).
    """
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    fr = np.asarray(firing_rates, dtype=np.float64)
    if fr.ndim == 1:
        fr = fr[:, None]
    d = np.asarray(distance, dtype=np.float64)
    n = len(d)

    if dist_edges is None:
        finite_d = d[np.isfinite(d)]
        lo = float(np.min(finite_d)) if finite_d.size else 0.0
        hi = float(np.max(finite_d)) if finite_d.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        dist_edges = np.linspace(lo, hi, n_distance_bins + 1)
    else:
        dist_edges = np.asarray(dist_edges, dtype=np.float64)
    dist_centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])

    y_pred = np.full(n, np.nan, dtype=np.float64)
    posterior = (np.full((n, len(dist_centers)), np.nan, dtype=np.float64)
                 if return_posterior else None)
    fold_errors = []

    if cv_folds and cv_folds > 1 and n >= cv_folds + 2:
        splits = list(KFold(n_splits=cv_folds, shuffle=False).split(fr))
    else:
        splits = [(np.arange(n), np.arange(n))]  # in-sample fallback

    for tr_idx, te_idx in splits:
        tc = compute_distance_tuning(
            fr[tr_idx], d[tr_idx], smoothing_sigma=tuning_smoothing_sigma,
            dist_edges=dist_edges,
        )
        pred, post = _bayesian_decode_1d(
            fr[te_idx], tc["tuning"], tc["occupancy"], bin_size, dist_centers,
            use_occupancy_prior=use_occupancy_prior, estimate=estimate,
        )
        y_pred[te_idx] = pred
        if return_posterior:
            posterior[te_idx] = post
        fold_errors.append(float(np.median(np.abs(d[te_idx] - pred))))

    valid = np.isfinite(y_pred) & np.isfinite(d)
    cv_r2 = (float(r2_score(d[valid], y_pred[valid]))
             if valid.sum() >= 2 and np.std(d[valid]) > 0 else float("nan"))
    abs_err = np.abs(d[valid] - y_pred[valid])
    median_error = float(np.median(abs_err)) if valid.any() else float("nan")
    mean_error = float(np.mean(abs_err)) if valid.any() else float("nan")

    out = {"y_pred": y_pred, "median_error": median_error,
           "mean_error": mean_error, "error_per_fold": fold_errors,
           "cv_r2": cv_r2, "dist_edges": dist_edges, "dist_centers": dist_centers}
    if return_posterior:
        out["posterior"] = posterior
    return out


def _null_bayesian_distance(firing_rates: np.ndarray, distance: np.ndarray, *,
                            bin_size: float, n_distance_bins: int = 15,
                            tuning_smoothing_sigma: float = 1.0, cv_folds: int = 5,
                            use_occupancy_prior: bool = True,
                            estimate: str = "expected",
                            null: str = "shuffle", n_shuffles: int = 100,
                            seed: int = 0) -> Dict:
    """Bayesian-decoder null from a broken rate↔distance pairing.

    Mirrors :func:`_null_distance_regression` (and
    ``decode_location._null_position_decode``): distance is rolled (not the
    rates) so each stream's autocorrelation is preserved. Returns ``null_r2`` /
    ``null_r2_std`` / ``null_r2_dist`` and ``null_median_error`` /
    ``null_median_error_std``.
    """
    fr = np.asarray(firing_rates, dtype=np.float64)
    if fr.ndim == 1:
        fr = fr[:, None]
    d = np.asarray(distance, dtype=np.float64)
    n = len(d)

    if null == "reverse":
        variants = [d[::-1]]
    elif null == "shuffle":
        rng = np.random.default_rng(seed)
        lo, hi = max(1, n // 10), max(2, n - n // 10)
        shifts = rng.integers(lo, hi, size=n_shuffles)
        variants = [np.roll(d, int(s)) for s in shifts]
    else:
        raise ValueError(f"Unknown null method {null!r} (use 'reverse' or 'shuffle').")

    r2s, meds = [], []
    for d_null in variants:
        r = _cv_bayesian_distance(
            fr, d_null, bin_size=bin_size, n_distance_bins=n_distance_bins,
            tuning_smoothing_sigma=tuning_smoothing_sigma, cv_folds=cv_folds,
            use_occupancy_prior=use_occupancy_prior, estimate=estimate,
        )
        r2s.append(r["cv_r2"])
        meds.append(r["median_error"])
    r2s = np.asarray(r2s, dtype=np.float64)
    meds = np.asarray(meds, dtype=np.float64)

    return {"null_r2": float(np.nanmean(r2s)),
            "null_r2_std": float(np.nanstd(r2s)),
            "null_r2_dist": r2s.tolist(),
            "null_median_error": float(np.nanmean(meds)),
            "null_median_error_std": float(np.nanstd(meds))}


# ---------------------------------------------------------------------------
# Pure-compute orchestrator (no I/O — used by CLI, GUI, and tests)
# ---------------------------------------------------------------------------

def _bin_size_from_centers(bin_centers: np.ndarray) -> float:
    """Infer the time-bin width (s) from bin centres; 1.0 if undeterminable.

    The Bayesian decoder needs Δt for the Poisson likelihood. Bins are an even
    grid (some may be dropped for NaNs), so the median consecutive spacing
    recovers the bin size robustly.
    """
    bc = np.asarray(bin_centers, dtype=np.float64)
    if bc.size >= 2:
        diffs = np.diff(bc)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.median(diffs))
    return 1.0


def _analyze(firing_rates: np.ndarray, distance: np.ndarray,
             nuisance: Optional[np.ndarray], bin_centers: np.ndarray,
             units: str, focal: str, partner: str, *,
             alpha: float = 1.0, cv_folds: int = 5,
             n_distance_bins: int = 15, tuning_smoothing_sigma: float = 1.0,
             null: Optional[str] = "shuffle", n_shuffles: int = 100,
             nuisance_names: Optional[Sequence[str]] = None,
             decoder: str = "ridge", bayesian_estimate: str = "expected",
             bayesian_occupancy_prior: bool = True,
             seed: int = 0) -> Dict:
    """Run the full single-cell + population distance decode on pre-binned arrays.

    I/O-free so the GUI and CLI share it and tests can exercise it on synthetic
    data. ``decode_partner_distance`` wraps ``build_distance_binned_data`` +
    this. See module docstring for the result-dict schema.

    ``decoder`` selects the population decoder. ``'ridge'`` (default) is the
    linear ridge regression — every top-level key keeps its original meaning.
    ``'both'`` additionally runs the Poisson :func:`_bayesian_decode_1d` over the
    distance tuning curves and nests its result (``cv_r2``, ``median_error``,
    ``y_pred``, its own null, optional ``posterior``) under ``result['bayesian']``
    — ridge stays primary, so the schema is unchanged. The Bayesian decoder
    exploits bump-shaped distance tuning that linear ridge cannot capture.
    """
    firing_rates = np.asarray(firing_rates, dtype=np.float64)
    if firing_rates.ndim == 1:
        firing_rates = firing_rates[:, None]
    distance = np.asarray(distance, dtype=np.float64)
    n_bins, n_cells = firing_rates.shape

    parameters = {
        "focal": focal, "partner": partner, "units": units,
        "alpha": alpha, "cv_folds": int(cv_folds),
        "n_distance_bins": int(n_distance_bins),
        "tuning_smoothing_sigma": tuning_smoothing_sigma,
        "null": null, "n_shuffles": int(n_shuffles), "seed": int(seed),
        "nuisance_names": list(nuisance_names) if nuisance_names is not None else None,
        "decoder": decoder, "bayesian_estimate": bayesian_estimate,
        "bayesian_occupancy_prior": bool(bayesian_occupancy_prior),
        "class_label": CLASS_LABEL, "analysis_title": ANALYSIS_TITLE,
    }

    if n_bins < cv_folds * 2:
        return {"status": "insufficient_data", "cv_r2": np.nan,
                "n_bins": n_bins, "n_cells": n_cells, "parameters": parameters}

    tuning = compute_distance_tuning(
        firing_rates, distance,
        n_distance_bins=n_distance_bins, smoothing_sigma=tuning_smoothing_sigma,
    )
    sc = single_cell_distance_scores(
        firing_rates, distance, alpha=alpha, cv_folds=cv_folds, nuisance=nuisance,
    )
    pop = population_distance_regression(
        firing_rates, distance, alpha=alpha, cv_folds=cv_folds, nuisance=nuisance,
    )

    result = {
        "status": "success",
        "tuning": tuning["tuning"], "occupancy": tuning["occupancy"],
        "dist_edges": tuning["dist_edges"], "dist_centers": tuning["dist_centers"],
        "r2_per_cell": sc["r2_per_cell"],
        "pearson_r_per_cell": sc["pearson_r_per_cell"],
        "r2_partial_per_cell": sc["r2_partial_per_cell"],
        "cell_ranking": sc["cell_ranking"],
        "cv_r2": pop["cv_r2"], "r2_per_fold": pop["r2_per_fold"],
        "rmse": pop["rmse"], "cv_r2_partial": pop["cv_r2_partial"],
        "y_true": distance, "y_pred": pop["y_pred"],
        "bin_centers": np.asarray(bin_centers, dtype=np.float64),
        "n_cells": n_cells, "n_bins": n_bins, "units": units,
        "focal": focal, "partner": partner, "parameters": parameters,
    }

    # Robust scoring of the ridge decode (pooled OOF R², per-fold spread).
    diag = pop.get("diagnostics")
    if diag is not None:
        result["diagnostics"] = diag
        result["cv_r2_pooled"] = diag["pooled_r2"]
        result["cv_r2_pearson"] = diag["pearson_r"]
        result["r2_fold_std"] = diag["r2_fold_std"]

    if null is not None and null != "none":
        result.update(_null_distance_regression(
            firing_rates, distance, alpha=alpha, cv_folds=cv_folds,
            null=null, n_shuffles=n_shuffles, seed=seed,
        ))
        # "Negative R²" must be judged against the null, not zero: the
        # circular-shift null is itself strongly negative here.
        null_r2 = result.get("null_r2")
        null_std = result.get("null_r2_std")
        if null_r2 is not None and np.isfinite(null_r2):
            result["cv_r2_vs_null_z"] = (
                float((result["cv_r2"] - null_r2) / null_std)
                if null_std and null_std > 0 else float("nan"))
            null_dist = np.asarray(result.get("null_r2_dist", []), dtype=np.float64)
            null_dist = null_dist[np.isfinite(null_dist)]
            result["cv_r2_null_percentile"] = (
                float((null_dist < result["cv_r2"]).mean() * 100.0)
                if null_dist.size else float("nan"))

    # Side-by-side Bayesian decoder over the distance tuning curves.
    if decoder == "both":
        bayes = _cv_bayesian_distance(
            firing_rates, distance, bin_size=_bin_size_from_centers(bin_centers),
            n_distance_bins=n_distance_bins,
            tuning_smoothing_sigma=tuning_smoothing_sigma, cv_folds=cv_folds,
            dist_edges=tuning["dist_edges"],
            use_occupancy_prior=bayesian_occupancy_prior,
            estimate=bayesian_estimate, return_posterior=True,
        )
        if null is not None and null != "none":
            bayes.update(_null_bayesian_distance(
                firing_rates, distance, bin_size=_bin_size_from_centers(bin_centers),
                n_distance_bins=n_distance_bins,
                tuning_smoothing_sigma=tuning_smoothing_sigma, cv_folds=cv_folds,
                use_occupancy_prior=bayesian_occupancy_prior,
                estimate=bayesian_estimate, null=null, n_shuffles=n_shuffles,
                seed=seed,
            ))
        result["bayesian"] = bayes
    return result


# ---------------------------------------------------------------------------
# Data assembly (I/O) + top-level convenience
# ---------------------------------------------------------------------------

def build_distance_binned_data(ks_focal, tracking, sync, focal: str, partner: str, *,
                               pixels_per_cm: Optional[float] = None,
                               bin_size: float = 0.5,
                               smoothing_sigma_sec: Optional[float] = None,
                               t_start: Optional[float] = None,
                               t_end: Optional[float] = None,
                               filter_kwargs: Optional[dict] = None) -> Dict:
    """Bin focal firing rates, focal↔partner distance, and self-motion nuisance.

    Takes plain data objects — a focal-animal
    :class:`~ingestion.kilosort_data_import.KilosortData`, the session
    :class:`~video.tracking_import.VideoTrackingData` (all animals), and a
    :class:`~ingestion.ephys_sync.DataSyncManager` — so the partner needs no
    ephys. Firing rates come from ``KilosortData.bin_spike_times``; the
    partner-distance target and the focal (speed, x, y) nuisance block are both
    derived from
    :func:`~video.tracking_import.resolve_tracking_on_ephys_clock` (the single
    canonical tracking↔ephys / pixels→cm path), so units are consistent.

    Bins where the distance, any cell's rate, or any nuisance column is NaN
    (e.g. outside the tracked range) are dropped, mirroring
    ``ephys.decode_location.build_binned_data``.

    Parameters
    ----------
    ks_focal : KilosortData
        Spike-sorting results for the focal (implanted) animal.
    tracking : VideoTrackingData
        Session tracking (every animal); not required to be pre-synchronized.
    sync : DataSyncManager
        Behavior↔ephys clock map for the session.
    focal, partner : str
        Animal ids (resolved against tracking by the substring-fallback resolver).
    pixels_per_cm : float, optional
        Calibration forwarded to ``resolve_tracking_on_ephys_clock``.

    Returns ``firing_rates`` (n_bins, n_cells), ``distance`` (n_bins,),
    ``nuisance`` (n_bins, 3), ``nuisance_names``, ``bin_centers``, ``n_cells``,
    ``bin_size``, ``units``, ``focal``, ``partner``.
    """
    from video.behavior_features import _interp
    from video.tracking_import import resolve_tracking_on_ephys_clock

    if filter_kwargs:
        ks_focal.filter_cells_by_firing_patterns(**filter_kwargs)
    rates, bin_centers = ks_focal.bin_spike_times(
        bin_size_sec=bin_size, t_start=t_start, t_end=t_end, filtered_only=True,
    )
    if smoothing_sigma_sec is not None and smoothing_sigma_sec > 0:
        from scipy.ndimage import gaussian_filter1d
        rates = gaussian_filter1d(
            rates, sigma=smoothing_sigma_sec / bin_size, axis=1, mode="reflect",
        )
    X_full = np.asarray(rates, dtype=np.float64).T  # (n_bins, n_cells)
    bin_centers = np.asarray(bin_centers, dtype=np.float64)

    tracking_by_animal = resolve_tracking_on_ephys_clock(
        tracking, sync, [focal, partner],
        pixels_per_cm=pixels_per_cm,
        t_start_ephys=t_start, t_end_ephys=t_end,
    )
    missing = [a for a in (focal, partner) if a not in tracking_by_animal]
    if missing:
        raise KeyError(f"No tracking resolved for {missing} in session "
                       f"{tracking.session_id}.")

    units = "cm" if pixels_per_cm else "pixels"

    fdf, pdf = tracking_by_animal[focal], tracking_by_animal[partner]
    ft = fdf["t"].to_numpy(dtype=np.float64)
    pt = pdf["t"].to_numpy(dtype=np.float64)
    fx = _interp(ft, fdf["x"].to_numpy(dtype=np.float64), bin_centers)
    fy = _interp(ft, fdf["y"].to_numpy(dtype=np.float64), bin_centers)
    fspeed = _interp(ft, fdf["speed"].to_numpy(dtype=np.float64), bin_centers)
    px = _interp(pt, pdf["x"].to_numpy(dtype=np.float64), bin_centers)
    py = _interp(pt, pdf["y"].to_numpy(dtype=np.float64), bin_centers)

    distance = np.hypot(px - fx, py - fy)
    nuisance = np.column_stack([fspeed, fx, fy])
    nuisance_names = ["focal_speed", "focal_x", "focal_y"]

    valid = (np.isfinite(distance)
             & np.all(np.isfinite(X_full), axis=1)
             & np.all(np.isfinite(nuisance), axis=1))

    return {
        "firing_rates": X_full[valid],
        "distance": distance[valid],
        "nuisance": nuisance[valid],
        "nuisance_names": nuisance_names,
        "bin_centers": bin_centers[valid],
        "n_cells": int(X_full.shape[1]),
        "bin_size": bin_size,
        "units": units,
        "focal": focal,
        "partner": partner,
    }


def decode_partner_distance(session_id: str, focal: str, partner: str, *,
                            config_path: Optional[str] = None,
                            dio_channel: int = 1,
                            bin_size: float = 0.5,
                            smoothing_sigma_sec: Optional[float] = None,
                            n_distance_bins: int = 15,
                            tuning_smoothing_sigma: float = 1.0,
                            alpha: float = 1.0, cv_folds: int = 5,
                            null: Optional[str] = "shuffle", n_shuffles: int = 100,
                            decoder: str = "ridge",
                            bayesian_estimate: str = "expected",
                            bayesian_occupancy_prior: bool = True,
                            t_start: Optional[float] = None,
                            t_end: Optional[float] = None,
                            filter_kwargs: Optional[dict] = None,
                            seed: int = 0) -> Dict:
    """End-to-end: load focal KilosortData + session tracking, then decode.

    Thin convenience wrapper = ``load_focal_session_inputs`` +
    ``build_distance_binned_data`` + ``_analyze``. The partner needs no ephys.
    Pass ``decoder='both'`` to additionally run the Bayesian decoder alongside
    ridge (see :func:`_analyze`).
    """
    from ingestion.focal_session import load_focal_session_inputs

    inputs = load_focal_session_inputs(
        session_id, focal, config_path=config_path, dio_channel=dio_channel,
    )
    data = build_distance_binned_data(
        inputs.ks_focal, inputs.tracking, inputs.sync, focal, partner,
        pixels_per_cm=inputs.pixels_per_cm,
        bin_size=bin_size, smoothing_sigma_sec=smoothing_sigma_sec,
        t_start=t_start, t_end=t_end, filter_kwargs=filter_kwargs,
    )
    return _analyze(
        data["firing_rates"], data["distance"], data["nuisance"],
        data["bin_centers"], data["units"], focal, partner,
        alpha=alpha, cv_folds=cv_folds,
        n_distance_bins=n_distance_bins,
        tuning_smoothing_sigma=tuning_smoothing_sigma,
        null=null, n_shuffles=n_shuffles,
        decoder=decoder, bayesian_estimate=bayesian_estimate,
        bayesian_occupancy_prior=bayesian_occupancy_prior,
        nuisance_names=data["nuisance_names"], seed=seed,
    )
