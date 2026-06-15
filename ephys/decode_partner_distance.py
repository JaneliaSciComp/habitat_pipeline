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
           "rmse": res["rmse"], "y_pred": res["y_pred"], "cv_r2_partial": None}

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
# Pure-compute orchestrator (no I/O — used by CLI, GUI, and tests)
# ---------------------------------------------------------------------------

def _analyze(firing_rates: np.ndarray, distance: np.ndarray,
             nuisance: Optional[np.ndarray], bin_centers: np.ndarray,
             units: str, focal: str, partner: str, *,
             alpha: float = 1.0, cv_folds: int = 5,
             n_distance_bins: int = 15, tuning_smoothing_sigma: float = 1.0,
             null: Optional[str] = "shuffle", n_shuffles: int = 100,
             nuisance_names: Optional[Sequence[str]] = None,
             seed: int = 0) -> Dict:
    """Run the full single-cell + population distance decode on pre-binned arrays.

    I/O-free so the GUI and CLI share it and tests can exercise it on synthetic
    data. ``decode_partner_distance`` wraps ``build_distance_binned_data`` +
    this. See module docstring for the result-dict schema.
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

    if null is not None and null != "none":
        result.update(_null_distance_regression(
            firing_rates, distance, alpha=alpha, cv_folds=cv_folds,
            null=null, n_shuffles=n_shuffles, seed=seed,
        ))
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
                            t_start: Optional[float] = None,
                            t_end: Optional[float] = None,
                            filter_kwargs: Optional[dict] = None,
                            seed: int = 0) -> Dict:
    """End-to-end: load focal KilosortData + session tracking, then decode.

    Thin convenience wrapper = ``load_focal_session_inputs`` +
    ``build_distance_binned_data`` + ``_analyze``. The partner needs no ephys.
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
        nuisance_names=data["nuisance_names"], seed=seed,
    )
