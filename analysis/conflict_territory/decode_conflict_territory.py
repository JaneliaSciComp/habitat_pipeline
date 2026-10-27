"""
Conflict-defined territory decoding.

Territory is defined from **where the focal animal won or lost conflicts**: every
point of the arena takes the outcome of its nearest localized conflict (a
Voronoi partition, :mod:`video.conflict_territory`) - ``own`` if that conflict
was a focal win, ``other`` if it was a loss. The question is whether binned
population activity decodes ``own`` vs ``other`` under cross-validation, *and*
whether it does so better for the real win/loss map than for maps built from
the same conflict locations with the outcomes reassigned.

Shape: regression/custom (pattern 2 in the coder brief) - an I/O-free core
:func:`_analyze` (arrays in, result dict out) plus a thin loading wrapper
:func:`decode_conflict_territory`, mirroring :mod:`analysis.partner_distance.decode_partner_distance`.
It is *not* a per-event classification, so it does not go through
``analysis.event_decoding._lda_decoding``; the decoder is the shrinkage LDA of
:func:`analysis.territory.territory_population._cv_balanced_accuracy`.

Statistical notes
------------------
* **The label is a deterministic function of position.** Any place-tuned
  population decodes it to some degree, so above-chance decoding, and a
  time-shift null, test *decodability of a spatial partition*, not territory
  coding. The comparison is therefore against **reassigned maps**: keep the
  conflict locations, permute the win/loss outcomes among them (preserving the
  W/L counts), rebuild the Voronoi labels, re-decode. The **primary test** is
  ``gain_null``: the neural balanced accuracy *minus* that of a smooth-position
  decoder of the same label, for the real map against the same reassigned maps
  (decision 2026-10-12). The plain neural ``map_null`` is secondary, for the
  reason in the next point.
* **What the map null breaks and preserves.** It preserves the neural data, the
  trajectory, the conflict locations and the W/L counts; it breaks only the
  pairing between locations and outcomes. It assumes that, under H0, outcomes
  are **exchangeable across conflict locations**. That fails if wins and losses
  are spatially segregated for reasons unrelated to the brain (e.g. all wins at
  the feeder): the real map is then spatially *simpler* (one compact region per
  class) than a typical reassigned map, and a simple partition is easier for
  any place code to decode. ``diagnostics['position_map_null']`` runs the same
  null with a smooth-position decoder in place of neurons so this can be seen:
  if position alone also places the real map in the null's upper tail, the
  neural map-null p is confounded by map geometry and must not be read as
  territory coding. This is not hypothetical: on synthetic data with a pure
  place-cell population and wins clustered on one half of the arena, losses on
  the other, the neural map-null p sits at its floor
  (``tests/test_decode_conflict_territory.py::TestMapGeometryConfound``) - and
  a spatially clustered win/loss layout is exactly what a real territory would
  produce. Subtracting the position decoder's accuracy cancels most of that
  geometric advantage, which is why ``gain_null`` is primary. Its calibration on
  synthetic pure-place populations (99 reassigned maps per session, 2026-10-12):
  false-positive rate 0/20 and 1/20 seeds with interleaved outcomes (broad and
  fine place fields), 0/20 for fine fields with clustered outcomes, but
  **11/100 (95% CI 5.6-18.8%) for broad place fields with clustered outcomes** -
  i.e. still about 2x nominal when the real map is geometrically *simpler* than
  the reassigned ones. Read ``diagnostics['position_map_null']`` (position-only
  accuracy of the real map against the reassigned maps; no neural data) to see
  which regime a real map is in. Power: 20/20 seeds for a planted territory
  population in both layouts; 20/20 (interleaved) and 14/20 (clustered) for
  place cells plus weak territory cells.
* **Degenerate maps.** A map where either class covers less than
  ``min_class_frac`` of the analysed bins, or where any training fold is
  single-class, is undecodable. Null draws meeting this criterion are redrawn;
  the *same* criterion is applied to the observed map before anything is
  decoded, and it reports ``status='insufficient_data'`` if it fails.
* **Secondary time-shift null.** The real label sequence is circularly shifted
  relative to the rates, independently within each contiguous segment of
  analysed bins (shifts in ``[0.1 n, 0.9 n)`` of the segment length, as
  ``analysis.rest_distance.rest_distance_coding._shifted``). It preserves the label's
  autocorrelation and bout structure and breaks its link to the rates. It is
  the place-coding-sensitive test: expected to be significant for a pure place
  population.
* **Cross-validation is blocked, not shuffled** (``HZ-STAT-005``): contiguous
  folds from :func:`analysis.territory.territory_encoding.blocked_folds`, with ``purge_bins``
  training bins dropped either side of every test block, because positions,
  labels and rates are strongly autocorrelated at 0.5 s. Long ``own``/``other``
  bouts make this close to cross-bout generalisation - conservative, possibly
  low-powered; see ``diagnostics['bouts']``.
* **Balanced accuracy** is the score, so chance is 0.5 regardless of class
  imbalance; ``baseline_accuracy`` (majority-class rate, ``HZ-STAT-002``) is
  reported alongside for anyone reading plain accuracy (``accuracy``).
* **p-values are add-one** ``(1+k)/(n+1)`` via
  :func:`analysis._common._stats_utils.empirical_p_value` (``HZ-STAT-003``); non-finite
  draws are excluded from both numerator and denominator (``HZ-STAT-004``).
  ``p_floor`` is reported; with ``k`` W and ``m`` L points the map null has only
  ``C(k+m, k)`` distinct maps (``n_possible_maps``) and the number actually
  drawn (``n_distinct_maps``), which bounds its real resolution.
* **Bins near a conflict are excluded** (``event_exclusion_sec`` either side of
  every scored conflict), so the decoder cannot read out the conflict itself
  (arousal, motor activity). Conflicts still both define the map and shape the
  session's behaviour; the exclusion removes only the most direct overlap.
* **Events** come only from the date-named directory ``DataStorageManager``
  resolves (``HZ-DATA-007``); the wrapper checks this and fails otherwise.
* **n = 1 session.** Only cohort-7 ``20251216`` has scored events, so any
  result is exploratory; no held-out confirmation is possible.

Units
-----
Positions (``x``, ``y``, conflict coordinates, ``dist``/``margin`` diagnostics)
are in **cm** when ``pixels_per_cm`` is set and the tracking is the manual
``mask_metrics`` format the calibration was made on, otherwise **pixels**
(``units`` field). APT tracking is forced to pixels (CLAUDE.md: the config's
calibration is wrong for the APT frame).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis._common._stats_utils import empirical_p_value, fdr_resolution, majority_class_baseline
from analysis.territory.run_territory import bin_counts, interp_on_grid
from analysis.territory.territory_encoding import blocked_folds
from analysis.territory.territory_population import (_balanced_accuracy, _cv_balanced_accuracy, _position_features,
                                        preprocess_rates)
from video.conflict_territory import OTHER, OWN, nearest_conflict, outcome_is_win, voronoi_labels

logger = logging.getLogger(__name__)

CLASS_LABEL = "territory (own vs other)"
ANALYSIS_TITLE = "Conflict-defined territory decoding"
SHIFT_BATCH = 100


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def segments_from_mask(keep: np.ndarray) -> np.ndarray:
    """Segment id per *kept* bin: consecutive kept bins share an id, a dropped bin starts a new one."""
    keep = np.asarray(keep, dtype=bool)
    starts = keep & ~np.concatenate([[False], keep[:-1]])
    return (np.cumsum(starts) - 1)[keep]


def _bouts(lab: np.ndarray, seg: np.ndarray) -> Dict[str, object]:
    """Runs of constant label, broken at segment boundaries."""
    lab, seg = np.asarray(lab), np.asarray(seg)
    if len(lab) == 0:
        return dict(n_own=0, n_other=0, median_len_bins=np.nan, median_own_len_bins=np.nan,
                    median_other_len_bins=np.nan)
    brk = np.concatenate([[True], (lab[1:] != lab[:-1]) | (seg[1:] != seg[:-1])])
    starts = np.flatnonzero(brk)
    lens = np.diff(np.concatenate([starts, [len(lab)]]))
    cls = lab[starts]
    med = lambda v: float(np.median(v)) if len(v) else float("nan")  # noqa: E731
    return dict(n_own=int(np.sum(cls == 1)), n_other=int(np.sum(cls == 0)), median_len_bins=med(lens),
                median_own_len_bins=med(lens[cls == 1]), median_other_len_bins=med(lens[cls == 0]))


def _fold_counts(lab: np.ndarray, n_folds: int, purge_bins: int):
    out = []
    for train, test in blocked_folds(len(lab), n_folds, purge_bins):
        out.append(dict(train_own=int(np.sum(lab[train] == 1)), train_other=int(np.sum(lab[train] == 0)),
                        test_own=int(np.sum(lab[test] == 1)), test_other=int(np.sum(lab[test] == 0)),
                        n_train=int(len(train)), test_start=int(test[0]), test_stop=int(test[-1]) + 1))
    return out


def _map_ok(lab: np.ndarray, *, min_class_frac: float, n_folds: int, purge_bins: int) -> Tuple[bool, str]:
    """The degeneracy criterion, applied identically to the observed map and every null draw."""
    frac = float(np.mean(lab == 1)) if len(lab) else 0.0
    if min(frac, 1.0 - frac) < min_class_frac:
        return False, f"class fraction {frac:.3f} outside [{min_class_frac}, {1 - min_class_frac}]"
    for f in _fold_counts(lab, n_folds, purge_bins):
        if f["train_own"] == 0 or f["train_other"] == 0 or f["n_train"] < 10:
            return False, "a training fold has a single class (or < 10 bins)"
    return True, ""


def _cv_predict(X: np.ndarray, y: np.ndarray, *, n_folds: int, purge: int, shrinkage: float = 0.3):
    """Out-of-fold predictions + per-fold balanced accuracy.

    Same model and folds as :func:`analysis.territory.territory_population._cv_balanced_accuracy`
    (shrinkage LDA, in-fold z-score, equal priors), which is what every null draw
    uses; this variant also returns the predictions. Pinned equal to it by a test.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    preds = np.full(len(y), -1)
    per_fold = []
    for train, test in blocked_folds(len(y), n_folds, purge):
        if len(np.unique(y[train])) < 2 or len(train) < 10:
            return None, None
        mu, sd = X[train].mean(axis=0), X[train].std(axis=0)
        sd[sd < 1e-9] = 1.0
        clf = LDA(solver="lsqr", shrinkage=shrinkage, priors=[0.5, 0.5])
        clf.fit((X[train] - mu) / sd, y[train])
        preds[test] = clf.predict((X[test] - mu) / sd)
        per_fold.append(_balanced_accuracy(y[test], preds[test]))
    return preds, per_fold


def _shifted_labels(lab: np.ndarray, seg: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    from analysis.rest_distance.rest_distance_coding import _shifted
    return _shifted(lab.astype(float), seg, n, rng).astype(int)


def _null_summary(obs: float, draws: np.ndarray, **extra) -> Dict[str, object]:
    draws = np.asarray(draws, dtype=float)
    valid = draws[np.isfinite(draws)]
    return dict(draws=draws, p_value=empirical_p_value(obs, draws),
                p_floor=1.0 / (valid.size + 1) if valid.size else float("nan"),
                percentile=float(np.mean(valid < obs) * 100.0) if valid.size and np.isfinite(obs) else float("nan"),
                null_mean=float(valid.mean()) if valid.size else float("nan"),
                null_std=float(valid.std()) if valid.size else float("nan"),
                n_valid=int(valid.size), n_invalid=int(draws.size - valid.size), **extra)


# ---------------------------------------------------------------------------
# I/O-free core
# ---------------------------------------------------------------------------

def _analyze(rates: np.ndarray, x: np.ndarray, y: np.ndarray, segment_id: np.ndarray,
             conflicts: pd.DataFrame, *, bin_centers: Optional[np.ndarray] = None, dt: float = 0.5,
             detrend_sec: Optional[float] = 120.0, n_folds: int = 5, purge_bins: int = 20,
             n_map_shuffles: int = 1000, n_time_shifts: int = 1000, min_points_per_class: int = 3,
             min_class_frac: float = 0.10, shrinkage: float = 0.3, position_grid: int = 6,
             position_map_null: bool = True, max_redraw_factor: int = 50,
             n_tests_for_correction: Optional[int] = None, alpha: float = 0.05,
             units: str = "pixels", focal: Optional[str] = None, seed: int = 0) -> Dict:
    """Decode the conflict-defined territory label from pre-binned rates.

    Parameters
    ----------
    rates : (n_bins, n_cells)
        Firing rates (Hz or counts) of the analysed bins only, in time order.
    x, y : (n_bins,)
        Focal position per bin (must be finite).
    segment_id : (n_bins,)
        Contiguous-run id (see :func:`segments_from_mask`); the time-shift null
        shifts within each segment.
    conflicts : DataFrame
        Localized conflicts with ``x, y, outcome`` (``'winner'``/``'loser'``,
        focal view), as returned by :func:`video.conflict_territory.localize_events`.
    n_map_shuffles, n_time_shifts : int
        Null budgets; 0 disables that null.
    n_tests_for_correction : int, optional
        Size of the *declared* test family (``LabNotebook.family_denominator``);
        when given, ``significance_resolution = fdr_resolution(...)`` for the map
        budget. Never pass a hand-counted number.
    seed : int
        Seeds ``np.random.default_rng`` for both nulls.

    Assumptions:
        Map null - under H0 the win/loss outcomes are exchangeable across the
        localized conflict locations, so every reassignment of the observed W/L
        counts to those locations is an equally likely map; the trajectory and
        neural data are held fixed. Exchangeability fails when outcomes are
        spatially segregated for non-neural reasons, in which case the real map
        is geometrically simpler than a typical draw and a place code decodes it
        better; ``diagnostics['position_map_null']`` exposes this (read it
        before the neural p). Draws failing
        the degeneracy criterion are redrawn, so the null is conditional on a
        decodable map - the same condition imposed on the observed map.
        Time-shift null - the circularly shifted label (within each segment) is
        a draw from the label's own autocorrelation structure, independent of
        the rates; this holds for stationary data and is violated by slow drift
        shared by rates and position (mitigated by the 120 s detrend in
        :func:`analysis.territory.territory_population.preprocess_rates`). Because the label
        is a function of position, this null is rejected by place coding alone.
        Both - folds are contiguous with a ``purge_bins`` gap; bins are not
        assumed independent, only blocks separated by the purge are.
    """
    rates = np.asarray(rates, dtype=float)
    if rates.ndim == 1:
        rates = rates[:, None]
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    seg = np.asarray(segment_id)
    n_bins, n_cells = rates.shape
    rng = np.random.default_rng(seed)

    parameters = dict(dt=dt, detrend_sec=detrend_sec, n_folds=int(n_folds), purge_bins=int(purge_bins),
                      n_map_shuffles=int(n_map_shuffles), n_time_shifts=int(n_time_shifts),
                      min_points_per_class=int(min_points_per_class), min_class_frac=float(min_class_frac),
                      shrinkage=shrinkage, position_grid=int(position_grid),
                      position_map_null=bool(position_map_null), max_redraw_factor=int(max_redraw_factor),
                      n_tests_for_correction=n_tests_for_correction, alpha=alpha, units=units, focal=focal,
                      seed=int(seed), class_label=CLASS_LABEL, analysis_title=ANALYSIS_TITLE)
    conflicts = pd.DataFrame(conflicts).reset_index(drop=True)
    win = outcome_is_win(conflicts["outcome"].to_numpy()) if len(conflicts) else np.zeros(0, bool)
    n_win, n_loss = int(win.sum()), int((~win).sum())
    base = dict(parameters=parameters, units=units, n_bins=int(n_bins), n_cells=int(n_cells),
                conflicts=conflicts, n_conflicts_win=n_win, n_conflicts_loss=n_loss,
                balanced_accuracy=float("nan"))

    def insufficient(reason: str, **kw):
        logger.warning("conflict territory: insufficient data - %s", reason)
        return dict(base, status="insufficient_data", reason=reason, **kw)

    if min(n_win, n_loss) < min_points_per_class:
        return insufficient(f"{n_win} win / {n_loss} loss localized conflicts; need >= {min_points_per_class} each")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite for every analysed bin; drop bins without a position first")
    if n_bins < max(2 * n_folds, 10) or n_cells == 0:
        return insufficient(f"{n_bins} bins x {n_cells} cells")

    cx, cy = conflicts["x"].to_numpy(float), conflicts["y"].to_numpy(float)
    vor = voronoi_labels(x, y, cx, cy, win)
    nearest = vor["nearest"]
    lab = vor["own"]
    ok, why = _map_ok(lab, min_class_frac=min_class_frac, n_folds=n_folds, purge_bins=purge_bins)
    fold_counts = _fold_counts(lab, n_folds, purge_bins)
    if not ok:
        return insufficient(f"observed map is degenerate: {why}", fold_class_counts=fold_counts,
                            own_fraction=float(np.mean(lab == 1)))

    Z = preprocess_rates(rates, dt, detrend_sec)
    y_pred, ba_per_fold = _cv_predict(Z, lab, n_folds=n_folds, purge=purge_bins, shrinkage=shrinkage)
    ba = _balanced_accuracy(lab, y_pred)
    cv = dict(n_folds=n_folds, purge=purge_bins, shrinkage=shrinkage)

    P = _position_features(x, y, None, position_grid)
    ba_pos = _cv_balanced_accuracy(P, lab, **cv)

    # ---- reassigned-map draws (neural -> map_null, secondary; neural - position -> gain_null, primary)
    map_draws, pos_draws, maps_seen = [], [], set()
    n_redrawn, attempts = 0, 0
    max_attempts = max(1, max_redraw_factor) * max(1, n_map_shuffles)
    while len(map_draws) < n_map_shuffles and attempts < max_attempts:
        attempts += 1
        perm = rng.permutation(win)
        lab_k = perm[nearest].astype(int)
        if not _map_ok(lab_k, min_class_frac=min_class_frac, n_folds=n_folds, purge_bins=purge_bins)[0]:
            n_redrawn += 1
            continue
        maps_seen.add(perm.tobytes())
        map_draws.append(_cv_balanced_accuracy(Z, lab_k, **cv))
        if position_map_null:
            pos_draws.append(_cv_balanced_accuracy(P, lab_k, **cv))
    n_possible = math.comb(n_win + n_loss, n_win)
    map_null = _null_summary(ba, map_draws, n_redrawn_degenerate=int(n_redrawn),
                             n_distinct_maps=len(maps_seen), n_possible_maps=int(n_possible),
                             n_requested=int(n_map_shuffles),
                             budget_exhausted=bool(len(map_draws) < n_map_shuffles))
    if n_map_shuffles and map_null["budget_exhausted"]:
        logger.warning("map null: only %d of %d draws were non-degenerate", len(map_draws), n_map_shuffles)

    # ---- secondary: within-segment circular time shift -------------------------
    shift_draws, agreement = [], []
    done = 0
    while done < n_time_shifts:
        k = min(SHIFT_BATCH, n_time_shifts - done)
        S = _shifted_labels(lab, seg, k, rng)
        for j in range(k):
            shift_draws.append(_cv_balanced_accuracy(Z, S[:, j], **cv))
            agreement.append(float(np.mean(S[:, j] == lab)))
        done += k
    shift_null = _null_summary(ba, shift_draws,
                               mean_label_agreement=float(np.mean(agreement)) if agreement else float("nan"))

    diagnostics = dict(
        own_fraction=float(np.mean(lab == 1)),
        bouts=_bouts(lab, seg),
        n_segments=int(len(np.unique(seg))),
        fold_class_counts=fold_counts,
        position_only_ba_ceiling=float(ba_pos),
        position_only_note="smooth-position decoder of the same label: a ceiling, near 1 by construction, not a control",
        median_dist_to_nearest_conflict=float(np.median(vor["dist"])),
        median_boundary_margin=float(np.median(vor["margin"][np.isfinite(vor["margin"])]))
        if np.isfinite(vor["margin"]).any() else float("nan"),
        bins_per_conflict=np.bincount(nearest, minlength=len(conflicts)).tolist(),
    )
    if position_map_null and n_map_shuffles:
        diagnostics["position_map_null"] = _null_summary(
            ba_pos, pos_draws,
            note="same permuted maps, smooth-position decoder instead of neurons; a small p here means the "
                 "real map is geometrically easier to decode than a typical reassignment, which confounds "
                 "the neural map-null p (no test - diagnostic only)")

    # ---- primary: gain over position, against the same reassigned maps ---------
    gain_null = None
    if position_map_null and n_map_shuffles:
        gain_obs = float(ba - ba_pos)
        gain_draws = np.asarray(map_draws, dtype=float) - np.asarray(pos_draws, dtype=float)
        gain_null = _null_summary(
            gain_obs, gain_draws, observed=gain_obs,
            note="PRIMARY: neural BA minus smooth-position BA of the same label, real map vs the same "
                 "reassigned maps as map_null")

    result = dict(
        base, status="success",
        balanced_accuracy=float(ba), accuracy=float(np.mean(y_pred == lab)),
        chance_balanced_accuracy=0.5, baseline_accuracy=majority_class_baseline(lab),
        ba_per_fold=[float(v) for v in ba_per_fold],
        y_true=lab, y_pred=y_pred, label_names={1: OWN, 0: OTHER},
        bin_centers=np.asarray(bin_centers, dtype=float) if bin_centers is not None else np.arange(n_bins) * dt,
        x=x, y=y, segment_id=seg, dist_to_conflict=vor["dist"], boundary_margin=vor["margin"],
        map_null=map_null, shift_null=shift_null, diagnostics=diagnostics,
    )
    if gain_null is not None:
        result["gain_null"] = gain_null
    if n_tests_for_correction is not None and n_map_shuffles:
        result["significance_resolution"] = fdr_resolution(int(n_tests_for_correction),
                                                            max(map_null["n_valid"], 1), alpha)
    return result


# ---------------------------------------------------------------------------
# Binning (takes loaded objects; no disk access)
# ---------------------------------------------------------------------------

def build_territory_binned_data(ks, tracking_df: pd.DataFrame, conflicts: pd.DataFrame, *,
                                bin_size: float = 0.5, window: Tuple[float, float],
                                event_exclusion_sec: float = 5.0, max_gap: float = 1.0,
                                exclusion_intervals: Optional[np.ndarray] = None,
                                quality_thresholds: Optional[dict] = None) -> Dict:
    """Bin quality-cell rates and focal position; drop untracked and near-conflict bins.

    Spikes come from ``ks.get_filtered_cells_spike_times(**_DEFAULT_QUALITY_THRESHOLDS)``
    binned with :func:`analysis.territory.run_territory.bin_counts` (not
    ``KilosortData.bin_spike_times``, which ignores the filter thresholds).
    Position is :func:`analysis.territory.run_territory.interp_on_grid` of ``tracking_df``
    (``t, x, y[, speed]`` on the ephys clock), NaN beyond ``max_gap`` seconds
    from a tracked sample.

    The grid spans ``window`` intersected with the tracked range. A bin is kept
    only if it has a position and its centre lies outside
    ``[t_start - event_exclusion_sec, t_end + event_exclusion_sec]`` of every
    conflict. ``exclusion_intervals`` (``(k, 2)`` start/end) overrides the
    intervals taken from ``conflicts`` (``t`` and, if present, ``t_end``); the
    wrapper passes *all* scored conflicts here, including those dropped for
    lacking a position, since they still happened.
    """
    from ingestion.kilosort_data_import import _DEFAULT_QUALITY_THRESHOLDS

    thresholds = dict(_DEFAULT_QUALITY_THRESHOLDS) if quality_thresholds is None else dict(quality_thresholds)
    cluster_ids, spike_times = ks.get_filtered_cells_spike_times(**thresholds)
    df = tracking_df.sort_values("t")
    if "speed" not in df.columns:
        df = df.assign(speed=np.nan)
    lo = max(float(window[0]), float(df["t"].iloc[0]))
    hi = min(float(window[1]), float(df["t"].iloc[-1]))
    n = max(int((hi - lo) // bin_size), 0)
    centers = lo + (np.arange(n) + 0.5) * bin_size
    counts = bin_counts(spike_times, lo, n, dt=bin_size) if (n and spike_times) else np.zeros((n, len(spike_times)))
    pos = interp_on_grid(df, centers, max_gap=max_gap) if n else dict(x=np.zeros(0), y=np.zeros(0))
    has_pos = np.isfinite(pos["x"]) & np.isfinite(pos["y"])

    if exclusion_intervals is None:
        c = pd.DataFrame(conflicts)
        t0 = c["t"].to_numpy(float) if len(c) else np.zeros(0)
        t1 = c["t_end"].to_numpy(float) if "t_end" in c.columns else t0
        exclusion_intervals = np.column_stack([t0, t1]) if len(t0) else np.zeros((0, 2))
    iv = np.asarray(exclusion_intervals, dtype=float).reshape(-1, 2)
    near = np.zeros(n, dtype=bool)
    for a, b in iv:
        lo_e, hi_e = min(a, b) - event_exclusion_sec, max(a, b) + event_exclusion_sec
        near |= (centers >= lo_e) & (centers <= hi_e)
    keep = has_pos & ~near
    return dict(rates=counts[keep] / bin_size, x=pos["x"][keep], y=pos["y"][keep],
                segment_id=segments_from_mask(keep), bin_centers=centers[keep], cluster_ids=list(cluster_ids),
                n_cells=len(cluster_ids), bin_size=bin_size, window=(lo, hi), n_bins_total=int(n),
                n_bins_no_position=int((~has_pos).sum()), n_bins_near_event=int((has_pos & near).sum()),
                n_bins_kept=int(keep.sum()), quality_thresholds=thresholds)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_conflict_territory(result: Dict, save_path: Optional[str] = None, *, grid_bins: int = 150):
    """Map + trajectory, label over time, position-only map null; gain (primary), map and shift nulls."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from video.conflict_territory import territory_grid

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    x, y, lab = result["x"], result["y"], result["y_true"]
    c = result["conflicts"]
    pad = 0.02 * max(np.ptp(x), np.ptp(y), 1.0)
    bounds = (min(x.min(), c["x"].min()) - pad, max(x.max(), c["x"].max()) + pad,
              min(y.min(), c["y"].min()) - pad, max(y.max(), c["y"].max()) + pad)
    g = territory_grid(c["x"], c["y"], c["outcome"], bounds=bounds, bins=grid_bins)
    cmap = ListedColormap(["#f4c7c3", "#c6e2c6"])
    a = ax[0, 0]
    a.pcolormesh(g["x_edges"], g["y_edges"], g["own"], cmap=cmap, vmin=0, vmax=1, shading="flat")
    a.scatter(x, y, c=np.where(lab == 1, "tab:green", "tab:red"), s=1, alpha=0.3, rasterized=True)
    w = outcome_is_win(c["outcome"].to_numpy())
    a.scatter(c["x"][w], c["y"][w], marker="^", s=90, c="darkgreen", edgecolor="k", label=f"win ({w.sum()})")
    a.scatter(c["x"][~w], c["y"][~w], marker="v", s=90, c="darkred", edgecolor="k", label=f"loss ({(~w).sum()})")
    a.set_title("Voronoi territory (green = own) + trajectory")
    a.set_xlabel(f"x ({result['units']})")
    a.set_ylabel(f"y ({result['units']})")
    a.invert_yaxis()
    a.legend(loc="upper right", fontsize=8)

    a = ax[0, 1]
    t = result["bin_centers"]
    a.plot(t, lab, drawstyle="steps-mid", lw=0.8, color="k")
    for f in result["diagnostics"]["fold_class_counts"][1:]:
        a.axvline(t[f["test_start"]], color="tab:blue", ls="--", lw=1)
    a.set_yticks([0, 1], [OTHER, OWN])
    a.set_xlabel("ephys time (s)")
    a.set_title(f"label over time (CV fold boundaries dashed); own frac "
                f"{result['diagnostics']['own_fraction']:.2f}")

    diag = result["diagnostics"]
    panels = [(ax[0, 2], diag.get("position_map_null"), diag.get("position_only_ba_ceiling"),
               "position-only vs reassigned maps (geometry check)", "position-only CV balanced accuracy", 0.5),
              (ax[1, 0], result.get("gain_null"), (result.get("gain_null") or {}).get("observed"),
               "gain over position (PRIMARY)", "neural BA - position BA", None),
              (ax[1, 1], result.get("map_null"), result["balanced_accuracy"],
               "reassigned-map null (secondary)", "CV balanced accuracy", 0.5),
              (ax[1, 2], result.get("shift_null"), result["balanced_accuracy"],
               "time-shift null (secondary)", "CV balanced accuracy", 0.5)]
    for a, nd, obs, name, xlabel, chance in panels:
        if not nd:
            a.set_axis_off()
            continue
        d = np.asarray(nd["draws"], dtype=float)
        d = d[np.isfinite(d)]
        if d.size:
            a.hist(d, bins=30, color="0.7", edgecolor="0.4")
        a.axvline(obs, color="tab:red", lw=2, label="observed")
        if chance is not None:
            a.axvline(chance, color="k", ls=":", lw=1, label=f"chance ({chance})")
        a.set_xlabel(xlabel)
        a.set_title(f"{name}\nobs {obs:.3f}, p = {nd['p_value']:.4f} (floor {nd['p_floor']:.4f})", fontsize=10)
        a.legend(fontsize=8)
    fig.suptitle(f"{ANALYSIS_TITLE} - {result['parameters'].get('focal')}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Thin loading wrapper
# ---------------------------------------------------------------------------

def decode_conflict_territory(session_id: str, animal_id: str, *, config_path: Optional[str] = None,
                              dio_channel: int = 1, behavior_type=None, min_events_per_class: int = 3,
                              bin_size: float = 0.5, event_exclusion_sec: float = 5.0, max_gap: float = 1.0,
                              n_folds: int = 5, purge_bins: int = 20, n_map_shuffles: int = 1000,
                              n_time_shifts: int = 1000, min_points_per_class: int = 3,
                              min_class_frac: float = 0.10, detrend_sec: Optional[float] = 120.0,
                              quality_thresholds: Optional[dict] = None, tracking_file_index: int = 0,
                              purpose: str = "exploratory", hypothesis_id: Optional[int] = None,
                              family_id: Optional[int] = None, output_dir: Optional[str] = None,
                              save_figure: bool = True, seed: int = 0) -> Dict:
    """Load one focal animal's session, build the conflict map and run :func:`_analyze`.

    Refuses held-out targets first (``LabNotebook.assert_not_held_out``,
    ``multi_animal=False`` - only the focal's ephys and tracking are read; the
    opponents appear only as names in the scored events). A
    :class:`~database.lab_notebook.HoldoutViolation` propagates; every later
    failure returns ``{'error': str(e), 'status': 'failed'}``.

    Events are synchronized (``synchronize_with_ephys``) *before*
    ``extract_outcome_labels`` (``HZ-API-002``), with ``animal_of_interest`` as a
    ``'rat<N>'`` string (``HZ-API-003``), ``behavior_type=None`` (every event
    with a winner and loser, ``HZ-API-004``), restricted to the quality cells'
    spike window. Position is the manual ``mask_metrics`` tracking
    (``dsm.get_tracking_files()[tracking_file_index]``) through
    ``resolve_tracking_on_ephys_clock``.
    """
    animal_id = str(animal_id)
    focal = animal_id if animal_id.startswith("rat") else f"rat{animal_id}"
    num_id = focal[3:]

    from database.lab_notebook import LabNotebook
    nb = LabNotebook()
    nb.assert_not_held_out(session_id, num_id, purpose=purpose, hypothesis_id=hypothesis_id, multi_animal=False)
    n_tests = None
    if family_id is not None:
        n_tests = int(nb.family_denominator(family_id)["n_tests_for_correction"])

    try:
        from discovery.detectors import resolved_from_dated_directory
        from ingestion.data_paths import DataStorageManager
        from ingestion.ephys_sync import DataSyncManager
        from ingestion.kilosort_data_import import _DEFAULT_QUALITY_THRESHOLDS, load_kilosort_data
        from video.behavioral_events import load_behavioral_events
        from video.conflict_territory import localize_events
        from video.tracking_import import (TRACKING_FORMAT_MASK_METRICS, load_tracking_data,
                                           resolve_tracking_on_ephys_clock)

        thresholds = dict(_DEFAULT_QUALITY_THRESHOLDS) if quality_thresholds is None else dict(quality_thresholds)
        dsm = DataStorageManager(num_id, session_id, config_path=config_path)
        ks_path = dsm.get_kilosort_path()
        if ks_path is None:
            raise FileNotFoundError(f"no kilosort output for {focal} / {session_id}")
        ks = load_kilosort_data(ks_path)
        win = ks.quality_ephys_window(**thresholds)
        if win == (0.0, 0.0):
            raise ValueError("no quality-filtered cell has spikes")

        event_files = dsm.get_behavioral_event_files()
        dated = resolved_from_dated_directory(event_files)
        if not dated["all_from_dated_directory"]:
            raise ValueError(f"HZ-DATA-007: events resolved outside a dated directory: {dated['loose_files']}")
        events = load_behavioral_events(event_files, session_id=dsm.session_id)
        sync = DataSyncManager(dsm, dio_channel=dio_channel)
        events.synchronize_with_ephys(sync, create_new_columns=True)
        starts, ends, labels = events.extract_outcome_labels(
            focal, behavior_type, min_events_per_class, ephys_time_range=(float(win[0]), float(win[1])))
        if len(starts) == 0:
            raise ValueError(f"no outcome events for {focal} with >= {min_events_per_class} per class "
                             f"inside the quality window {win}")

        tracking_files = dsm.get_tracking_files()
        if not tracking_files:
            raise FileNotFoundError(f"no tracking for {session_id}")
        tracking = load_tracking_data(dsm, file_index=tracking_file_index)
        ppcm = dsm.get_pixels_per_cm()
        if tracking.tracking_format != TRACKING_FORMAT_MASK_METRICS and ppcm is not None:
            logger.warning("tracking format %s: pixels_per_cm=%s was calibrated on the manual frame; "
                           "running in pixels", tracking.tracking_format, ppcm)
            ppcm = None
        tracks = resolve_tracking_on_ephys_clock(tracking, sync, [focal], pixels_per_cm=ppcm)
        if focal not in tracks or len(tracks[focal]) < 2:
            raise KeyError(f"no tracking resolved for {focal}")
        focal_df = tracks[focal]
        units = "cm" if ppcm else "pixels"

        conflicts, n_dropped = localize_events(starts, labels, focal_df, max_gap=max_gap, end_times=ends)
        window = (max(float(win[0]), float(focal_df["t"].min())), min(float(win[1]), float(focal_df["t"].max())))
        data = build_territory_binned_data(
            ks, focal_df, conflicts, bin_size=bin_size, window=window, event_exclusion_sec=event_exclusion_sec,
            max_gap=max_gap, exclusion_intervals=np.column_stack([starts, ends]), quality_thresholds=thresholds)
        result = _analyze(
            data["rates"], data["x"], data["y"], data["segment_id"], conflicts, bin_centers=data["bin_centers"],
            dt=bin_size, detrend_sec=detrend_sec, n_folds=n_folds, purge_bins=purge_bins,
            n_map_shuffles=n_map_shuffles, n_time_shifts=n_time_shifts, min_points_per_class=min_points_per_class,
            min_class_frac=min_class_frac, n_tests_for_correction=n_tests, units=units, focal=focal, seed=seed)
    except Exception as e:  # noqa: BLE001 - same failure contract as every wrapper
        logger.exception("decode_conflict_territory failed")
        return {"error": str(e), "status": "failed"}

    lab_u, lab_n = np.unique(labels, return_counts=True)
    loc_u, loc_n = (np.unique(conflicts["outcome"], return_counts=True) if len(conflicts)
                    else (np.array([]), np.array([])))
    result["parameters"].update(
        session_id=session_id, animal_id=num_id, animal_of_interest=focal, behavior_type=behavior_type,
        min_events_per_class=int(min_events_per_class), bin_size=bin_size, event_exclusion_sec=event_exclusion_sec,
        max_gap=max_gap, quality_thresholds=thresholds, tracking_file_index=int(tracking_file_index),
        config_path=config_path, dio_channel=int(dio_channel), purpose=purpose, hypothesis_id=hypothesis_id,
        family_id=family_id, pixels_per_cm=ppcm, events_synchronized=True)
    result["behavioral_summary"] = dict(
        event_files=[str(p) for p in event_files], events_from_dated_directory=True,
        n_events_total=int(len(events.events_data)), n_outcome_events=int(len(starts)),
        outcome_counts={str(k): int(v) for k, v in zip(lab_u, lab_n)},
        n_localized=int(len(conflicts)), n_dropped_no_position=int(n_dropped),
        localized_counts={str(k): int(v) for k, v in zip(loc_u, loc_n)},
        unique_classes=[OWN, OTHER], quality_ephys_window=[float(win[0]), float(win[1])],
        tracking_file=str(tracking_files[tracking_file_index]), tracking_format=tracking.tracking_format,
        tracking_range=[float(focal_df["t"].min()), float(focal_df["t"].max())], analysis_window=list(window),
        n_bins_total=data["n_bins_total"], n_bins_no_position=data["n_bins_no_position"],
        n_bins_near_event=data["n_bins_near_event"], n_bins_analyzed=data["n_bins_kept"],
        n_quality_cells=data["n_cells"], recording_id=getattr(dsm, "recording_id", None))

    if output_dir:
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        if save_figure and result.get("status") == "success":
            import matplotlib
            matplotlib.use("Agg")
            result["figure_path"] = str(d / "conflict_territory.png")
            plot_conflict_territory(result, result["figure_path"])
        with open(d / "results.pkl", "wb") as fh:
            pickle.dump(result, fh)
        (d / "summary.json").write_text(json.dumps(summarize(result), indent=2, default=_json_default))
    return result


def summarize(result: Dict) -> Dict:
    """JSON-friendly headline fields (no per-bin arrays, no null draws)."""
    keep = ("status", "reason", "error", "balanced_accuracy", "accuracy", "baseline_accuracy", "ba_per_fold",
            "n_bins", "n_cells", "n_conflicts_win", "n_conflicts_loss", "units", "parameters",
            "behavioral_summary", "significance_resolution", "figure_path")
    out = {k: result[k] for k in keep if k in result}
    for k in ("gain_null", "map_null", "shift_null"):
        if k in result:
            out[k] = {kk: v for kk, v in result[k].items() if kk != "draws"}
    if "diagnostics" in result:
        diag = dict(result["diagnostics"])
        for k in ("position_map_null",):
            if k in diag:
                diag[k] = {kk: v for kk, v in diag[k].items() if kk != "draws"}
        out["diagnostics"] = diag
    if "conflicts" in result:
        out["conflicts"] = pd.DataFrame(result["conflicts"]).to_dict(orient="records")
    return out


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Conflict-defined territory decoding (see module docstring).")
    p.add_argument("--animal_id", required=True, type=str, help="focal rat number, e.g. 631")
    p.add_argument("--session_id", required=True, type=str, help="e.g. 20251216")
    p.add_argument("--config_path", default=None)
    p.add_argument("--dio_channel", type=int, default=1)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--bin_size", type=float, default=0.5)
    p.add_argument("--event_exclusion_sec", type=float, default=5.0)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--purge_bins", type=int, default=20)
    p.add_argument("--n_map_shuffles", type=int, default=1000)
    p.add_argument("--n_time_shifts", type=int, default=1000)
    p.add_argument("--min_points_per_class", type=int, default=3)
    p.add_argument("--min_class_frac", type=float, default=0.10)
    p.add_argument("--min_events_per_class", type=int, default=3)
    p.add_argument("--purpose", choices=["exploratory", "confirmatory"], default="exploratory")
    p.add_argument("--hypothesis_id", type=int, default=None)
    p.add_argument("--family_id", type=int, default=None,
                   help="declared test family; its denominator feeds fdr_resolution")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = args.output_dir or f"results/conflict_territory/{args.session_id}_{args.animal_id}"
    res = decode_conflict_territory(
        args.session_id, args.animal_id, config_path=args.config_path, dio_channel=args.dio_channel,
        min_events_per_class=args.min_events_per_class, bin_size=args.bin_size,
        event_exclusion_sec=args.event_exclusion_sec, n_folds=args.n_folds, purge_bins=args.purge_bins,
        n_map_shuffles=args.n_map_shuffles, n_time_shifts=args.n_time_shifts,
        min_points_per_class=args.min_points_per_class, min_class_frac=args.min_class_frac,
        purpose=args.purpose, hypothesis_id=args.hypothesis_id, family_id=args.family_id,
        output_dir=out_dir, seed=args.seed)
    print(json.dumps(summarize(res), indent=2, default=_json_default))
    print(f"saved to {out_dir}")
    return 0 if res.get("status") in ("success", "insufficient_data") else 1


if __name__ == "__main__":
    raise SystemExit(main())
