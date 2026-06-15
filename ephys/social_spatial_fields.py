"""
Allocentric social place fields.

For each sorted unit in a *focal* animal's brain, build occupancy-normalized
firing-rate maps as a function of *another* animal's allocentric ``(x, y)``
position ("social place fields"), quantify spatial tuning (Skaggs information,
sparsity, coherence, split-half stability), attach shuffle-based significance,
and classify cells by which conspecific(s) they encode (self vs partner-specific
vs broadcast).

This module is label-agnostic in the same spirit as :mod:`ephys._lda_decoding`:
the low-level ``compute_rate_map`` / ``spatial_*`` / ``field_significance``
functions take plain arrays and DataFrames, and the multi-target sweep
``compute_social_place_fields`` is the wrapper that knows about
``KilosortData`` (focal) / ``VideoTrackingData`` (session) and stamps the
result-dict ``parameters`` with ``class_label='target_position'`` and
``analysis_title`` so the plot module ([ephys/social_spatial_plots.py]) can be
driven from the dataclass without per-target branches. Only the focal animal
needs ephys; the target animals contribute tracking trajectories only.

Conventions
-----------
- ``RateMap.rates`` is ``(n_y_bins, n_x_bins)`` (image convention) — note this is
  transposed relative to ``ephys.decode_location``'s ``(n_x, n_y)``.
- Occupancy is **dwell time in seconds** (the per-spatial-bin sum of frame
  intervals), not a count of time bins.
- All times are ephys seconds. Tracking↔ephys conversion lives only in
  :func:`video.tracking_import.resolve_tracking_on_ephys_clock`.
- "cm" parameter names refer to whatever spatial unit the tracking is in; if no
  ``pixels_per_cm`` calibration is configured, that unit is pixels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from ingestion.kilosort_data_import import KilosortData
    from video.tracking_import import VideoTrackingData

logger = logging.getLogger(__name__)

CLASS_LABEL = "target_position"
ANALYSIS_TITLE = "Social Place Fields"

ArenaBounds = Tuple[Tuple[float, float], Tuple[float, float]]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RateMap:
    """Occupancy-normalized firing-rate map of one unit over one animal's (x, y)."""

    rates: np.ndarray             # (n_y_bins, n_x_bins), Hz, NaN where occupancy < min
    occupancy: np.ndarray         # (n_y_bins, n_x_bins), seconds (raw dwell time)
    spike_counts: np.ndarray      # (n_y_bins, n_x_bins), counts (raw)
    x_edges: np.ndarray
    y_edges: np.ndarray
    focal_animal: str
    target_animal: str
    cluster_id: int
    parameters: dict = field(default_factory=dict)


@dataclass
class FieldStats:
    cluster_id: int
    target_animal: str
    skaggs_bits_per_spike: float
    skaggs_bits_per_sec: float
    sparsity: float
    coherence: float
    split_half_corr: float
    peak_rate_hz: float
    mean_rate_hz: float
    n_spikes_in_window: int


@dataclass
class FieldSignificance:
    cluster_id: int
    target_animal: str
    null_method: str              # 'circular_shift' | 'position_shuffle'
    n_shuffles: int
    p_skaggs: float
    p_sparsity: float
    p_split_half: float
    shuffle_skaggs: np.ndarray    # length n_shuffles, kept for plotting


@dataclass
class SocialFieldResults:
    rate_maps: Dict[str, Dict[int, RateMap]]
    stats: Dict[str, Dict[int, FieldStats]]
    signif: Dict[str, Dict[int, FieldSignificance]]
    cell_classification: pd.DataFrame
    population_field_similarity: Dict[str, Dict[str, np.ndarray]]
    parameters: dict


# ---------------------------------------------------------------------------
# Spatial-binning helpers
# ---------------------------------------------------------------------------

def _edges_from_bounds(arena_bounds: ArenaBounds, bin_size_cm: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    (xmin, xmax), (ymin, ymax) = arena_bounds
    x_edges = np.arange(xmin, xmax + bin_size_cm, bin_size_cm, dtype=np.float64)
    y_edges = np.arange(ymin, ymax + bin_size_cm, bin_size_cm, dtype=np.float64)
    # Guard against degenerate (single-edge) axes.
    if len(x_edges) < 2:
        x_edges = np.array([xmin, xmin + bin_size_cm], dtype=np.float64)
    if len(y_edges) < 2:
        y_edges = np.array([ymin, ymin + bin_size_cm], dtype=np.float64)
    return x_edges, y_edges


def _infer_bounds(target_xy: pd.DataFrame, pad_cm: float = 5.0) -> ArenaBounds:
    x = target_xy["x"].to_numpy()
    y = target_xy["y"].to_numpy()
    return (
        (float(np.nanmin(x)) - pad_cm, float(np.nanmax(x)) + pad_cm),
        (float(np.nanmin(y)) - pad_cm, float(np.nanmax(y)) + pad_cm),
    )


# ---------------------------------------------------------------------------
# Rate map
# ---------------------------------------------------------------------------

def compute_rate_map(
    spike_times: np.ndarray,
    target_xy: pd.DataFrame,
    bin_size_cm: float = 5.0,
    arena_bounds: Optional[ArenaBounds] = None,
    smoothing_sigma_cm: Optional[float] = 5.0,
    min_occupancy_sec: float = 0.1,
    speed_xy: Optional[pd.DataFrame] = None,
    speed_threshold_cms: Optional[float] = 5.0,
    t_window_ephys: Optional[Tuple[float, float]] = None,
    speed_filter_subject: Literal["focal", "target", "none"] = "target",
    focal_animal: str = "",
    target_animal: str = "",
    cluster_id: int = -1,
) -> RateMap:
    """Occupancy-normalized rate map of ``spike_times`` over ``target_xy`` (x, y).

    ``target_xy`` must have columns ``t`` (ephys seconds), ``x``, ``y`` (cm).
    Each tracking sample owns a dwell interval (central-difference of ``t``);
    occupancy is the per-spatial-bin sum of those intervals in seconds. Spikes
    are assigned to the spatial bin of their nearest tracking sample. Occupancy
    and spike-count maps are Gaussian-smoothed **before** dividing; bins whose
    raw occupancy is below ``min_occupancy_sec`` are set to NaN.

    Speed gating (when ``speed_xy`` and ``speed_threshold_cms`` are given)
    removes tracking samples whose gating-subject speed is below threshold
    *before* binning, and drops spikes assigned to removed samples.
    ``speed_filter_subject`` is recorded in ``parameters`` only; the caller
    chooses which animal's speed to pass as ``speed_xy``.
    """
    t = target_xy["t"].to_numpy(dtype=np.float64)
    x = target_xy["x"].to_numpy(dtype=np.float64)
    y = target_xy["y"].to_numpy(dtype=np.float64)

    if t_window_ephys is not None:
        w0, w1 = t_window_ephys
        in_win = (t >= w0) & (t <= w1)
        t, x, y = t[in_win], x[in_win], y[in_win]
    else:
        w0 = float(t.min()) if t.size else 0.0
        w1 = float(t.max()) if t.size else 0.0

    finite = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[finite], x[finite], y[finite]
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x[order], y[order]

    if arena_bounds is None:
        arena_bounds = _infer_bounds(target_xy) if target_xy.shape[0] else ((0.0, 1.0), (0.0, 1.0))
    x_edges, y_edges = _edges_from_bounds(arena_bounds, bin_size_cm)
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1

    occupancy = np.zeros((n_y, n_x), dtype=np.float64)
    spike_counts = np.zeros((n_y, n_x), dtype=np.float64)
    n_spikes_in_window = 0

    if t.size >= 2:
        # Dwell interval each sample represents (sums to ~ total duration).
        dt = np.gradient(t)
        dt[dt < 0] = 0.0

        # Speed gate over tracking samples.
        keep = np.ones(t.size, dtype=bool)
        if speed_xy is not None and speed_threshold_cms is not None:
            sp_t = speed_xy["t"].to_numpy(dtype=np.float64)
            sp_v = speed_xy["speed"].to_numpy(dtype=np.float64)
            sp_at_t = np.interp(t, sp_t, sp_v, left=np.nan, right=np.nan)
            keep = np.isfinite(sp_at_t) & (sp_at_t >= speed_threshold_cms)

        ix = np.clip(np.digitize(x, x_edges) - 1, 0, n_x - 1)
        iy = np.clip(np.digitize(y, y_edges) - 1, 0, n_y - 1)
        np.add.at(occupancy, (iy[keep], ix[keep]), dt[keep])

        # Spikes -> nearest tracking sample -> spatial bin.
        st = np.asarray(spike_times, dtype=np.float64)
        st = st[(st >= w0) & (st <= w1) & (st >= t[0]) & (st <= t[-1])]
        n_spikes_in_window = int(st.size)
        if st.size:
            right = np.clip(np.searchsorted(t, st), 1, t.size - 1)
            left = right - 1
            choose_left = (st - t[left]) <= (t[right] - st)
            idx = np.where(choose_left, left, right)
            idx = idx[keep[idx]]
            if idx.size:
                np.add.at(spike_counts, (iy[idx], ix[idx]), 1.0)

    # Smooth then divide.
    if smoothing_sigma_cm is not None and smoothing_sigma_cm > 0 and n_x and n_y:
        sigma_bins = smoothing_sigma_cm / bin_size_cm
        occ_s = gaussian_filter(occupancy, sigma=sigma_bins, mode="constant")
        sc_s = gaussian_filter(spike_counts, sigma=sigma_bins, mode="constant")
    else:
        occ_s, sc_s = occupancy, spike_counts

    with np.errstate(divide="ignore", invalid="ignore"):
        rates = sc_s / occ_s
    rates[occupancy < min_occupancy_sec] = np.nan
    rates[~np.isfinite(rates)] = np.nan

    parameters = {
        "bin_size_cm": bin_size_cm,
        "smoothing_sigma_cm": smoothing_sigma_cm,
        "speed_threshold_cms": speed_threshold_cms,
        "arena_bounds": arena_bounds,
        "min_occupancy_sec": min_occupancy_sec,
        "t_window_ephys": (w0, w1),
        "speed_filter_subject": speed_filter_subject,
        "class_label": CLASS_LABEL,
        "analysis_title": ANALYSIS_TITLE,
    }
    return RateMap(
        rates=rates,
        occupancy=occupancy,
        spike_counts=spike_counts,
        x_edges=x_edges,
        y_edges=y_edges,
        focal_animal=focal_animal,
        target_animal=target_animal,
        cluster_id=cluster_id,
        parameters=parameters,
    )


# ---------------------------------------------------------------------------
# Spatial statistics
# ---------------------------------------------------------------------------

def _valid_pr(rate_map: RateMap) -> Tuple[np.ndarray, np.ndarray]:
    """Return (p_i, r_i) over valid bins: occupancy probability and rate."""
    rates = rate_map.rates
    occ = rate_map.occupancy
    valid = np.isfinite(rates) & (occ > 0)
    r = rates[valid].astype(np.float64)
    o = occ[valid].astype(np.float64)
    total = o.sum()
    if total <= 0 or r.size == 0:
        return np.array([]), np.array([])
    return o / total, r


def spatial_information(rate_map: RateMap) -> Tuple[float, float]:
    """Skaggs spatial information: ``(bits_per_spike, bits_per_second)``."""
    p, r = _valid_pr(rate_map)
    if p.size == 0:
        return 0.0, 0.0
    mean_rate = float(np.sum(p * r))
    if mean_rate <= 0:
        return 0.0, 0.0
    pos = r > 0
    ratio = r[pos] / mean_rate
    bits_per_sec = float(np.sum(p[pos] * r[pos] * np.log2(ratio)))
    bits_per_spike = bits_per_sec / mean_rate
    return bits_per_spike, bits_per_sec


def spatial_sparsity(rate_map: RateMap) -> float:
    """Occupancy-weighted sparsity ``(<r>)^2 / <r^2>`` (lower = more selective)."""
    p, r = _valid_pr(rate_map)
    if p.size == 0:
        return np.nan
    num = float(np.sum(p * r)) ** 2
    den = float(np.sum(p * r ** 2))
    if den <= 0:
        return np.nan
    return num / den


def spatial_coherence(rate_map: RateMap) -> float:
    """Coherence: Fisher-z of the Pearson r between each bin's rate and its
    8-neighborhood mean, over bins where both are defined."""
    rates = rate_map.rates
    valid = np.isfinite(rates)
    if valid.sum() < 3:
        return np.nan

    filled = np.where(valid, rates, 0.0)
    mask = valid.astype(np.float64)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float64)
    from scipy.ndimage import convolve
    neigh_sum = convolve(filled, kernel, mode="constant", cval=0.0)
    neigh_cnt = convolve(mask, kernel, mode="constant", cval=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        neigh_mean = neigh_sum / neigh_cnt

    both = valid & np.isfinite(neigh_mean) & (neigh_cnt > 0)
    if both.sum() < 3:
        return np.nan
    a = rates[both].astype(np.float64)
    b = neigh_mean[both].astype(np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r = float(np.corrcoef(a, b)[0, 1])
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def split_half_stability(spike_times: np.ndarray, target_xy: pd.DataFrame,
                         **rate_map_kwargs) -> float:
    """Pearson r between rate maps of the first vs second half of the window."""
    t_window = rate_map_kwargs.pop("t_window_ephys", None)
    t = target_xy["t"].to_numpy(dtype=np.float64)
    if t.size == 0:
        return np.nan
    if t_window is None:
        t0, t1 = float(t.min()), float(t.max())
    else:
        t0, t1 = t_window
    tmid = 0.5 * (t0 + t1)

    # Share a fixed arena across halves so bin grids align.
    bounds = rate_map_kwargs.get("arena_bounds")
    if bounds is None:
        bounds = _infer_bounds(target_xy)
        rate_map_kwargs = {**rate_map_kwargs, "arena_bounds": bounds}

    rm1 = compute_rate_map(spike_times, target_xy, t_window_ephys=(t0, tmid),
                           **rate_map_kwargs)
    rm2 = compute_rate_map(spike_times, target_xy, t_window_ephys=(tmid, t1),
                           **rate_map_kwargs)
    both = np.isfinite(rm1.rates) & np.isfinite(rm2.rates)
    if both.sum() < 3:
        return np.nan
    a = rm1.rates[both]
    b = rm2.rates[both]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compute_field_stats(rate_map: RateMap, spike_times: np.ndarray,
                        target_xy: pd.DataFrame, **rate_map_kwargs) -> FieldStats:
    """Bundle the per-field summary statistics for a single rate map."""
    bits_spike, bits_sec = spatial_information(rate_map)
    valid = np.isfinite(rate_map.rates)
    peak = float(np.nanmax(rate_map.rates)) if valid.any() else 0.0
    p, r = _valid_pr(rate_map)
    mean_rate = float(np.sum(p * r)) if p.size else 0.0
    return FieldStats(
        cluster_id=rate_map.cluster_id,
        target_animal=rate_map.target_animal,
        skaggs_bits_per_spike=bits_spike,
        skaggs_bits_per_sec=bits_sec,
        sparsity=spatial_sparsity(rate_map),
        coherence=spatial_coherence(rate_map),
        split_half_corr=split_half_stability(spike_times, target_xy, **rate_map_kwargs),
        peak_rate_hz=peak,
        mean_rate_hz=mean_rate,
        n_spikes_in_window=int(np.sum(rate_map.spike_counts)),
    )


# ---------------------------------------------------------------------------
# Shuffle significance
# ---------------------------------------------------------------------------

def field_significance(
    spike_times: np.ndarray,
    target_xy: pd.DataFrame,
    n_shuffles: int = 500,
    null_method: Literal["circular_shift", "position_shuffle"] = "circular_shift",
    seed: int = 0,
    cluster_id: int = -1,
    target_animal: str = "",
    **rate_map_kwargs,
) -> FieldSignificance:
    """Shuffle-based significance for a single rate map.

    ``circular_shift`` rigidly time-shifts the spike train within the window
    (preserves firing rate + autocorrelation); ``position_shuffle`` cyclically
    rolls the target ``(x, y)`` relative to the spikes. In both cases the shift
    magnitude is drawn from ``[0.1 T, 0.9 T]`` of the window.

    For each shuffle the Skaggs bits/spike, sparsity, and split-half correlation
    are recomputed. P-values are one-tailed in the meaningful direction:
    ``p_skaggs`` and ``p_split`` are ``fraction(shuffle >= true)``; ``p_sparsity``
    is ``fraction(shuffle <= true)`` because lower sparsity = more selective.
    """
    rng = np.random.default_rng(seed)

    if rate_map_kwargs.get("arena_bounds") is None:
        rate_map_kwargs = {**rate_map_kwargs, "arena_bounds": _infer_bounds(target_xy)}

    t = target_xy["t"].to_numpy(dtype=np.float64)
    t_window = rate_map_kwargs.get("t_window_ephys")
    if t_window is None:
        w0, w1 = float(t.min()), float(t.max())
    else:
        w0, w1 = t_window
    span = max(w1 - w0, 1e-9)
    median_dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0

    def _stats(sp, xy):
        rm = compute_rate_map(sp, xy, cluster_id=cluster_id,
                              target_animal=target_animal, **rate_map_kwargs)
        bits, _ = spatial_information(rm)
        return bits, spatial_sparsity(rm), split_half_stability(sp, xy, **rate_map_kwargs)

    st = np.asarray(spike_times, dtype=np.float64)
    true_skaggs, true_sparsity, true_split = _stats(st, target_xy)

    sh_skaggs = np.full(n_shuffles, np.nan)
    sh_sparsity = np.full(n_shuffles, np.nan)
    sh_split = np.full(n_shuffles, np.nan)
    x0 = target_xy["x"].to_numpy()
    y0 = target_xy["y"].to_numpy()
    for i in range(n_shuffles):
        tau = rng.uniform(0.1 * span, 0.9 * span)
        if null_method == "circular_shift":
            sp_i = w0 + np.mod(st - w0 + tau, span)
            xy_i = target_xy
        elif null_method == "position_shuffle":
            k = int(round(tau / max(median_dt, 1e-9)))
            xy_i = target_xy.copy()
            xy_i["x"] = np.roll(x0, k)
            xy_i["y"] = np.roll(y0, k)
            sp_i = st
        else:
            raise ValueError(f"Unknown null_method: {null_method!r}")
        sh_skaggs[i], sh_sparsity[i], sh_split[i] = _stats(sp_i, xy_i)

    def _p_geq(true_val, shuffles):
        valid = shuffles[np.isfinite(shuffles)]
        if not np.isfinite(true_val) or valid.size == 0:
            return np.nan
        return float(np.mean(valid >= true_val))

    def _p_leq(true_val, shuffles):
        valid = shuffles[np.isfinite(shuffles)]
        if not np.isfinite(true_val) or valid.size == 0:
            return np.nan
        return float(np.mean(valid <= true_val))

    return FieldSignificance(
        cluster_id=cluster_id,
        target_animal=target_animal,
        null_method=null_method,
        n_shuffles=n_shuffles,
        p_skaggs=_p_geq(true_skaggs, sh_skaggs),
        p_sparsity=_p_leq(true_sparsity, sh_sparsity),
        p_split_half=_p_geq(true_split, sh_split),
        shuffle_skaggs=sh_skaggs,
    )


# ---------------------------------------------------------------------------
# Arena bounds from a whole session
# ---------------------------------------------------------------------------

def compute_arena_bounds_from_tracking(tracking_by_animal: Dict[str, pd.DataFrame],
                                       pad_cm: float = 5.0) -> ArenaBounds:
    """Aggregate (min, max) x and y across all animals' tracking, padded.

    ``tracking_by_animal`` is a resolved ``{animal_id: (t, x, y, speed) df}`` dict
    (e.g. from :func:`video.tracking_import.resolve_tracking_on_ephys_clock`).
    """
    return _bounds_from_tracking_dict(tracking_by_animal, pad_cm)


def _bounds_from_tracking_dict(tracking: Dict[str, pd.DataFrame],
                               pad_cm: float = 5.0) -> ArenaBounds:
    xs, ys = [], []
    for df in tracking.values():
        if df.shape[0]:
            xs.append(df["x"].to_numpy())
            ys.append(df["y"].to_numpy())
    if not xs:
        raise ValueError("No tracking available to infer arena bounds.")
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return (
        (float(np.nanmin(x)) - pad_cm, float(np.nanmax(x)) + pad_cm),
        (float(np.nanmin(y)) - pad_cm, float(np.nanmax(y)) + pad_cm),
    )


# ---------------------------------------------------------------------------
# Cross-target field similarity
# ---------------------------------------------------------------------------

def _masked_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over entries where both arrays are finite."""
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < 3:
        return np.nan
    av, bv = a[both], b[both]
    if np.std(av) == 0 or np.std(bv) == 0:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def field_similarity_across_targets(rate_maps: Dict[str, RateMap]) -> pd.DataFrame:
    """Pairwise Pearson r between one cluster's rate maps under each target.

    Correlations use only bins where both maps have valid (non-NaN) occupancy.
    Returns a square DataFrame indexed and columned by target animal.
    """
    targets = list(rate_maps.keys())
    n = len(targets)
    mat = np.full((n, n), np.nan)
    flats = {t: rate_maps[t].rates.ravel() for t in targets}
    for i, ti in enumerate(targets):
        for j, tj in enumerate(targets):
            if i == j:
                mat[i, j] = 1.0
            elif j > i:
                r = _masked_corr(flats[ti], flats[tj])
                mat[i, j] = mat[j, i] = r
    return pd.DataFrame(mat, index=targets, columns=targets)


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (q-values). NaNs map to 1.0."""
    p = np.asarray(pvals, dtype=np.float64).copy()
    nan_mask = ~np.isfinite(p)
    p[nan_mask] = 1.0
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest rank down.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=np.float64)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


# ---------------------------------------------------------------------------
# Multi-target sweep
# ---------------------------------------------------------------------------

def compute_social_place_fields(
    ks: "KilosortData",
    tracking: "VideoTrackingData",
    sync,
    focal_animal: str,
    target_animals: Optional[List[str]] = None,
    *,
    pixels_per_cm: Optional[float] = None,
    bin_size_cm: float = 5.0,
    smoothing_sigma_cm: float = 5.0,
    speed_threshold_cms: float = 5.0,
    speed_filter_subject: Literal["focal", "target", "none"] = "target",
    n_shuffles: int = 500,
    min_n_spikes: int = 50,
    min_occupancy_sec: float = 0.1,
    use_quality_cells: bool = True,
    quality_thresholds: Optional[dict] = None,
    t_window_ephys: Optional[Tuple[float, float]] = None,
    arena_bounds: Optional[ArenaBounds] = None,
    null_method: Literal["circular_shift", "position_shuffle"] = "circular_shift",
    sig_alpha: float = 0.01,
    seed: int = 0,
) -> SocialFieldResults:
    """Compute social place fields for every focal cell over every target animal.

    Takes the **focal** animal's :class:`KilosortData` (only the focal needs
    ephys), the session :class:`~video.tracking_import.VideoTrackingData` (which
    already contains every animal), and a
    :class:`~ingestion.ephys_sync.DataSyncManager`; tracking is resolved onto the
    ephys clock once via :func:`video.tracking_import.resolve_tracking_on_ephys_clock`.

    ``target_animals`` defaults to every animal present in the tracking file
    (including the focal animal, whose map is the self place field). For each
    focal cluster and each target the function builds a :class:`RateMap`,
    :class:`FieldStats`, and :class:`FieldSignificance`, then classifies each cell
    by which targets it is significantly tuned to (Benjamini–Hochberg FDR across
    targets on Skaggs bits/spike).
    """
    from ingestion.kilosort_data_import import _DEFAULT_QUALITY_THRESHOLDS
    from video.tracking_import import resolve_tracking_on_ephys_clock

    session_id = tracking.session_id

    if target_animals is None:
        target_animals = list(tracking.parsed_data.keys())

    t_start = t_window_ephys[0] if t_window_ephys else None
    t_end = t_window_ephys[1] if t_window_ephys else None
    animals = list(dict.fromkeys([focal_animal, *target_animals]))
    tracking_by_animal = resolve_tracking_on_ephys_clock(
        tracking, sync, animals, pixels_per_cm=pixels_per_cm,
        t_start_ephys=t_start, t_end_ephys=t_end,
    )

    if arena_bounds is None:
        arena_bounds = _bounds_from_tracking_dict(tracking_by_animal)

    # Focal cells and their spike trains.
    if use_quality_cells:
        thresholds = quality_thresholds or dict(_DEFAULT_QUALITY_THRESHOLDS)
        cluster_ids, spike_lists = ks.get_filtered_cells_spike_times(**thresholds)
    else:
        thresholds = None
        cluster_ids = list(ks.ks_ids)
        spike_lists = list(ks.spike_times_by_cell)

    if focal_animal not in tracking_by_animal:
        logger.warning("Focal animal %s has no tracking; self-map unavailable.", focal_animal)

    rate_maps: Dict[str, Dict[int, RateMap]] = {}
    stats: Dict[str, Dict[int, FieldStats]] = {}
    signif: Dict[str, Dict[int, FieldSignificance]] = {}

    for target in target_animals:
        target_xy = tracking_by_animal.get(target)
        if target_xy is None or target_xy.shape[0] < 2:
            logger.warning("No usable tracking for target %s; skipping.", target)
            continue

        if speed_filter_subject == "none":
            speed_xy = None
            thr = None
        elif speed_filter_subject == "focal":
            fdf = tracking_by_animal.get(focal_animal)
            speed_xy = fdf[["t", "speed"]] if fdf is not None else None
            thr = speed_threshold_cms if speed_xy is not None else None
        else:  # 'target'
            speed_xy = target_xy[["t", "speed"]]
            thr = speed_threshold_cms

        bin_kwargs = dict(
            bin_size_cm=bin_size_cm,
            arena_bounds=arena_bounds,
            smoothing_sigma_cm=smoothing_sigma_cm,
            min_occupancy_sec=min_occupancy_sec,
            speed_xy=speed_xy,
            speed_threshold_cms=thr,
            t_window_ephys=t_window_ephys,
            speed_filter_subject=speed_filter_subject,
        )

        rate_maps[target] = {}
        stats[target] = {}
        signif[target] = {}
        for cid, st in zip(cluster_ids, spike_lists):
            rm = compute_rate_map(st, target_xy, focal_animal=focal_animal,
                                  target_animal=target, cluster_id=cid, **bin_kwargs)
            fs = compute_field_stats(rm, st, target_xy, **bin_kwargs)
            rate_maps[target][cid] = rm
            stats[target][cid] = fs

            if fs.n_spikes_in_window < min_n_spikes:
                signif[target][cid] = FieldSignificance(
                    cluster_id=cid, target_animal=target, null_method=null_method,
                    n_shuffles=0, p_skaggs=np.nan, p_sparsity=np.nan,
                    p_split_half=np.nan, shuffle_skaggs=np.array([]),
                )
            else:
                signif[target][cid] = field_significance(
                    st, target_xy, n_shuffles=n_shuffles, null_method=null_method,
                    seed=seed, cluster_id=cid, target_animal=target, **bin_kwargs,
                )

        # Informative warning when speed gating leaves no occupancy.
        if rate_maps[target]:
            probe = next(iter(rate_maps[target].values()))
            if probe.occupancy.sum() <= 0:
                logger.warning(
                    "Target %s has zero occupancy after speed gating "
                    "(subject=%s, threshold=%s); all rate maps are empty.",
                    target, speed_filter_subject, thr,
                )

    used_targets = list(rate_maps.keys())
    cell_classification = _classify_cells(
        cluster_ids, used_targets, focal_animal, stats, signif, sig_alpha,
    )
    population_field_similarity = _population_similarity(
        cluster_ids, used_targets, focal_animal, rate_maps,
    )

    parameters = {
        "focal_animal": focal_animal,
        "target_animals": used_targets,
        "session_id": session_id,
        "bin_size_cm": bin_size_cm,
        "smoothing_sigma_cm": smoothing_sigma_cm,
        "speed_threshold_cms": speed_threshold_cms,
        "speed_filter_subject": speed_filter_subject,
        "n_shuffles": n_shuffles,
        "min_n_spikes": min_n_spikes,
        "min_occupancy_sec": min_occupancy_sec,
        "use_quality_cells": use_quality_cells,
        "quality_thresholds": thresholds,
        "t_window_ephys": t_window_ephys,
        "arena_bounds": arena_bounds,
        "null_method": null_method,
        "sig_alpha": sig_alpha,
        "class_label": CLASS_LABEL,
        "analysis_title": ANALYSIS_TITLE,
    }
    return SocialFieldResults(
        rate_maps=rate_maps,
        stats=stats,
        signif=signif,
        cell_classification=cell_classification,
        population_field_similarity=population_field_similarity,
        parameters=parameters,
    )


def _classify_cells(cluster_ids, targets, focal_animal, stats, signif, sig_alpha):
    """Build the cell-classification DataFrame (FDR across targets per cell)."""
    rows = []
    for cid in cluster_ids:
        p_by_target = {t: signif[t][cid].p_skaggs for t in targets if cid in signif[t]}
        target_list = list(p_by_target.keys())
        pvals = np.array([p_by_target[t] for t in target_list], dtype=np.float64)
        qvals = _benjamini_hochberg(pvals) if pvals.size else np.array([])
        sig_targets = {t for t, q in zip(target_list, qvals) if np.isfinite(q) and q < sig_alpha}

        self_sig = focal_animal in sig_targets
        partner_sig = sig_targets - {focal_animal}
        n_partner = len(partner_sig)

        if not sig_targets:
            category = "none"
        elif self_sig and n_partner == 0:
            category = "self_only"
        elif self_sig and n_partner >= 1:
            category = "conjunctive"
        elif not self_sig and n_partner >= 2:
            category = "broadcast"
        else:  # exactly one partner, no self
            category = "partner_only"

        # Dominant target: argmax Skaggs among significant targets.
        dominant = np.nan
        if sig_targets:
            dominant = max(
                sig_targets, key=lambda t: stats[t][cid].skaggs_bits_per_spike,
            )

        row = {
            "cluster_id": cid,
            "n_target_significant": len(sig_targets),
            "category": category,
            "dominant_target": dominant,
        }
        for t in targets:
            if cid in stats[t]:
                fs = stats[t][cid]
                row[f"bits_per_spike_{t}"] = fs.skaggs_bits_per_spike
                row[f"sparsity_{t}"] = fs.sparsity
                row[f"split_half_{t}"] = fs.split_half_corr
                row[f"p_value_{t}"] = signif[t][cid].p_skaggs
        rows.append(row)
    return pd.DataFrame(rows)


def _population_similarity(cluster_ids, targets, focal_animal, rate_maps):
    """(n_cells, n_cells) rate-map correlation between cells under target pairs.

    Pairs are (focal/self, partner) so the plot module can sort by the self
    target. Returns ``{f'{focal}__{partner}': {'similarity_matrix', 'diag_distribution'}}``.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    if focal_animal not in rate_maps:
        return out
    self_flats = [rate_maps[focal_animal][c].rates.ravel() for c in cluster_ids]
    n = len(cluster_ids)
    for partner in targets:
        if partner == focal_animal:
            continue
        partner_flats = [rate_maps[partner][c].rates.ravel() for c in cluster_ids]
        mat = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                mat[i, j] = _masked_corr(self_flats[i], partner_flats[j])
        out[f"{focal_animal}__{partner}"] = {
            "similarity_matrix": mat,
            "diag_distribution": np.diag(mat).copy(),
        }
    return out
