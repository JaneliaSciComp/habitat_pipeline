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

Self-position confound
----------------------
The default nulls (``circular_shift`` / ``position_shuffle``) break the
spike↔target-position pairing *entirely*, so the hypothesis they reject is "this
cell's firing is unrelated in time to the target's position". A pure **self**
place cell violates that hypothesis whenever the two animals' trajectories are
temporally correlated — which is the defining property of a shared arena. Such a
cell will be reported as tuned to the partner even though it encodes only its
own position.

:func:`self_position_stratum` is the direct control: restrict the analysis to
samples where the focal animal sat within ``radius_cm`` of one location, so its
own position has almost no variance and cannot generate a map over the target's
position. Pass ``self_stratum_radius_cm`` to
:func:`compute_social_place_fields`, or use
:func:`compare_self_stratum_control` to run the sweep with and without the
restriction and read off which cells survive. Three things to keep in mind:

- ``radius_cm`` must be small relative to a place field or self tuning leaks
  through anyway; the trade-off is that small strata retain little data. The
  returned diagnostics carry the retained fraction and the residual spread of
  the focal's position so the tightness of the control is auditable.
- A tight stratum keeps well under 1% of samples, so cells drop below
  ``min_n_spikes`` and stop being testable. The loss is *uneven*: a cell whose
  field coincides with the stratum keeps most of its spikes, while one that
  fires elsewhere is starved. So the control is best powered against exactly the
  cells it should kill, and worst powered for genuine partner cells — a
  disappearance is only evidence of leakage if the spike count held up. Read
  ``n_spikes_stratum`` before reading the p-value.
- Skaggs bits/spike are **not** comparable across the two runs. Fewer retained
  spikes bias it upward through noise while removing real signal pushes it down,
  and neither effect is calibrated between runs. The permutation test is still
  valid because each run builds its null from its own retained data — so compare
  p-values, not magnitudes.
- Conditioning on self *position* does not condition on self *behavioural
  state*. A cell modulated by arousal as a partner approaches is not excluded by
  this (or by any position-based) control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING

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

# Sample-level restriction mask, carried as a column on ``target_xy`` so it rides
# through the window/finite/sort transforms alongside t/x/y, and propagates for
# free through the ``**rate_map_kwargs`` chain into split_half_stability and
# field_significance.
STRATUM_COLUMN = "_in_self_stratum"

# Default self-position stratum radius. Deliberately permissive. A radius at
# place-field scale (~5 cm) is the tighter control, but measured on synthetic
# sessions it retains under 1% of samples, which starves most cells below
# min_n_spikes and leaves the control unable to detect a real partner field.
# 12 cm keeps enough data to have power, at the cost of more residual
# self-position variance inside the disc — read ``residual_spread_cm`` from the
# diagnostics to see how much, and tighten it if the retained budget allows.
DEFAULT_STRATUM_RADIUS_CM = 12.0


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


@dataclass
class StratumComparison:
    """Unrestricted vs self-position-restricted sweeps, and their per-test diff."""

    table: pd.DataFrame        # one row per (cluster_id, target)
    unrestricted: SocialFieldResults
    stratified: SocialFieldResults
    parameters: dict


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

#: ``False``/``None`` for silence, ``True`` for a bar, or ``fn(done, total, label)``.
ProgressArg = Union[bool, Callable[[int, int, str], None], None]


class _Progress:
    """Counts completed cell-target fits for the long sweeps.

    ``True`` gives a tqdm bar, falling back to periodic logging because tqdm is
    not a declared dependency of this project. A callable is invoked as
    ``fn(done, total, label)`` so a host such as the Streamlit tab can drive its
    own widget instead. The default is silence, so nothing changes for callers
    that do not ask.
    """

    def __init__(self, total: int, label: str, mode: ProgressArg = False):
        self.total = max(int(total), 0)
        self.label = label
        self.done = 0
        self._bar = None
        self._callback = mode if callable(mode) else None
        self._log = False
        # Starts at 0, not -1: the "N fits to run" line already marks the start,
        # so the first logged tick should be 10%, not 0% after one fit.
        self._last_decile = 0
        if self._callback is None and mode:
            try:
                from tqdm.auto import tqdm
                self._bar = tqdm(total=self.total, desc=label, unit="fit")
            except ImportError:
                self._log = True
                logger.info("%s: %d cell-target fits to run.", label, self.total)

    def set_label(self, label: str) -> None:
        self.label = label
        if self._bar is not None:
            self._bar.set_description(label)

    def update(self, n: int = 1) -> None:
        self.done += n
        if self._bar is not None:
            self._bar.update(n)
        elif self._callback is not None:
            self._callback(self.done, self.total, self.label)
        elif self._log and self.total:
            decile = int(10 * self.done / self.total)
            if decile > self._last_decile:
                self._last_decile = decile
                logger.info("%s: %d%% (%d/%d)", self.label,
                            int(100 * self.done / self.total),
                            self.done, self.total)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


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
# Self-position stratum (confound control)
# ---------------------------------------------------------------------------

def read_stratum_mask(xy: pd.DataFrame) -> Optional[np.ndarray]:
    """Boolean restriction mask carried on ``xy``, or ``None`` if absent."""
    if STRATUM_COLUMN not in xy.columns:
        return None
    col = pd.to_numeric(xy[STRATUM_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(col, nan=0.0) > 0.5


def _disc_kernel(radius_cm: float, bin_size_cm: float) -> np.ndarray:
    """Uniform disc of ``radius_cm``, on a ``bin_size_cm`` grid.

    A radius below one bin yields a 1x1 kernel, i.e. the identity.
    """
    r_bins = float(radius_cm) / float(bin_size_cm)
    half = int(np.floor(r_bins))
    offs = np.arange(-half, half + 1)
    yy, xx = np.meshgrid(offs, offs, indexing="ij")
    return (np.hypot(xx, yy) <= r_bins).astype(np.float64)


def modal_occupancy_center(focal_xy: pd.DataFrame, bin_size_cm: float = 5.0,
                           arena_bounds: Optional[ArenaBounds] = None,
                           smoothing_radius_cm: Optional[float] = None,
                           ) -> Tuple[float, float]:
    """Centre of the focal animal's highest dwell-time region.

    With ``smoothing_radius_cm`` set, occupancy is first convolved with a
    **uniform disc of that radius**, so the returned bin centre is the point
    whose radius-``smoothing_radius_cm`` disc holds the most dwell time. Pass the
    stratum radius you intend to use and the centre then maximises exactly the
    quantity the control is short of — retained seconds — instead of chasing a
    single tall bin, which a raw argmax will happily do on one tracking glitch.
    ``compute_social_place_fields`` passes ``self_stratum_radius_cm`` here.

    Left at ``None`` the occupancy is unsmoothed and this is a plain argmax over
    single bins; that is mostly useful for inspecting the raw dwell peak.

    Note the result is quantised to the bin grid either way, and that occupancy
    is **not** speed-gated, so the winner is wherever the animal spent the most
    wall-clock time — typically wherever it rests.
    """
    t = focal_xy["t"].to_numpy(dtype=np.float64)
    x = focal_xy["x"].to_numpy(dtype=np.float64)
    y = focal_xy["y"].to_numpy(dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[finite], x[finite], y[finite]
    if t.size < 2:
        raise ValueError("Need at least 2 finite focal tracking samples.")
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x[order], y[order]

    if arena_bounds is None:
        arena_bounds = _infer_bounds(focal_xy)
    x_edges, y_edges = _edges_from_bounds(arena_bounds, bin_size_cm)
    n_x, n_y = len(x_edges) - 1, len(y_edges) - 1

    dt = np.gradient(t)
    dt[dt < 0] = 0.0
    occ = np.zeros((n_y, n_x), dtype=np.float64)
    ix = np.clip(np.digitize(x, x_edges) - 1, 0, n_x - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, n_y - 1)
    np.add.at(occ, (iy, ix), dt)

    if smoothing_radius_cm is not None and smoothing_radius_cm > 0:
        from scipy.ndimage import convolve
        # Total dwell time inside the candidate disc centred on each bin. Zero
        # padding is correct rather than merely convenient: there is no
        # occupancy outside the arena, so edge discs really do hold less.
        occ = convolve(occ, _disc_kernel(smoothing_radius_cm, bin_size_cm),
                       mode="constant", cval=0.0)

    py, px = np.unravel_index(int(np.argmax(occ)), occ.shape)
    return (float(0.5 * (x_edges[px] + x_edges[px + 1])),
            float(0.5 * (y_edges[py] + y_edges[py + 1])))


def self_position_stratum(
    focal_xy: pd.DataFrame,
    sample_t: np.ndarray,
    *,
    radius_cm: float,
    center: Optional[Tuple[float, float]] = None,
    bin_size_cm: float = 5.0,
    arena_bounds: Optional[ArenaBounds] = None,
    max_gap_sec: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    """Mask of ``sample_t`` where the focal animal sat within ``radius_cm`` of one spot.

    This is the single-stratum form of conditioning on self position: rather than
    modelling the focal animal's contribution, hold it approximately constant and
    throw the rest away. Within the mask the focal's own position has little
    variance, so it cannot generate a rate map over a *target's* position — any
    surviving tuning is not self-position leakage.

    ``center`` defaults to the centre of the focal's modal dwell-time bin
    (:func:`modal_occupancy_center`). The focal trajectory is interpolated onto
    ``sample_t``; samples whose nearest focal observation is more than
    ``max_gap_sec`` away are excluded rather than trusted, because tracking drops
    undetected frames per animal and interpolating across a long gap invents a
    position.

    Returns ``(mask, diagnostics)``. ``diagnostics`` carries the retained
    fraction and seconds, and ``residual_spread_cm`` — the RMS distance of the
    focal's retained positions from ``center``. That last number is what makes
    the control auditable: it must be small relative to a place field, or self
    tuning leaks through the restriction anyway.
    """
    sample_t = np.asarray(sample_t, dtype=np.float64)
    ft = focal_xy["t"].to_numpy(dtype=np.float64)
    fx = focal_xy["x"].to_numpy(dtype=np.float64)
    fy = focal_xy["y"].to_numpy(dtype=np.float64)
    finite = np.isfinite(ft) & np.isfinite(fx) & np.isfinite(fy)
    ft, fx, fy = ft[finite], fx[finite], fy[finite]
    order = np.argsort(ft, kind="stable")
    ft, fx, fy = ft[order], fx[order], fy[order]

    if center is None:
        center = modal_occupancy_center(focal_xy, bin_size_cm, arena_bounds)
    cx, cy = center

    if ft.size == 0 or sample_t.size == 0:
        return np.zeros(sample_t.size, dtype=bool), {
            "center": (float(cx), float(cy)), "radius_cm": float(radius_cm),
            "n_retained": 0, "fraction_retained": 0.0, "retained_seconds": 0.0,
            "residual_spread_cm": np.nan,
        }

    x_at = np.interp(sample_t, ft, fx, left=np.nan, right=np.nan)
    y_at = np.interp(sample_t, ft, fy, left=np.nan, right=np.nan)

    # Exclude samples that fall in a tracking gap rather than interpolating it.
    right = np.clip(np.searchsorted(ft, sample_t), 1, ft.size - 1)
    gap = np.minimum(np.abs(sample_t - ft[right - 1]), np.abs(ft[right] - sample_t))
    if ft.size == 1:
        gap = np.abs(sample_t - ft[0])

    dist = np.hypot(x_at - cx, y_at - cy)
    mask = np.isfinite(dist) & (dist <= float(radius_cm)) & (gap <= float(max_gap_sec))

    dt = np.gradient(sample_t) if sample_t.size >= 2 else np.zeros(sample_t.size)
    dt[dt < 0] = 0.0
    n_ret = int(mask.sum())
    diagnostics = {
        "center": (float(cx), float(cy)),
        "radius_cm": float(radius_cm),
        "n_retained": n_ret,
        "n_samples": int(sample_t.size),
        "fraction_retained": float(n_ret / sample_t.size),
        "retained_seconds": float(dt[mask].sum()),
        "residual_spread_cm": float(np.sqrt(np.mean(dist[mask] ** 2))) if n_ret else np.nan,
        "excluded_by_gap": int(np.sum(np.isfinite(dist) & (dist <= float(radius_cm))
                                      & (gap > float(max_gap_sec)))),
    }
    return mask, diagnostics


# ---------------------------------------------------------------------------
# Binning internals
#
# Everything here is spike-independent, which is the whole point: a shuffle
# changes only which spatial bin each spike lands in, never the tracking-side
# work (dwell intervals, speed gate, digitizing, occupancy, occupancy
# smoothing). Hoisting that out of the null loop — and out of the per-cell loop
# — is where the speed comes from. ``compute_rate_map`` and the fast null path
# in ``field_significance`` both go through these functions so there is exactly
# one definition of the binning maths.
# ---------------------------------------------------------------------------

@dataclass
class _BinPrep:
    """Spike-independent binning state for one target trajectory + parameters."""

    t: np.ndarray                 # kept samples' times, ascending
    dt: np.ndarray                # dwell interval per kept sample
    keep: np.ndarray              # speed gate AND self-position stratum
    ix: np.ndarray                # x bin per kept sample
    iy: np.ndarray                # y bin per kept sample
    ix_raw: np.ndarray            # x bin per *original* row (for position rolls)
    iy_raw: np.ndarray
    sel: np.ndarray               # original-row index of each kept sample
    x_edges: np.ndarray
    y_edges: np.ndarray
    n_x: int
    n_y: int
    w0: float
    w1: float
    arena_bounds: ArenaBounds
    sigma_bins: Optional[float]   # None when smoothing is disabled
    min_occupancy_sec: float
    has_stratum: bool
    occupancy: np.ndarray         # raw dwell time per bin
    occ_smoothed: np.ndarray      # occupancy after Gaussian smoothing


def _smooth(prep: "_BinPrep", arr: np.ndarray) -> np.ndarray:
    if prep.sigma_bins is None:
        return arr
    return gaussian_filter(arr, sigma=prep.sigma_bins, mode="constant")


def _accumulate(prep: "_BinPrep", iy: np.ndarray, ix: np.ndarray,
                weights) -> np.ndarray:
    out = np.zeros((prep.n_y, prep.n_x), dtype=np.float64)
    if iy.size:
        np.add.at(out, (iy, ix), weights)
    return out


def _prepare_binning(
    target_xy: pd.DataFrame,
    bin_size_cm: float,
    arena_bounds: Optional[ArenaBounds],
    smoothing_sigma_cm: Optional[float],
    min_occupancy_sec: float,
    speed_xy: Optional[pd.DataFrame],
    speed_threshold_cms: Optional[float],
    t_window_ephys: Optional[Tuple[float, float]],
) -> _BinPrep:
    """Window, filter, sort, speed-gate, digitize and accumulate occupancy."""
    t_all = target_xy["t"].to_numpy(dtype=np.float64)
    x_all = target_xy["x"].to_numpy(dtype=np.float64)
    y_all = target_xy["y"].to_numpy(dtype=np.float64)
    stratum = read_stratum_mask(target_xy)
    n_rows = t_all.size

    if t_window_ephys is not None:
        w0, w1 = t_window_ephys
        sel = np.flatnonzero((t_all >= w0) & (t_all <= w1))
    else:
        w0 = float(t_all.min()) if n_rows else 0.0
        w1 = float(t_all.max()) if n_rows else 0.0
        sel = np.arange(n_rows)

    finite = (np.isfinite(t_all[sel]) & np.isfinite(x_all[sel])
              & np.isfinite(y_all[sel]))
    sel = sel[finite]
    sel = sel[np.argsort(t_all[sel], kind="stable")]

    t = t_all[sel]
    x = x_all[sel]
    y = y_all[sel]

    if arena_bounds is None:
        arena_bounds = _infer_bounds(target_xy) if n_rows else ((0.0, 1.0), (0.0, 1.0))
    x_edges, y_edges = _edges_from_bounds(arena_bounds, bin_size_cm)
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1

    sigma_bins = None
    if smoothing_sigma_cm is not None and smoothing_sigma_cm > 0 and n_x and n_y:
        sigma_bins = smoothing_sigma_cm / bin_size_cm

    if t.size >= 2:
        dt = np.gradient(t)
        dt[dt < 0] = 0.0

        keep = np.ones(t.size, dtype=bool)
        if stratum is not None:
            keep &= stratum[sel]
        if speed_xy is not None and speed_threshold_cms is not None:
            sp_t = speed_xy["t"].to_numpy(dtype=np.float64)
            sp_v = speed_xy["speed"].to_numpy(dtype=np.float64)
            sp_at_t = np.interp(t, sp_t, sp_v, left=np.nan, right=np.nan)
            keep &= np.isfinite(sp_at_t) & (sp_at_t >= speed_threshold_cms)

        # Digitize over every original row so a position roll can be applied as
        # a roll of bin indices (digitize is elementwise, so rolling before or
        # after it is identical) instead of re-digitizing each shuffle.
        ix_raw = np.clip(np.digitize(x_all, x_edges) - 1, 0, max(n_x - 1, 0))
        iy_raw = np.clip(np.digitize(y_all, y_edges) - 1, 0, max(n_y - 1, 0))
        ix = ix_raw[sel]
        iy = iy_raw[sel]
    else:
        dt = np.zeros(t.size)
        keep = np.zeros(t.size, dtype=bool)
        ix_raw = np.zeros(n_rows, dtype=np.int64)
        iy_raw = np.zeros(n_rows, dtype=np.int64)
        ix = ix_raw[sel]
        iy = iy_raw[sel]

    prep = _BinPrep(
        t=t, dt=dt, keep=keep, ix=ix, iy=iy, ix_raw=ix_raw, iy_raw=iy_raw,
        sel=sel, x_edges=x_edges, y_edges=y_edges, n_x=n_x, n_y=n_y,
        w0=w0, w1=w1, arena_bounds=arena_bounds, sigma_bins=sigma_bins,
        min_occupancy_sec=min_occupancy_sec, has_stratum=stratum is not None,
        occupancy=np.zeros((n_y, n_x), dtype=np.float64),
        occ_smoothed=np.zeros((n_y, n_x), dtype=np.float64),
    )
    prep.occupancy = _accumulate(prep, iy[keep], ix[keep], dt[keep])
    prep.occ_smoothed = _smooth(prep, prep.occupancy)
    return prep


# Mirrors compute_rate_map's binning-parameter defaults. A drift between the two
# would make the fast null path disagree with the observed map, so
# tests/test_social_spatial_fields.py::TestFastNullPath pins them together.
_BINNING_DEFAULTS = dict(
    bin_size_cm=5.0, arena_bounds=None, smoothing_sigma_cm=5.0,
    min_occupancy_sec=0.1, speed_xy=None, speed_threshold_cms=5.0,
    t_window_ephys=None,
)


def _prep_from_kwargs(target_xy: pd.DataFrame, rate_map_kwargs: dict) -> _BinPrep:
    """Build a :class:`_BinPrep` from a partial ``compute_rate_map`` kwarg dict."""
    kw = {k: rate_map_kwargs.get(k, v) for k, v in _BINNING_DEFAULTS.items()}
    return _prepare_binning(target_xy, **kw)


def _spike_sample_indices(prep: _BinPrep, spike_times: np.ndarray) -> np.ndarray:
    """Kept-sample index nearest each in-window spike, gated samples dropped."""
    t = prep.t
    if t.size < 2:
        return np.empty(0, dtype=np.int64)
    st = np.asarray(spike_times, dtype=np.float64)
    st = st[(st >= prep.w0) & (st <= prep.w1) & (st >= t[0]) & (st <= t[-1])]
    if not st.size:
        return np.empty(0, dtype=np.int64)
    right = np.clip(np.searchsorted(t, st), 1, t.size - 1)
    left = right - 1
    idx = np.where((st - t[left]) <= (t[right] - st), left, right)
    return idx[prep.keep[idx]]


def _rates_from_counts(prep: _BinPrep, spike_counts: np.ndarray,
                       occupancy: np.ndarray, occ_smoothed: np.ndarray
                       ) -> np.ndarray:
    """Smoothed counts / smoothed occupancy, NaN where raw occupancy is too low."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = _smooth(prep, spike_counts) / occ_smoothed
    rates[occupancy < prep.min_occupancy_sec] = np.nan
    rates[~np.isfinite(rates)] = np.nan
    return rates


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
    prep = _prepare_binning(
        target_xy, bin_size_cm, arena_bounds, smoothing_sigma_cm,
        min_occupancy_sec, speed_xy, speed_threshold_cms, t_window_ephys,
    )
    idx = _spike_sample_indices(prep, spike_times)
    spike_counts = _accumulate(prep, prep.iy[idx], prep.ix[idx], 1.0)
    rates = _rates_from_counts(prep, spike_counts, prep.occupancy,
                               prep.occ_smoothed)

    parameters = {
        "bin_size_cm": bin_size_cm,
        "smoothing_sigma_cm": smoothing_sigma_cm,
        "speed_threshold_cms": speed_threshold_cms,
        "arena_bounds": prep.arena_bounds,
        "min_occupancy_sec": min_occupancy_sec,
        "t_window_ephys": (prep.w0, prep.w1),
        "speed_filter_subject": speed_filter_subject,
        "self_stratum_applied": prep.has_stratum,
        "class_label": CLASS_LABEL,
        "analysis_title": ANALYSIS_TITLE,
    }
    return RateMap(
        rates=rates,
        occupancy=prep.occupancy,
        spike_counts=spike_counts,
        x_edges=prep.x_edges,
        y_edges=prep.y_edges,
        focal_animal=focal_animal,
        target_animal=target_animal,
        cluster_id=cluster_id,
        parameters=parameters,
    )


# ---------------------------------------------------------------------------
# Spatial statistics
# ---------------------------------------------------------------------------

def _valid_pr_arrays(rates: np.ndarray, occ: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (p_i, r_i) over valid bins: occupancy probability and rate."""
    valid = np.isfinite(rates) & (occ > 0)
    r = rates[valid].astype(np.float64)
    o = occ[valid].astype(np.float64)
    total = o.sum()
    if total <= 0 or r.size == 0:
        return np.array([]), np.array([])
    return o / total, r


def _valid_pr(rate_map: RateMap) -> Tuple[np.ndarray, np.ndarray]:
    return _valid_pr_arrays(rate_map.rates, rate_map.occupancy)


def _skaggs_from_pr(p: np.ndarray, r: np.ndarray) -> Tuple[float, float]:
    if p.size == 0:
        return 0.0, 0.0
    mean_rate = float(np.sum(p * r))
    if mean_rate <= 0:
        return 0.0, 0.0
    pos = r > 0
    ratio = r[pos] / mean_rate
    bits_per_sec = float(np.sum(p[pos] * r[pos] * np.log2(ratio)))
    return bits_per_sec / mean_rate, bits_per_sec


def _sparsity_from_pr(p: np.ndarray, r: np.ndarray) -> float:
    if p.size == 0:
        return np.nan
    num = float(np.sum(p * r)) ** 2
    den = float(np.sum(p * r ** 2))
    if den <= 0:
        return np.nan
    return num / den


def spatial_information(rate_map: RateMap) -> Tuple[float, float]:
    """Skaggs spatial information: ``(bits_per_spike, bits_per_second)``."""
    p, r = _valid_pr(rate_map)
    return _skaggs_from_pr(p, r)


def spatial_sparsity(rate_map: RateMap) -> float:
    """Occupancy-weighted sparsity ``(<r>)^2 / <r^2>`` (lower = more selective)."""
    p, r = _valid_pr(rate_map)
    return _sparsity_from_pr(p, r)


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
    shuffle_split_half: bool = False,
    _prep: Optional[_BinPrep] = None,
    **rate_map_kwargs,
) -> FieldSignificance:
    """Shuffle-based significance for a single rate map.

    ``circular_shift`` rigidly time-shifts the spike train within the window
    (preserves firing rate + autocorrelation); ``position_shuffle`` cyclically
    rolls the target ``(x, y)`` relative to the spikes. In both cases the shift
    magnitude is drawn from ``[0.1 T, 0.9 T]`` of the window.

    For each shuffle the Skaggs bits/spike and sparsity are recomputed.
    P-values are one-tailed in the meaningful direction: ``p_skaggs`` and
    ``p_split`` are ``fraction(shuffle >= true)``; ``p_sparsity`` is
    ``fraction(shuffle <= true)`` because lower sparsity = more selective.

    ``shuffle_split_half`` defaults to **False**: recomputing the split-half
    correlation for every surrogate costs two extra rate maps per shuffle —
    measured at roughly half the total runtime — and nothing in the repo reads
    the resulting ``p_split_half`` (the plots use the *observed*
    ``FieldStats.split_half_corr``, which is always computed). Turn it on only if
    you specifically want that null; ``p_split_half`` is NaN otherwise.

    The null loop hoists every spike-independent quantity out of the iteration
    (see :func:`_prepare_binning`), and ``_prep`` lets a caller sweeping many
    cells over one target share that work across cells too.

    When ``target_xy`` carries a restriction mask (:data:`STRATUM_COLUMN`),
    prefer ``position_shuffle``: it rolls positions *within* the retained samples
    only, so every surrogate keeps the observed occupancy and the observed
    retained spike count. ``circular_shift`` moves spikes in wall-clock time, so
    under a fragmented mask a surrogate retains a different number of spikes than
    the observed train — noisier maps, higher Skaggs in the null, and a
    correspondingly conservative test.
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

    if null_method not in ("circular_shift", "position_shuffle"):
        raise ValueError(f"Unknown null_method: {null_method!r}")

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

    prep = _prep if _prep is not None else _prep_from_kwargs(target_xy, rate_map_kwargs)
    stratum0 = read_stratum_mask(target_xy)
    roll_rows = np.flatnonzero(stratum0) if stratum0 is not None else None
    # Spike times are fixed under position_shuffle, so their nearest-sample
    # indices are too — resolve them once instead of per shuffle.
    idx_fixed = (_spike_sample_indices(prep, st)
                 if null_method == "position_shuffle" else None)

    for i in range(n_shuffles):
        tau = rng.uniform(0.1 * span, 0.9 * span)
        if null_method == "circular_shift":
            # Positions are untouched, so occupancy and its smoothing are the
            # observed ones and need no recomputing.
            sp_i = w0 + np.mod(st - w0 + tau, span)
            idx = _spike_sample_indices(prep, sp_i)
            occ_i, occ_s_i = prep.occupancy, prep.occ_smoothed
            iy_i, ix_i = prep.iy, prep.ix
        else:
            k = int(round(tau / max(median_dt, 1e-9)))
            if roll_rows is None:
                ix_r, iy_r = np.roll(prep.ix_raw, k), np.roll(prep.iy_raw, k)
            else:
                # Roll only within the retained samples. Rolling the full array
                # would pair in-stratum times with out-of-stratum positions, so
                # the surrogate maps would no longer share the observed
                # occupancy. Rolling within keeps occupancy and the retained
                # spike count exactly fixed, making the null a pure re-pairing.
                k_eff = int(round(k * roll_rows.size / max(prep.ix_raw.size, 1)))
                if roll_rows.size > 1:
                    k_eff = max(k_eff, 1)
                ix_r, iy_r = prep.ix_raw.copy(), prep.iy_raw.copy()
                ix_r[roll_rows] = np.roll(prep.ix_raw[roll_rows], k_eff)
                iy_r[roll_rows] = np.roll(prep.iy_raw[roll_rows], k_eff)
            # Rolling bin indices == rolling positions then digitizing, since
            # digitize is elementwise.
            ix_i, iy_i = ix_r[prep.sel], iy_r[prep.sel]
            occ_i = _accumulate(prep, iy_i[prep.keep], ix_i[prep.keep],
                                prep.dt[prep.keep])
            occ_s_i = _smooth(prep, occ_i)
            idx = idx_fixed

        counts = _accumulate(prep, iy_i[idx], ix_i[idx], 1.0)
        rates = _rates_from_counts(prep, counts, occ_i, occ_s_i)
        p_i, r_i = _valid_pr_arrays(rates, occ_i)
        sh_skaggs[i] = _skaggs_from_pr(p_i, r_i)[0]
        sh_sparsity[i] = _sparsity_from_pr(p_i, r_i)

        if shuffle_split_half:
            # Opt-in only: this rebuilds two more rate maps per shuffle and was
            # measured at ~half the total runtime.
            if null_method == "circular_shift":
                sh_split[i] = split_half_stability(sp_i, target_xy, **rate_map_kwargs)
            else:
                x0 = target_xy["x"].to_numpy()
                y0 = target_xy["y"].to_numpy()
                if roll_rows is None:
                    xi, yi = np.roll(x0, k), np.roll(y0, k)
                else:
                    xi, yi = x0.copy(), y0.copy()
                    xi[roll_rows] = np.roll(x0[roll_rows], k_eff)
                    yi[roll_rows] = np.roll(y0[roll_rows], k_eff)
                xy_i = target_xy.copy()
                xy_i["x"], xy_i["y"] = xi, yi
                sh_split[i] = split_half_stability(st, xy_i, **rate_map_kwargs)

    # Add-one ("plus-one") estimator: (1 + #exceedances) / (1 + n_shuffles).
    # A finite permutation test cannot justify p == 0 — the plain k/n form
    # returns exactly 0.0 when no shuffle beats the observed value, which then
    # survives Benjamini-Hochberg as q == 0 and reads as infinite confidence.
    # The floor here is 1/(n+1), matching
    # ``ephys._lda_decoding.compute_population_significance``.
    def _p_geq(true_val, shuffles):
        valid = shuffles[np.isfinite(shuffles)]
        if not np.isfinite(true_val) or valid.size == 0:
            return np.nan
        return float((1 + np.sum(valid >= true_val)) / (1 + valid.size))

    def _p_leq(true_val, shuffles):
        valid = shuffles[np.isfinite(shuffles)]
        if not np.isfinite(true_val) or valid.size == 0:
            return np.nan
        return float((1 + np.sum(valid <= true_val)) / (1 + valid.size))

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
    shuffle_split_half: bool = False,
    self_stratum_radius_cm: Optional[float] = None,
    self_stratum_center: Optional[Tuple[float, float]] = None,
    self_stratum_max_gap_sec: float = 1.0,
    progress: ProgressArg = False,
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

    Setting ``self_stratum_radius_cm`` restricts every map to samples where the
    focal animal sat within that radius of ``self_stratum_center`` (default: the
    bin whose disc of that same radius holds the most focal dwell time, via
    :func:`modal_occupancy_center`), which is the confound control described
    in the module docstring — see :func:`self_position_stratum`. Two notes on
    reading the output under a stratum: the **self** target is then a positive
    control that is *expected* to collapse, since the focal is confined by
    construction; and because occupancy is sparser, bits/spike is inflated
    relative to the unrestricted run, so compare p-values rather than effect
    sizes. :func:`compare_self_stratum_control` runs both and tabulates the
    difference.

    The FDR family here is targets within a cell, not cells. It does not control
    the false-discovery rate over the cell population; take the denominator for
    any population claim from ``LabNotebook.family_denominator``.

    ``progress`` reports completed cell-target fits: ``True`` for a bar, or a
    ``fn(done, total, label)`` callable to drive your own widget. Off by default.
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

    # Resolve the self-position stratum centre once, so every target is
    # restricted to the same region of the focal animal's trajectory.
    stratum_center = self_stratum_center
    if self_stratum_radius_cm is not None:
        focal_df = tracking_by_animal.get(focal_animal)
        if focal_df is None or focal_df.shape[0] < 2:
            raise ValueError(
                f"self_stratum_radius_cm requires tracking for the focal animal "
                f"{focal_animal!r}, which has none in session {session_id}."
            )
        if stratum_center is None:
            # Disc kernel matched to the stratum radius, so the centre maximises
            # the dwell time the stratum will actually retain.
            stratum_center = modal_occupancy_center(
                focal_df, bin_size_cm, arena_bounds,
                smoothing_radius_cm=self_stratum_radius_cm,
            )
        if null_method == "circular_shift":
            logger.warning(
                "null_method='circular_shift' under a self-position stratum gives "
                "a conservative test: shifted spike trains land on a different "
                "number of retained samples than the observed train. Prefer "
                "null_method='position_shuffle'."
            )

    rate_maps: Dict[str, Dict[int, RateMap]] = {}
    stats: Dict[str, Dict[int, FieldStats]] = {}
    signif: Dict[str, Dict[int, FieldSignificance]] = {}
    stratum_diagnostics: Dict[str, dict] = {}

    # Resolve which targets are actually runnable first, so the progress total
    # is the real number of cell-target fits rather than an optimistic one.
    usable_targets = []
    for target in target_animals:
        tdf = tracking_by_animal.get(target)
        if tdf is None or tdf.shape[0] < 2:
            logger.warning("No usable tracking for target %s; skipping.", target)
        else:
            usable_targets.append(target)

    base_label = f"social fields: focal {focal_animal}"
    if self_stratum_radius_cm is not None:
        base_label += " [stratum]"
    reporter = _Progress(len(usable_targets) * len(cluster_ids),
                         base_label, progress)

    for target in usable_targets:
        target_xy = tracking_by_animal[target]
        reporter.set_label(f"{base_label} -> {target}")

        if self_stratum_radius_cm is not None:
            mask, diag = self_position_stratum(
                tracking_by_animal[focal_animal], target_xy["t"].to_numpy(),
                radius_cm=self_stratum_radius_cm, center=stratum_center,
                bin_size_cm=bin_size_cm, arena_bounds=arena_bounds,
                max_gap_sec=self_stratum_max_gap_sec,
            )
            target_xy = target_xy.assign(**{STRATUM_COLUMN: mask})
            stratum_diagnostics[target] = diag
            if diag["n_retained"] == 0:
                logger.warning(
                    "Self-position stratum retains no samples for target %s "
                    "(center=%s, radius=%s).", target, stratum_center,
                    self_stratum_radius_cm,
                )
            elif diag["retained_seconds"] < 60.0:
                logger.warning(
                    "Self-position stratum retains only %.1f s for target %s "
                    "(%.1f%% of samples); consider a larger radius, and raise "
                    "min_occupancy_sec above its permissive %.2f s default.",
                    diag["retained_seconds"], target,
                    100.0 * diag["fraction_retained"], min_occupancy_sec,
                )

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

        # One binning prep per target, shared by every cell and every shuffle:
        # the tracking-side work does not depend on which cell we are looking at.
        prep = _prep_from_kwargs(target_xy, bin_kwargs)

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
                    seed=seed, cluster_id=cid, target_animal=target,
                    shuffle_split_half=shuffle_split_half, _prep=prep,
                    **bin_kwargs,
                )
            reporter.update()

        # Informative warning when speed gating leaves no occupancy.
        if rate_maps[target]:
            probe = next(iter(rate_maps[target].values()))
            if probe.occupancy.sum() <= 0:
                logger.warning(
                    "Target %s has zero occupancy after speed gating "
                    "(subject=%s, threshold=%s); all rate maps are empty.",
                    target, speed_filter_subject, thr,
                )

    reporter.close()

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
        "shuffle_split_half": shuffle_split_half,
        "self_stratum_radius_cm": self_stratum_radius_cm,
        "self_stratum_center": stratum_center,
        "self_stratum_diagnostics": stratum_diagnostics,
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


# ---------------------------------------------------------------------------
# Self-position confound control: paired sweep
# ---------------------------------------------------------------------------

def compare_self_stratum_control(
    ks: "KilosortData",
    tracking: "VideoTrackingData",
    sync,
    focal_animal: str,
    target_animals: Optional[List[str]] = None,
    *,
    self_stratum_radius_cm: float = DEFAULT_STRATUM_RADIUS_CM,
    self_stratum_center: Optional[Tuple[float, float]] = None,
    null_method: Literal["circular_shift", "position_shuffle"] = "position_shuffle",
    progress: ProgressArg = False,
    **kwargs,
) -> StratumComparison:
    """Run the sweep with and without the self-position restriction, and diff them.

    This is the readout for the confound: a cell whose partner tuning is really
    its own place field loses significance once the focal animal is held in one
    place, while a cell that genuinely tracks the partner survives. The returned
    ``table`` has one row per ``(cluster_id, target)`` with ``p_full`` /
    ``p_stratum``, the two Skaggs values, retained spike counts, and
    ``survives_stratum``.

    Two things deserve separate reading. The **self** target is a positive
    control that should collapse — the focal animal is confined by construction,
    so its own map cannot be significant. And ``lost_to_stratum`` is only
    evidence of self-position leakage when ``n_spikes_stratum`` stayed well above
    ``min_n_spikes``; a tight stratum starves cells whose fields sit away from
    it, and a cell that vanishes for want of spikes has not been shown to be a
    confound. Compare ``p_full`` against ``p_stratum``, never ``bits_full``
    against ``bits_stratum`` — the two runs bias Skaggs differently and the
    magnitudes are not on a common scale.

    Note this doubles the test count for the multiple-comparisons ledger. Declare
    both runs up front via ``LabNotebook.declare_family_tests`` rather than
    reporting whichever one is cleaner.
    """
    common = dict(kwargs)
    common.pop("self_stratum_radius_cm", None)

    # Two sequential sweeps, so two bars; the second labels itself [stratum].
    res_full = compute_social_place_fields(
        ks, tracking, sync, focal_animal, target_animals,
        null_method=null_method, progress=progress, **common,
    )
    res_strat = compute_social_place_fields(
        ks, tracking, sync, focal_animal, target_animals,
        null_method=null_method,
        self_stratum_radius_cm=self_stratum_radius_cm,
        self_stratum_center=self_stratum_center,
        progress=progress, **common,
    )

    alpha = res_full.parameters["sig_alpha"]
    sig_full = _significant_targets_by_cell(res_full, alpha)
    sig_strat = _significant_targets_by_cell(res_strat, alpha)

    rows = []
    for target in res_full.parameters["target_animals"]:
        for cid in res_full.stats.get(target, {}):
            in_strat = cid in res_strat.stats.get(target, {})
            rows.append({
                "cluster_id": cid,
                "target": target,
                "is_self": target == focal_animal,
                "p_full": res_full.signif[target][cid].p_skaggs,
                "p_stratum": res_strat.signif[target][cid].p_skaggs if in_strat else np.nan,
                "bits_full": res_full.stats[target][cid].skaggs_bits_per_spike,
                "bits_stratum": (res_strat.stats[target][cid].skaggs_bits_per_spike
                                 if in_strat else np.nan),
                "n_spikes_full": res_full.stats[target][cid].n_spikes_in_window,
                "n_spikes_stratum": (res_strat.stats[target][cid].n_spikes_in_window
                                     if in_strat else 0),
                "sig_full": target in sig_full.get(cid, set()),
                "survives_stratum": target in sig_strat.get(cid, set()),
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["lost_to_stratum"] = table["sig_full"] & ~table["survives_stratum"]

    parameters = {
        "focal_animal": focal_animal,
        "self_stratum_radius_cm": self_stratum_radius_cm,
        "self_stratum_center": res_strat.parameters["self_stratum_center"],
        "self_stratum_diagnostics": res_strat.parameters["self_stratum_diagnostics"],
        "null_method": null_method,
        "sig_alpha": alpha,
        "class_label": CLASS_LABEL,
        "analysis_title": f"{ANALYSIS_TITLE} — self-position stratum control",
    }
    return StratumComparison(table=table, unrestricted=res_full,
                             stratified=res_strat, parameters=parameters)


def _significant_targets_by_cell(res: SocialFieldResults, alpha: float
                                 ) -> Dict[int, set]:
    """``{cluster_id: {significant targets}}``, reusing the run's own BH family."""
    out: Dict[int, set] = {}
    targets = res.parameters["target_animals"]
    for _, row in res.cell_classification.iterrows():
        cid = row["cluster_id"]
        pvals = np.array(
            [row.get(f"p_value_{t}", np.nan) for t in targets], dtype=np.float64)
        qvals = _benjamini_hochberg(pvals) if pvals.size else np.array([])
        out[cid] = {t for t, q in zip(targets, qvals)
                    if np.isfinite(q) and q < alpha}
    return out
