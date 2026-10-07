"""Numeric territory maps from multi-animal tracking.

Turns the descriptive occupancy plots in ``plot_trajectory`` into quantities an
analysis can use: an owner map, a signed distance to each animal's own-territory
boundary, and stability metrics between maps.

Conventions
-----------
* Input is ``{animal: DataFrame[t, x, y, ...]}`` in *ephys seconds*, as returned by
  ``video.tracking_import.resolve_tracking_on_ephys_clock``. Units of ``x``/``y``
  are whatever the caller passed (pixels for APT until ``pixels_per_cm`` is
  re-measured); every distance here is in those same units.
* Occupancy is **dwell time in seconds**, not frame counts, so a gap in one
  animal's tracking does not read as absence of dwell and frame-rate differences
  between tracking formats do not matter.
* By default each animal's map is normalised by that animal's own tracked time
  (``normalize=True``), so "owner" means "where this animal spends its time
  relative to the others" rather than "who was tracked longest".
* A territory map must never be built from the data it is later used to test:
  use :func:`build_territory_map_loo`, which refuses a test chunk that is also in
  the training set.

Signed boundary distance is positive inside an animal's own territory and
negative outside; it is computed per animal against that animal's own mask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

logger = logging.getLogger(__name__)

Tracking = Mapping[str, pd.DataFrame]

UNCLAIMED = -1


@dataclass
class TerritoryMap:
    """Owner map plus the per-animal occupancy it was built from."""

    animal_ids: List[str]                  # order matches the first axis of ``share``
    x_edges: np.ndarray
    y_edges: np.ndarray
    dwell_sec: Dict[str, np.ndarray]       # raw seconds per bin, (nx, ny)
    share: np.ndarray                      # (n_animals, nx, ny), smoothed, normalised
    owner: np.ndarray                      # (nx, ny) int index into animal_ids, UNCLAIMED if none
    excluded: Dict[str, str] = field(default_factory=dict)   # animal -> reason
    parameters: Dict[str, object] = field(default_factory=dict)

    @property
    def bin_size(self) -> Tuple[float, float]:
        return float(np.mean(np.diff(self.x_edges))), float(np.mean(np.diff(self.y_edges)))

    def owner_mask(self, animal: str) -> np.ndarray:
        if animal not in self.animal_ids:
            raise KeyError(f"{animal!r} has no territory in this map "
                           f"(mapped: {self.animal_ids}; excluded: {self.excluded})")
        return self.owner == self.animal_ids.index(animal)

    def territory_fraction(self) -> Dict[str, float]:
        total = self.owner.size
        return {a: float(np.sum(self.owner == i)) / total for i, a in enumerate(self.animal_ids)}


# ---------------------------------------------------------------------------
# Occupancy
# ---------------------------------------------------------------------------

def _xy_dt(df: pd.DataFrame, max_gap_sec: float,
           t_window: Optional[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid (x, y) samples with the time each one represents.

    A sample's dwell is the gap to the next sample, capped at ``max_gap_sec`` so
    a tracking dropout contributes neither a huge dwell nor a fabricated path.
    """
    for col in ("t", "x", "y"):
        if col not in df.columns:
            raise ValueError(f"tracking frame needs a {col!r} column; has {list(df.columns)}")
    t = df["t"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x[order], y[order]
    dt = np.diff(t, append=np.nan)
    dt = np.where(np.isfinite(dt) & (dt > 0), np.minimum(dt, max_gap_sec), 0.0)
    keep = np.isfinite(x) & np.isfinite(y) & (dt > 0)
    if t_window is not None:
        keep &= (t >= t_window[0]) & (t < t_window[1])
    return x[keep], y[keep], dt[keep]


def dwell_time_maps(tracking: Tracking, x_edges: np.ndarray, y_edges: np.ndarray, *,
                    max_gap_sec: float = 0.5,
                    t_window: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
    """Per-animal dwell time (seconds) on the given grid."""
    maps: Dict[str, np.ndarray] = {}
    for name, df in tracking.items():
        x, y, dt = _xy_dt(df, max_gap_sec, t_window)
        if x.size == 0:
            continue
        h, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=dt)
        maps[name] = h
    return maps


def _bounds(tracking: Tracking) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for df in tracking.values():
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.any():
            xs.append((x[ok].min(), x[ok].max()))
            ys.append((y[ok].min(), y[ok].max()))
    if not xs:
        raise ValueError("no valid coordinates in any animal's tracking")
    return (min(a for a, _ in xs), max(b for _, b in xs),
            min(a for a, _ in ys), max(b for _, b in ys))


# ---------------------------------------------------------------------------
# Territory map
# ---------------------------------------------------------------------------

def compute_territory_map(
    tracking: Tracking, *,
    bins: int = 40,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    smoothing_sigma_bins: float = 1.0,
    min_dwell_sec: float = 1.0,
    min_tracked_sec: float = 300.0,
    min_spread: float = 0.0,
    normalize: bool = True,
    max_gap_sec: float = 0.5,
    t_window: Optional[Tuple[float, float]] = None,
    dwell_override: Optional[Dict[str, np.ndarray]] = None,
    edges: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> TerritoryMap:
    """Build an owner map from all animals' tracking.

    Parameters
    ----------
    min_dwell_sec : a bin is UNCLAIMED unless the winning animal's *raw* dwell in
        it (before smoothing/normalisation) reaches this many seconds.
    min_tracked_sec : animals tracked for less than this are left out of the map
        (a mostly-missing animal would otherwise own nothing, or worse, noise).
    min_spread : animals whose x or y standard deviation is below this are left
        out - a near-stationary animal wins every bin it sits in (HZ-STAT-007).
        Same units as the coordinates; 0 disables.
    dwell_override / edges : supply pre-pooled dwell maps (used by the
        leave-one-chunk-out builder); ``tracking`` is then only consulted for
        the exclusion criteria and may be a pooled frame.
    """
    if not tracking and dwell_override is None:
        raise ValueError("no tracking data provided")

    if edges is not None:
        x_edges, y_edges = (np.asarray(e, dtype=float) for e in edges)
    else:
        x_min, x_max, y_min, y_max = bounds if bounds is not None else _bounds(tracking)
        x_edges = np.linspace(x_min, x_max, bins + 1)
        y_edges = np.linspace(y_min, y_max, bins + 1)

    excluded: Dict[str, str] = {}
    if dwell_override is None:
        raw = dwell_time_maps(tracking, x_edges, y_edges,
                              max_gap_sec=max_gap_sec, t_window=t_window)
    else:
        raw = dict(dwell_override)

    names: List[str] = []
    for name in (list(tracking.keys()) if tracking else list(raw.keys())):
        if name not in raw:
            excluded[name] = "no valid samples"
            continue
        tracked = float(raw[name].sum())
        if tracked < min_tracked_sec:
            excluded[name] = f"tracked {tracked:.0f}s < min_tracked_sec={min_tracked_sec:g}"
            continue
        if min_spread > 0 and tracking and name in tracking:
            x, y, _ = _xy_dt(tracking[name], max_gap_sec, t_window)
            spread = float(min(np.std(x), np.std(y))) if x.size else 0.0
            if spread < min_spread:
                excluded[name] = (f"spread {spread:.1f} < min_spread={min_spread:g} "
                                  "(near-stationary)")
                continue
        names.append(name)
    for name, reason in excluded.items():
        logger.warning("territory map: %s excluded (%s)", name, reason)
    if len(names) < 2:
        raise ValueError(f"need >=2 animals to define territories; usable={names}, "
                         f"excluded={excluded}")

    stack = np.stack([raw[n] for n in names], axis=0)               # seconds
    share = np.empty_like(stack)
    for i in range(len(names)):
        s = ndimage.gaussian_filter(stack[i], sigma=smoothing_sigma_bins, mode="constant") \
            if smoothing_sigma_bins > 0 else stack[i].copy()
        if normalize:
            s = s / max(float(stack[i].sum()), 1e-12)
        share[i] = s

    winner = share.argmax(axis=0)
    raw_at_winner = np.take_along_axis(stack, winner[None], axis=0)[0]
    owner = np.where(raw_at_winner >= min_dwell_sec, winner, UNCLAIMED).astype(int)

    return TerritoryMap(
        animal_ids=names, x_edges=x_edges, y_edges=y_edges,
        dwell_sec={n: raw[n] for n in names}, share=share, owner=owner, excluded=excluded,
        parameters=dict(bins=len(x_edges) - 1, smoothing_sigma_bins=smoothing_sigma_bins,
                        min_dwell_sec=min_dwell_sec, min_tracked_sec=min_tracked_sec,
                        min_spread=min_spread, normalize=normalize, max_gap_sec=max_gap_sec,
                        t_window=t_window),
    )


def build_territory_map_loo(
    chunks: Mapping[str, Tracking], test_chunk: str, *,
    bins: int = 40,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    train_chunks: Optional[Sequence[str]] = None,
    **kwargs,
) -> TerritoryMap:
    """Territory map pooled over every chunk *except* ``test_chunk``.

    The territory used to label a chunk must come from other data, otherwise
    "territory" is partly defined by the very samples whose neural activity is
    being tested against it. ``train_chunks`` may narrow the pool but may never
    contain ``test_chunk``.
    """
    if test_chunk not in chunks:
        raise KeyError(f"test chunk {test_chunk!r} not among chunks {sorted(chunks)}")
    if train_chunks is None:
        train_chunks = [c for c in chunks if c != test_chunk]
    if test_chunk in train_chunks:
        raise ValueError(f"leakage: test chunk {test_chunk!r} is in the territory "
                         "training set")
    if not train_chunks:
        raise ValueError("no training chunks left after excluding the test chunk")

    # one shared grid, from the training chunks only
    pooled_bounds = bounds
    if pooled_bounds is None:
        bs = [_bounds(chunks[c]) for c in train_chunks]
        pooled_bounds = (min(b[0] for b in bs), max(b[1] for b in bs),
                         min(b[2] for b in bs), max(b[3] for b in bs))
    x_edges = np.linspace(pooled_bounds[0], pooled_bounds[1], bins + 1)
    y_edges = np.linspace(pooled_bounds[2], pooled_bounds[3], bins + 1)

    max_gap = kwargs.get("max_gap_sec", 0.5)
    pooled: Dict[str, np.ndarray] = {}
    for c in train_chunks:
        for name, h in dwell_time_maps(chunks[c], x_edges, y_edges, max_gap_sec=max_gap).items():
            pooled[name] = pooled.get(name, 0.0) + h

    # spread criterion needs coordinates: concatenate the training chunks per animal
    coords: Dict[str, pd.DataFrame] = {}
    for c in train_chunks:
        for name, df in chunks[c].items():
            coords.setdefault(name, []).append(df[["t", "x", "y"]])
    coords = {n: pd.concat(v, ignore_index=True) for n, v in coords.items()}

    tmap = compute_territory_map(coords, edges=(x_edges, y_edges), dwell_override=pooled,
                                 **kwargs)
    tmap.parameters.update(test_chunk=test_chunk, train_chunks=list(train_chunks))
    return tmap


# ---------------------------------------------------------------------------
# Per-sample labels and boundary distance
# ---------------------------------------------------------------------------

def signed_distance_from_mask(mask: np.ndarray, sampling: Tuple[float, float]) -> np.ndarray:
    """Signed distance (coordinate units) to the boundary of a boolean grid mask.

    Positive inside, negative outside; the boundary sits half a bin either side.
    An empty mask gives ``-inf`` everywhere. Usable on displaced/shifted masks.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, -np.inf)
    d_in = ndimage.distance_transform_edt(mask, sampling=sampling)
    d_out = ndimage.distance_transform_edt(~mask, sampling=sampling)
    half = 0.5 * min(sampling)
    return np.where(mask, d_in - half, -(d_out - half))


def signed_boundary_distance(tmap: TerritoryMap, animal: str) -> np.ndarray:
    """Signed distance (coordinate units) to the boundary of ``animal``'s territory.

    Positive inside the animal's own territory, negative outside; each axis is
    scaled by its own bin size.
    """
    return signed_distance_from_mask(tmap.owner_mask(animal), tmap.bin_size)


def exclusivity_map(tmap: TerritoryMap, animal: str, *, min_total_frac: float = 0.05) -> np.ndarray:
    """Continuous territory measure: the animal's share of everyone's occupancy, per bin.

    ``share[animal] / sum(share)`` in [0, 1] using the smoothed, per-animal-normalised
    occupancy the owner map is built from; ``1/n_animals`` means the bin is used
    like everyone else's, 1 means only this animal uses it. Unlike the winner-takes-all
    owner map, it does not flip when two animals with similar occupancy swap rank.
    Bins where total occupancy is below ``min_total_frac`` of the median occupied
    bin are NaN (nobody uses them, so "exclusivity" is meaningless).
    """
    if animal not in tmap.animal_ids:
        raise KeyError(f"{animal!r} has no territory in this map "
                       f"(mapped: {tmap.animal_ids}; excluded: {tmap.excluded})")
    total = tmap.share.sum(axis=0)
    occupied = total[total > 0]
    floor = min_total_frac * float(np.median(occupied)) if occupied.size else np.inf
    with np.errstate(invalid="ignore", divide="ignore"):
        ex = tmap.share[tmap.animal_ids.index(animal)] / total
    return np.where(total > floor, ex, np.nan)


def label_positions(tmap: TerritoryMap, animal: str, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Territory labels for sample positions of ``animal``.

    Columns: ``owner`` (animal name or ``None`` when unclaimed / off grid),
    ``own`` (bool), ``foreign`` (bool, in someone else's territory),
    ``signed_dist``, ``exclusivity`` (see :func:`exclusivity_map`; NaN for an unmapped
    animal or an unused bin), ``in_grid``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ix = np.searchsorted(tmap.x_edges, x, side="right") - 1
    iy = np.searchsorted(tmap.y_edges, y, side="right") - 1
    # the right-most edge belongs to the last bin
    ix = np.where(x == tmap.x_edges[-1], len(tmap.x_edges) - 2, ix)
    iy = np.where(y == tmap.y_edges[-1], len(tmap.y_edges) - 2, iy)
    in_grid = (np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (iy >= 0)
               & (ix < len(tmap.x_edges) - 1) & (iy < len(tmap.y_edges) - 1))
    ix_c = np.clip(ix, 0, len(tmap.x_edges) - 2)
    iy_c = np.clip(iy, 0, len(tmap.y_edges) - 2)

    own_idx = tmap.animal_ids.index(animal) if animal in tmap.animal_ids else None
    dist_grid = signed_boundary_distance(tmap, animal) if own_idx is not None else None
    cell_owner = tmap.owner[ix_c, iy_c]

    owner_name = np.array([tmap.animal_ids[i] if i != UNCLAIMED else None for i in cell_owner],
                          dtype=object)
    owner_name[~in_grid] = None
    own = in_grid & (cell_owner == own_idx) if own_idx is not None else np.zeros(x.shape, bool)
    foreign = in_grid & (cell_owner != UNCLAIMED) & ~own
    signed = np.where(in_grid, dist_grid[ix_c, iy_c], np.nan) if dist_grid is not None \
        else np.full(x.shape, np.nan)
    if own_idx is not None:
        ex_grid = exclusivity_map(tmap, animal)
        excl = np.where(in_grid, ex_grid[ix_c, iy_c], np.nan)
    else:
        excl = np.full(x.shape, np.nan)
    return pd.DataFrame({"owner": owner_name, "own": own, "foreign": foreign,
                         "signed_dist": signed, "exclusivity": excl, "in_grid": in_grid})


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def territory_stability(a: TerritoryMap, b: TerritoryMap) -> Dict[str, object]:
    """Agreement between two territory maps built from independent data.

    Both maps must share a grid. Returns per-animal Jaccard overlap of the own
    masks, the fraction of bins claimed in both that agree on the owner, and an
    adjusted Rand index over bins claimed in both. ``None`` values mean the
    quantity was undefined (no common animals / no jointly claimed bins).
    """
    if a.owner.shape != b.owner.shape or not (np.allclose(a.x_edges, b.x_edges)
                                              and np.allclose(a.y_edges, b.y_edges)):
        raise ValueError("territory maps must share the same grid")

    common = [n for n in a.animal_ids if n in b.animal_ids]
    jaccard: Dict[str, Optional[float]] = {}
    for n in common:
        ma, mb = a.owner_mask(n), b.owner_mask(n)
        union = np.sum(ma | mb)
        jaccard[n] = float(np.sum(ma & mb) / union) if union else None

    # compare owners by name so differing animal orderings don't matter
    na = np.array([a.animal_ids[i] if i != UNCLAIMED else "" for i in a.owner.ravel()])
    nb = np.array([b.animal_ids[i] if i != UNCLAIMED else "" for i in b.owner.ravel()])
    both = (na != "") & (nb != "")
    agree = float(np.mean(na[both] == nb[both])) if both.any() else None
    ari = None
    if both.sum() >= 2:
        from sklearn.metrics import adjusted_rand_score
        ari = float(adjusted_rand_score(na[both], nb[both]))
    return dict(common_animals=common, jaccard_by_animal=jaccard,
                owner_agreement=agree, adjusted_rand_index=ari,
                n_jointly_claimed_bins=int(both.sum()))
