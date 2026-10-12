"""Conflict-defined territory: a win/loss Voronoi map of the arena.

Each point in the arena is labelled by the outcome of the **nearest** conflict
(Euclidean nearest neighbour, i.e. a Voronoi partition of the localized
conflict points): ``'own'`` if the focal animal *won* that conflict, ``'other'``
if it *lost* it.

Pure functions, no I/O. Positions are whatever units the caller's tracking is in
(pixels or cm); every function is unit-agnostic.

Outcome convention (the self/outcome reference that the guardrail test in
``tests/test_decode_conflict_territory.py`` pins): the outcome strings are the
ones :meth:`video.behavioral_events.BehavioralEventsData.extract_outcome_labels`
returns **from the focal animal's point of view** - ``'winner'`` means the focal
animal was the winner. ``'winner'`` -> ``'own'``, ``'loser'`` -> ``'other'``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

OWN = "own"
OTHER = "other"
WIN = "winner"
LOSS = "loser"


def outcome_is_win(c_outcome: Sequence) -> np.ndarray:
    """Boolean ``True`` where a conflict outcome is a focal **win**.

    Accepts ``'winner'``/``'loser'`` (what ``extract_outcome_labels`` returns) or
    booleans. Anything else raises: silently mapping an unknown string to
    ``'loser'`` is exactly the kind of self/outcome swap this module must not make.
    """
    arr = np.asarray(c_outcome)
    if arr.dtype == bool:
        return arr.copy()
    s = np.array([str(v).strip().lower() for v in arr.ravel()], dtype=object)
    bad = ~np.isin(s, [WIN, LOSS])
    if bad.any():
        raise ValueError(f"outcomes must be 'winner'/'loser' (focal view); got {sorted(set(s[bad]))}")
    return (s == WIN).reshape(arr.shape)


def localize_events(times: Sequence[float], labels: Sequence, focal_df: pd.DataFrame, *,
                    max_gap: float = 1.0, end_times: Optional[Sequence[float]] = None
                    ) -> Tuple[pd.DataFrame, int]:
    """Place each conflict at the focal animal's position at the event time.

    ``focal_df`` is one animal's ``(t, x, y[, speed])`` frame on the ephys clock
    (as returned by :func:`video.tracking_import.resolve_tracking_on_ephys_clock`).
    Positions are linearly interpolated with the same nearest-sample gap rule as
    :func:`ephys.run_territory.interp_on_grid`: an event whose nearest tracked
    sample is more than ``max_gap`` seconds away (or that lies outside the
    tracked range) has no position and is **dropped**.

    Returns ``(conflicts, n_dropped)`` where ``conflicts`` has columns
    ``t, x, y, outcome`` (plus ``t_end`` when ``end_times`` is given), sorted by
    ``t``.
    """
    from ephys.run_territory import interp_on_grid

    times = np.asarray(times, dtype=float)
    labels = np.asarray(labels)
    if len(times) != len(labels):
        raise ValueError(f"{len(times)} times but {len(labels)} labels")
    outcome = np.where(outcome_is_win(labels), WIN, LOSS) if len(labels) else np.array([], dtype=object)
    cols = ["t", "x", "y", "outcome"] + (["t_end"] if end_times is not None else [])
    if len(times) == 0 or focal_df is None or len(focal_df) < 2:
        return pd.DataFrame(columns=cols), int(len(times))

    df = focal_df.sort_values("t")
    if "speed" not in df.columns:
        df = df.assign(speed=np.nan)
    pos = interp_on_grid(df, times, max_gap=max_gap)
    ok = np.isfinite(pos["x"]) & np.isfinite(pos["y"])
    out = pd.DataFrame({"t": times[ok], "x": pos["x"][ok], "y": pos["y"][ok], "outcome": outcome[ok]})
    if end_times is not None:
        out["t_end"] = np.asarray(end_times, dtype=float)[ok]
    return out.sort_values("t").reset_index(drop=True)[cols], int((~ok).sum())


def nearest_conflict(px, py, cx, cy) -> Tuple[np.ndarray, np.ndarray]:
    """Index of, and distance to, the nearest conflict point for every query point.

    Non-finite query points get index ``-1`` and distance ``nan``. The nearest
    index depends only on *where* conflicts happened, never on their outcomes,
    which is what lets a map-specificity null permute outcomes without
    re-querying the tree.
    """
    from scipy.spatial import cKDTree

    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    pts = np.column_stack([np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)])
    idx = np.full(px.shape, -1, dtype=int)
    dist = np.full(px.shape, np.nan)
    ok = np.isfinite(px) & np.isfinite(py)
    if len(pts) and ok.any():
        d, i = cKDTree(pts).query(np.column_stack([px[ok], py[ok]]))
        idx[ok], dist[ok] = i, d
    return idx, dist


def voronoi_labels(px, py, cx, cy, c_outcome) -> Dict[str, np.ndarray]:
    """Label query points by the outcome of their nearest conflict.

    Returns a dict:

    - ``label``  : object array, ``'own'`` (nearest conflict was a focal win),
      ``'other'`` (a loss), ``None`` for non-finite query points;
    - ``own``    : int array, 1 = own, 0 = other, -1 = no position;
    - ``nearest``: index of the nearest conflict point (-1 = no position);
    - ``dist``   : distance to the nearest conflict point;
    - ``margin`` : distance to the nearest conflict of the *opposite* class
      minus ``dist`` (>= 0; small = near a territory boundary; ``inf`` when only
      one class exists).
    """
    from scipy.spatial import cKDTree

    win = outcome_is_win(c_outcome)
    cx, cy = np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)
    idx, dist = nearest_conflict(px, py, cx, cy)
    ok = idx >= 0
    own = np.full(idx.shape, -1, dtype=int)
    own[ok] = win[idx[ok]].astype(int)
    label = np.full(idx.shape, None, dtype=object)
    label[own == 1] = OWN
    label[own == 0] = OTHER

    margin = np.full(idx.shape, np.nan)
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    for cls in (True, False):
        sel = ok & (own == int(cls))
        opp = ~win if cls else win
        if not sel.any():
            continue
        if not opp.any():
            margin[sel] = np.inf
            continue
        d_opp, _ = cKDTree(np.column_stack([cx[opp], cy[opp]])).query(np.column_stack([px[sel], py[sel]]))
        margin[sel] = d_opp - dist[sel]
    return dict(label=label, own=own, nearest=idx, dist=dist, margin=margin)


def territory_grid(cx, cy, c_outcome, *, bounds: Tuple[float, float, float, float],
                   bins: int = 200) -> Dict[str, np.ndarray]:
    """Raster of the Voronoi map for plotting.

    ``bounds = (x_min, x_max, y_min, y_max)``. Returns ``x_edges``, ``y_edges``
    and ``own`` with shape ``(len(y_edges)-1, len(x_edges)-1)`` (row = y, so it
    can go straight into ``pcolormesh(x_edges, y_edges, own)``), 1.0 = own,
    0.0 = other.
    """
    x0, x1, y0, y1 = map(float, bounds)
    xe, ye = np.linspace(x0, x1, bins + 1), np.linspace(y0, y1, bins + 1)
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    gx, gy = np.meshgrid(xc, yc)
    lab = voronoi_labels(gx.ravel(), gy.ravel(), cx, cy, c_outcome)["own"].reshape(gx.shape)
    return dict(x_edges=xe, y_edges=ye, own=np.where(lab >= 0, lab, np.nan).astype(float))
