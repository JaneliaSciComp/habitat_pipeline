"""Behavioural territoriality screen from tracking alone (no ephys, no sync).

Asks, per rat and per day, whether *territory is a stable behavioural variable*
before any neural analysis is attempted. Three quantities, all computed per 30-min
APT chunk and compared **across chunks** (never within the data that defined them):

* **home base** - peak of the smoothed *low-speed* dwell map (where the rat rests).
  Stability = distance between a rat's home bases in different chunks, in units of
  the arena diagonal.
* **utilisation distribution (UD)** - Gaussian-smoothed dwell time normalised to sum 1.
  Self-reliability = correlation of a rat's UD between chunks.
* **pairwise overlap** - Bhattacharyya coefficient between two rats' UDs in the same
  chunk. A territorial rat should be more similar to *itself* in another chunk than to
  any neighbour in the same chunk (``territoriality = self_reliability - max_overlap``).

Assumptions:
    - Pixels, not cm (APT calibration unresolved, see CLAUDE.md); lengths are reported
      relative to the arena diagonal so they are unit-free.
    - ``timestamp`` is the Linux-ns column; time only matters for dwell and speed, so
      no ephys sync is needed.
    - Rats absent for most of a chunk are excluded from that chunk (``min_tracked_sec``),
      not zero-filled.
    - Arena bounds are the pooled 0.5-99.5 percentile of all positions on the day, so
      a tracking glitch does not stretch the grid.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

Tracks = Dict[str, pd.DataFrame]          # animal -> DataFrame[t (s), x, y]


def tracks_from_centers(centers: pd.DataFrame) -> Tracks:
    """Long ``(timestamp, object_name, center_x, center_y)`` -> ``{animal: DataFrame[t,x,y]}``."""
    out: Tracks = {}
    for name, g in centers.groupby("object_name"):
        g = g.dropna(subset=["center_x", "center_y", "timestamp"]).sort_values("timestamp")
        out[str(name)] = pd.DataFrame({"t": g["timestamp"].to_numpy(float) / 1e9,
                                       "x": g["center_x"].to_numpy(float),
                                       "y": g["center_y"].to_numpy(float)})
    return out


def arena_bounds(chunks: List[Tracks], pct: float = 0.5) -> Tuple[float, float, float, float]:
    xs = np.concatenate([df["x"].to_numpy() for c in chunks for df in c.values()])
    ys = np.concatenate([df["y"].to_numpy() for c in chunks for df in c.values()])
    return (float(np.percentile(xs, pct)), float(np.percentile(xs, 100 - pct)),
            float(np.percentile(ys, pct)), float(np.percentile(ys, 100 - pct)))


def speed_px_s(df: pd.DataFrame, smooth_sec: float = 2.0) -> np.ndarray:
    """Speed (px/s) from displacement over ~``smooth_sec``, robust to tracking jitter."""
    t, x, y = df["t"].to_numpy(), df["x"].to_numpy(), df["y"].to_numpy()
    if len(t) < 3:
        return np.zeros(len(t))
    fs = 1.0 / max(np.median(np.diff(t)), 1e-3)
    k = max(int(round(smooth_sec * fs)), 1)
    xs = pd.Series(x).rolling(k, center=True, min_periods=1).median().to_numpy()
    ys = pd.Series(y).rolling(k, center=True, min_periods=1).median().to_numpy()
    gx, gy = np.gradient(xs, t), np.gradient(ys, t)
    return np.hypot(gx, gy)


def _dwell(df: pd.DataFrame, edges, weights_mask: Optional[np.ndarray], max_gap: float) -> np.ndarray:
    t, x, y = df["t"].to_numpy(), df["x"].to_numpy(), df["y"].to_numpy()
    dt = np.diff(t, append=np.nan)
    dt = np.where(np.isfinite(dt) & (dt > 0), np.minimum(dt, max_gap), 0.0)
    if weights_mask is not None:
        dt = dt * weights_mask
    h, _, _ = np.histogram2d(x, y, bins=edges, weights=dt)
    return h


def chunk_summary(tracks: Tracks, bounds, *, bins: int = 30, smooth_bins: float = 1.5,
                  min_tracked_sec: float = 300.0, still_px_s: float = 20.0,
                  max_gap: float = 1.0) -> Dict[str, Dict[str, object]]:
    """Per-rat UD, still-dwell map and home base for one chunk."""
    edges = [np.linspace(bounds[0], bounds[1], bins + 1), np.linspace(bounds[2], bounds[3], bins + 1)]
    out: Dict[str, Dict[str, object]] = {}
    for name, df in tracks.items():
        if len(df) < 10:
            continue
        dwell = _dwell(df, edges, None, max_gap)
        tracked = float(dwell.sum())
        if tracked < min_tracked_sec:
            continue
        sp = speed_px_s(df)
        still = _dwell(df, edges, (sp < still_px_s).astype(float), max_gap)
        ud = gaussian_filter(dwell, smooth_bins)
        ud = ud / ud.sum()
        still_s = gaussian_filter(still, smooth_bins)
        i, j = np.unravel_index(np.argmax(still_s), still_s.shape)
        cx = 0.5 * (edges[0][i] + edges[0][i + 1])
        cy = 0.5 * (edges[1][j] + edges[1][j + 1])
        out[name] = dict(tracked_sec=tracked, still_frac=float(still.sum() / max(tracked, 1e-9)),
                         ud=ud, home=(float(cx), float(cy)), median_speed=float(np.median(sp)))
    return out


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(p * q)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def day_screen(summaries: Dict[str, Dict[str, Dict[str, object]]], bounds) -> Dict[str, object]:
    """Cross-chunk reliability, overlap and home-base stability for one day.

    ``summaries``: ``{chunk: chunk_summary(...)}``. Returns per-rat rows and a pair table.
    """
    diag = float(np.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2]))
    chunks = sorted(summaries)
    animals = sorted({a for c in chunks for a in summaries[c]})
    rows, pairs = [], []
    for a in animals:
        present = [c for c in chunks if a in summaries[c]]
        rel, hd = [], []
        for i, c1 in enumerate(present):
            for c2 in present[i + 1:]:
                rel.append(_corr(summaries[c1][a]["ud"], summaries[c2][a]["ud"]))
                h1, h2 = summaries[c1][a]["home"], summaries[c2][a]["home"]
                hd.append(float(np.hypot(h1[0] - h2[0], h1[1] - h2[1]) / diag))
        # strongest same-chunk overlap with any neighbour, averaged over this rat's chunks
        mx = []
        for c in present:
            ov = [bhattacharyya(summaries[c][a]["ud"], summaries[c][b]["ud"])
                  for b in summaries[c] if b != a]
            if ov:
                mx.append(max(ov))
        # self-similarity on the same scale as overlap (BC between chunks)
        self_bc = [bhattacharyya(summaries[c1][a]["ud"], summaries[c2][a]["ud"])
                   for i, c1 in enumerate(present) for c2 in present[i + 1:]]
        rows.append(dict(
            animal=a, n_chunks=len(present),
            tracked_min=float(sum(summaries[c][a]["tracked_sec"] for c in present) / 60.0),
            still_frac=float(np.mean([summaries[c][a]["still_frac"] for c in present])),
            self_reliability=float(np.nanmedian(rel)) if rel else np.nan,
            self_bc=float(np.median(self_bc)) if self_bc else np.nan,
            max_neighbour_bc=float(np.mean(mx)) if mx else np.nan,
            home_shift_frac_diag=float(np.median(hd)) if hd else np.nan))
        rows[-1]["territoriality"] = rows[-1]["self_bc"] - rows[-1]["max_neighbour_bc"]
    for c in chunks:
        names = sorted(summaries[c])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs.append(dict(chunk=c, a=a, b=b,
                                  bc=bhattacharyya(summaries[c][a]["ud"], summaries[c][b]["ud"]),
                                  home_dist_frac_diag=float(np.hypot(
                                      summaries[c][a]["home"][0] - summaries[c][b]["home"][0],
                                      summaries[c][a]["home"][1] - summaries[c][b]["home"][1]) / diag)))
    return dict(per_rat=pd.DataFrame(rows), pairs=pd.DataFrame(pairs), chunks=chunks,
                arena_diag_px=diag)
