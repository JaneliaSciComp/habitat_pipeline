"""Per-partner hub co-occupancy minutes and arrival/departure events for a focal rat (tracking only).

    python scripts/hub_event_counts.py --animal 630

Uses chunks with full ephys overlap from results/territory_behavior/coverage_<animal>.csv. 1 Hz grid.
Arrival of partner P: P outside the hub for the previous ``pre`` s, then inside for ``post`` s, with the focal
resting at the hub throughout [t - pre, t + post]. Departures are the mirror image.
"""

import argparse
import pickle

import numpy as np
import pandas as pd

from video.territory_behavior import arena_bounds, chunk_summary, speed_px_s

CACHE = "results/territory_behavior/cache"


def grid(df, t0, t1, max_gap=2.0):
    tc = np.arange(t0, t1, 1.0)
    t = df["t"].to_numpy()
    x = np.interp(tc, t, df["x"].to_numpy(), left=np.nan, right=np.nan)
    y = np.interp(tc, t, df["y"].to_numpy(), left=np.nan, right=np.nan)
    sp = np.interp(tc, t, speed_px_s(df), left=np.nan, right=np.nan)
    idx = np.clip(np.searchsorted(t, tc), 1, len(t) - 1)
    gap = np.minimum(np.abs(tc - t[idx - 1]), np.abs(t[idx] - tc))
    bad = gap > max_gap
    x[bad] = y[bad] = sp[bad] = np.nan
    return x, y, sp


def runs_all(mask, lo, hi):
    """True where mask is True over the whole window [i+lo, i+hi]."""
    n = len(mask)
    c = np.concatenate([[0], np.cumsum(mask.astype(int))])
    out = np.zeros(n, bool)
    for i in range(max(0, -lo), min(n, n - hi)):
        out[i] = c[i + hi + 1] - c[i + lo] == hi - lo + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="630")
    ap.add_argument("--hub", type=float, nargs=2, default=[3450.0, 610.0])
    ap.add_argument("--hub_frac_diag", type=float, default=0.04)
    ap.add_argument("--pre", type=int, default=30)
    ap.add_argument("--post", type=int, default=10)
    ap.add_argument("--still_px_s", type=float, default=20.0)
    ap.add_argument("--own_home", action="store_true", help="use the focal's own per-chunk home base as the hub")
    a = ap.parse_args()

    cov = pd.read_csv(f"results/territory_behavior/coverage_{a.animal}.csv")
    cov = cov[(cov.status == "ok") & (cov.overlap_frac >= 0.99) & (cov.focal_tracked)]
    names = sorted(set(cov.chunk))
    tracks = {n: pickle.load(open(f"{CACHE}/{n}.pkl", "rb")) for n in names}
    import glob, os
    allt = [pickle.load(open(f, "rb")) for f in glob.glob(f"{CACHE}/*.pkl")]
    b = arena_bounds(allt)
    rad = a.hub_frac_diag * float(np.hypot(b[1] - b[0], b[3] - b[2]))
    focal = f"rat{a.animal}"
    rows = []
    for _, r in cov.iterrows():
        tr = tracks[r.chunk]
        hub = a.hub
        if a.own_home:
            sm = chunk_summary({focal: tr[focal]}, b, min_tracked_sec=60.0)
            if focal not in sm:      # focal barely tracked in this chunk
                continue
            hub = sm[focal]["home"]
        t0 = max(tr[o]["t"].min() for o in [focal])
        t1 = tr[focal]["t"].max()
        fx, fy, fsp = grid(tr[focal], t0, t1)
        rest = (np.hypot(fx - hub[0], fy - hub[1]) < rad) & (fsp < a.still_px_s)
        for p in tr:
            if p == focal:
                continue
            px, py, _ = grid(tr[p], t0, t1)
            valid = np.isfinite(px)
            inhub = valid & (np.hypot(px - hub[0], py - hub[1]) < rad)
            outhub = valid & ~inhub
            both = rest & inhub
            absent = rest & outhub
            # events: state flips at i (outside for pre s before, inside for post s from i), focal resting throughout
            arr = runs_all(outhub, -a.pre, -1) & runs_all(inhub, 0, a.post - 1) & runs_all(rest, -a.pre, a.post - 1)
            dep = runs_all(inhub, -a.pre, -1) & runs_all(outhub, 0, a.post - 1) & runs_all(rest, -a.pre, a.post - 1)
            # count each event once (collapse consecutive hits)
            def n_ev(m):
                return int(np.sum(m & ~np.concatenate([[False], m[:-1]])))
            rows.append(dict(recording=r.recording, chunk=r.chunk, partner=p[3:], co_rest_min=both.sum() / 60,
                             absent_rest_min=absent.sum() / 60, n_arrivals=n_ev(arr), n_departures=n_ev(dep)))
    d = pd.DataFrame(rows)
    d.to_csv(f"results/territory_behavior/hub_events_{a.animal}{"_ownhome" if a.own_home else ""}.csv", index=False)
    pd.set_option("display.width", 200)
    pr = d.groupby("partner").agg(chunks_cooc=("co_rest_min", lambda v: int((v >= 2).sum())),
                                  co_rest_min=("co_rest_min", "sum"), absent_rest_min=("absent_rest_min", "sum"),
                                  arrivals=("n_arrivals", "sum"), departures=("n_departures", "sum")).round(1)
    print(pr.sort_values("arrivals", ascending=False).to_string())
    d16 = d[d.recording.str.startswith("20251216")]
    print("\n20251216 only:")
    print(d16.groupby("partner").agg(co_rest_min=("co_rest_min", "sum"), absent_rest_min=("absent_rest_min", "sum"),
                                     arrivals=("n_arrivals", "sum"), departures=("n_departures", "sum")).round(1)
          .sort_values("arrivals", ascending=False).to_string())
    print("\nfocal rest minutes per chunk:")
    print(d.groupby("chunk").apply(lambda g: round(float((g.co_rest_min + g.absent_rest_min).max()), 1)).to_string())


if __name__ == "__main__":
    main()
