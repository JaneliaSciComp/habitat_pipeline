"""How much partner-distance variation exists while the focal rests at a fixed spot? (tracking only)

    python scripts/hub_distance_variation.py --animal 630 [--own_home]

Per chunk with full ephys overlap: 0.5 s bins where the focal is still within ``hub_frac_diag`` of the spot.
Reports distance to the nearest other rat, its SD, how much of its variance is slow (a 120 s running mean) vs
fast, the autocorrelation time, and how much of the variance is between partners (who is nearest) vs within.
"""

import argparse
import glob
import pickle

import numpy as np
import pandas as pd

from video.territory_behavior import arena_bounds, chunk_summary, speed_px_s

CACHE = "results/territory_behavior/cache"
DT = 0.5


def grid_dt(df, t0, t1, max_gap=1.0):
    tc = t0 + np.arange(int((t1 - t0) // DT)) * DT
    t = df["t"].to_numpy()
    out = {k: np.interp(tc, t, v, left=np.nan, right=np.nan)
           for k, v in (("x", df["x"].to_numpy()), ("y", df["y"].to_numpy()), ("s", speed_px_s(df)))}
    idx = np.clip(np.searchsorted(t, tc), 1, len(t) - 1)
    gap = np.minimum(np.abs(tc - t[idx - 1]), np.abs(t[idx] - tc))
    for k in out:
        out[k] = np.where(gap <= max_gap, out[k], np.nan)
    return tc, out


def acf_time(v, max_lag=240):
    v = v - v.mean()
    if v.std() == 0:
        return np.nan
    ac = np.array([np.mean(v[:-k] * v[k:]) / v.var() if k else 1.0 for k in range(max_lag)])
    below = np.where(ac < 1 / np.e)[0]
    return float(below[0] * DT) if below.size else float(max_lag * DT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="630")
    ap.add_argument("--hub", type=float, nargs=2, default=[3450.0, 610.0])
    ap.add_argument("--hub_frac_diag", type=float, default=0.04)
    ap.add_argument("--own_home", action="store_true")
    ap.add_argument("--still_px_s", type=float, default=20.0)
    a = ap.parse_args()
    cov = pd.read_csv(f"results/territory_behavior/coverage_{a.animal}.csv")
    cov = cov[(cov.status == "ok") & (cov.overlap_frac >= 0.99) & (cov.focal_tracked)]
    allt = {f: pickle.load(open(f, "rb")) for f in glob.glob(f"{CACHE}/*.pkl")}
    b = arena_bounds(list(allt.values()))
    rad = a.hub_frac_diag * float(np.hypot(b[1] - b[0], b[3] - b[2]))
    focal = f"rat{a.animal}"
    rows = []
    for chunk in sorted(set(cov.chunk)):
        tr = pickle.load(open(f"{CACHE}/{chunk}.pkl", "rb"))
        if focal not in tr:
            continue
        hub = a.hub
        if a.own_home:
            sm = chunk_summary({focal: tr[focal]}, b, min_tracked_sec=60.0)
            if focal not in sm:
                continue
            hub = sm[focal]["home"]
        t0, t1 = tr[focal]["t"].min(), tr[focal]["t"].max()
        tc, f = grid_dt(tr[focal], t0, t1)
        rest = (np.hypot(f["x"] - hub[0], f["y"] - hub[1]) < rad) & (f["s"] < a.still_px_s)
        if rest.sum() * DT < 300:
            continue
        D = {}
        for p, df in tr.items():
            if p == focal or len(df) < 10:
                continue
            _, g = grid_dt(df, t0, t1)
            D[p] = np.hypot(g["x"] - f["x"], g["y"] - f["y"])
        names = list(D)
        M = np.column_stack([D[n] for n in names])
        ok = rest & np.isfinite(M).any(axis=1)
        nearest = np.nanmin(np.where(np.isfinite(M), M, np.inf), axis=1)
        who = np.array(names)[np.nanargmin(np.where(np.isfinite(M), M, np.inf), axis=1)]
        v = nearest[ok]
        # work on the contiguous series of rest bins (gaps removed) for slow/fast split
        k = int(120 / DT)
        slow = pd.Series(v).rolling(k, center=True, min_periods=k // 4).mean().to_numpy()
        fast_var = float(np.nanvar(v - slow))
        # between-partner share: variance of per-partner mean nearest distance (weighted) over total
        w = who[ok]
        pm = pd.Series(v).groupby(w).transform("mean").to_numpy()
        rows.append(dict(chunk=chunk[-9:], rest_min=round(ok.sum() * DT / 60, 1), median_px=round(float(np.median(v))),
                         sd_px=round(float(v.std())), p10_p90=f"{np.percentile(v,10):.0f}-{np.percentile(v,90):.0f}",
                         frac_var_fast=round(fast_var / v.var(), 2), acf_sec=round(acf_time(v), 1),
                         frac_var_between_partners=round(float(pm.var() / v.var()), 2), n_nearest_ids=len(set(w))))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
