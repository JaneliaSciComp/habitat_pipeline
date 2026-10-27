"""Which APT chunks have ephys for one rat, and how much hub-resting time is in them.

Metadata only (spike-window extent + sync); no neural analysis is run. Writes
``results/territory_behavior/coverage_<animal>.csv``.

    python scripts/territory_coverage_table.py --animal 630
"""

import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video.territory_behavior import arena_bounds, speed_px_s  # noqa: E402

ROOT = "//nearline/karpova/TervoLab/analysis/Tracking/APT/cohort7"
CACHE = "results/territory_behavior/cache"


def chunk_window(path, sync):
    ts = pd.read_csv(path, usecols=["timestamp"], encoding="utf-8-sig")["timestamp"].to_numpy()
    e = sync.convert_behavior_to_ephys(np.array([ts.min(), ts.max()]) / 1e9)
    return float(e[0]), float(e[1])


def hub_rest_sec(df, t_lo, t_hi, hub, radius, still_px_s=20.0, max_gap=1.0):
    """Seconds the rat was still within ``radius`` px of ``hub`` and inside ``[t_lo, t_hi]`` (tracking clock)."""
    t = df["t"].to_numpy()
    dt = np.diff(t, append=np.nan)
    dt = np.where(np.isfinite(dt) & (dt > 0), np.minimum(dt, max_gap), 0.0)
    ok = (t >= t_lo) & (t <= t_hi) & (speed_px_s(df) < still_px_s) \
        & (np.hypot(df["x"].to_numpy() - hub[0], df["y"].to_numpy() - hub[1]) < radius)
    return float(dt[ok].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="630")
    ap.add_argument("--hub", type=float, nargs=2, default=[3450.0, 610.0])
    ap.add_argument("--hub_frac_diag", type=float, default=0.04)
    ap.add_argument("--dio_channel", type=int, default=1)
    a = ap.parse_args()

    from ingestion.data_paths import DataStorageManager
    from ingestion.ephys_sync import DataSyncManager
    from ingestion.kilosort_data_import import load_kilosort_data

    chunks = {}
    for name in sorted(os.listdir(ROOT)):
        m = re.fullmatch(r"cohort7_(\d{8})_(\d{4})", name)
        p = os.path.join(ROOT, name, "solution", "TQT_named.csv")
        if m and os.path.exists(p):
            chunks.setdefault(m.group(1), []).append((name, p))
    tracks = {n: pickle.load(open(f"{CACHE}/{n}.pkl", "rb")) for d in chunks for n, _ in chunks[d]}
    b = arena_bounds(list(tracks.values()))
    radius = a.hub_frac_diag * float(np.hypot(b[1] - b[0], b[3] - b[2]))
    focal = f"rat{a.animal}"

    rows = []
    for date, cl in chunks.items():
        try:
            rec_ids = DataStorageManager(a.animal, date).recording_ids_on_date
            rec_ids = rec_ids() if callable(rec_ids) else rec_ids
        except Exception as e:  # no ephys for this animal on this date
            rows.append(dict(date=date, recording="-", status=f"no_recording: {type(e).__name__}"))
            continue
        for rec in rec_ids:
            base = dict(date=date, recording=rec)
            try:
                dsm = DataStorageManager(a.animal, rec)
                if dsm.get_kilosort_path() is None:
                    rows.append({**base, "status": "no_spike_files"})
                    continue
                ks = load_kilosort_data(dsm.get_kilosort_path())
                sync = DataSyncManager(dsm, dio_channel=a.dio_channel)
                win = ks.quality_ephys_window()
                n_cells = len(ks.get_filtered_cells_spike_times()[0])
            except Exception as e:
                rows.append({**base, "status": f"load_failed: {type(e).__name__}: {str(e)[:80]}"})
                continue
            for name, p in cl:
                lo, hi = chunk_window(p, sync)
                ov = max(0.0, min(hi, win[1]) - max(lo, win[0])) / max(hi - lo, 1e-9)
                tr = tracks[name]
                row = {**base, "chunk": name, "status": "ok", "n_quality_cells": n_cells,
                       "quality_window": [round(win[0]), round(win[1])], "chunk_ephys_window": [round(lo), round(hi)],
                       "overlap_frac": round(ov, 3), "n_rats_tracked": len(tr), "focal_tracked": focal in tr}
                if focal in tr and ov > 0:
                    # overlap window mapped back to the tracking clock
                    t_lo = float(tr[focal]["t"].min()) + max(win[0] - lo, 0.0)
                    t_hi = float(tr[focal]["t"].min()) + (min(win[1], hi) - lo)
                    row["hub_rest_min"] = round(hub_rest_sec(tr[focal], t_lo, t_hi, a.hub, radius) / 60, 1)
                    others = [o for o in tr if o != focal]
                    row["others_at_hub"] = ",".join(
                        o[3:] for o in others
                        if hub_rest_sec(tr[o], t_lo, t_hi, a.hub, radius) > 120)
                rows.append(row)
            print(date, rec, "done", flush=True)
    df = pd.DataFrame(rows)
    out = f"results/territory_behavior/coverage_{a.animal}.csv"
    df.to_csv(out, index=False)
    json.dump(vars(a), open(out.replace(".csv", "_params.json"), "w"))
    pd.set_option("display.width", 250)
    print(df.drop(columns=["quality_window"], errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
