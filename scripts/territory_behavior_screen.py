"""Behavioural territory screen over every APT day (tracking only; see video/territory_behavior.py).

    python scripts/territory_behavior_screen.py [--dates 20251216 20251217] [--out results/territory_behavior]

Per-chunk centre positions are cached as pickles so reruns are cheap.
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video.territory_behavior import arena_bounds, chunk_summary, day_screen, tracks_from_centers  # noqa: E402
from video.tracking_import import (_APT_COLUMN_RE, _read_tracking_csv, parse_apt_tracking,  # noqa: E402
                                   read_tracking_header)


def read_centers_with_time(path):
    """Long (timestamp, object_name, center_x, center_y); read_tracking_centers drops the timestamp."""
    path = Path(path)
    header = read_tracking_header(path)
    cols = ['timestamp'] + [c for c in header if _APT_COLUMN_RE.match(c)
                            and _APT_COLUMN_RE.match(c).group('field') == 'center']
    parsed = parse_apt_tracking(_read_tracking_csv(path, usecols=cols))
    return pd.concat([o.assign(object_name=n) for n, o in parsed.items()], ignore_index=True)

ROOT = "//nearline/karpova/TervoLab/analysis/Tracking/APT/cohort7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--out", default="results/territory_behavior")
    ap.add_argument("--min_tracked_sec", type=float, default=300.0)
    ap.add_argument("--still_px_s", type=float, default=20.0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "cache"), exist_ok=True)

    by_date = {}
    for name in sorted(os.listdir(a.root)):
        m = re.fullmatch(r"cohort7_(\d{8})_(\d{4})", name)
        p = os.path.join(a.root, name, "solution", "TQT_named.csv")
        if m and os.path.exists(p):
            by_date.setdefault(m.group(1), []).append((name, p))
    dates = a.dates or sorted(by_date)
    print("days:", {d: len(by_date.get(d, [])) for d in dates}, flush=True)

    all_rat, all_pairs = [], []
    for d in dates:
        chunks = {}
        for name, p in by_date.get(d, []):
            cp = os.path.join(a.out, "cache", f"{name}.pkl")
            if os.path.exists(cp):
                tr = pickle.load(open(cp, "rb"))
            else:
                t0 = time.time()
                tr = tracks_from_centers(read_centers_with_time(p))
                pickle.dump(tr, open(cp, "wb"))
                print(f"  loaded {name} in {time.time() - t0:.0f}s", flush=True)
            chunks[name] = tr
        if len(chunks) < 2:
            print(d, "skipped: <2 chunks", flush=True)
            continue
        bounds = arena_bounds(list(chunks.values()))
        summ = {c: chunk_summary(tr, bounds, min_tracked_sec=a.min_tracked_sec, still_px_s=a.still_px_s)
                for c, tr in chunks.items()}
        res = day_screen(summ, bounds)
        r, pr = res["per_rat"].assign(date=d), res["pairs"].assign(date=d)
        all_rat.append(r)
        all_pairs.append(pr)
        print(f"\n== {d}: {len(chunks)} chunks, arena diag {res['arena_diag_px']:.0f}px", flush=True)
        print(r[r.n_chunks >= 2].round(2).to_string(index=False), flush=True)
    if all_rat:
        pd.concat(all_rat).to_csv(os.path.join(a.out, "per_rat.csv"), index=False)
        pd.concat(all_pairs).to_csv(os.path.join(a.out, "pairs.csv"), index=False)
        json.dump(vars(a), open(os.path.join(a.out, "params.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
