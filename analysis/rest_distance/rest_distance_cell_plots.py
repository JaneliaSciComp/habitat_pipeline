"""Plots for the strongest rest-distance cells of a finished run (re-prepares the same inputs as the run).

    python analysis/rest_distance/rest_distance_cell_plots.py --run_dir results/rest_distance/20251216_145034_630_exploratory

Selection (fixed, not tuned): cells with q <= 0.05 in every tested chunk, then the cells with the largest mean |r|
among those whose r has the same sign in every chunk, up to ``--n_cells`` in total. One row per cell, one column per
chunk: high-passed rate against high-passed log distance (hexbin-free scatter + 10 equal-count bin means), with the
partial r and p from the run, and the cell's and the distance's correlation with each high-passed nuisance column
(a cell that tracks the nuisance as strongly as it tracks distance is the arousal-leak signature).
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.rest_distance import run_rest_distance as rr  # noqa: E402


def select_cells(cells: dict, n: int):
    chunks = list(cells)
    q = pd.concat([c.set_index("cluster_id").q_value.rename(k) for k, c in cells.items()], axis=1)
    r = pd.concat([c.set_index("cluster_id").r.rename(k) for k, c in cells.items()], axis=1)
    core = list(q.index[(q <= 0.05).all(axis=1)])
    same = r.index[(np.sign(r) == np.sign(r.iloc[:, 0].to_numpy()[:, None])).all(axis=1)]
    rest = r.loc[[c for c in same if c not in core]].abs().mean(axis=1).sort_values(ascending=False)
    return (core + list(rest.index))[:n], set(core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_cells", type=int, default=6)
    a = ap.parse_args()
    summ = json.load(open(os.path.join(a.run_dir, "summary.json")))
    args = rr.build_parser().parse_args(["--session_id", summ["session_id"], "--animal_id", summ["animal"][3:]])
    for k, v in summ["parameters"].items():
        if hasattr(args, k):
            setattr(args, k, v)
    cells = {os.path.basename(f)[6:-4]: pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.run_dir, "cells_*.csv")))}
    sel, core = select_cells(cells, a.n_cells)

    from ingestion.data_paths import DataStorageManager
    from ingestion.ephys_sync import DataSyncManager
    from ingestion.kilosort_data_import import load_kilosort_data
    focal = summ["animal"]
    dsm = DataStorageManager(args.animal_id, args.session_id, config_path=args.config_path)
    ks = load_kilosort_data(dsm.get_kilosort_path())
    sync = DataSyncManager(dsm, dio_channel=args.dio_channel)
    cluster_ids, spike_times = ks.get_filtered_cells_spike_times()
    win = ks.quality_ephys_window()
    cfg = json.load(open(args.config_path))
    roots = cfg["tracking"] if isinstance(cfg["tracking"], list) else [cfg["tracking"]]
    apt = next(r for r in roots if "APT" in str(r))
    paths = dict(rr.apt_chunk_dirs(apt, args.session_id[:8]))
    loaded = {n: rr.load_chunk_with_motion(paths[n], sync, focal) for n in cells}
    bounds = rr.arena_bounds([tr for tr, _ in loaded.values()])

    prepared = {}
    for n, (tr, mo) in loaded.items():
        res, P, keep = rr.prepare_chunk(n, tr, mo, focal, win, cluster_ids, spike_times, bounds=bounds, args=args)
        prepared[n] = (P, np.asarray(cluster_ids)[keep], res["nuisance"])

    fig, axes = plt.subplots(len(sel), len(cells), figsize=(4.3 * len(cells), 3.0 * len(sel)), squeeze=False)
    for i, cid in enumerate(sel):
        for j, n in enumerate(cells):
            ax = axes[i, j]
            P, ids, nn = prepared[n]
            col = int(np.flatnonzero(ids == cid)[0])
            y, d = P["Y"][:, col], P["d"]
            Z = P["N"][:, 1:1 + len(nn)]
            ax.scatter(d, y, s=3, alpha=0.15, color="0.5", rasterized=True)
            edges = np.quantile(d, np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, 9)
            mx = np.array([d[b == k].mean() for k in range(10)])
            my = np.array([y[b == k].mean() for k in range(10)])
            se = np.array([y[b == k].std() / np.sqrt((b == k).sum()) for k in range(10)])
            ax.errorbar(mx, my, se, color="C3", marker="o", ms=4, lw=1.5)
            ax.axhline(0, color="k", lw=0.4)
            row = cells[n].set_index("cluster_id").loc[cid]
            cy = [np.corrcoef(y, Z[:, k])[0, 1] for k in range(Z.shape[1])]
            cd = [np.corrcoef(d, Z[:, k])[0, 1] for k in range(Z.shape[1])]
            short = [s.replace("log1p_", "").replace("_speed", "") for s in nn]
            txt = f"partial r={row.r:+.3f}  p={row.p_value:.3f}  q={row.q_value:.3f}\n" + \
                  "r(cell,nuis) " + " ".join(f"{s}:{v:+.2f}" for s, v in zip(short, cy)) + "\n" + \
                  "r(dist,nuis) " + " ".join(f"{s}:{v:+.2f}" for s, v in zip(short, cd))
            ax.set_title(txt, fontsize=6.5, loc="left")
            if i == len(sel) - 1:
                ax.set_xlabel("high-passed log nearest-partner distance")
            if j == 0:
                ax.set_ylabel(f"cluster {cid}{' *' if cid in core else ''}\nhigh-passed sqrt rate")
            if i == 0:
                ax.annotate(n[-4:] + f"  ({P['rest_min']:.0f} min rest)", (0.5, 1.45), xycoords="axes fraction",
                            ha="center", fontsize=9, weight="bold")
    fig.suptitle(f"{focal} {summ['recording_id']}: strongest rest-distance cells (* = q<=0.05 in every chunk). "
                 "Exploratory; red = 10 equal-count bin means +-SEM", y=1.0, fontsize=10)
    fig.tight_layout()
    out = os.path.join(a.run_dir, "best_cells.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out, "cells:", sel)


if __name__ == "__main__":
    main()
