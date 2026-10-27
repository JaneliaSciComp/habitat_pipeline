"""Example-cell diagnostics for an exploratory territory run (rate vs exclusivity, rate map, exclusivity map).

    python scripts/territory_cell_examples.py --animal 630 --session 20251216_145034 \
        --cells 415:1659 270:1659 143:1829 294:1659 294:1829 --out results/territory/..._examples.png

Re-derives the same covariates as ephys.run_territory.analyse_chunk (leave-one-chunk-out map), so what is
plotted is exactly what the GLM saw. Diagnostic only: no test is run.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ephys import run_territory as rt  # noqa: E402
from video.territory import build_territory_map_loo, exclusivity_map, label_positions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="630")
    ap.add_argument("--session", default="20251216_145034")
    ap.add_argument("--cells", nargs="+", required=True, help="cluster_id:chunk_hhmm")
    ap.add_argument("--chunks", nargs="+", default=["1459", "1659", "1829"])
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from ingestion.data_paths import DataStorageManager
    from ingestion.ephys_sync import DataSyncManager
    from ingestion.kilosort_data_import import load_kilosort_data

    focal = f"rat{a.animal}"
    dsm = DataStorageManager(a.animal, a.session)
    ks = load_kilosort_data(dsm.get_kilosort_path())
    sync = DataSyncManager(dsm, dio_channel=1)
    cids, spikes = ks.get_filtered_cells_spike_times()
    cids = list(cids)
    win = ks.quality_ephys_window()
    cfg = json.load(open("config/default_paths.json"))
    apt = next(r for r in cfg["tracking"] if "APT" in str(r))
    date = a.session[:8]
    paths = dict(rt.apt_chunk_dirs(apt, date))
    chunks = {f"cohort7_{date}_{h}": rt.load_chunk(paths[f"cohort7_{date}_{h}"], sync) for h in a.chunks}
    bounds = (min(df["x"].min() for t in chunks.values() for df in t.values()),
              max(df["x"].max() for t in chunks.values() for df in t.values()),
              min(df["y"].min() for t in chunks.values() for df in t.values()),
              max(df["y"].max() for t in chunks.values() for df in t.values()))
    map_kw = dict(smoothing_sigma_bins=1.0, min_dwell_sec=5.0, min_tracked_sec=60.0, min_spread=0.0)
    # same map parameters as the run: read them back from its saved parameters
    sj = Path("results/territory") / f"{date}_{a.session.split('_')[1]}_{a.animal}_exploratory" / "summary.json"
    if sj.exists():
        p = json.load(open(sj))["parameters"]
        map_kw = dict(smoothing_sigma_bins=1.0, min_dwell_sec=p["min_dwell_sec"],
                      min_tracked_sec=p["min_tracked_sec"], min_spread=p["min_spread_px"])
        a.bins = p["bins"]

    cache = {}
    n = len(a.cells)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.2 * n), squeeze=False)
    for r, spec in enumerate(a.cells):
        cid, hh = spec.split(":")
        cname = f"cohort7_{date}_{hh}"
        if cname not in cache:
            tmap = build_territory_map_loo(chunks, cname, bounds=bounds, bins=a.bins, **map_kw)
            inp = rt.chunk_inputs(chunks[cname], focal, win)
            lab = label_positions(tmap, focal, inp["x"], inp["y"])
            cache[cname] = (tmap, inp, lab, exclusivity_map(tmap, focal))
        tmap, inp, lab, exm = cache[cname]
        k = cids.index(int(cid))
        counts = rt.bin_counts([spikes[k]], inp["t0"], inp["n_bins"])[:, 0]
        ex = lab["exclusivity"].to_numpy()
        ok = np.isfinite(ex) & np.isfinite(inp["x"])
        # A: rate vs exclusivity in quantile bins
        ax = axes[r, 0]
        qs = np.unique(np.quantile(ex[ok], np.linspace(0, 1, 9)))
        idx = np.clip(np.digitize(ex[ok], qs[1:-1]), 0, len(qs) - 2)
        xs, ys, lo, hi = [], [], [], []
        for b in range(len(qs) - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            tot = m.sum() * rt.DT
            cnt = counts[ok][m].sum()
            xs.append(ex[ok][m].mean())
            ys.append(cnt / tot)
            se = np.sqrt(max(cnt, 1)) / tot
            lo.append(se)
        ax.errorbar(xs, ys, yerr=lo, fmt="o-", color="k")
        ax.set_xlabel("exclusivity (focal share of dwell)")
        ax.set_ylabel("rate (Hz), +-1 Poisson SE")
        ax.set_title(f"cell {cid} / chunk {hh}: rate vs exclusivity")
        ax.text(0.02, 0.95, f"{ok.sum() * rt.DT / 60:.0f} min; exclusivity {np.nanmin(ex):.2f}-{np.nanmax(ex):.2f}",
                transform=ax.transAxes, va="top", fontsize=8)
        # B: spatial rate map (dwell-normalised)
        ax = axes[r, 1]
        xe = np.linspace(bounds[0], bounds[1], a.bins + 1)
        ye = np.linspace(bounds[2], bounds[3], a.bins + 1)
        dw, _, _ = np.histogram2d(inp["x"][ok], inp["y"][ok], bins=[xe, ye])
        sp, _, _ = np.histogram2d(inp["x"][ok], inp["y"][ok], bins=[xe, ye], weights=counts[ok])
        rm = np.where(dw * rt.DT >= 5, sp / np.maximum(dw * rt.DT, 1e-9), np.nan)
        im = ax.imshow(rm.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto", cmap="viridis")
        plt.colorbar(im, ax=ax, label="Hz")
        ax.set_title("spatial rate map (>=5 s dwell)")
        # C: exclusivity map the covariate came from
        ax = axes[r, 2]
        im = ax.imshow(exm.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto",
                       cmap="magma", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="exclusivity")
        ax.set_title(f"{focal} exclusivity map (from other chunks)")
    fig.suptitle(f"{focal} {a.session}: example cells (diagnostic; exploratory, not a result)", y=1.0)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
