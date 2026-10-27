#!/usr/bin/env python3
"""
Per-cell decoding accuracy against position on the probe: one point per cell.

The x axis (``--x depth``, the default) is the distance in mm below the
probe's topmost recording site. That is ``max(channel_positions[:, 1]) -
KilosortData.DV`` (um -> mm), where DV is the peak channel's vertical position. It reads
the same way in every rat. ``--x channel`` plots the hardware channel number
instead, which does **not**: rat4678 and rat4679 have a reversed
``channel_map`` (hardware channel 383 sits at the lowest site), so for them
a higher channel is lower on the probe. The y axis is the cached per-cell
balanced accuracy, or plain accuracy with ``--metric accuracy``.

Colours follow the significance histograms. Light blue is every other cell.
Dark blue is uncorrected p < ``--mid_p``. Red is BH q < ``--alpha``, using
``scripts/refine_permutation_fdr.py``'s CSV. Red points are drawn on top.

Writes ``<stem>_by_rat`` (one panel per rat) and ``<stem>_combined`` (every
rat pooled), as .png and .pdf.

Example
-------
::

    python scripts/plot_accuracy_by_channel.py
    python scripts/plot_accuracy_by_channel.py --label_mode group
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ingestion.data_paths import DataStorageManager  # noqa: E402
from ingestion.kilosort_data_import import load_kilosort_data  # noqa: E402
import plot_decoding_accuracy_violins as violins  # noqa: E402
from plot_decoding_auroc_histograms import (INK, MID_COLOR, NONSIG_COLOR,  # noqa: E402
                                            SIG_COLOR)
from refine_permutation_fdr import decode_params  # noqa: E402

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

WHAT = {"opponent": "opponent identity", "group": "group identity"}
STEM = {"opponent": "ec_top2", "group": "ec_group"}


def collect(mode: str, metric: str, alpha: float, mid_p: float,
            cache_dir: Path) -> List[Dict]:
    fdr = pd.read_csv(REPO_ROOT / f"results/refined_fdr_{mode}.csv")
    params = decode_params(mode)
    groups = []
    for animal, session in violins.DEFAULT_TARGETS:
        res = violins._load_result_pickle(
            cache_dir / violins._cache_key(animal, session, params))
        g = res["_group"]
        dm = DataStorageManager(animal, session, config_path=g["config_path"])
        ks = load_kilosort_data(dm.get_kilosort_path())
        index = {cid: i for i, cid in enumerate(ks.ks_ids)}
        rows = fdr[fdr["animal_id"] == animal].set_index("cluster_id")
        positions = next(Path(dm.get_kilosort_path()).rglob("channel_positions.npy"))
        top = float(np.load(positions)[:, 1].max())
        cids = [c for c in res["significance"]
                if np.isfinite(res["cell_results"][c].get(metric, np.nan))]
        ii = [index[c] for c in cids]
        groups.append({
            "animal_id": animal,
            "cohort": violins.cohort_of_config(g["config_path"]),
            "channel": np.asarray(ks.channel)[ii].astype(int),
            "dv": np.asarray(ks.DV)[ii].astype(float),
            "depth": (top - np.asarray(ks.DV)[ii].astype(float)) / 1000.0,  # mm
            "value": np.array([res["cell_results"][c][metric] for c in cids], float),
            "sig": np.array([rows.loc[c, "q_value"] < alpha for c in cids]),
            "mid": np.array([rows.loc[c, "p_value"] < mid_p for c in cids]),
        })
    return groups


def _scatter(ax, g: Dict, xkey: str, big: bool = False) -> None:
    s = 14 if big else 10
    rest = ~g["sig"] & ~g["mid"]
    mid = g["mid"] & ~g["sig"]
    for mask, color, z in ((rest, NONSIG_COLOR, 2), (mid, MID_COLOR, 3),
                           (g["sig"], SIG_COLOR, 4)):
        ax.scatter(g[xkey][mask], g["value"][mask], s=s, color=color,
                   edgecolors="white", linewidths=0.3, zorder=z)
    ax.axhline(0.5, color=INK, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    if xkey == "channel":
        ax.set_xlim(-8, 392)
        ax.xaxis.set_major_locator(MultipleLocator(100))
    else:
        ax.set_xlim(-0.08, 3.92)
        ax.xaxis.set_major_locator(MultipleLocator(1))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=3.5, width=1.0, labelsize=9 if not big else 10)


def _legend(ax, g: Dict, title: str, alpha: float, mid_p: float, fs: float):
    n = g["value"].size
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.3)
               for c in (NONSIG_COLOR, MID_COLOR, SIG_COLOR)]
    labels = ["All cells",
              f"Uncorrected significant (p < {mid_p:g}, {(g['mid'] | g['sig']).mean():.0%})",
              f"Significant (q < {alpha:g}, BH, {g['sig'].mean():.0%})"]
    leg = ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                    frameon=False, title=f"{title}, n = {n} cells",
                    title_fontsize=fs, fontsize=fs - 1.5, alignment="left",
                    handletextpad=0.3, labelspacing=0.25, borderaxespad=0.2,
                    borderpad=0.1)
    return leg


def figure_by_rat(groups: List[Dict], ylabel: str, alpha: float, mid_p: float,
                  xkey: str, xlabel: str):
    ncol = 3
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.3 * nrow),
                             squeeze=False, sharey=True)
    vals = np.concatenate([g["value"] for g in groups])
    ylim = (np.floor(vals.min() * 10) / 10, np.ceil(vals.max() * 10) / 10)
    for ax, g in zip(axes.flat, groups):
        _scatter(ax, g, xkey)
        ax.set_ylim(*ylim)
        reversed_ch = xkey == "channel" and spearmanr(g["channel"], g["dv"])[0] < 0
        _legend(ax, g, f"{g['animal_id']} ({g['cohort'].replace('cohort', 'cohort ')})"
                       f"{', channels reversed' if reversed_ch else ''}",
                alpha, mid_p, 9)
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel, fontsize=10)
    fig.tight_layout()
    return fig


def figure_combined(groups: List[Dict], ylabel: str, alpha: float, mid_p: float,
                    xkey: str, xlabel: str):
    g = {k: np.concatenate([x[k] for x in groups])
         for k in ("channel", "dv", "depth", "value", "sig", "mid")}
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    _scatter(ax, g, xkey, big=True)
    _legend(ax, g, f"{len(groups)} rats pooled", alpha, mid_p, 10.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--label_mode", choices=("opponent", "group"), default="opponent")
    p.add_argument("--metric", choices=("balanced_accuracy", "accuracy"),
                   default="balanced_accuracy")
    p.add_argument("--x", choices=("depth", "channel"), default="depth",
                   help="depth = mm below the topmost recording site (default).")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mid_p", type=float, default=0.05)
    p.add_argument("--cache_dir", default=".gui_cache/accuracy_violins")
    args = p.parse_args(argv)

    groups = collect(args.label_mode, args.metric, args.alpha, args.mid_p,
                     REPO_ROOT / args.cache_dir)
    name = {"balanced_accuracy": "Balanced accuracy", "accuracy": "Accuracy"}[args.metric]
    ylabel = f"{name}\n({WHAT[args.label_mode]})"
    for g in groups:
        rho = spearmanr(g[args.x], g["value"])[0]
        print(f"{g['animal_id']:8} n={g['value'].size:4} BH={g['sig'].sum():3} "
              f"p<{args.mid_p:g}={(g['mid'] | g['sig']).sum():3}  "
              f"rho({args.x}, {args.metric})={rho:+.2f}  "
              f"{args.x} {g[args.x].min():g}-{g[args.x].max():g}")
    xlabel = ("Channel" if args.x == "channel"
              else "Distance from top channel (mm)")
    stem = f"{STEM[args.label_mode]}_{args.metric}_by_{args.x}"
    for suffix, fig in (("_by_rat", figure_by_rat(groups, ylabel, args.alpha, args.mid_p,
                                                  args.x, xlabel)),
                        ("_combined", figure_combined(groups, ylabel, args.alpha,
                                                      args.mid_p, args.x, xlabel))):
        base = REPO_ROOT / "results" / f"{stem}{suffix}"
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {base.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
