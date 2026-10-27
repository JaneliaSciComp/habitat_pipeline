#!/usr/bin/env python3
"""
Single-bar summary of the BH-significant cells' balanced accuracy.

For each decode (EC top-2 opponent, EC group), pools every rat's cells with
BH q < ``--alpha`` from ``analysis/event_decoding/refine_permutation_fdr.py``'s CSV. It draws one
red bar at their mean balanced accuracy with a mean +- SD error bar.
Balanced accuracy is read from the cached decode (the same values as the
histograms), and nothing is recomputed.

Example
-------
::

    python analysis/event_decoding/plot_bh_significant_bar.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from analysis._paths import results_dir  # noqa: E402

from analysis.event_decoding import plot_decoding_accuracy_violins as violins  # noqa: E402
from analysis.event_decoding.plot_decoding_auroc_histograms import INK, SIG_COLOR  # noqa: E402
from analysis.event_decoding.refine_permutation_fdr import decode_params  # noqa: E402

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

WHAT = {"opponent": "opponent identity", "group": "group identity"}
STEM = {"opponent": "ec_top2", "group": "ec_group"}


def significant_values(mode: str, alpha: float, cache_dir: Path) -> np.ndarray:
    fdr = pd.read_csv(results_dir("event_decoding") / f"refined_fdr_{mode}.csv")
    params = decode_params(mode)
    vals = []
    for (animal, session), rows in fdr[fdr["q_value"] < alpha].groupby(
            ["animal_id", "session_id"], sort=False):
        res = violins._load_result_pickle(
            cache_dir / violins._cache_key(animal, str(session), params))
        cr = res["cell_results"]
        vals += [cr[cid]["balanced_accuracy"] for cid in rows["cluster_id"]]
    return np.asarray(vals, dtype=float)


def figure(vals: np.ndarray, mode: str, alpha: float) -> plt.Figure:
    mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    fig, ax = plt.subplots(figsize=(1.35, 3.2))
    ax.bar([0], [mean - 0.5], bottom=0.5, width=0.35, color=SIG_COLOR,
           edgecolor="none", zorder=2)
    ax.errorbar([0], [mean], yerr=[sd], color=INK, capsize=4, capthick=1.0,
                linewidth=1.0, zorder=3)
    ax.axhline(0.5, color=INK, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(0.45, 0.8)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xticks([0])
    ax.set_xticklabels([f"q < {alpha:g}\n(BH)\nn = {vals.size}"], fontsize=9)
    ax.set_ylabel(f"Balanced accuracy\n({WHAT[mode]})", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=3.5, width=1.0, labelsize=9)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--cache_dir", default=".gui_cache/accuracy_violins")
    args = p.parse_args(argv)
    for mode in ("opponent", "group"):
        vals = significant_values(mode, args.alpha, REPO_ROOT / args.cache_dir)
        print(f"{mode}: n={vals.size} BH-significant cells, balanced accuracy "
              f"mean {vals.mean():.3f} +- SD {vals.std(ddof=1):.3f} "
              f"(range {vals.min():.3f}-{vals.max():.3f})")
        fig = figure(vals, mode, args.alpha)
        base = results_dir("event_decoding") / f"{STEM[mode]}_balanced_accuracy_bh_bar"
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {base.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
