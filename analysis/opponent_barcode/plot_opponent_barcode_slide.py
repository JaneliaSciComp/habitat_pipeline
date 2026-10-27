#!/usr/bin/env python3
"""
Slide figures for the per-opponent barcode test (``scripts/opponent_barcode.py``).

* ``<tag>_slide_fraction``: one panel per rat. Each opponent (sorted by
  number of encounters) shows the observed fraction of cells with
  d > 0.5 (dot) against the free-shuffle null 95% range (grey bar, tick at
  the null mean). The dot is red when p < ``--alpha`` under both the free
  and the time-block null.
* ``<tag>_slide_heatmap`` (all rats) and ``<tag>_slide_heatmap_<example>``
  (one rat): barcode cells only. ``*_all_events`` variants sort and show d
  from every encounter; those are not cross-validated and look diagonal
  even for noise, so read them next to the odd/even version. A cell is a barcode cell
  when its best opponent has d > 0.5 on odd encounters; rows are sorted by
  that opponent and show d on the held-out even encounters.

Reads ``results/<tag>.csv`` and the cached response matrices; nothing is
recomputed except the odd/even d.

Example
-------
::

    python scripts/plot_opponent_barcode_slide.py
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

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import opponent_barcode as ob  # noqa: E402
import plot_decoding_accuracy_violins as violins  # noqa: E402
from plot_decoding_auroc_histograms import INK, NONSIG_COLOR, SIG_COLOR  # noqa: E402

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

NULL_COLOR = "#c9c9c9"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=3.5, width=1.0, labelsize=9)


# group mode: 'self' = opponent in the focal rat's own ID half (_assign_id_groups)
GROUP_NAMES = {"self": "Same group", "others": "Other group"}


def _order(names: list, counts: np.ndarray) -> np.ndarray:
    """Column order: same/other group fixed; opponents by encounter count."""
    if set(names) <= set(GROUP_NAMES):
        return np.array([names.index(n) for n in GROUP_NAMES if n in names])
    return np.argsort(-np.asarray(counts), kind="stable")


def _short(name: str) -> str:
    if name in GROUP_NAMES:
        return GROUP_NAMES[name]
    return name[3:] if name.startswith("rat") else name


def fig_fraction(df: pd.DataFrame, cohorts: dict, alpha: float, thr: float,
                 xlabel: str = "Opponent (rat ID / encounters)"):
    rats = list(dict.fromkeys(df["animal_id"]))
    sub = df[(df.metric == "d") & (df.threshold == thr) & (df.item != "single_opponent_cells")]
    ncol = 3
    nrow = int(np.ceil(len(rats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow),
                             squeeze=False, sharey=True)
    for ax, rat in zip(axes.flat, rats):
        free = sub[(sub.animal_id == rat) & (sub.null == "free")]
        free = free.iloc[_order(free.item.tolist(), free.n_events.to_numpy())]
        block = sub[(sub.animal_id == rat) & (sub.null == "block")].set_index("item")
        x = np.arange(len(free))
        ax.bar(x, 100 * (free.null_hi - free.null_lo), bottom=100 * free.null_lo,
               width=0.6, color=NULL_COLOR, edgecolor="none", zorder=1)
        ax.hlines(100 * free.null_mean, x - 0.3, x + 0.3, color="#8a8a8a",
                  linewidth=1.0, zorder=2)
        sig = np.array([max(pf, block.loc[it, "p"]) < alpha
                        for it, pf in zip(free.item, free.p)])
        ax.scatter(x, 100 * free.observed, s=28, zorder=3,
                   color=np.where(sig, SIG_COLOR, NONSIG_COLOR),
                   edgecolors="white", linewidths=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{_short(i)}\n{n}" for i, n in zip(free.item, free.n_events)],
                           fontsize=7.5)
        ax.set_xlim(-0.6, len(free) - 0.4)
        _style(ax)
        ax.set_title(f"{rat} ({cohorts[rat].replace('cohort', 'cohort ')}), "
                     f"n = {free.n_cells.iloc[0]} cells", fontsize=9.5, loc="left")
    axes[0, 0].set_ylim(0, 1.05 * 100 * max(sub.null_hi.max(), sub.observed.max()))
    for ax in axes.flat[len(rats):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Cells with d > {thr:g} (%)", fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel, fontsize=9.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=6,
                          markerfacecolor=c, markeredgecolor="white")
               for c in (SIG_COLOR, NONSIG_COLOR)]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=NULL_COLOR))
    fig.legend(handles, [f"Observed, p < {alpha:g} (free and time-block shuffle)",
                         "Observed, n.s.", "Shuffle null, 95% range"],
               loc="upper right", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(1.0, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_heatmap(datas: dict, df: pd.DataFrame, thr: float, xlabel: str = "Opponent",
                all_events: bool = False):
    """Barcode-cell heatmap. ``all_events`` sorts and shows d from every
    encounter (not cross-validated: diagonal by construction)."""
    rats = list(datas)
    width = 1.6 * len(rats) + 0.9 if len(rats) > 1 else 2.6
    fig, axes = plt.subplots(1, len(rats), figsize=(width, 4.0), squeeze=False)
    im = None
    for ax, rat in zip(axes.flat, rats):
        data = datas[rat]
        names, codes = np.unique(data["labels"], return_inverse=True)
        K = names.size
        # columns sorted by encounter count, as in the fraction figure
        col = _order(list(names), np.bincount(codes, minlength=K))
        if all_events:
            do, _ = ob.selectivity(data["R"], codes, K)
            de = do
        else:
            a, b = ob.split_halves(codes, data["event_times"])
            do, _ = ob.selectivity(data["R"][:, a], codes[a], K)
            de, _ = ob.selectivity(data["R"][:, b], codes[b], K)
        do, de = do[:, col], de[:, col]
        ok = np.isfinite(do).all(1) & np.isfinite(de).all(1)
        pref = np.where(ok, np.nanargmax(np.where(ok[:, None], do, -np.inf), 1), -1)
        bar = ok & (do[np.arange(len(do)), pref.clip(0)] > thr)
        idx = np.flatnonzero(bar)
        idx = idx[np.lexsort((-do[idx, pref[idx]], pref[idx]))]
        im = ax.imshow(de[idx], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                       interpolation="nearest")
        # divider between preferred-opponent groups
        for r in np.flatnonzero(np.diff(pref[idx])) + 0.5:
            ax.axhline(r, color=INK, linewidth=0.6)
        rep = np.mean(de[idx].argmax(1) == pref[idx]) if idx.size else np.nan
        ax.set_xticks(range(K))
        ax.set_xticklabels([_short(n) for n in names[col]],
                           rotation=90 if K > 2 else 0, fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_yticks([])
        ax.set_title(f"{rat}\n{idx.size} cells" if all_events
                     else f"{rat}\n{idx.size} cells, {rep:.0%} replicate", fontsize=8.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    which = "all" if all_events else "odd"
    axes[0, 0].set_ylabel(f"Barcode cells (d > {thr:g} on {which} encounters),\n"
                          "sorted by preferred opponent", fontsize=8.5)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015)
    cb.set_label("d (all encounters)" if all_events
                 else "d on held-out even encounters", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--label_mode", choices=("opponent", "group"), default="opponent")
    p.add_argument("--response", choices=("evoked", "raw"), default="evoked")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--example", default="rat631",
                   help="Rat for the single-panel heatmap.")
    p.add_argument("--out_dir", default=".gui_cache/opponent_barcode")
    args = p.parse_args(argv)
    tag = f"barcode_{args.label_mode}" + ("" if args.response == "evoked" else "_raw")
    df = pd.read_csv(REPO_ROOT / "results" / f"{tag}.csv")
    datas = {a: ob.load_events_and_responses(a, s, args.label_mode, args.response,
                                             REPO_ROOT / args.out_dir)
             for a, s in violins.DEFAULT_TARGETS}
    cohorts = {a: violins.cohort_of_config(d["config_path"]) for a, d in datas.items()}
    group = args.label_mode == "group"
    xl_frac = "Opponent group (encounters)" if group else "Opponent (rat ID / encounters)"
    xl_heat = "Opponent group" if group else "Opponent"
    for name, fig in ((f"{tag}_slide_fraction", fig_fraction(df, cohorts, args.alpha,
                                                             ob.HEADLINE, xl_frac)),
                      (f"{tag}_slide_heatmap", fig_heatmap(datas, df, ob.HEADLINE, xl_heat)),
                      (f"{tag}_slide_heatmap_{args.example}",
                       fig_heatmap({args.example: datas[args.example]}, df, ob.HEADLINE,
                                   xl_heat)),
                      (f"{tag}_slide_heatmap_all_events",
                       fig_heatmap(datas, df, ob.HEADLINE, xl_heat, all_events=True)),
                      (f"{tag}_slide_heatmap_all_events_{args.example}",
                       fig_heatmap({args.example: datas[args.example]}, df, ob.HEADLINE,
                                   xl_heat, all_events=True))):
        base = REPO_ROOT / "results" / name
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {base.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
