#!/usr/bin/env python3
"""
Slide composites for single-cell opponent decoding, as vector figures.

One figure per decode (EC top-2 opponent identity, EC self/others group), with
the layout of the hand-assembled ``slides_decoding_{identity,group}_631.png``:

* left column: peri-event firing rate (mean +- SEM, split by class) of example
  cells whose classes overlap before event onset and separate after it;
* top middle: pooled histogram of every rat's per-cell balanced accuracy,
  stacked by uncorrected p < 0.05 and BH q < 0.05 (same as
  ``plot_decoding_auroc_histograms.py --hist_metric balanced_accuracy
  --refined_fdr ... --mid_p 0.05``);
* top right: mean +- SD balanced accuracy of the BH-significant cells (same as
  ``plot_bh_significant_bar.py``);
* bottom middle: confusion matrix of the top example cell.

Example cells (``EXAMPLE_CELLS``) were picked by eye from the top of the
``onset_ranking`` list: per-event rates in 0.5 s bins, Welch t between the two
classes per bin, score = max |t| over (0, 2) s minus max |t| over [-5, 0) s.
``--cells`` overrides them; ``--cells auto`` takes the top of that ranking.

Nothing is decoded here. Decodes come from the violins cache
(``.gui_cache/accuracy_violins``, parameters from
``refine_permutation_fdr.decode_params``) and significance from
``results/event_decoding/refined_fdr_<mode>.csv``. Only the example rat's
Kilosort data is loaded, for the firing-rate curves.

Cells are ranked by balanced accuracy, not plain accuracy: the group decode is
imbalanced (165 self / 87 others for rat631), so plain accuracy has a ~65%
majority-class floor.

Example
-------
::

    python analysis/event_decoding/plot_decoding_slides.py --animal rat631
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
from analysis.event_decoding import plot_decoding_auroc_histograms as hist  # noqa: E402
from analysis.event_decoding.decoding_plots import _cluster_index_map, _draw_confusion_matrix  # noqa: E402
from analysis.event_decoding.plot_bh_significant_bar import significant_values  # noqa: E402
from analysis.event_decoding.refine_permutation_fdr import decode_params  # noqa: E402
from analysis.event_decoding._lda_decoding import align_spikes_to_events  # noqa: E402
from ingestion.data_paths import DataStorageManager  # noqa: E402
from ingestion.kilosort_data_import import load_kilosort_data  # noqa: E402

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42   # editable text in the PDF
plt.rcParams["svg.fonttype"] = "none"  # text stays text in the SVG

WHAT = {"opponent": "opponent identity", "group": "group identity"}
STEM = {"opponent": "identity", "group": "group"}
CLASS_COLORS = ("#e41a1c", "#999999")  # Set1 ends, as in the original PETHs
INK = hist.INK
# rat631 / 20251216, picked from onset_ranking (see module docstring)
EXAMPLE_CELLS = {"opponent": [1119, 1016, 892, 1170],
                 "group": [816, 955, 1175, 970]}


def load_decode(animal: str, session: str, mode: str, cache_dir: Path) -> dict:
    return violins._load_result_pickle(
        cache_dir / violins._cache_key(animal, session, decode_params(mode)))


def pooled_groups(mode: str, cache_dir: Path, alpha: float, mid_p: float):
    """Per-rat balanced accuracy + BH / uncorrected flags, in the CSV's rat order."""
    fdr = pd.read_csv(results_dir("event_decoding") / f"refined_fdr_{mode}.csv")
    vals, sig, mid = [], [], []
    n_rats = 0
    for (animal, session), rows in fdr.groupby(["animal_id", "session_id"], sort=False):
        cr = load_decode(animal, str(session), mode, cache_dir)["cell_results"]
        rows = rows.set_index("cluster_id")
        cids = [c for c in rows.index if c in cr]
        if len(cids) != len(rows):
            raise SystemExit(f"{animal}: {len(rows) - len(cids)} CSV cells not in the cached decode")
        vals.append([cr[c]["balanced_accuracy"] for c in cids])
        sig.append((rows.loc[cids, "q_value"] < alpha).to_numpy())
        mid.append((rows.loc[cids, "p_value"] < mid_p).to_numpy())
        n_rats += 1
    return (np.concatenate(vals).astype(float), np.concatenate(sig),
            np.concatenate(mid), n_rats)


def onset_ranking(ks, res, window=(-5.0, 5.0), bin_size=0.5, post=(0.0, 2.0),
                  min_rate=0.5) -> pd.DataFrame:
    """Cells ranked by post-onset minus pre-onset class separation (|Welch t|)."""
    from scipy.stats import ttest_ind
    idx = _cluster_index_map(ks)
    ev = np.asarray(res["event_times"], dtype=float)
    lab = np.asarray(res["labels"])
    c0, c1 = list(res["behavioral_summary"]["unique_classes"])
    edges = np.arange(window[0], window[1] + bin_size / 2, bin_size)
    cen = edges[:-1] + bin_size / 2
    pre, aft = cen < 0, (cen > post[0]) & (cen < post[1])
    rows = []
    for cid in res["successful_cells"]:
        R = np.array([np.histogram(s, bins=edges)[0] / bin_size for s in
                      align_spikes_to_events(ks.spike_times_by_cell[idx[cid]], ev, window)])
        t = np.abs(np.nan_to_num(ttest_ind(R[lab == c0], R[lab == c1],
                                           equal_var=False).statistic))
        rows.append(dict(cluster_id=cid, pre_t=t[pre].max(), post_t=t[aft].max(),
                         rate=R.mean()))
    d = pd.DataFrame(rows)
    d["score"] = d.post_t - d.pre_t
    return d[d.rate > min_rate].sort_values("score", ascending=False)


def draw_peth(ax, spike_times, event_times, labels, classes, window, bin_size,
              legend: bool) -> None:
    edges = np.arange(window[0], window[1] + bin_size / 2, bin_size)
    centers = edges[:-1] + bin_size / 2
    top = 0.0
    for cls, color in zip(classes, CLASS_COLORS):
        ev = event_times[labels == cls]
        rates = np.array([np.histogram(s, bins=edges)[0] / bin_size
                          for s in align_spikes_to_events(spike_times, ev, window)])
        mean = rates.mean(axis=0)
        sem = rates.std(axis=0) / np.sqrt(len(rates))
        ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.3,
                        linewidth=0)
        ax.plot(centers, mean, color=color, linewidth=1.6, label=f"{cls} (n={len(ev)})")
        top = max(top, float(np.max(mean + sem)))
    ax.axvline(0, color="0.35", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.15 * top if top > 0 else 1.0)
    ax.set_xlim(window)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=8)
    if legend:
        ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9,
                  handlelength=1.5)


def draw_bar(ax, vals: np.ndarray, mode: str, alpha: float) -> None:
    """``plot_bh_significant_bar.figure`` drawn on a given axis."""
    mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    ax.bar([0], [mean - 0.5], bottom=0.5, width=0.35, color=hist.SIG_COLOR,
           edgecolor="none", zorder=2)
    ax.errorbar([0], [mean], yerr=[sd], color=INK, capsize=4, capthick=1.0,
                linewidth=1.0, zorder=3)
    ax.axhline(0.5, color=INK, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(0.45, max(0.8, np.ceil((mean + sd) * 10) / 10))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xticks([0])
    ax.set_xticklabels([f"q < {alpha:g}\n(BH)\nn = {vals.size}"], fontsize=10)
    ax.set_ylabel(f"Balanced accuracy\n({WHAT[mode]})", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3.5, width=1.0, labelsize=10)
    ax.tick_params(axis="x", length=0)


def figure(mode: str, animal: str, session: str, ks, cache_dir: Path,
           alpha: float, mid_p: float, n_cells: int, window, bin_size,
           cells=None) -> plt.Figure:
    res = load_decode(animal, session, mode, cache_dir)
    cr = res["cell_results"]
    if cells is None:
        cells = onset_ranking(ks, res).cluster_id.tolist()[:n_cells]
    ranked = sorted(cells, key=lambda c: cr[c]["balanced_accuracy"], reverse=True)
    n_cells = len(ranked)
    classes = list(res["behavioral_summary"]["unique_classes"])
    event_times = np.asarray(res["event_times"], dtype=float)
    labels = np.asarray(res["labels"])
    idx = _cluster_index_map(ks)

    fig = plt.figure(figsize=(12.5, 8.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.22,
                             left=0.07, right=0.98, top=0.95, bottom=0.08)
    left = outer[0].subgridspec(n_cells, 1, hspace=0.45)
    peth_axes = []
    for i, cid in enumerate(ranked):
        ax = fig.add_subplot(left[i], sharex=peth_axes[0] if peth_axes else None)
        draw_peth(ax, ks.spike_times_by_cell[idx[cid]], event_times, labels, classes,
                  window, bin_size, legend=(i == 0))
        ax.set_title(f"Cell {cid}\nBalanced accuracy: {cr[cid]['balanced_accuracy']:.1%}",
                     fontsize=9)
        if i < n_cells - 1:
            ax.tick_params(labelbottom=False)
        peth_axes.append(ax)
    peth_axes[-1].set_xlabel("Time from event (s)", fontsize=11)
    fig.text(0.015, 0.5, "Firing rate (Hz)", rotation=90, va="center", fontsize=11)

    # columns: histogram / confusion matrix, legend gutter, BH bar
    right = outer[1].subgridspec(2, 3, width_ratios=[2.6, 2.1, 0.75],
                                 height_ratios=[1, 1], hspace=0.38, wspace=0.15)
    # Pooled histogram (all rats)
    vals, sig, mid, n_rats = pooled_groups(mode, cache_dir, alpha, mid_p)
    axh = fig.add_subplot(right[0, 0])
    hist.draw_hist(axh, vals, sig, f"{n_rats} rats pooled", f"q < {alpha:g}, BH",
                   hist._xlim(vals), big=True, mid=mid, mid_text=f"p < {mid_p:g}")
    axh.set_ylabel("Cell count", fontsize=12)
    axh.set_xlabel(f"Balanced accuracy\n({WHAT[mode]})", fontsize=12)
    # Bars fill the y range; the legend is rebuilt in its own gutter right of
    # the axes, so it never covers a bar.
    axh.set_ylim(0, axh._bar_tops.max() * 1.05)
    old = axh.get_legend()
    handles, texts = old.legend_handles, [t.get_text() for t in old.get_texts()]
    title = old.get_title().get_text()
    old.remove()
    axh.legend(handles, texts, loc="upper left", bbox_to_anchor=(1.0, 1.02),
               frameon=False, title=title, title_fontsize=11.5, alignment="left",
               fontsize=10, handlelength=1.0, handleheight=0.7, labelspacing=0.35,
               borderaxespad=0.2, borderpad=0.1)

    # BH-significant bar
    axb = fig.add_subplot(right[0, 2])
    draw_bar(axb, significant_values(mode, alpha, cache_dir), mode, alpha)

    # Top example cell's confusion matrix
    best = ranked[0]
    axc = fig.add_subplot(right[1, 0])
    im = _draw_confusion_matrix(
        axc, np.asarray(cr[best]["confusion_matrix"]), classes,
        title=f"Confusion matrix — cell {best}",
        xlabel="Predicted opponent" if mode == "opponent" else "Predicted group",
        ylabel="True opponent" if mode == "opponent" else "True group")
    axc.title.set_fontsize(12)
    fig.colorbar(im, ax=axc, fraction=0.046, pad=0.04)

    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--animal", default="rat631")
    p.add_argument("--session", default="20251216")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mid_p", type=float, default=0.05)
    p.add_argument("--n_cells", type=int, default=4)
    p.add_argument("--window", type=float, nargs=2, default=(-5.0, 5.0))
    p.add_argument("--bin_size", type=float, default=0.2)
    p.add_argument("--cache_dir", default=".gui_cache/accuracy_violins")
    p.add_argument("--cells", nargs="+", default=None,
                   help="cluster ids for the PETH column (one mode only), or 'auto' "
                        "for the top of onset_ranking; default EXAMPLE_CELLS")
    p.add_argument("--modes", nargs="+", default=["opponent", "group"],
                   choices=["opponent", "group"])
    args = p.parse_args(argv)

    cache_dir = REPO_ROOT / args.cache_dir
    ks = load_kilosort_data(DataStorageManager(args.animal, args.session).get_kilosort_path())
    for mode in args.modes:
        if args.cells == ["auto"]:
            cells = None
        elif args.cells:
            cells = [int(c) for c in args.cells]
        else:
            cells = EXAMPLE_CELLS[mode] if args.animal == "rat631" else None
        fig = figure(mode, args.animal, args.session, ks, cache_dir, args.alpha,
                     args.mid_p, args.n_cells, tuple(args.window), args.bin_size,
                     cells=cells)
        base = results_dir("event_decoding") / (
            f"slides_decoding_{STEM[mode]}_{args.animal.removeprefix('rat')}_vector")
        for ext in ("pdf", "svg", "png"):
            fig.savefig(base.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None)
        plt.close(fig)
        print(f"wrote {base}.{{pdf,svg,png}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
