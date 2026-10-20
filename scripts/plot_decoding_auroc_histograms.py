#!/usr/bin/env python3
"""
Per-cell opponent-identity decoding auROC, as stacked count histograms.

Companion to ``scripts/plot_decoding_accuracy_violins.py``. It reads the
cached results of that script's significance run and re-scores each cell with
**auROC** instead of accuracy. The decoder is the same per-cell LDA: same
features, same scaling, and the same ``StratifiedKFold(shuffle=True,
random_state=42)`` folds. auROC is computed on the pooled cross-validated
LDA decision scores (``cross_val_predict(method='decision_function')``).
The auROC needs only one pass per cell; **no shuffles are rerun**.
Significance is the per-cell permutation ``p_value`` already in the cache
(uncorrected, add-one form). That test scored *plain accuracy* with the same
model and folds, not auROC.

The decoder auROC does not depend on which class is called "positive".
Swapping the opponents also swaps the sign of the LDA axis, which the fit
learns anyway. So 0.5 is chance, and values below 0.5 mean the
cross-validated prediction anti-generalises. They do not mean the cell
prefers the other opponent.

Two figures:

* ``<output>_by_rat.png``: one histogram per rat. Grey bars are cells that
  did not pass; orange bars at the base of each stack are cells with uncorrected
  p < ``--select_uncorrected_p``. The panel reports the share that passed.
* ``<output>_combined.png``: every cell of every rat pooled into one
  histogram.

Each rat's cached decode pkl must already exist for the same parameters as
the violins run. The cached pkl stores the event times (ephys s) and labels,
so only the Kilosort data is reloaded and nothing is re-synchronised.
The auROC results are cached under ``--cache_dir``.

Example
-------
::

    python scripts/plot_decoding_auroc_histograms.py \\
        --n_shuffles 100 --null_mode per_cell --select_uncorrected_p 0.05 \\
        --output results/ec_top2_auroc.png

    # self- vs other-group opponent (label_mode group)
    python scripts/plot_decoding_auroc_histograms.py --label_mode group \
        --n_shuffles 100 --null_mode per_cell --select_uncorrected_p 0.05 \
        --output results/ec_group_auroc.png
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ephys._lda_decoding import _prepare_lda_features  # noqa: E402
from ingestion.data_paths import DataStorageManager  # noqa: E402
from ephys.decode_event_outcome import decode_event_outcome_population  # noqa: E402
from ingestion.ephys_sync import DataSyncManager  # noqa: E402
from ingestion.kilosort_data_import import load_kilosort_data  # noqa: E402
from video.behavioral_events import load_behavioral_events  # noqa: E402
import plot_decoding_accuracy_violins as violins  # noqa: E402

SIG_COLOR = "#d1352b"
NONSIG_COLOR = "#8db4e2"
MID_COLOR = "#1f4e9c"  # uncorrected p only, between the two above
INK = "#0b0b0b"
BINS = np.linspace(0.0, 1.0, 41)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42  # editable text in the PDFs


def _build_parser() -> argparse.ArgumentParser:
    p = violins._build_parser()
    p.description = ("Stacked per-cell auROC histograms for opponent-identity "
                     "decoding, significance reused from the violins cache.")
    p.set_defaults(output="results/ec_top2_auroc.png",
                   select_uncorrected_p=0.05)
    p.add_argument("--auroc_cache_dir", type=str,
                   default=".gui_cache/auroc_histograms")
    p.add_argument("--decoder", choices=("identity", "outcome"),
                   default="identity",
                   help="'identity' reads the violins script's opponent/group "
                        "cache; 'outcome' decodes winner vs loser and caches "
                        "it here first (needs --n_shuffles for significance).")
    p.add_argument("--outcome_behavior_type", default="any",
                   help="Outcome mode only: behaviour type to restrict to "
                        "(default any = every event with a winner and a loser).")
    p.add_argument("--outcome_alignment", choices=("start", "end"),
                   default="end", help="Outcome mode only (default end).")
    p.add_argument("--refined_fdr", type=str, default=None, metavar="CSV",
                   help="Mark cells significant by BH q < --alpha from "
                        "scripts/refine_permutation_fdr.py's CSV instead of "
                        "the uncorrected cached p < --select_uncorrected_p.")
    p.add_argument("--mid_p", type=float, default=None, metavar="P",
                   help="With --refined_fdr: also mark, in dark blue, cells "
                        "with that file's (uncorrected) p < P that are not "
                        "BH-significant.")
    p.add_argument("--legend_at_peak", action="store_true",
                   help="Combined figure: put the legend's top edge level "
                        "with the tallest bar, right of the axes.")
    p.add_argument("--hist_metric",
                   choices=("auroc", "accuracy", "balanced_accuracy"),
                   default="auroc",
                   help="Per-cell value to histogram (default auroc). The "
                        "accuracies are read straight from the cached decode; "
                        "for plain accuracy each rat's majority-class baseline "
                        "is drawn as well, since 0.5 is not its chance level.")
    return p


def cell_aurocs(results: Dict, ks_data, params: Dict) -> Dict:
    """Cross-validated LDA auROC per successful cell, plus a re-derived
    accuracy that is checked against the cached one."""
    event_times = np.asarray(results["event_times"], dtype=float)
    labels = np.asarray(results["labels"])
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"auROC needs exactly 2 classes, got {list(classes)}")
    y = (labels == classes[1]).astype(int)
    cv = StratifiedKFold(n_splits=min(int(params["cv_folds"]), int(counts.min())),
                         shuffle=True, random_state=42)

    index = {cid: i for i, cid in enumerate(ks_data.ks_ids)}
    out = {}
    for cid in results["successful_cells"]:
        feats = _prepare_lda_features(
            np.asarray(ks_data.spike_times_by_cell[index[cid]]), event_times,
            tuple(params["time_window"]), params["time_bin_size"])
        if feats is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_predict(LinearDiscriminantAnalysis(), feats, labels,
                                       cv=cv, method="decision_function")
            preds = cross_val_predict(LinearDiscriminantAnalysis(), feats, labels,
                                      cv=cv)
        out[cid] = {"auroc": float(roc_auc_score(y, scores)),
                    "pooled_accuracy": float(np.mean(preds == labels))}
    return out


def run_outcome(animal_id: str, session_id: str, args, params: Dict) -> bool:
    """Make sure an outcome decode with significance is cached for this rat.

    Writes to the same cache directory and key scheme as the violins script
    (``params`` carries ``decoder='outcome'``, so keys never collide with
    the identity runs), after which ``collect`` reads it like any other.
    Returns False when the decoder refuses (e.g. too few winner/loser events).
    """
    cache_file = Path(args.cache_dir) / violins._cache_key(animal_id, session_id, params)
    if not cache_file.is_absolute():
        cache_file = REPO_ROOT / cache_file
    if cache_file.exists() and not args.force:
        return True

    config_path = violins.resolve_cohort_config(session_id, args.config_path)
    dm = DataStorageManager(animal_id, session_id, config_path=config_path)
    ks_data = load_kilosort_data(dm.get_kilosort_path())
    behavior_data = load_behavioral_events(dm.get_behavioral_event_files(),
                                           session_id=dm.session_id)
    sync = DataSyncManager(dm, dio_channel=1)
    behavior_data.synchronize_with_ephys(sync, create_new_columns=True)
    results = decode_event_outcome_population(
        ks_data=ks_data, behavior_data=behavior_data,
        animal_of_interest=animal_id,
        behavior_type=params["behavior_type"],
        use_quality_cells=params["use_quality_cells"],
        alignment=params["alignment"],
        time_window=tuple(params["time_window"]),
        time_bin_size=params["time_bin_size"],
        cv_folds=params["cv_folds"],
        min_events_per_class=params["min_events_per_class"],
        n_shuffles=params["n_shuffles"], null_mode=params["null_mode"],
        alpha=params["alpha"], seed=params["seed"])
    if results.get("status") != "success":
        print(f"  outcome decode FAILED: {results.get('error', 'unknown error')}")
        return False
    results["_group"] = {"animal_id": animal_id, "session_id": session_id,
                         "recording_id": dm.recording_id,
                         "config_path": config_path}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as fh:
        pickle.dump(results, fh)
    print(f"  cached ->   {cache_file.name}")
    return True


def collect(animal_id: str, session_id: str, args, params: Dict) -> Optional[Dict]:
    cache_file = Path(args.cache_dir) / violins._cache_key(animal_id, session_id, params)
    if not cache_file.is_absolute():
        cache_file = REPO_ROOT / cache_file
    if not cache_file.exists():
        print(f"  no cached decode {cache_file.name}; run the violins script "
              f"with the same parameters first")
        return None
    results = violins._load_result_pickle(cache_file)
    sig = results.get("significance") or {}
    if not sig:
        print(f"  {cache_file.name} has no significance; need --n_shuffles > 0")
        return None

    auc_dir = Path(args.auroc_cache_dir)
    if not auc_dir.is_absolute():
        auc_dir = REPO_ROOT / auc_dir
    auc_file = auc_dir / cache_file.name
    if auc_file.exists() and not args.force:
        print(f"  cached auROC {auc_file.name}")
        with open(auc_file, "rb") as fh:
            aucs = pickle.load(fh)
    else:
        group = results.get("_group", {})
        config_path = group.get("config_path") or violins.resolve_cohort_config(
            session_id, args.config_path)
        dm = DataStorageManager(animal_id, session_id, config_path=config_path)
        ks_data = load_kilosort_data(dm.get_kilosort_path())
        aucs = cell_aurocs(results, ks_data, params)
        auc_dir.mkdir(parents=True, exist_ok=True)
        with open(auc_file, "wb") as fh:
            pickle.dump(aucs, fh)
        print(f"  auROC ->    {auc_file.name}")

    # The pooled-prediction accuracy is not the mean of the fold accuracies,
    # but it should be close; a large gap would mean different features/folds.
    cr = results["cell_results"]
    gaps = [abs(v["pooled_accuracy"] - cr[c]["accuracy"]) for c, v in aucs.items()]
    print(f"  check: |pooled - cached accuracy| max {max(gaps):.3f}, "
          f"median {np.median(gaps):.4f} over {len(gaps)} cells")

    cids = list(aucs)
    p_thr = args.select_uncorrected_p
    mid = None
    if args.hist_metric == "auroc":
        vals = np.array([aucs[c]["auroc"] for c in cids])
    else:
        vals = np.array([cr[c][args.hist_metric] for c in cids], dtype=float)
    if args.refined_fdr:
        fdr = pd.read_csv(args.refined_fdr)
        fdr = fdr[fdr["animal_id"] == animal_id].set_index("cluster_id")
        missing = [c for c in cids if c not in fdr.index]
        if missing:
            raise SystemExit(f"{animal_id}: {len(missing)} cells missing from "
                             f"{args.refined_fdr}")
        passed = np.array([bool(fdr.loc[c, "q_value"] < args.alpha) for c in cids])
        if args.mid_p is not None:
            mid = np.array([bool(fdr.loc[c, "p_value"] < args.mid_p) for c in cids])
    else:
        passed = np.array([sig.get(c, {}).get("p_value", 1.0) < p_thr for c in cids])
    bs = results["behavioral_summary"]
    return {"animal_id": animal_id, "session_id": session_id,
            "cohort": violins.cohort_of_config(results.get("_group", {}).get("config_path")),
            "auroc": vals, "significant": passed, "mid": mid,
            "baseline": float(results.get("population_baseline_accuracy", np.nan)),
            "class_counts": {str(k): int(v) for k, v in bs["class_counts"].items()}}


def _xlim(values: np.ndarray) -> tuple:
    """Data range padded by one bin, snapped to 0.05, always showing 0.5."""
    lo = min(float(np.nanmin(values)), 0.5) - (BINS[1] - BINS[0])
    hi = max(float(np.nanmax(values)), 0.5) + (BINS[1] - BINS[0])
    return max(0.0, np.floor(lo * 20) / 20), min(1.0, np.ceil(hi * 20) / 20)


def _style(ax, xlim: tuple, labelsize: float) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=3.5, width=1.0, labelsize=labelsize)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def draw_hist(ax, auc: np.ndarray, significant: np.ndarray, title: str,
              p_thr: str, xlim: tuple, big: bool = False,
              baseline: Optional[float] = None,
              mid: Optional[np.ndarray] = None, mid_text: str = "") -> None:
    # `mid` (optional): cells passing a looser test, drawn dark blue between
    # the significant base and the rest. Significant cells are never counted
    # in it.
    mid = np.zeros_like(significant) if mid is None else (mid & ~significant)
    n_ns, _ = np.histogram(auc[~significant & ~mid], bins=BINS)
    n_m, _ = np.histogram(auc[mid], bins=BINS)
    n_s, _ = np.histogram(auc[significant], bins=BINS)
    width = np.diff(BINS)
    frac = significant.mean() if auc.size else 0.0
    fs = 12 if big else 9.5
    # Stacked: the full bar height is every cell in the bin, the red base
    # is the significant part of it.
    ax.bar(BINS[:-1], n_s, width=width, align="edge", color=SIG_COLOR,
           edgecolor="white", linewidth=0.6,
           label=f"Significant cells\n({p_thr}, {frac:.0%})")
    if mid.any():
        ax.bar(BINS[:-1], n_m, width=width, align="edge", bottom=n_s,
               color=MID_COLOR, edgecolor="white", linewidth=0.6,
               label=f"Uncorrected significant\n({mid_text}, "
                     f"{(mid | significant).mean():.0%})")
    ax.bar(BINS[:-1], n_ns, width=width, align="edge", bottom=n_s + n_m,
           color=NONSIG_COLOR, edgecolor="white", linewidth=0.6,
           label="All cells")
    ax.axvline(0.5, color=INK, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    if baseline is not None and np.isfinite(baseline):
        ax.axvline(baseline, color=INK, linewidth=1.1, zorder=3,
                   label="Majority class")
    ax._bar_tops = n_ns + n_m + n_s  # read back by _fit_legends
    ax.set_ylim(0, int(ax._bar_tops.max()) * 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True,
                                           steps=[1, 2, 2.5, 5, 10]))
    _style(ax, xlim, labelsize=fs - 1)
    ax.set_box_aspect(1)  # square plotting area
    handles, labels = ax.get_legend_handles_labels()
    order = ([labels.index("All cells")]
             + [i for i, l in enumerate(labels) if l.startswith("Uncorrected")]
             + [i for i, l in enumerate(labels) if l.startswith("Significant")]
             + [i for i, l in enumerate(labels) if l == "Majority class"])
    # Opaque so the 0.5 / majority guide lines don't run through the text.
    leg = ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper right", frameon=True, fancybox=False,
              framealpha=1.0, edgecolor="none",
              title=f"{title}\nn = {auc.size} cells",
              title_fontsize=fs - 0.5, alignment="left", fontsize=fs - 2,
              handlelength=1.0, handleheight=0.7, labelspacing=0.35,
              borderaxespad=0.2, borderpad=0.1)
    leg.set_zorder(5)


def _fit_legends(fig: plt.Figure, axes) -> None:
    """Raise each panel's y limit just enough that no bar reaches the legend.

    Run after ``tight_layout``: the legend is anchored in axes coordinates, so
    its size as a fraction of the axes is only final once the layout is.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        leg = ax.get_legend()
        if leg is None or not ax.get_visible():
            continue
        bb = leg.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        x0, x1 = ax.get_xlim()
        # One bin of slack each side: a bar only touching the box counts.
        pad = BINS[1] - BINS[0]
        lo, hi = x0 + bb.x0 * (x1 - x0) - pad, x0 + bb.x1 * (x1 - x0) + pad
        under = (BINS[1:] > lo) & (BINS[:-1] < hi)
        tallest = ax._bar_tops[under].max() if under.any() else 0
        room = bb.y0 - 0.03  # axes fraction left below the legend
        if tallest > 0 and room > 0.1:
            ax.set_ylim(0, max(ax.get_ylim()[1], tallest / room))
        # Guide lines (axvline: y in axes fraction) that pass behind the
        # legend stop at its bottom, so no stub pokes out above the box.
        lx0, lx1 = x0 + bb.x0 * (x1 - x0), x0 + bb.x1 * (x1 - x0)
        for line in ax.lines:
            xs = line.get_xdata()
            if len(xs) == 2 and xs[0] == xs[1] and lx0 <= xs[0] <= lx1:
                line.set_ydata([0, bb.y0 - 0.01])


def figure_by_rat(groups: List[Dict], p_thr: str, xlabel: str,
                  show_baseline: bool = False, mid_text: str = "") -> plt.Figure:
    ncol = min(3, len(groups))
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow),
                             squeeze=False)
    # One x range for every panel, so the rats can be compared by eye.
    xlim = _xlim(np.concatenate([g["auroc"] for g in groups]
                                + [[g["baseline"]] for g in groups
                                   if show_baseline and np.isfinite(g["baseline"])]))
    for ax, g in zip(axes.flat, groups):
        draw_hist(ax, g["auroc"], g["significant"],
                  f"{g['animal_id']} ({g['cohort'].replace('cohort', 'cohort ')})",
                  p_thr, xlim, baseline=g["baseline"] if show_baseline else None,
                  mid=g.get("mid"), mid_text=mid_text)
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cell count", fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel, fontsize=10)
    for _ in range(2):  # the layout and the limits depend on each other
        fig.tight_layout()
        _fit_legends(fig, axes.flat)
    return fig


def figure_combined(groups: List[Dict], p_thr: str, xlabel: str,
                    mid_text: str = "", legend_at_peak: bool = False) -> plt.Figure:
    auc = np.concatenate([g["auroc"] for g in groups])
    sig = np.concatenate([g["significant"] for g in groups])
    mid = (np.concatenate([g["mid"] for g in groups])
           if all(g.get("mid") is not None for g in groups) else None)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    title = (f"{len(groups)} rats pooled" if len(groups) > 1 else
             f"{groups[0]['animal_id']} "
             f"({groups[0]['cohort'].replace('cohort', 'cohort ')})")
    draw_hist(ax, auc, sig, title, p_thr, _xlim(auc),
              big=True, mid=mid, mid_text=mid_text)
    ax.set_ylabel("Cell count", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    if legend_at_peak:
        # Legend top level with the tallest bar, pushed out past the right
        # edge of the axes (there is no right spine), so the y range only
        # needs to fit the bars.
        leg = ax.get_legend()
        leg.borderaxespad = 0.0
        leg.borderpad = 0.0
        frac = ax._bar_tops.max() / ax.get_ylim()[1]
        leg.set_bbox_to_anchor((1.12, frac), transform=ax.transAxes)
        fig.tight_layout()
        return fig
    for _ in range(2):
        fig.tight_layout()
        _fit_legends(fig, [ax])
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    targets = args.targets or list(violins.DEFAULT_TARGETS)
    behavior_type = None if args.behavior_type.lower() == "any" else args.behavior_type
    # Must match violins.main exactly, or the cache key misses.
    params = {
        "behavior_type": behavior_type,
        "max_opponents": args.max_opponents,
        "use_quality_cells": args.use_quality_cells,
        "alignment": args.alignment,
        "time_window": list(args.time_window),
        "time_bin_size": args.time_bin_size,
        "cv_folds": args.cv_folds,
        "min_events_per_class": args.min_events_per_class,
        "n_shuffles": args.n_shuffles,
        "null_mode": args.null_mode,
        "alpha": args.alpha,
        "seed": args.seed,
    }
    if args.label_mode == "group":
        # Same convention as violins.main: the key is only present in group
        # mode, which pools every opponent into self/others.
        params["label_mode"] = "group"
        params["max_opponents"] = None
    if args.decoder == "outcome":
        # Winner vs loser of the focal rat's own events. The --behavior_type
        # default (EC) is the identity runs' choice; outcome uses every event
        # with a winner and a loser unless --outcome_behavior_type says
        # otherwise. Aligned to event end by default, like the decoder.
        params.update(decoder="outcome", max_opponents=None,
                      behavior_type=(None if args.outcome_behavior_type == "any"
                                     else args.outcome_behavior_type),
                      alignment=args.outcome_alignment)
        params.pop("label_mode", None)
    metric_name = {"auroc": "auROC", "accuracy": "Accuracy",
                   "balanced_accuracy": "Balanced accuracy"}[args.hist_metric]
    what = {"outcome": "win / loss outcome"}.get(
        args.decoder, "group identity" if args.label_mode == "group"
        else "opponent identity")
    xlabel = f"{metric_name}\n({what})"

    if args.decoder == "outcome" and not args.skip_holdout_check:
        violins.assert_targets_not_held_out(targets)

    groups = []
    for animal_id, session_id in targets:
        print(f"\n=== {animal_id} / {session_id}")
        if args.decoder == "outcome" and not run_outcome(
                animal_id, session_id, args, params):
            continue
        g = collect(animal_id, session_id, args, params)
        if g is not None:
            groups.append(g)
    if not groups:
        return 1

    # Legend text for the red bars.
    p_thr = (f"q < {args.alpha:g}, BH" if args.refined_fdr
             else f"p < {args.select_uncorrected_p:g}")
    mid_text = f"p < {args.mid_p:g}" if args.mid_p is not None else ""
    print(f"\n{'rat':<9}{'cells':>6}{'sig':>5}{'%sig':>7}{'median':>8}"
          f"{'median sig':>12}  classes   [{args.hist_metric}]")
    for g in groups:
        s = g["significant"]
        print(f"{g['animal_id']:<9}{g['auroc'].size:>6}{s.sum():>5}{s.mean():>7.1%}"
              f"{np.median(g['auroc']):>8.3f}"
              f"{(np.median(g['auroc'][s]) if s.any() else np.nan):>12.3f}  "
              f"{g['class_counts']}")

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, fig in (("_by_rat", figure_by_rat(
                            groups, p_thr, xlabel,
                            show_baseline=args.hist_metric == "accuracy",
                            mid_text=mid_text)),
                        ("_combined", figure_combined(
                            groups, p_thr, xlabel, mid_text,
                            legend_at_peak=args.legend_at_peak))):
        base = out.with_name(out.stem + suffix)
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {base.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
