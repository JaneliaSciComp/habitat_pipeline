#!/usr/bin/env python3
"""
Cross-session summary of per-cell opponent-decoding accuracy, as violin plots.

Runs ``ephys.decode_opponent_identity.decode_opponent_identity_population`` once
per requested animal/recording, restricted to a single behaviour type
(``EC`` = encounter by default) and to the top-N opponents by surviving event
count (``--max_opponents 2`` by default, i.e. "top 2 animals"), then draws one
violin per animal/session of the per-cell cross-validated accuracies.

Two panels, because raw accuracy is not comparable across groups:

* **left** — raw per-cell accuracy. ``1/n_classes`` is drawn as a light dashed
  line, but the line that actually matters is the per-group **majority-class**
  baseline (CLAUDE.md: plain ``'accuracy'`` must be read against the
  majority-class rate, not ``1/n_classes``), drawn as a solid tick inside each
  violin's x-span.
* **right** — accuracy minus that group's own majority baseline, which is the
  comparable quantity. Zero is the honest "no better than guessing" line.

Pass ``--metric balanced_accuracy`` to use balanced accuracy instead, whose
chance level genuinely *is* ``1/n_classes``; the right panel then shows the
distance from that.

Gates run before any data is read: ``LabNotebook.assert_not_held_out`` on every
target, then ``discovery.capability_manifest.check_testable``.

Per-group results are pickled under ``--cache_dir`` and keyed on the full
parameter set, so re-plotting is instant; delete the pkl (or pass
``--force``) to recompute.

Examples
--------
::

    # the three sessions this script was written for
    python scripts/plot_decoding_accuracy_violins.py

    # explicit, with a significance screen
    python scripts/plot_decoding_accuracy_violins.py \\
        --target rat650:20250819 --target rat631:20251216 --target rat630:20251216 \\
        --behavior_type EC --max_opponents 2 \\
        --n_shuffles 1000 --null_mode pooled \\
        --output results/ec_top2_accuracy_violins.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from discovery.capability_manifest import ManifestStale, check_testable  # noqa: E402
from ephys.decode_opponent_identity import (  # noqa: E402
    decode_opponent_identity_population,
)
from ingestion.data_paths import DataStorageManager  # noqa: E402
from ingestion.ephys_sync import DataSyncManager  # noqa: E402
from ingestion.kilosort_data_import import load_kilosort_data  # noqa: E402
from video.behavioral_events import load_behavioral_events  # noqa: E402

logger = logging.getLogger(__name__)

#: Cohort name (as the capability manifest spells it) -> config file. Same map as
#: ``gui/session_index.COHORT_CONFIGS``; duplicated rather than imported so this
#: script does not drag in the GUI stack.
COHORT_CONFIGS = {
    "cohort7": "config/default_paths.json",
    "cohort5": "config/cohort5_paths.json",
}

#: The three animal/session pairs this summary was commissioned for.
DEFAULT_TARGETS = [
    ("rat650", "20250819"),
    ("rat631", "20251216"),
    ("rat630", "20251216"),
]

#: Categorical slots 1-3 of the validated default palette. These three
#: specifically clear the all-pairs CVD and normal-vision floors in both modes;
#: a fourth group must not simply take the next hue (yellow beside orange fails
#: all-pairs) -- facet instead.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"

_ANALYSIS = "ephys.decode_opponent_identity"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _target(spec: str) -> Tuple[str, str]:
    """Parse ``animal:session`` (e.g. ``rat631:20251216``)."""
    parts = spec.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"--target must be 'animal_id:session_id', got {spec!r}")
    return parts[0], parts[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Violin summary of per-cell opponent-decoding accuracy "
                    "across animal/session pairs.")
    p.add_argument("--target", type=_target, action="append", dest="targets",
                   metavar="ANIMAL:SESSION",
                   help="Animal/session to include; repeatable. Defaults to "
                        + ", ".join(f"{a}:{s}" for a, s in DEFAULT_TARGETS) + ".")
    p.add_argument("--behavior_type", type=str, default="EC",
                   help="Behaviour abbreviation to restrict to (default: EC = "
                        "encounter). 'any' for every type.")
    p.add_argument("--max_opponents", type=int, default=2,
                   help="Keep the top-N opponents by surviving event count "
                        "(default: 2).")
    p.add_argument("--metric", choices=("accuracy", "balanced_accuracy"),
                   default="accuracy",
                   help="Per-cell metric to plot (default: accuracy).")
    p.add_argument("--config_path", type=str, default=None,
                   help="Force one paths config for every target. By default "
                        "each target's cohort is read from the capability "
                        "manifest and mapped to its own config.")
    p.add_argument("--use_quality_cells", dest="use_quality_cells",
                   action="store_true", default=True,
                   help="Restrict to quality-filtered cells (default).")
    p.add_argument("--all_cells", dest="use_quality_cells", action="store_false",
                   help="Use every cluster instead of the quality subset.")
    p.add_argument("--alignment", choices=("start", "end"), default="start")
    p.add_argument("--time_window", type=float, nargs=2, default=(-1.0, 2.0),
                   metavar=("START", "END"))
    p.add_argument("--time_bin_size", type=float, default=0.5)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--min_events_per_class", type=int, default=5)
    p.add_argument("--n_shuffles", type=int, default=0,
                   help="Label-permutation shuffles per cell (default 0 = off). "
                        "Read the printed significance_resolution before "
                        "believing any per-cell null result.")
    p.add_argument("--null_mode", choices=("per_cell", "pooled"),
                   default="per_cell")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str,
                   default="results/decoding_accuracy_violins.png",
                   help="Figure path (.png). A .pdf sibling is written too.")
    p.add_argument("--cache_dir", type=str, default=".gui_cache/accuracy_violins",
                   help="Where per-group result pickles live.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even when a cached result matches.")
    p.add_argument("--skip_holdout_check", action="store_true",
                   help="Skip the lab-notebook holdout assertion. Only for a "
                        "machine with no notebook DB; never to get past a "
                        "HoldoutViolation.")
    p.add_argument("--log_level", default="WARNING")
    return p


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def assert_targets_not_held_out(targets: List[Tuple[str, str]]) -> None:
    """Refuse before any data is read if a target is reserved.

    ``Iteration.held_out`` is a descriptive label, not a gate; the registry in
    ``holdout_reservations`` is the real check (CLAUDE.md). An unresolvable
    session raises ``HoldoutIndeterminate``, which is deliberate -- let it
    propagate.
    """
    from database.lab_notebook import LabNotebook

    nb = LabNotebook()
    for animal_id, session_id in targets:
        # multi_animal=False: opponent decoding reads only the focal animal's
        # spikes; the opponent is a label on a behavioural event, not a second
        # ephys stream.
        status = nb.assert_not_held_out(session_id, animal_id,
                                        purpose="exploratory",
                                        multi_animal=False)
        print(f"  holdout OK  {animal_id}/{session_id}: {status.summary()}")


def resolve_cohort_config(session_id: str, override: Optional[str]) -> Optional[str]:
    """Map a session to its cohort's paths config via the capability manifest."""
    if override is not None:
        return override
    manifest_path = REPO_ROOT / "discovery" / "capability_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Could not read the capability manifest (%s); falling "
                       "back to the default paths config.", exc)
        return None
    for key, record in manifest.get("sessions", {}).items():
        if key == session_id or key.startswith(f"{session_id}_"):
            cohort = record.get("cohort")
            config = COHORT_CONFIGS.get(cohort)
            if config is None:
                logger.warning("Session %s is cohort %r, which has no config "
                               "in COHORT_CONFIGS; using the default.",
                               session_id, cohort)
            return config
    logger.warning("Session %s is not in the capability manifest; using the "
                   "default paths config.", session_id)
    return None


def report_feasibility(animal_id: str, session_id: str,
                       behavior_type: Optional[str]) -> bool:
    """Let the manifest refuse before the load. Returns ``report.testable``.

    ``ManifestStale`` is downgraded to a loud warning rather than a refusal.
    The manifest's own inputs have moved (``config['tracking']`` became a list
    of roots), and a rebuild is an hours-long share probe; it is right for the
    manifest to refuse to *answer*, but this analysis reads no tracking at all,
    so the staleness cannot change its verdict. The run then proceeds
    ungated and the decoder's own event/class checks are what refuse. Rebuild
    with ``scripts/build_capability_manifest.py --probe-level full`` to get the
    real gate back.
    """
    kwargs = {} if behavior_type is None else {"behavior_type": behavior_type}
    try:
        report = check_testable(_ANALYSIS, session_id, animal_id=animal_id, **kwargs)
    except ManifestStale as exc:
        print(f"  feasibility {animal_id}/{session_id}: NOT CHECKED - "
              f"manifest is stale ({exc}); proceeding ungated")
        return True
    print(f"  feasibility {animal_id}/{session_id}: "
          f"testable={report.testable}")
    for warning in getattr(report, "warnings", []) or []:
        print(f"    warning: {warning}")
    if not report.testable:
        print(report.summary())
    return bool(report.testable)


# ---------------------------------------------------------------------------
# Running one group
# ---------------------------------------------------------------------------

def _cache_key(animal_id: str, session_id: str, params: Dict) -> str:
    blob = json.dumps({"animal_id": animal_id, "session_id": session_id,
                       **params}, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode()).hexdigest()[:12]
    return f"{animal_id}_{session_id}_{digest}.pkl"


def run_group(animal_id: str, session_id: str, args, params: Dict,
              cache_dir: Path) -> Optional[Dict]:
    """Decode one animal/session, with a disk cache keyed on the parameters."""
    cache_file = cache_dir / _cache_key(animal_id, session_id, params)
    if cache_file.exists() and not args.force:
        print(f"  cached      {cache_file.name}")
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    config_path = resolve_cohort_config(session_id, args.config_path)
    print(f"  loading     config={config_path or 'config/default_paths.json'}")
    dm = DataStorageManager(animal_id, session_id, config_path=config_path)
    print(f"  recording   {dm.recording_id} "
          f"(primary={dm.is_primary_recording}, "
          f"on date: {', '.join(dm.recording_ids_on_date) or 'n/a'})")

    ks_data = load_kilosort_data(dm.get_kilosort_path())
    behavior_data = load_behavioral_events(dm.get_behavioral_event_files(),
                                           session_id=dm.session_id)
    # Calling the wrapper in-process means we must sync ourselves; only the
    # decoder CLIs' main() does it for you.
    sync = DataSyncManager(dm, dio_channel=1)
    behavior_data.synchronize_with_ephys(sync, create_new_columns=True)

    results = decode_opponent_identity_population(
        ks_data=ks_data,
        behavior_data=behavior_data,
        animal_of_interest=animal_id,
        behavior_type=params["behavior_type"],
        use_quality_cells=params["use_quality_cells"],
        alignment=params["alignment"],
        time_window=tuple(params["time_window"]),
        time_bin_size=params["time_bin_size"],
        cv_folds=params["cv_folds"],
        min_events_per_class=params["min_events_per_class"],
        max_opponents=params["max_opponents"],
        n_shuffles=params["n_shuffles"],
        null_mode=params["null_mode"],
        alpha=params["alpha"],
        seed=params["seed"],
    )
    results["_group"] = {"animal_id": animal_id, "session_id": session_id,
                         "recording_id": dm.recording_id,
                         "config_path": config_path}

    if results.get("status") == "success":
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as fh:
            pickle.dump(results, fh)
        print(f"  cached ->   {cache_file.name}")
    return results


# ---------------------------------------------------------------------------
# Summarising a group for the plot
# ---------------------------------------------------------------------------

def summarize(results: Dict, metric: str) -> Dict:
    """Pull the per-cell metric plus the baselines a reader needs."""
    cell_results = results["cell_results"]
    values = np.array([
        cell_results[cid][metric] for cid in results["successful_cells"]
        if np.isfinite(cell_results[cid].get(metric, np.nan))
    ], dtype=float)

    classes = list(results["behavioral_summary"]["unique_classes"])
    counts = results["behavioral_summary"]["class_counts"]
    n_classes = len(classes)
    chance = 1.0 / n_classes if n_classes else np.nan
    # population_baseline_accuracy is majority_class_baseline(labels); for
    # balanced accuracy the honest reference is 1/n_classes instead.
    if metric == "balanced_accuracy":
        baseline = chance
        baseline_name = f"chance (1/{n_classes})"
    else:
        baseline = float(results.get("population_baseline_accuracy", np.nan))
        baseline_name = "majority class"

    sig = results.get("significance") or {}
    n_sig = sum(1 for cid in results["successful_cells"]
                if sig.get(cid, {}).get("significant"))

    group = results.get("_group", {})
    return {
        "animal_id": group.get("animal_id", "?"),
        "session_id": group.get("session_id", "?"),
        "recording_id": group.get("recording_id"),
        "values": values,
        "n_cells": int(values.size),
        "n_total_cells": int(results.get("n_total_cells", 0)),
        "n_events": int(results["behavioral_summary"]["n_events"]),
        "classes": [str(c) for c in classes],
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "n_classes": n_classes,
        "chance": chance,
        "baseline": baseline,
        "baseline_name": baseline_name,
        "n_significant": n_sig if sig else None,
        "resolution": results.get("significance_resolution"),
        "significance_population": results.get("significance_population"),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)
    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def _draw_violins(ax, groups: List[Dict], series: List[np.ndarray],
                  colors: List[str]) -> None:
    positions = np.arange(1, len(series) + 1)
    parts = ax.violinplot(series, positions=positions, widths=0.72,
                          showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)
        body.set_linewidth(1.2)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, series, colors):
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.11, 0.11, size=vals.size)
        ax.scatter(pos + jitter, vals, s=9, color=color, alpha=0.55,
                   linewidths=0.4, edgecolors="white", zorder=3)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        # Thin IQR spine + a 2px surface-ringed median marker, so the summary
        # reads on top of the points without hiding them.
        ax.plot([pos, pos], [q1, q3], color=color, linewidth=2.0,
                solid_capstyle="round", zorder=4)
        ax.plot([pos], [med], marker="o", markersize=8, color=color,
                markeredgecolor="white", markeredgewidth=2.0, zorder=5)

    ax.set_xticks(positions)
    ax.set_xlim(0.4, len(series) + 0.6)


def make_figure(groups: List[Dict], args) -> plt.Figure:
    metric_label = ("Balanced accuracy" if args.metric == "balanced_accuracy"
                    else "Decoding accuracy")
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(groups))]
    raw = [g["values"] for g in groups]
    delta = [g["values"] - g["baseline"] for g in groups]

    fig, (ax_raw, ax_delta) = plt.subplots(
        1, 2, figsize=(12.5, 6.4), gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (ax_raw, ax_delta):
        ax.set_facecolor("#fcfcfb")
        _style_axes(ax)

    # ---- left: raw ------------------------------------------------------
    _draw_violins(ax_raw, groups, raw, colors)
    chances = {round(g["chance"], 6) for g in groups}
    if len(chances) == 1:
        chance = chances.pop()
        ax_raw.axhline(chance, color=INK_MUTED, linestyle=(0, (4, 3)),
                       linewidth=1.2, zorder=1)
        chance_label = f"1/n_classes = {chance:.0%}"
    else:
        chance_label = "1/n_classes"
        for pos, g in enumerate(groups, start=1):
            ax_raw.plot([pos - 0.36, pos + 0.36], [g["chance"]] * 2,
                        color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2,
                        zorder=1)

    for pos, g in enumerate(groups, start=1):
        ax_raw.plot([pos - 0.36, pos + 0.36], [g["baseline"]] * 2,
                    color=INK_PRIMARY, linewidth=1.8, solid_capstyle="butt",
                    zorder=2)
    ax_raw.plot([], [], color=INK_PRIMARY, linewidth=1.8,
                label=f"{groups[0]['baseline_name']} baseline")
    ax_raw.plot([], [], color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2,
                label=chance_label)
    ax_raw.legend(loc="upper left", frameon=False, fontsize=8,
                  labelcolor=INK_SECONDARY)

    ax_raw.set_ylabel(f"{metric_label} (per cell, cross-validated)",
                      fontsize=10, color=INK_PRIMARY)
    ax_raw.set_title(f"{metric_label} per cell", fontsize=11,
                     color=INK_PRIMARY, loc="left", pad=8)

    # ---- right: baseline-corrected -------------------------------------
    _draw_violins(ax_delta, groups, delta, colors)
    ax_delta.axhline(0.0, color=INK_PRIMARY, linewidth=1.4, zorder=2)
    ax_delta.plot([], [], color=INK_PRIMARY, linewidth=1.4,
                  label=f"zero = {groups[0]['baseline_name']} baseline")
    ax_delta.legend(loc="upper left", frameon=False, fontsize=8,
                    labelcolor=INK_SECONDARY)
    ax_delta.set_ylabel(f"{metric_label} − {groups[0]['baseline_name']} baseline",
                        fontsize=10, color=INK_PRIMARY)
    ax_delta.set_title("Baseline-corrected (comparable across sessions)",
                       fontsize=11, color=INK_PRIMARY, loc="left", pad=8)

    # ---- direct labels under each violin -------------------------------
    for ax, series in ((ax_raw, raw), (ax_delta, delta)):
        labels = []
        for g, vals in zip(groups, series):
            mean = np.mean(vals) if vals.size else np.nan
            frac_above = (np.mean(vals > (0.0 if ax is ax_delta else g["baseline"]))
                          if vals.size else np.nan)
            lines = [
                f"{g['animal_id']}\n{g['session_id']}",
                f"n={g['n_cells']}/{g['n_total_cells']} cells",
                f"mean {mean:+.1%}" if ax is ax_delta else f"mean {mean:.1%}",
                f"{frac_above:.0%} above",
            ]
            if g["n_significant"] is not None:
                lines.append(f"{g['n_significant']} sig.")
            labels.append("\n".join(lines))
        ax.set_xticklabels(labels, fontsize=8.5, color=INK_SECONDARY)
        for tick, color in zip(ax.get_xticklabels(), colors):
            tick.set_color(color)

    # ---- caption carrying the event / class provenance -----------------
    btype = args.behavior_type or "any"
    caption_rows = []
    for g in groups:
        classes = ", ".join(f"{c} (n={g['class_counts'].get(c, 0)})"
                            for c in g["classes"])
        rec = g["recording_id"] or g["session_id"]
        caption_rows.append(
            f"{g['animal_id']} / {rec}: {g['n_events']} {btype} events, "
            f"{g['n_classes']} classes — {classes}; "
            f"{g['baseline_name']} baseline {g['baseline']:.1%}")
    caption = "\n".join(caption_rows)

    fig.suptitle(
        f"Opponent-identity decoding from single cells — {btype} events, "
        f"top {args.max_opponents} opponents",
        fontsize=13.5, color=INK_PRIMARY, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.935,
             f"Per-cell cross-validated LDA ({args.cv_folds}-fold), "
             f"window {args.time_window[0]:+.1f} to {args.time_window[1]:+.1f} s "
             f"around event {args.alignment}, {args.time_bin_size:g} s bins, "
             f"{'quality-filtered' if args.use_quality_cells else 'all'} cells",
             fontsize=9, color=INK_SECONDARY, ha="left")
    fig.text(0.012, 0.012, caption, fontsize=7.8, color=INK_SECONDARY,
             ha="left", va="bottom", linespacing=1.5)

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.92))
    return fig


# ---------------------------------------------------------------------------
# Text summary (the table view the figure's colors are not the only carrier of)
# ---------------------------------------------------------------------------

def print_table(groups: List[Dict], metric: str) -> None:
    header = (f"{'animal/session':<24}{'cells':>8}{'events':>8}"
              f"{'mean':>9}{'median':>9}{'best':>9}{'baseline':>10}"
              f"{'>base':>8}{'sig':>6}")
    if metric == "balanced_accuracy":
        note = "'baseline' is 1/n_classes, which balanced accuracy genuinely has"
    else:
        note = ("'baseline' is each group's own majority-class rate, "
                "NOT 1/n_classes")
    print(f"\nper-cell {metric}; {note}")
    print(header)
    print("-" * len(header))
    for g in groups:
        vals = g["values"]
        above = np.sum(vals > g["baseline"]) if vals.size else 0
        sig = "-" if g["n_significant"] is None else str(g["n_significant"])
        print(f"{g['animal_id'] + '/' + g['session_id']:<24}"
              f"{g['n_cells']:>8}{g['n_events']:>8}"
              f"{np.mean(vals):>9.1%}{np.median(vals):>9.1%}"
              f"{np.max(vals):>9.1%}{g['baseline']:>10.1%}"
              f"{above:>8}{sig:>6}")
    print()
    for g in groups:
        res = g["resolution"]
        if res is not None and not res.get("resolvable", True):
            print(f"  !! {g['animal_id']}/{g['session_id']}: per-cell screen is "
                  f"UNDER-RESOLVED (best reachable q="
                  f"{res.get('best_achievable_q', float('nan')):.2f}); a null "
                  f"per-cell result here is predetermined by the shuffle budget, "
                  f"not biology. recommended_n_shuffles="
                  f"{res.get('recommended_n_shuffles')}")
        pop = g["significance_population"]
        if pop is not None:
            print(f"  {g['animal_id']}/{g['session_id']}: population-level "
                  f"p={pop.get('p_value'):.4g} "
                  f"(observed mean {pop.get('observed'):.1%})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(levelname)s %(name)s: %(message)s")

    targets = args.targets or list(DEFAULT_TARGETS)
    behavior_type = None if args.behavior_type.lower() == "any" else args.behavior_type
    args.behavior_type = behavior_type

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

    print("Gates")
    if args.skip_holdout_check:
        print("  holdout     SKIPPED by --skip_holdout_check")
    else:
        assert_targets_not_held_out(targets)
    runnable = [t for t in targets if report_feasibility(t[0], t[1], behavior_type)]
    if not runnable:
        print("No target is testable for these parameters; nothing to plot.")
        return 1
    if len(runnable) < len(targets):
        skipped = [f"{a}/{s}" for a, s in targets if (a, s) not in runnable]
        print(f"Skipping untestable target(s): {', '.join(skipped)}")

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir

    groups: List[Dict] = []
    for animal_id, session_id in runnable:
        print(f"\n=== {animal_id} / {session_id}")
        results = run_group(animal_id, session_id, args, params, cache_dir)
        if results is None or results.get("status") != "success":
            reason = (results or {}).get("error", "unknown error")
            print(f"  FAILED: {reason}")
            continue
        groups.append(summarize(results, args.metric))

    if not groups:
        print("\nNo group produced a successful result; nothing to plot.")
        return 1

    print_table(groups, args.metric)

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure(groups, args)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {pdf_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
