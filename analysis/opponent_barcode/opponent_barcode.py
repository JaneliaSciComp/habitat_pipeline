#!/usr/bin/env python3
"""
Per-opponent "barcode" test: does each opponent have its own small set of
cells that fire more during encounters with it?

For every quality cell and every encounter (EC events, aligned to onset), the
response is the mean rate in [0, 2] s minus that encounter's own baseline
rate in ``--baseline`` (default [-2, -1] s: the scored onset can be off, so
[-1, 0] may already hold the response; ``--response raw`` drops the
baseline). Encounters whose baseline starts before the cells' first spike
are dropped. All opponents that clear
``min_events_per_class`` are used (``max_opponents=None``), not the top two.

Per cell and opponent the selectivity is an opponent-vs-rest effect size::

    d[c, o] = (mean_o - mean_rest) / pooled within-opponent SD

It is not capped by the number of opponents, unlike a z-score across the N
opponent means (max (N-1)/sqrt(N)). That z is kept as a secondary metric.
For each opponent and threshold, the observed fraction of cells with
``d > threshold`` is compared with two label-permutation nulls (counts per
opponent kept):

* ``free``: labels permuted across all encounters;
* ``block``: labels permuted only within ``--block_min``-minute blocks of
  encounter time, so slow drift cannot pose as identity.

Specificity: the number of opponents each cell passes for (a barcode means
an excess of single-opponent cells), plus Jaccard overlap between opponents'
cell sets. Replication: preferred opponent picked on odd encounters (per
opponent, in time order) and checked on even ones, and vice versa. The
heatmap is sorted on odd encounters and shows even ones, so it does not
look diagonal just from regression to the mean.

Example
-------
::

    python analysis/opponent_barcode/opponent_barcode.py
    python analysis/opponent_barcode/opponent_barcode.py --label_mode group
    python analysis/opponent_barcode/opponent_barcode.py --response raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from analysis._paths import results_dir  # noqa: E402

from analysis.event_decoding import plot_decoding_accuracy_violins as violins  # noqa: E402
from analysis.event_decoding.plot_decoding_auroc_histograms import INK, NONSIG_COLOR, SIG_COLOR  # noqa: E402
from analysis.event_decoding.refine_permutation_fdr import decode_params  # noqa: E402

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

THRESHOLDS = np.round(np.arange(0.25, 1.51, 0.125), 3)
HEADLINE = 0.5
OPP_COLORS = ["#d1352b", "#1f4e9c", "#2a9d5c", "#e08a1e", "#7b4fa8", "#555555"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DEFAULT_BASELINE = (-2.0, -1.0)


def load_events_and_responses(animal: str, session: str, mode: str, response: str,
                              out_dir: Path, baseline=DEFAULT_BASELINE) -> Dict:
    """Encounters (all opponents) and the cells x events response matrix."""
    params = decode_params(mode)
    params["max_opponents"] = None
    baseline = [float(b) for b in baseline]
    key = hashlib.sha1(json.dumps({"animal": animal, "session": session,
                                   "response": response, "baseline": baseline,
                                   **params},
                                  sort_keys=True, default=str).encode()).hexdigest()[:12]
    cache = out_dir / f"{animal}_{session}_{mode}_{response}_{key}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    from analysis.event_decoding.decode_opponent_identity import decode_opponent_identity_population
    from ingestion.data_paths import DataStorageManager
    from ingestion.ephys_sync import DataSyncManager
    from ingestion.kilosort_data_import import (DEFAULT_QUALITY_THRESHOLDS,
                                                load_kilosort_data)
    from video.behavioral_events import load_behavioral_events

    config_path = violins.resolve_cohort_config(session, None)
    dm = DataStorageManager(animal, session, config_path=config_path)
    ks = load_kilosort_data(dm.get_kilosort_path())
    beh = load_behavioral_events(dm.get_behavioral_event_files(), session_id=dm.session_id)
    beh.synchronize_with_ephys(DataSyncManager(dm, dio_channel=1), create_new_columns=True)
    # n_shuffles=0: this only reuses the decoder's event/label filtering
    # (ephys-range, min_events_per_class), so the encounters are exactly the
    # ones a max_opponents=None decode would use.
    res = decode_opponent_identity_population(
        ks_data=ks, behavior_data=beh, animal_of_interest=animal,
        behavior_type=params["behavior_type"],
        use_quality_cells=params["use_quality_cells"], alignment=params["alignment"],
        time_window=tuple(params["time_window"]), time_bin_size=params["time_bin_size"],
        cv_folds=params["cv_folds"], min_events_per_class=params["min_events_per_class"],
        max_opponents=None, label_mode=params.get("label_mode", "opponent"),
        n_shuffles=0)
    if res.get("status") == "failed":
        raise RuntimeError(f"{animal}/{session}: {res.get('error')}")
    cids, spikes = ks.get_filtered_cells_spike_times(**DEFAULT_QUALITY_THRESHOLDS)
    t = np.asarray(res["event_times"], float)
    labels = np.asarray(res["labels"]).astype(str)
    first = min(float(np.min(st)) for st in spikes if len(st))
    keep = t + min(baseline[0], 0.0) >= first
    if not keep.all():
        print(f"  dropping {int((~keep).sum())} encounters whose baseline starts "
              f"before the first spike ({first:.1f} s)")
    t, labels = t[keep], labels[keep]

    def rate(st, lo, hi):
        st = np.sort(np.asarray(st, float))
        return (np.searchsorted(st, t + hi) - np.searchsorted(st, t + lo)) / (hi - lo)

    evoked = np.vstack([rate(st, 0.0, 2.0) for st in spikes])
    if response == "evoked":
        evoked = evoked - np.vstack([rate(st, *baseline) for st in spikes])
    out = {"animal_id": animal, "session_id": session, "config_path": config_path,
           "cluster_ids": list(cids), "event_times": t,
           "labels": labels, "R": evoked, "baseline": baseline}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def selectivity(R: np.ndarray, codes: np.ndarray, K: int):
    """d (opponent vs rest, pooled within-opponent SD) and across-opponent z.

    R is cells x events, codes are integer labels 0..K-1. Returns two
    cells x K arrays; cells with zero within-opponent variance are NaN.
    """
    onehot = np.eye(K)[codes]                        # events x K
    n = onehot.sum(0)                                # K
    sums = R @ onehot                                # cells x K
    means = sums / n
    total = R.sum(1, keepdims=True)
    rest = (total - sums) / (codes.size - n)
    resid = R - means[:, codes]
    sd = np.sqrt((resid ** 2).sum(1, keepdims=True) / (codes.size - K))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (means - rest) / sd
        msd = means.std(1, keepdims=True)
        z = (means - means.mean(1, keepdims=True)) / msd
    d[~np.isfinite(d)] = np.nan
    z[~np.isfinite(z)] = np.nan
    return d, z


def permutations(codes: np.ndarray, times: np.ndarray, n: int, kind: str,
                 block_s: float, rng: np.random.Generator) -> List[np.ndarray]:
    if kind == "free":
        return [codes[rng.permutation(codes.size)] for _ in range(n)]
    block = np.floor((times - times.min()) / block_s).astype(int)
    groups = [np.flatnonzero(block == b) for b in np.unique(block)]
    out = []
    for _ in range(n):
        c = codes.copy()
        for g in groups:
            c[g] = codes[g[rng.permutation(g.size)]]
        out.append(c)
    return out


def split_halves(codes: np.ndarray, times: np.ndarray):
    """Odd/even encounters within each opponent, in time order."""
    odd = np.zeros(codes.size, bool)
    for k in np.unique(codes):
        idx = np.flatnonzero(codes == k)
        idx = idx[np.argsort(times[idx])]
        odd[idx[::2]] = True
    return odd, ~odd


def replication(R, codes, times, K, thr):
    """Fraction of barcode cells (d_pref > thr on one half) whose preferred
    opponent is the same on the other half, averaged over both directions."""
    a, b = split_halves(codes, times)
    hits = tot = 0
    for x, y in ((a, b), (b, a)):
        dx, _ = selectivity(R[:, x], codes[x], K)
        dy, _ = selectivity(R[:, y], codes[y], K)
        ok = np.isfinite(dx).all(1) & np.isfinite(dy).all(1)
        pref = np.nanargmax(np.where(ok[:, None], dx, -np.inf), 1)
        sel = ok & (dx[np.arange(len(dx)), pref] > thr)
        hits += int((np.argmax(np.nan_to_num(dy[sel], nan=-np.inf), 1) == pref[sel]).sum())
        tot += int(sel.sum())
    return hits / tot if tot else np.nan, tot


def summarize(d: np.ndarray, valid: np.ndarray, thr: float):
    passed = d[valid] > thr                          # cells x K
    frac = passed.mean(0)
    n_opp = passed.sum(1)
    hist = np.bincount(n_opp, minlength=d.shape[1] + 1)
    K = d.shape[1]
    jac = []
    for i, j in combinations(range(K), 2):
        u = (passed[:, i] | passed[:, j]).sum()
        jac.append((passed[:, i] & passed[:, j]).sum() / u if u else np.nan)
    return frac, hist, np.array(jac)


def analyze(data: Dict, n_perm: int, block_min: float, seed: int) -> Dict:
    R, labels, times = data["R"], data["labels"], data["event_times"]
    names, codes = np.unique(labels, return_inverse=True)
    K = names.size
    d, z = selectivity(R, codes, K)
    valid = np.isfinite(d).all(1)
    rng = np.random.default_rng(seed)
    out = {"names": names, "counts": np.bincount(codes, minlength=K), "K": K,
           "n_cells": int(valid.sum()), "d": d, "z": z, "valid": valid,
           "cluster_ids": np.asarray(data["cluster_ids"]), "obs": {}, "null": {}}
    for metric, arr, thrs in (("d", d, THRESHOLDS), ("z", z, np.array([0.5, 1.0]))):
        out["obs"][metric] = {t: summarize(arr, valid, t) for t in thrs}
    perm_sets = {kind: permutations(codes, times, n_perm, kind, block_min * 60, rng)
                 for kind in ("free", "block")}
    for kind, perms in perm_sets.items():
        res = {"d": {t: [] for t in THRESHOLDS}, "z": {0.5: [], 1.0: []}, "rep": []}
        for c in perms:
            dn, zn = selectivity(R, c, K)
            for t in THRESHOLDS:
                res["d"][t].append(summarize(dn, valid, t))
            for t in (0.5, 1.0):
                res["z"][t].append(summarize(zn, valid, t))
        for c in perms[:200]:
            res["rep"].append(replication(R, c, times, K, HEADLINE)[0])
        out["null"][kind] = res
    out["rep_obs"] = replication(R, codes, times, K, HEADLINE)
    # cross-validated heatmap: sort on odd encounters, show even ones
    a, b = split_halves(codes, times)
    out["d_odd"], _ = selectivity(R[:, a], codes[a], K)
    out["d_even"], _ = selectivity(R[:, b], codes[b], K)
    return out


def _p(null_vals: np.ndarray, obs: float) -> float:
    null_vals = null_vals[np.isfinite(null_vals)]
    return (1 + np.sum(null_vals >= obs - 1e-12)) / (null_vals.size + 1)


def table_rows(rat: str, a: Dict) -> List[Dict]:
    rows = []
    for metric in ("d", "z"):
        for t, (frac, hist, jac) in a["obs"][metric].items():
            for kind in ("free", "block"):
                nf = np.array([s[0] for s in a["null"][kind][metric][t]])
                nh = np.array([s[1] for s in a["null"][kind][metric][t]])
                for k, name in enumerate(a["names"]):
                    rows.append({"animal_id": rat, "metric": metric, "threshold": t,
                                 "null": kind, "item": name,
                                 "n_events": int(a["counts"][k]), "n_cells": a["n_cells"],
                                 "observed": frac[k], "null_mean": nf[:, k].mean(),
                                 "null_lo": np.quantile(nf[:, k], 0.025),
                                 "null_hi": np.quantile(nf[:, k], 0.975),
                                 "p": _p(nf[:, k], frac[k])})
                if a["K"] > 2:
                    rows.append({"animal_id": rat, "metric": metric, "threshold": t,
                                 "null": kind, "item": "single_opponent_cells",
                                 "n_events": int(a["counts"].sum()), "n_cells": a["n_cells"],
                                 "observed": hist[1] / a["n_cells"],
                                 "null_mean": nh[:, 1].mean() / a["n_cells"],
                                 "null_lo": np.quantile(nh[:, 1], 0.025) / a["n_cells"],
                                 "null_hi": np.quantile(nh[:, 1], 0.975) / a["n_cells"],
                                 "p": _p(nh[:, 1].astype(float), hist[1])})
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=3.5, width=1.0, labelsize=9)


def _title(ax, rat, a, extra=""):
    ax.set_title(f"{rat} ({a['cohort'].replace('cohort', 'cohort ')}), "
                 f"n = {a['n_cells']} cells{extra}", fontsize=9.5, loc="left")


def fig_fraction(results: Dict, null: str, what: str):
    ncol = 3
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.0 * nrow), squeeze=False)
    for ax, (rat, a) in zip(axes.flat, results.items()):
        for k, name in enumerate(a["names"]):
            col = OPP_COLORS[k % len(OPP_COLORS)]
            obs = [a["obs"]["d"][t][0][k] for t in THRESHOLDS]
            nf = np.array([[s[0][k] for s in a["null"][null]["d"][t]] for t in THRESHOLDS])
            ax.fill_between(THRESHOLDS, np.quantile(nf, 0.025, 1), np.quantile(nf, 0.975, 1),
                            color=col, alpha=0.15, linewidth=0)
            ax.plot(THRESHOLDS, obs, color=col, linewidth=1.4,
                    label=f"{name} ({a['counts'][k]} enc.)")
        ax.set_yscale("log")
        ax.axvline(HEADLINE, color=INK, linewidth=0.7, linestyle=(0, (2, 2)))
        _style(ax)
        _title(ax, rat, a)
        ax.legend(frameon=False, fontsize=7.5, handlelength=1.2, labelspacing=0.2)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of cells above threshold", fontsize=9.5)
    for ax in axes[-1, :]:
        ax.set_xlabel(f"Selectivity threshold, d ({what} vs rest)", fontsize=9.5)
    fig.suptitle(f"Lines: observed; bands: {null}-shuffle null 95%", fontsize=9, x=0.01,
                 ha="left")
    fig.tight_layout()
    return fig


def fig_specificity(results: Dict, null: str):
    rats = [r for r, a in results.items() if a["K"] > 2]
    if not rats:
        return None
    fig, axes = plt.subplots(1, len(rats), figsize=(2.3 * len(rats), 2.9), squeeze=False)
    for ax, rat in zip(axes.flat, rats):
        a = results[rat]
        _, hist, _ = a["obs"]["d"][HEADLINE]
        nh = np.array([s[1] for s in a["null"][null]["d"][HEADLINE]])
        x = np.arange(1, a["K"] + 1)
        ax.bar(x - 0.18, nh[:, 1:].mean(0), width=0.36, color=NONSIG_COLOR,
               yerr=np.clip([nh[:, 1:].mean(0) - np.quantile(nh[:, 1:], 0.025, 0),
                             np.quantile(nh[:, 1:], 0.975, 0) - nh[:, 1:].mean(0)], 0, None),
               error_kw={"linewidth": 0.8, "capsize": 2}, label=f"{null} null")
        ax.bar(x + 0.18, hist[1:], width=0.36, color=SIG_COLOR, label="Observed")
        ax.set_xticks(x)
        ax.set_xlabel(f"Opponents passed (d > {HEADLINE:g})", fontsize=9)
        _style(ax)
        ax.set_title(f"{rat}\nn = {a['n_cells']} cells", fontsize=9)
    axes[0, 0].set_ylabel("Number of cells", fontsize=9.5)
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_heatmap(results: Dict, what: str):
    ncol = len(results)
    fig, axes = plt.subplots(1, ncol, figsize=(1.55 * ncol + 0.8, 4.6), squeeze=False)
    im = None
    for ax, (rat, a) in zip(axes.flat, results.items()):
        ok = np.isfinite(a["d_odd"]).all(1) & np.isfinite(a["d_even"]).all(1)
        do, de = a["d_odd"][ok], a["d_even"][ok]
        pref = do.argmax(1)
        order = np.lexsort((-do.max(1), pref))
        im = ax.imshow(de[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                       interpolation="nearest")
        ax.set_xticks(range(a["K"]))
        ax.set_xticklabels(a["names"], rotation=60, fontsize=7.5)
        ax.set_yticks([])
        ax.set_title(f"{rat}\nn = {ok.sum()}", fontsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    axes[0, 0].set_ylabel("Cells (sorted by preferred\nopponent on odd encounters)",
                          fontsize=9)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cb.set_label(f"d on even encounters ({what} vs rest)", fontsize=9)
    return fig


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--label_mode", choices=("opponent", "group"), default="opponent")
    p.add_argument("--response", choices=("evoked", "raw"), default="evoked")
    p.add_argument("--baseline", type=float, nargs=2, default=list(DEFAULT_BASELINE),
                   metavar=("LO", "HI"), help="Baseline window (s) for --response evoked.")
    p.add_argument("--target", type=violins._target, action="append", dest="targets",
                   metavar="ANIMAL:SESSION")
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--block_min", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=".gui_cache/opponent_barcode")
    args = p.parse_args(argv)
    targets = args.targets or violins.DEFAULT_TARGETS
    violins.assert_targets_not_held_out(targets)

    what = "opponent" if args.label_mode == "opponent" else "group"
    results, rows = {}, []
    for animal, session in targets:
        data = load_events_and_responses(animal, session, args.label_mode, args.response,
                                         REPO_ROOT / args.out_dir, args.baseline)
        a = analyze(data, args.n_perm, args.block_min, args.seed)
        a["cohort"] = violins.cohort_of_config(data["config_path"])
        results[animal] = a
        rows += table_rows(animal, a)
        frac, hist, jac = a["obs"]["d"][HEADLINE]
        print(f"\n{animal}/{session}: {a['n_cells']} cells, K={a['K']}, "
              f"{data['labels'].size} encounters")
        for kind in ("free", "block"):
            nf = np.array([s[0] for s in a["null"][kind]["d"][HEADLINE]])
            cells = "  ".join(
                f"{n}({c}): {f:.3f} vs {m:.3f} [{lo:.3f},{hi:.3f}] p={_p(nf[:, k], f):.3f}"
                for k, (n, c, f, m, lo, hi) in enumerate(zip(
                    a["names"], a["counts"], frac, nf.mean(0),
                    np.quantile(nf, 0.025, 0), np.quantile(nf, 0.975, 0))))
            rep = np.array(a["null"][kind]["rep"])
            print(f"  [{kind:5}] d>{HEADLINE:g}: {cells}")
            print(f"  [{kind:5}] replication {a['rep_obs'][0]:.3f} (n={a['rep_obs'][1]}) "
                  f"vs null {np.nanmean(rep):.3f} p={_p(rep, a['rep_obs'][0]):.3f}")
            if a["K"] > 2:
                nh = np.array([s[1] for s in a["null"][kind]["d"][HEADLINE]])
                nj = np.array([s[2] for s in a["null"][kind]["d"][HEADLINE]])
                print(f"  [{kind:5}] single-opponent cells {hist[1]} vs null "
                      f"{nh[:, 1].mean():.1f} p={_p(nh[:, 1].astype(float), hist[1]):.3f}; "
                      f"mean Jaccard {np.nanmean(jac):.3f} vs {np.nanmean(nj):.3f}")

    tag = f"barcode_{args.label_mode}" + ("" if args.response == "evoked" else "_raw")
    pd.DataFrame(rows).to_csv(results_dir("opponent_barcode") / f"{tag}.csv", index=False)
    cells = []
    for rat, a in results.items():
        for i, cid in enumerate(a["cluster_ids"]):
            cells.append({"animal_id": rat, "cluster_id": cid,
                          **{f"d_{n}": a["d"][i, k] for k, n in enumerate(a["names"])}})
    pd.DataFrame(cells).to_csv(results_dir("opponent_barcode") / f"{tag}_cells.csv", index=False)
    figs = {f"{tag}_fraction_vs_threshold_by_rat": fig_fraction(results, "free", what),
            f"{tag}_fraction_vs_threshold_block_by_rat": fig_fraction(results, "block", what),
            f"{tag}_specificity": fig_specificity(results, "free"),
            f"{tag}_heatmap_by_rat": fig_heatmap(results, what)}
    for name, fig in figs.items():
        if fig is None:
            continue
        base = results_dir("opponent_barcode") / name
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {base.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
