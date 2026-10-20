#!/usr/bin/env python3
"""
Staged permutation refinement and per-session BH-FDR for per-cell identity
decoding.

The violins cache (``scripts/plot_decoding_accuracy_violins.py``) holds a
per-cell permutation p-value from 100 shuffles, so p >= 1/101. With m
cells in a rat, BH gives q >= m/101: nothing can be significant. A lone
strong cell needs about ``m / alpha`` shuffles
(``fdr_resolution(...)['recommended_n_shuffles']``, called ``N_req``
below). Running that many for every cell is wasteful, so only cells that
could still pass get more shuffles:

1. Stage 0 (cached, nothing rerun): a cell goes on if its cached p is below
   ``--screen_p`` (default 0.02, i.e. at most 1 of 100 shuffles beat it).
2. Each later stage has ``--growth`` times as many shuffles (1,000, then
   10,000, ...), capped at ``N_req``. So every rat's last stage is exactly
   its budget.
3. After a stage that did not reach ``N_req``, a cell goes on only if its
   p-value could still pass BH in the best case. C is the number of cells
   that entered the stage, and the most lenient BH cutoff is C * alpha / m
   (reached only if every candidate is a discovery). A cell goes on if the
   Clopper-Pearson lower 95% bound on its exceedance rate is at or below
   that cutoff.
4. Every stage uses fresh permutations (``default_rng([seed, stage])``), one
   per shuffle index, shared across cells as in
   ``compute_population_significance``. A cell's final p comes only from the
   draws of the last stage it reached. Those draws do not depend on the
   selection, so the p-value stays valid. A dropped cell keeps the p from the
   stage that dropped it.
5. BH (``benjamini_hochberg``) is applied per rat, over **all** m tested
   cells, including those dropped at stage 0. Significant means q < alpha.

The screen is conservative. It can only lose power, never add false
positives. A cell with true p = 0.002 fails stage 0 about 2% of the time.

Scoring is ``_cv_lda_accuracy`` on ``_prepare_lda_features``: the same
model, features and folds as the cached decode. Each candidate's observed
accuracy is checked against the cached value before any shuffle runs.

Example
-------
::

    python scripts/refine_permutation_fdr.py --label_mode opponent
    python scripts/refine_permutation_fdr.py --label_mode group
    # winner vs loser, from the auROC script's outcome cache
    python scripts/refine_permutation_fdr.py --label_mode outcome \
        --target rat631:20251216
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import beta

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ephys._lda_decoding import _cv_lda_accuracy, _prepare_lda_features  # noqa: E402
from ephys._stats_utils import (benjamini_hochberg, empirical_p_value,  # noqa: E402
                                fdr_resolution)
from ingestion.data_paths import DataStorageManager  # noqa: E402
from ingestion.kilosort_data_import import load_kilosort_data  # noqa: E402
import plot_decoding_accuracy_violins as violins  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--label_mode", choices=("opponent", "group", "outcome"),
                   default="opponent",
                   help="'outcome' reads the winner/loser decode that "
                        "plot_decoding_auroc_histograms.py --decoder outcome cached "
                        "(every event with a winner and a loser, aligned to end).")
    p.add_argument("--target", type=violins._target, action="append", dest="targets",
                   metavar="ANIMAL:SESSION",
                   help="Repeatable; defaults to violins.DEFAULT_TARGETS.")
    p.add_argument("--alpha", type=float, default=0.05, help="Final BH level.")
    p.add_argument("--screen_p", type=float, default=0.02,
                   help="Stage-0 cut on the cached 100-shuffle p-value.")
    p.add_argument("--growth", type=int, default=10,
                   help="Shuffle-count multiplier per stage.")
    p.add_argument("--first_stage", type=int, default=1000)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache_dir", default=".gui_cache/accuracy_violins",
                   help="Where the cached 100-shuffle decodes live (read only).")
    p.add_argument("--out_dir", default=".gui_cache/refined_significance")
    p.add_argument("--csv", default=None,
                   help="Default results/refined_fdr_<label_mode>.csv")
    p.add_argument("--verify_only", action="store_true",
                   help="Only check that 100 shuffles at seed 0 reproduce the "
                        "cached p-values for a few cells per rat.")
    return p


def decode_params(label_mode: str) -> Dict:
    """The cached decode's params. Must match violins.main exactly, or the key misses."""
    params = {"behavior_type": "EC", "max_opponents": 2, "use_quality_cells": True,
              "alignment": "start", "time_window": [-1.0, 2.0], "time_bin_size": 0.5,
              "cv_folds": 5, "min_events_per_class": 5, "n_shuffles": 100,
              "null_mode": "per_cell", "alpha": 0.05, "seed": 0}
    if label_mode == "outcome":
        # Same as plot_decoding_auroc_histograms.main with --decoder outcome
        # at its defaults (--outcome_behavior_type any, --outcome_alignment end).
        params.update(decoder="outcome", max_opponents=None, behavior_type=None,
                      alignment="end")
    if label_mode == "group":
        params["label_mode"] = "group"
        params["max_opponents"] = None
    return params


def _null_row(feats: np.ndarray, labels: np.ndarray, counts: np.ndarray,
              cv_folds, perms: np.ndarray) -> np.ndarray:
    return np.array([np.mean(_cv_lda_accuracy(feats, labels[perm], counts, cv_folds))
                     for perm in perms], dtype=np.float64)


def _null_rows(features: Dict, cids: List, labels, counts, cv_folds, perms,
               n_jobs: int) -> Dict:
    # Split each cell's shuffles into chunks so the pool stays busy even
    # when only a handful of cells are left.
    n_chunks = max(1, min(len(perms), math.ceil(4 * (os.cpu_count() or 1) / max(len(cids), 1))))
    chunks = np.array_split(perms, n_chunks)
    jobs = [(cid, i) for cid in cids for i in range(len(chunks))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = Parallel(n_jobs=n_jobs)(
            delayed(_null_row)(features[cid], labels, counts, cv_folds, chunks[i])
            for cid, i in jobs)
    rows: Dict = {cid: [] for cid in cids}
    for (cid, _), part in zip(jobs, out):
        rows[cid].append(part)
    return {cid: np.concatenate(parts) for cid, parts in rows.items()}


def _cp_lower(k: int, n: int, level: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound on an exceedance rate k/n."""
    return 0.0 if k == 0 else float(beta.ppf(1 - level, k, n - k + 1))


def load_cached(animal: str, session: str, params: Dict, cache_dir: Path) -> Dict:
    f = cache_dir / violins._cache_key(animal, session, params)
    if not f.exists():
        raise FileNotFoundError(f"No cached decode for {animal}/{session}: {f}")
    return violins._load_result_pickle(f)


def cell_features(results: Dict, params: Dict, cids: List) -> Dict:
    g = results["_group"]
    dm = DataStorageManager(g["animal_id"], g["session_id"], config_path=g["config_path"])
    ks = load_kilosort_data(dm.get_kilosort_path())
    index = {cid: i for i, cid in enumerate(ks.ks_ids)}
    ev = np.asarray(results["event_times"], dtype=float)
    feats = {}
    for cid in cids:
        f = _prepare_lda_features(np.asarray(ks.spike_times_by_cell[index[cid]]), ev,
                                  tuple(params["time_window"]), params["time_bin_size"])
        if f is None:
            raise RuntimeError(f"cell {cid} has no spikes but was tested in the cache")
        feats[cid] = f
    return feats


def verify(animal: str, results: Dict, params: Dict, n_jobs: int) -> None:
    """100 shuffles at seed 0 drawn as compute_population_significance draws
    them must reproduce the cached p-values exactly."""
    sig = results["significance"]
    labels = np.asarray(results["labels"])
    _, counts = np.unique(labels, return_counts=True)
    ps = {cid: d["p_value"] for cid, d in sig.items()}
    order = sorted(ps, key=ps.get)
    cids = [order[0], order[len(order) // 2], order[-1]]
    feats = cell_features(results, params, cids)
    rng = np.random.default_rng(params["seed"])
    perms = np.array([rng.permutation(len(labels)) for _ in range(params["n_shuffles"])])
    rows = _null_rows(feats, cids, labels, counts, params["cv_folds"], perms, n_jobs)
    for cid in cids:
        obs = float(np.mean(_cv_lda_accuracy(feats[cid], labels, counts, params["cv_folds"])))
        p = empirical_p_value(obs, rows[cid])
        status = "OK" if p == ps[cid] else "MISMATCH"
        print(f"  verify {animal} cell {cid}: cached p={ps[cid]:.4f} rerun p={p:.4f} {status}")
        if status != "OK":
            raise SystemExit(f"Reproduction failed for {animal} cell {cid}")


def refine(animal: str, session: str, results: Dict, params: Dict, args,
           ckpt_dir: Path) -> pd.DataFrame:
    sig = results["significance"]
    labels = np.asarray(results["labels"])
    _, counts = np.unique(labels, return_counts=True)
    cids = list(sig)
    m = len(cids)
    n_req = fdr_resolution(m, 1, args.alpha)["recommended_n_shuffles"]
    cell_results = results["cell_results"]

    state = {cid: {"p_value": sig[cid]["p_value"], "n_shuffles": params["n_shuffles"],
                   "last_stage": 0} for cid in cids}
    alive = [cid for cid in cids if sig[cid]["p_value"] < args.screen_p]
    print(f"\n=== {animal}/{session}: m={m} tested cells, N_req={n_req}, "
          f"stage 0 (cached {params['n_shuffles']}): {len(alive)} with p < {args.screen_p}")

    feats = cell_features(results, params, alive) if alive else {}
    for cid in alive:
        obs = float(np.mean(_cv_lda_accuracy(feats[cid], labels, counts, params["cv_folds"])))
        cached = cell_results[cid]["accuracy"]
        if not np.isclose(obs, cached, rtol=0, atol=1e-12):
            raise SystemExit(f"{animal} cell {cid}: rerun accuracy {obs} != cached {cached}")
        state[cid]["observed"] = obs

    stage, n = 0, params["n_shuffles"]
    while alive and n < n_req:
        stage += 1
        n = min(n_req, args.first_stage if stage == 1 else n * args.growth)
        ckpt = ckpt_dir / f"{animal}_{session}_stage{stage}_n{n}.pkl"
        if ckpt.exists():
            with open(ckpt, "rb") as fh:
                saved = pickle.load(fh)
            assert saved["cids"] == alive, f"checkpoint {ckpt.name} is for other cells"
            rows = saved["rows"]
            print(f"  stage {stage}: n={n} for {len(alive)} cells (checkpoint)")
        else:
            t0 = time.time()
            rng = np.random.default_rng([args.seed, stage])
            perms = np.array([rng.permutation(len(labels)) for _ in range(n)])
            rows = _null_rows(feats, alive, labels, counts, params["cv_folds"], perms,
                              args.n_jobs)
            with open(ckpt, "wb") as fh:
                pickle.dump({"cids": alive, "rows": rows, "n": n}, fh)
            print(f"  stage {stage}: n={n} for {len(alive)} cells "
                  f"({time.time() - t0:.0f} s)")
        for cid in alive:
            row = rows[cid]
            state[cid].update(p_value=empirical_p_value(state[cid]["observed"], row),
                              n_shuffles=int(np.isfinite(row).sum()), last_stage=stage)
        if n >= n_req:
            break
        cutoff = len(alive) * args.alpha / m
        nxt = []
        for cid in alive:
            row = rows[cid][np.isfinite(rows[cid])]
            k = int(np.sum(row >= state[cid]["observed"]))
            if _cp_lower(k, row.size) <= cutoff:
                nxt.append(cid)
        print(f"    best-case BH cutoff {len(alive)}*{args.alpha}/{m} = {cutoff:.2e}: "
              f"{len(nxt)} of {len(alive)} go on")
        alive = nxt

    p = np.array([state[cid]["p_value"] for cid in cids])
    q = benjamini_hochberg(p)
    df = pd.DataFrame({
        "animal_id": animal, "session_id": session, "cluster_id": cids,
        "accuracy": [cell_results[cid]["accuracy"] for cid in cids],
        "p_cached_100": [sig[cid]["p_value"] for cid in cids],
        "p_value": p, "q_value": q,
        "n_shuffles": [state[cid]["n_shuffles"] for cid in cids],
        "last_stage": [state[cid]["last_stage"] for cid in cids],
        "significant": q < args.alpha,
    })
    res = fdr_resolution(m, int(df["n_shuffles"].max()), args.alpha)
    print(f"  final: {int(df['significant'].sum())}/{m} cells at q < {args.alpha} "
          f"(max n_shuffles {df['n_shuffles'].max()}, resolvable={res['resolvable']}, "
          f"best lone q {res['best_achievable_q']:.3f})")
    return df


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    targets = args.targets or list(violins.DEFAULT_TARGETS)
    params = decode_params(args.label_mode)
    cache_dir = REPO_ROOT / args.cache_dir
    out_dir = REPO_ROOT / args.out_dir / args.label_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    violins.assert_targets_not_held_out(targets)

    frames = []
    for animal, session in targets:
        results = load_cached(animal, session, params, cache_dir)
        if args.verify_only:
            verify(animal, results, params, args.n_jobs)
            continue
        frames.append(refine(animal, session, results, params, args, out_dir))
    if args.verify_only:
        return 0

    df = pd.concat(frames, ignore_index=True)
    csv = Path(args.csv or REPO_ROOT / f"results/refined_fdr_{args.label_mode}.csv")
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    with open(out_dir / "refined_fdr.pkl", "wb") as fh:
        pickle.dump({"table": df, "meta": {**vars(args), "decode_params": params}}, fh)

    print(f"\n{args.label_mode}: significant cells at q < {args.alpha} (BH per rat)")
    summary = df.groupby("animal_id", sort=False).agg(
        m=("cluster_id", "size"), sig=("significant", "sum"),
        max_n=("n_shuffles", "max"))
    summary["pct"] = (100 * summary["sig"] / summary["m"]).round(1)
    print(summary.to_string())
    print(f"pooled: {int(df['significant'].sum())}/{len(df)} "
          f"({100 * df['significant'].mean():.1f}%)")
    print(f"-> {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
