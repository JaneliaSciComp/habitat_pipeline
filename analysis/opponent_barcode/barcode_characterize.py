#!/usr/bin/env python3
"""
Characterise one rat's opponent barcode: which cells, how many, how sparse.

Builds on ``scripts/opponent_barcode.py`` (same encounters, same evoked
response: 0-2 s minus the [-2, -1] s baseline, same opponent-vs-rest d):

1. Per-cell test. For every cell x opponent, d is compared with a
   label-permutation null (``--n_perm``, free and 15-min time-block). The
   p-values are one-sided: excitation (d >= obs) and, as a separate family,
   suppression (d <= obs). BH runs over all cells x opponents of the family.
   The block null is the primary one.
2. How many cells code each opponent, without a threshold: Storey's
   pi1 = 1 - pi0 (lambda = 0.5) from that opponent's per-cell **two-sided**
   p-values, times the number of cells. (A one-sided p is ~1 for every cell
   that prefers a different opponent, which inflates pi0 past 1.) So pi1
   counts cells whose response differs for that opponent, either way. It has a bootstrap CI over cells and a null (the
   same estimator on permuted label sets).
3. Sparseness. Lifetime sparseness (Vinje-Gallant) per cell, over the
   per-opponent mean raw 0-2 s rates. Population sparseness (Treves-Rolls)
   per opponent, over cells' mean rate for that opponent / overall mean
   rate (lower = sparser). Kurtosis of each opponent's d across cells.
   Compared with the free-shuffle null; lifetime sparseness also with the
   block null, since slow drift alone raises S.
4. Overlap. Signal correlation (across cells) between opponents' d
   vectors, as observed minus null mean (d's share the "rest" group, so
   they are anti-correlated even under the null).

Example
-------
::

    python scripts/barcode_characterize.py                 # rat631 / 20251216
    python scripts/barcode_characterize.py --target rat4679:20260929
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
from scipy.stats import kurtosis

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ephys._stats_utils import benjamini_hochberg, fdr_resolution  # noqa: E402
import opponent_barcode as ob  # noqa: E402
import plot_decoding_accuracy_violins as violins  # noqa: E402
from plot_decoding_auroc_histograms import INK, NONSIG_COLOR, SIG_COLOR  # noqa: E402
from plot_opponent_barcode_slide import NULL_COLOR, _order, _short, _style  # noqa: E402

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

N_STORE = 1000   # permutations kept in memory for the sparseness / pi1 / corr nulls


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def lifetime_sparseness(M: np.ndarray) -> np.ndarray:
    """Vinje-Gallant S per row of a non-negative cells x K matrix (1 = one opponent)."""
    K = M.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        a = M.mean(1) ** 2 / (M ** 2).mean(1)
    return (1 - a) / (1 - 1 / K)


def treves_rolls(X: np.ndarray) -> np.ndarray:
    """Population sparseness a per column of a non-negative cells x K matrix."""
    return X.mean(0) ** 2 / (X ** 2).mean(0)


def class_means(R: np.ndarray, codes: np.ndarray, K: int) -> np.ndarray:
    onehot = np.eye(K)[codes]
    return (R @ onehot) / onehot.sum(0)


def storey_pi1(p: np.ndarray, lam: float = 0.5) -> float:
    return 1.0 - min(1.0, np.mean(p > lam) / (1 - lam))


def per_cell_nulls(R, Rraw, codes, times, K, n_perm, block_s, seed):
    """Exceedance counts for every cell x opponent, plus a stored subset of nulls."""
    rng = np.random.default_rng(seed)
    d_obs, _ = ob.selectivity(R, codes, K)
    out = {"d_obs": d_obs}
    for kind in ("free", "block"):
        hi = np.zeros_like(d_obs)
        lo = np.zeros_like(d_obs)
        stored, stored_means = [], []
        # generate in chunks so memory stays flat
        done = 0
        while done < n_perm:
            n = min(2000, n_perm - done)
            for c in ob.permutations(codes, times, n, kind, block_s, rng):
                dn, _ = ob.selectivity(R, c, K)
                hi += dn >= d_obs - 1e-12
                lo += dn <= d_obs + 1e-12
                if len(stored) < N_STORE:
                    stored.append(dn.astype(np.float32))
                    stored_means.append(class_means(Rraw, c, K).astype(np.float32))
            done += n
        out[kind] = {"p_hi": (1 + hi) / (n_perm + 1), "p_lo": (1 + lo) / (n_perm + 1)}
        if kind == "free":
            out["d_null"] = np.stack(stored)            # n x cells x K
            out["means_null"] = np.stack(stored_means)
        else:
            out["means_null_block"] = np.stack(stored_means)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_bh_heatmap(rat, names, col, d, sig, cids):
    rows = np.flatnonzero(sig.any(1))
    if rows.size == 0:
        return None
    # group by the significant opponent with the largest d
    best = np.array([np.argmax(np.where(sig[r], d[r], -np.inf)) for r in rows])
    rank = np.argsort(col)                     # opponent index -> column position
    order = np.lexsort((-d[rows, best], rank[best]))
    rows, best = rows[order], best[order]
    fig, ax = plt.subplots(figsize=(2.9, 0.9 + 0.16 * rows.size))
    im = ax.imshow(d[rows][:, col], aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2,
                   interpolation="nearest")
    for r, i in enumerate(rows):
        for j, k in enumerate(col):
            if sig[i, k]:
                ax.text(j, r, "*", ha="center", va="center", fontsize=7, color=INK)
    for r in np.flatnonzero(np.diff(rank[best])) + 0.5:
        ax.axhline(r, color=INK, linewidth=0.6)
    ax.set_xticks(range(len(col)))
    ax.set_xticklabels([_short(n) for n in names[col]], rotation=90, fontsize=7.5)
    ax.set_yticks(range(rows.size))
    ax.set_yticklabels([str(cids[i]) for i in rows], fontsize=6.5)
    ax.set_xlabel("Opponent", fontsize=9)
    ax.set_ylabel("Cluster ID", fontsize=9)
    ax.set_title(f"{rat}: {rows.size} barcode cells\n(* q < 0.05, BH, block shuffle)",
                 fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cb.set_label("d (all encounters)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    return fig


def fig_pi1(rat, names, col, counts, est, ci, null_hi, null_mean):
    x = np.arange(len(col))
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.bar(x, null_hi[col], color=NULL_COLOR, width=0.6, label="Null 95%", zorder=1)
    ax.hlines(null_mean[col], x - 0.3, x + 0.3, color="#8a8a8a", linewidth=1.0, zorder=2)
    ax.errorbar(x, est[col], yerr=np.clip([est[col] - ci[col, 0], ci[col, 1] - est[col]],
                                          0, None),
                fmt="o", color=SIG_COLOR, markersize=5, capsize=3, linewidth=1.0,
                zorder=3, label="Estimate, 95% CI")
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_short(n)}\n{c}" for n, c in zip(names[col], counts[col])],
                       fontsize=7.5)
    ax.set_xlabel("Opponent (rat ID / encounters)", fontsize=9)
    ax.set_ylabel("Estimated modulated cells\n(Storey $\\pi_1$ × n, two-sided)",
                  fontsize=9)
    _style(ax)
    ax.set_title(rat, fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_sparseness(rat, names, col, counts, S, S_null, tr, tr_null):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(S_null.ravel(), bins=bins, density=True, color=NULL_COLOR, label="Shuffle")
    ax.hist(S, bins=bins, density=True, histtype="step", color=SIG_COLOR, linewidth=1.5,
            label="Observed")
    ax.set_xlabel("Lifetime sparseness S", fontsize=9)
    ax.set_ylabel("Density (cells)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    _style(ax)
    ax = axes[1]
    x = np.arange(len(col))
    lo, hi = np.quantile(tr_null, [0.025, 0.975], axis=0)
    ax.bar(x, hi[col] - lo[col], bottom=lo[col], width=0.6, color=NULL_COLOR, zorder=1)
    below = tr < lo
    ax.scatter(x, tr[col], s=28, zorder=3, color=np.where(below[col], SIG_COLOR, NONSIG_COLOR),
               edgecolors="white", linewidths=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_short(n)}\n{c}" for n, c in zip(names[col], counts[col])],
                       fontsize=7.5)
    ax.set_xlabel("Opponent (rat ID / encounters)", fontsize=9)
    ax.set_ylabel("Population sparseness a\n(Treves–Rolls; lower = sparser)", fontsize=9)
    _style(ax)
    fig.suptitle(rat, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_signal_corr(rat, names, col, C, C_null):
    D = (C - C_null.mean(0))[np.ix_(col, col)]
    lo, hi = np.quantile(C_null, [0.025, 0.975], axis=0)
    out = ((C < lo) | (C > hi))[np.ix_(col, col)]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    np.fill_diagonal(D, np.nan)
    m = np.nanmax(np.abs(D))
    im = ax.imshow(D, cmap="RdBu_r", vmin=-m, vmax=m)
    for i in range(len(col)):
        for j in range(len(col)):
            if i != j and out[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8)
    lab = [_short(n) for n in names[col]]
    ax.set_xticks(range(len(col)))
    ax.set_xticklabels(lab, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(col)))
    ax.set_yticklabels(lab, fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Signal correlation of d\n(observed − null mean)", fontsize=8.5)
    ax.set_title(f"{rat} (* outside null 95%)", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", type=violins._target, default=("rat631", "20251216"))
    p.add_argument("--n_perm", type=int, default=30000)
    p.add_argument("--block_min", type=float, default=15.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=".gui_cache/opponent_barcode")
    args = p.parse_args(argv)
    animal, session = args.target
    violins.assert_targets_not_held_out([args.target])

    out_dir = REPO_ROOT / args.out_dir
    data = ob.load_events_and_responses(animal, session, "opponent", "evoked", out_dir)
    raw = ob.load_events_and_responses(animal, session, "opponent", "raw", out_dir)
    assert np.array_equal(data["event_times"], raw["event_times"])
    names, codes = np.unique(data["labels"], return_inverse=True)
    K = names.size
    counts = np.bincount(codes, minlength=K)
    col = _order(list(names), counts)
    d_all, _ = ob.selectivity(data["R"], codes, K)
    valid = np.isfinite(d_all).all(1)
    R, Rraw = data["R"][valid], raw["R"][valid]
    cids = np.asarray(data["cluster_ids"])[valid]
    N = int(valid.sum())
    m = N * K
    res_check = fdr_resolution(m, args.n_perm, args.alpha)
    print(f"{animal}/{session}: {N} cells x {K} opponents = {m} tests, "
          f"{data['labels'].size} encounters, baseline {data.get('baseline')}")
    print(f"  fdr_resolution: {res_check}")

    nulls = per_cell_nulls(R, Rraw, codes, data["event_times"], K, args.n_perm,
                           args.block_min * 60, args.seed)
    d = nulls["d_obs"]
    q = {kind: {side: benjamini_hochberg(nulls[kind][f"p_{side}"].ravel()).reshape(N, K)
                for side in ("hi", "lo")} for kind in ("free", "block")}
    sig = q["block"]["hi"] < args.alpha
    sup = q["block"]["lo"] < args.alpha

    # Storey pi1: observed, bootstrap over cells, null from stored permutations
    rng = np.random.default_rng(args.seed + 1)
    p_obs = np.minimum(1.0, 2 * np.minimum(nulls["free"]["p_hi"], nulls["free"]["p_lo"]))
    est = np.array([storey_pi1(p_obs[:, k]) for k in range(K)]) * N
    boot = np.array([[storey_pi1(p_obs[rng.integers(0, N, N), k]) for k in range(K)]
                     for _ in range(args.n_boot)]) * N
    ci = np.quantile(boot, [0.025, 0.975], axis=0).T
    dn = nulls["d_null"]                              # n x cells x K
    n_store = dn.shape[0]
    pi1_null = []
    for j in range(min(200, n_store)):
        # two-sided rank of perm j among the stored nulls
        pj = np.minimum(1.0, 2 * np.minimum((dn >= dn[j] - 1e-6).sum(0),
                                            (dn <= dn[j] + 1e-6).sum(0)) / n_store)
        pi1_null.append([storey_pi1(pj[:, k]) for k in range(K)])
    pi1_null = np.array(pi1_null) * N

    # sparseness on raw per-opponent means
    M = class_means(Rraw, codes, K)
    ok_rate = (M > 0).any(1)
    S = lifetime_sparseness(M[ok_rate])
    mn = nulls["means_null"][:, ok_rate]
    S_null = np.stack([lifetime_sparseness(x) for x in mn])
    S_null_block = np.stack([lifetime_sparseness(x)
                             for x in nulls["means_null_block"][:, ok_rate]])
    overall = Rraw[ok_rate].mean(1, keepdims=True)
    tr = treves_rolls(M[ok_rate] / overall)
    tr_null = np.stack([treves_rolls(x / overall) for x in mn])
    kurt = kurtosis(d, axis=0)
    kurt_null = kurtosis(dn, axis=1)                  # n x K
    C = np.corrcoef(d.T)
    C_null = np.stack([np.corrcoef(x.T) for x in dn])

    # --- report
    print(f"\n  BH (block null, q < {args.alpha}): {sig.any(1).sum()} excitatory barcode "
          f"cells, {sup.any(1).sum()} suppressed; free null: "
          f"{(q['free']['hi'] < args.alpha).any(1).sum()} / "
          f"{(q['free']['lo'] < args.alpha).any(1).sum()}")
    S_cell_hi = np.quantile(S_null, 0.95, axis=0)
    S_cell_hi_b = np.quantile(S_null_block, 0.95, axis=0)
    print(f"  lifetime S: median {np.median(S):.3f} vs free null {np.median(S_null):.3f}, "
          f"block null {np.median(S_null_block):.3f}; cells above their own null 95% "
          f"(5% expected): free {np.mean(S > S_cell_hi):.1%}, block {np.mean(S > S_cell_hi_b):.1%}")
    rows = []
    for k in col:
        nm = names[k]
        rows.append({
            "opponent": nm, "n_encounters": int(counts[k]),
            "bh_excited_block": int(sig[:, k].sum()), "bh_suppressed_block": int(sup[:, k].sum()),
            "bh_excited_free": int((q["free"]["hi"][:, k] < args.alpha).sum()),
            "frac_d_gt_0.5": float(np.mean(d[:, k] > 0.5)),
            "frac_d_lt_-0.5": float(np.mean(d[:, k] < -0.5)),
            "frac_d_lt_-0.5_null95": float(np.quantile((dn[:, :, k] < -0.5).mean(1), 0.975)),
            "pi1_cells": est[k], "pi1_lo": ci[k, 0], "pi1_hi": ci[k, 1],
            "pi1_null_mean": pi1_null[:, k].mean(),
            "pi1_null_hi": np.quantile(pi1_null[:, k], 0.975),
            "treves_rolls": tr[k], "tr_null_lo": np.quantile(tr_null[:, k], 0.025),
            "tr_null_mean": tr_null[:, k].mean(),
            "kurtosis_d": kurt[k], "kurtosis_null_hi": np.quantile(kurt_null[:, k], 0.975)})
        r = rows[-1]
        print(f"  {nm:7} n={counts[k]:3}  BH+ {r['bh_excited_block']:2} (free {r['bh_excited_free']:2})"
              f"  BH- {r['bh_suppressed_block']:2}  pi1*N {est[k]:5.1f} [{ci[k, 0]:.1f},{ci[k, 1]:.1f}]"
              f" null {r['pi1_null_mean']:.1f}/{r['pi1_null_hi']:.1f}  a {tr[k]:.3f}"
              f" (null {r['tr_null_mean']:.3f}, lo {r['tr_null_lo']:.3f})"
              f"  kurt {kurt[k]:.2f} (null hi {r['kurtosis_null_hi']:.2f})"
              f"  d<-0.5 {r['frac_d_lt_-0.5']:.3f} (null hi {r['frac_d_lt_-0.5_null95']:.3f})")
    n_multi = (sig.sum(1) > 1).sum()
    print(f"  barcode cells significant for >1 opponent: {n_multi}")
    lo, hi = np.quantile(C_null, [0.025, 0.975], axis=0)
    for i, j in zip(*np.triu_indices(K, 1)):
        if C[i, j] < lo[i, j] or C[i, j] > hi[i, j]:
            print(f"  signal corr {names[i]}-{names[j]}: {C[i, j]:+.3f} "
                  f"(null {C_null[:, i, j].mean():+.3f} [{lo[i, j]:+.3f},{hi[i, j]:+.3f}])")

    # --- link to decoding + depth
    fdr = pd.read_csv(REPO_ROOT / "results/refined_fdr_opponent.csv")
    dec = set(fdr[(fdr.animal_id == animal) & (fdr.q_value < args.alpha)].cluster_id)
    from ingestion.data_paths import DataStorageManager
    from ingestion.kilosort_data_import import load_kilosort_data
    dm = DataStorageManager(animal, session, config_path=data["config_path"])
    ks = load_kilosort_data(dm.get_kilosort_path())
    top = float(np.load(next(Path(dm.get_kilosort_path()).rglob("channel_positions.npy")))
                [:, 1].max())
    index = {c: i for i, c in enumerate(ks.ks_ids)}
    depth = np.array([(top - float(ks.DV[index[c]])) / 1000 for c in cids])
    rate = Rraw.mean(1)
    S_full = np.full(N, np.nan)
    S_full[ok_rate] = S
    bc = sig.any(1)
    print(f"  barcode cells also BH-significant in top-2 decoding: "
          f"{len(dec & set(cids[bc]))}/{bc.sum()} (decoding BH cells: {len(dec)})")
    print(f"  barcode-cell depth {np.median(depth[bc]) if bc.any() else np.nan:.2f} mm "
          f"(all {np.median(depth):.2f}); 0-2 s rate {np.median(rate[bc]) if bc.any() else np.nan:.2f}"
          f" Hz (all {np.median(rate):.2f})")

    cell_rows = []
    for i, c in enumerate(cids):
        row = {"cluster_id": c, "depth_mm": depth[i], "rate_0_2s_hz": rate[i],
               "lifetime_S": S_full[i], "decoding_bh": c in dec,
               "barcode": bool(bc[i]), "suppressed": bool(sup[i].any())}
        for k in col:
            row[f"d_{names[k]}"] = d[i, k]
            row[f"q_exc_{names[k]}"] = q["block"]["hi"][i, k]
            row[f"q_sup_{names[k]}"] = q["block"]["lo"][i, k]
        cell_rows.append(row)
    stem = f"barcode_{animal}"
    pd.DataFrame(cell_rows).to_csv(REPO_ROOT / "results" / f"{stem}_cells.csv", index=False)
    pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / f"{stem}_summary.csv", index=False)

    figs = {f"{stem}_bh_heatmap": fig_bh_heatmap(animal, names, col, d, sig, cids),
            f"{stem}_pi1": fig_pi1(animal, names, col, counts, est, ci,
                                   np.quantile(pi1_null, 0.975, axis=0), pi1_null.mean(0)),
            f"{stem}_sparseness": fig_sparseness(animal, names, col, counts, S, S_null,
                                                 tr, tr_null),
            f"{stem}_signal_corr": fig_signal_corr(animal, names, col, C, C_null)}
    for name, fig in figs.items():
        if fig is None:
            continue
        base = REPO_ROOT / "results" / name
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {base.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
