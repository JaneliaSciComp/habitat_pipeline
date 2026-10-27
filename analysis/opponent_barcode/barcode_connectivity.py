#!/usr/bin/env python3
"""
Are identity-coding (barcode) cells functionally coupled to each other?

Short-lag spike cross-correlograms (CCGs) between every pair of quality cells,
over the whole recording, tested for putative monosynaptic excitation and
inhibition with the Stark & Abeles (2012) hollowed-Gaussian predictor:

* CCG in 0.5 ms bins over +-100 ms.
* Predictor = CCG convolved with a Gaussian (sigma 10 ms) whose centre is
  60 % hollowed; it follows slow co-modulation (shared event responses,
  drift, theta) but not a few-ms synaptic peak or trough.
* Causal window +1..+4 ms (post fires after pre). Counts summed over the
  window are compared with the summed predictor: Poisson P(>= obs) for
  excitation, P(<= obs) for inhibition (mid-p). The window must also exceed
  (excitation) or fall below (inhibition) the mirrored anti-causal window.
  BH over all ordered pairs, per sign. Excess z = (obs - pred) / sqrt(pred).
* Lags < 1 ms are never used (Kilosort overlap suppression and refractory
  shadow on nearby channels); pairs whose peak channels are < 25 um apart
  are flagged and dropped in the ``_far`` variants.

Barcode cells and their preferred opponent come from
``results/opponent_barcode/barcode_<rat>_cells.csv`` (BH q_exc < alpha, block null, written by
``analysis/opponent_barcode/barcode_characterize.py``). Pair classes (unordered):
co-tuned (same preferred opponent), competing (different), barcode x
non-barcode, non x non.

Nulls
-----
1. co-tuned minus competing: preferred-opponent labels permuted among the
   barcode cells (keeps the cells, their CCGs and the group sizes).
2. barcode x barcode minus non x non: random cell sets of the same size,
   matched in depth tertile and rate tertile.
3. Continuous: Spearman rho between signal correlation (Pearson of the two
   cells' d vectors across opponents) and excess z over all pairs; null from
   time-block permutations of the encounter labels (d recomputed, CCGs fixed).
Each statistic is also computed on excess z residualised on |delta depth|,
|delta x| and log rate product, so co-tuned cells merely sitting closer
together can't produce it.

Example
-------
::

    python analysis/opponent_barcode/barcode_connectivity.py                  # rat631 / 20251216
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, poisson, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from analysis._paths import results_dir  # noqa: E402

from analysis._common._stats_utils import benjamini_hochberg  # noqa: E402
from analysis.opponent_barcode import opponent_barcode as ob  # noqa: E402
from analysis.event_decoding import plot_decoding_accuracy_violins as violins  # noqa: E402
from analysis.event_decoding.plot_decoding_auroc_histograms import INK, NONSIG_COLOR, SIG_COLOR  # noqa: E402
from analysis.opponent_barcode.plot_opponent_barcode_slide import NULL_COLOR, _short, _style  # noqa: E402

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

BIN = 0.0005                 # s
HALF = 0.100                 # s, CCG half-width
SIGMA = 0.010                # s, predictor Gaussian
HOLLOW = 0.6
WIN = (0.001, 0.004)         # s, causal window
MIN_SPIKES = 1000
NEAR_UM = 25.0
SAMPLE_RATE = 30000.0
CLASSES = ("co-tuned", "competing", "barcode x non", "non x non")
CLASS_COLORS = {"co-tuned": "#1a9e77", "competing": "#d95f02",
                "barcode x non": "#7f7f7f", "non x non": "#bdbdbd"}

NB = int(round(2 * HALF / BIN))                 # bins
LAGS = (np.arange(NB) + 0.5) * BIN - HALF       # bin centres, s
CAUSAL = (LAGS > WIN[0]) & (LAGS < WIN[1])
ANTI = (LAGS < -WIN[0]) & (LAGS > -WIN[1])


# ---------------------------------------------------------------------------
# CCG + Stark-Abeles test
# ---------------------------------------------------------------------------

def ccg(ti: np.ndarray, tj: np.ndarray) -> np.ndarray:
    """Counts of tj - ti in NB bins over [-HALF, HALF); both sorted."""
    lo = np.searchsorted(tj, ti - HALF)
    hi = np.searchsorted(tj, ti + HALF)
    n = hi - lo
    keep = n > 0
    lo, n, t0 = lo[keep], n[keep], ti[keep]
    if n.sum() == 0:
        return np.zeros(NB, np.int64)
    start = np.repeat(lo - np.concatenate(([0], np.cumsum(n)[:-1])), n)
    idx = start + np.arange(n.sum())
    dt = tj[idx] - np.repeat(t0, n)
    b = np.floor((dt + HALF) / BIN).astype(np.int64)
    b = b[(b >= 0) & (b < NB)]
    return np.bincount(b, minlength=NB)


def all_ccgs(spikes: list) -> tuple:
    """Unordered pairs i < j; C[p] counts t_j - t_i (j after i = positive lag)."""
    I, J = np.triu_indices(len(spikes), 1)
    C = np.zeros((I.size, NB), np.int32)
    for p, (i, j) in enumerate(zip(I, J)):
        C[p] = ccg(spikes[i], spikes[j])
    return I, J, C


def hollow_kernel() -> np.ndarray:
    x = np.arange(-int(4 * SIGMA / BIN), int(4 * SIGMA / BIN) + 1) * BIN
    k = np.exp(-x ** 2 / (2 * SIGMA ** 2))
    k[x.size // 2] *= 1 - HOLLOW
    return k / k.sum()


def predictor(C: np.ndarray) -> np.ndarray:
    """Hollowed-Gaussian convolution along lags (reflected edges).

    Bins with |lag| < WIN[1] + 0.5 ms are excluded from the convolution input
    (normalised convolution), so a strong peak at +2 ms can't raise the
    predictor in the mirrored -4..-1 ms window and fake an inhibitory trough
    in the reverse direction.
    """
    k = hollow_kernel()
    pad = k.size // 2
    mask = (np.abs(LAGS) >= WIN[1] + 0.0005).astype(float)

    def reflect(A):
        return np.concatenate([A[..., pad:0:-1], A, A[..., -2:-pad - 2:-1]], axis=-1)
    mp = reflect(mask)
    norm = np.convolve(mp, k, mode="valid")
    Cp = reflect(C.astype(float)) * mp
    out = np.empty(C.shape, float)
    for r in range(C.shape[0]):
        out[r] = np.convolve(Cp[r], k, mode="valid") / norm
    return out


def window_test(C: np.ndarray, P: np.ndarray, causal: np.ndarray, anti: np.ndarray):
    """Summed-window mid-p Poisson tests; returns obs, pred, z, p_exc, p_inh."""
    obs = C[:, causal].sum(1).astype(float)
    pred = P[:, causal].sum(1)
    mirror = C[:, anti].sum(1).astype(float)
    p_exc = poisson.sf(obs, pred) + 0.5 * poisson.pmf(obs, pred)
    p_inh = poisson.cdf(obs - 1, pred) + 0.5 * poisson.pmf(obs, pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (obs - pred) / np.sqrt(pred)
    # directional requirement against the mirrored window
    p_exc = np.where(obs > mirror, p_exc, 1.0)
    p_inh = np.where(obs < mirror, p_inh, 1.0)
    return obs, pred, z, np.clip(p_exc, 0, 1), np.clip(p_inh, 0, 1)


def ordered_pairs(I, J, C, P):
    """Both directions as rows: pre -> post with post-after-pre at + lags."""
    rows = []
    for pre, post, flip in ((I, J, False), (J, I, True)):
        Cd = C[:, ::-1] if flip else C
        Pd = P[:, ::-1] if flip else P
        obs, pred, z, pe, pi = window_test(Cd, Pd, CAUSAL, ANTI)
        rows.append(pd.DataFrame({"pre": pre, "post": post, "pair": np.arange(I.size),
                                  "obs": obs, "pred": pred, "z": z,
                                  "p_exc": pe, "p_inh": pi}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Cell properties
# ---------------------------------------------------------------------------

def trough_to_peak(ks_dir: Path, cids) -> np.ndarray:
    """Template trough-to-peak (ms) on the peak channel; NaN if unavailable."""
    try:
        tf = next(ks_dir.rglob("templates.npy"))
        templates = np.load(tf, mmap_mode="r")
        st = np.load(tf.parent / "spike_templates.npy").ravel()
        sc = np.load(tf.parent / "spike_clusters.npy").ravel()
    except (StopIteration, FileNotFoundError):
        return np.full(len(cids), np.nan)
    out = np.full(len(cids), np.nan)
    for n, c in enumerate(cids):
        t = st[sc == c]
        if t.size == 0:
            continue
        w = np.asarray(templates[np.bincount(t).argmax()])
        ch = np.argmax(w.max(0) - w.min(0))
        trough = int(np.argmin(w[:, ch]))
        out[n] = (int(np.argmax(w[trough:, ch])) / SAMPLE_RATE) * 1000
    return out


def load_cells(animal, session, config_path, cids, cache_dir: Path):
    key = hashlib.sha1(f"{animal}{session}{list(cids)}".encode()).hexdigest()[:12]
    cache = cache_dir / f"cells_{animal}_{session}_{key}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    from ingestion.data_paths import DataStorageManager
    from ingestion.kilosort_data_import import load_kilosort_data
    dm = DataStorageManager(animal, session, config_path=config_path)
    ks_dir = Path(dm.get_kilosort_path())
    ks = load_kilosort_data(ks_dir)
    index = {c: i for i, c in enumerate(ks.ks_ids)}
    out = {"spikes": [np.sort(np.asarray(ks.spike_times_by_cell[index[c]], float))
                      for c in cids],
           "dv": np.array([float(ks.DV[index[c]]) for c in cids]),
           "xx": np.array([float(ks.XX[index[c]]) for c in cids]),
           "t2p_ms": trough_to_peak(ks_dir, cids)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


# ---------------------------------------------------------------------------
# Pair statistics
# ---------------------------------------------------------------------------

def pair_class(pref_a, pref_b):
    """pref = -1 for non-barcode cells."""
    a, b = pref_a >= 0, pref_b >= 0
    return np.select([a & b & (pref_a == pref_b), a & b, a | b],
                     ["co-tuned", "competing", "barcode x non"], "non x non")


def residualise(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta, beta


def signal_corr(d: np.ndarray) -> np.ndarray:
    return np.corrcoef(d)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", type=violins._target, default=("rat631", "20251216"))
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--n_perm_labels", type=int, default=2000)
    p.add_argument("--block_min", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=".gui_cache/opponent_barcode")
    p.add_argument("--cache_dir", default=".gui_cache/barcode_connectivity")
    args = p.parse_args(argv)
    animal, session = args.target
    violins.assert_targets_not_held_out([args.target])
    rng = np.random.default_rng(args.seed)
    cache_dir = REPO_ROOT / args.cache_dir

    # --- tuning
    data = ob.load_events_and_responses(animal, session, "opponent", "evoked",
                                        REPO_ROOT / args.out_dir)
    names, codes = np.unique(data["labels"], return_inverse=True)
    K = names.size
    d_all, _ = ob.selectivity(data["R"], codes, K)
    valid = np.isfinite(d_all).all(1)
    cids = np.asarray(data["cluster_ids"])[valid]
    R = data["R"][valid]
    d = d_all[valid]
    cells_csv = pd.read_csv(results_dir("opponent_barcode") / f"barcode_{animal}_cells.csv")
    cells_csv = cells_csv.set_index("cluster_id").loc[cids]
    qcols = [f"q_exc_{n}" for n in names]
    sigm = cells_csv[qcols].to_numpy() < args.alpha
    assert (sigm.sum(1) <= 1).all(), "a barcode cell passes for >1 opponent"
    pref = np.where(sigm.any(1), sigm.argmax(1), -1)
    print(f"{animal}/{session}: {cids.size} cells, {int((pref >= 0).sum())} barcode cells "
          + ", ".join(f"{_short(names[k])}:{int((pref == k).sum())}"
                      for k in range(K) if (pref == k).any()))

    # --- spikes, CCGs
    cell = load_cells(animal, session, data["config_path"], cids, cache_dir)
    nsp = np.array([s.size for s in cell["spikes"]])
    span = max(s[-1] for s in cell["spikes"] if s.size) - min(s[0] for s in cell["spikes"] if s.size)
    rate = nsp / span
    key = hashlib.sha1(f"{animal}{session}{list(cids)}{BIN}{HALF}".encode()).hexdigest()[:12]
    ccg_cache = cache_dir / f"ccg_{animal}_{session}_{key}.npz"
    if ccg_cache.exists():
        z_ = np.load(ccg_cache)
        I, J, C = z_["I"], z_["J"], z_["C"]
    else:
        print(f"  computing {cids.size * (cids.size - 1) // 2} CCGs ...")
        I, J, C = all_ccgs(cell["spikes"])
        np.savez_compressed(ccg_cache, I=I, J=J, C=C)
    P = predictor(C)
    op = ordered_pairs(I, J, C, P)

    # --- pair attributes
    dist = np.hypot(cell["dv"][I] - cell["dv"][J], cell["xx"][I] - cell["xx"][J])
    op["dist_um"] = dist[op.pair]
    op["ddepth_um"] = np.abs(cell["dv"][I] - cell["dv"][J])[op.pair]
    op["dx_um"] = np.abs(cell["xx"][I] - cell["xx"][J])[op.pair]
    op["enough"] = (nsp[op.pre] >= MIN_SPIKES) & (nsp[op.post] >= MIN_SPIKES)
    op["far"] = op.dist_um >= NEAR_UM
    op["log_rate_prod"] = np.log(rate[op.pre] * rate[op.post] + 1e-9)
    SC = signal_corr(d)
    op["signal_corr"] = SC[op.pre, op.post]
    op["class"] = pair_class(pref[op.pre], pref[op.post])
    op["pre_cluster"] = cids[op.pre]
    op["post_cluster"] = cids[op.post]
    op["pre_t2p_ms"] = cell["t2p_ms"][op.pre]
    op = op[op.enough & np.isfinite(op.z)].reset_index(drop=True)
    op["q_exc"] = benjamini_hochberg(op.p_exc.to_numpy())
    op["q_inh"] = benjamini_hochberg(op.p_inh.to_numpy())
    op["conn"] = np.select([op.q_exc < args.alpha, op.q_inh < args.alpha],
                           ["exc", "inh"], "none")
    covars = op[["ddepth_um", "dx_um", "log_rate_prod"]].to_numpy()
    op["z_resid"], beta = residualise(op.z.to_numpy(), covars)
    print(f"  {len(op)} ordered pairs tested ({int((~op.far).sum())} < {NEAR_UM:g} um); "
          f"BH connections: exc {int((op.conn == 'exc').sum())}, "
          f"inh {int((op.conn == 'inh').sum())}")
    print(f"  z ~ ddepth, dx, log rate: beta = {np.round(beta, 4)}")

    rows = []
    for variant, sub in (("all", op), ("far", op[op.far])):
        sub = sub.reset_index(drop=True)
        for col in ("z", "z_resid"):
            y = sub[col].to_numpy()
            pre_, post_ = sub.pre.to_numpy(), sub.post.to_numpy()
            cls = sub["class"].to_numpy()
            for c in CLASSES:
                m = cls == c
                rows.append({"variant": variant, "measure": col, "stat": f"mean_{c}",
                             "observed": y[m].mean() if m.any() else np.nan, "n": int(m.sum()),
                             "frac_exc": (sub.conn[m] == "exc").mean() if m.any() else np.nan,
                             "frac_inh": (sub.conn[m] == "inh").mean() if m.any() else np.nan})

            # null 1: permute preference labels among barcode cells
            bc = np.flatnonzero(pref >= 0)
            both = (pref[pre_] >= 0) & (pref[post_] >= 0)

            def diff(pr):
                same = pr[pre_[both]] == pr[post_[both]]
                return y[both][same].mean() - y[both][~same].mean()
            obs1 = diff(pref)
            null1 = np.empty(args.n_perm)
            for b in range(args.n_perm):
                pr = pref.copy()
                pr[bc] = pref[bc][rng.permutation(bc.size)]
                null1[b] = diff(pr)
            rows.append(_row(variant, col, "co-tuned minus competing (pref-label perm)",
                             obs1, null1, int(both.sum())))

            # null 2: random matched cell sets
            strata = (np.digitize(cell["dv"], np.quantile(cell["dv"], [1 / 3, 2 / 3])) * 3
                      + np.digitize(rate, np.quantile(rate, [1 / 3, 2 / 3])))

            def diff2(isbc):
                a, b_ = isbc[pre_], isbc[post_]
                return y[a & b_].mean() - y[~a & ~b_].mean()
            isbc = pref >= 0
            obs2 = diff2(isbc)
            null2 = np.empty(args.n_perm)
            for b in range(args.n_perm):
                sel = np.zeros_like(isbc)
                for s in np.unique(strata[isbc]):
                    pool = np.flatnonzero(strata == s)
                    sel[rng.choice(pool, int((strata[isbc] == s).sum()), replace=False)] = True
                null2[b] = diff2(sel)
            rows.append(_row(variant, col, "barcode-barcode minus non-non (matched sets)",
                             obs2, null2, int((isbc[pre_] & isbc[post_]).sum())))

            # null 3: signal correlation vs coupling, label permutations
            obs3 = spearmanr(sub.signal_corr, y).statistic
            null3 = np.empty(args.n_perm_labels)
            perms = ob.permutations(codes, data["event_times"], args.n_perm_labels, "block",
                                    args.block_min * 60, np.random.default_rng(args.seed + 7))
            for b, c in enumerate(perms):
                dn, _ = ob.selectivity(R, c, K)
                null3[b] = spearmanr(np.corrcoef(dn)[pre_, post_], y).statistic
            rows.append(_row(variant, col, "spearman(signal corr, coupling) (block label perm)",
                             obs3, null3, len(sub)))

        # connection counts, co-tuned vs competing
        ct = sub[sub["class"] == "co-tuned"]
        cp = sub[sub["class"] == "competing"]
        for kind in ("exc", "inh"):
            tab = [[(ct.conn == kind).sum(), (ct.conn != kind).sum()],
                   [(cp.conn == kind).sum(), (cp.conn != kind).sum()]]
            rows.append({"variant": variant, "measure": "conn", "stat": f"fisher_{kind}",
                         "observed": fisher_exact(tab)[0], "p": fisher_exact(tab)[1],
                         "n": f"{tab}"})

    summ = pd.DataFrame(rows)
    stem = f"barcode_connectivity_{animal}"
    op.to_csv(results_dir("opponent_barcode") / f"{stem}_pairs.csv", index=False)
    summ.to_csv(results_dir("opponent_barcode") / f"{stem}_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    conn = op[op.conn != "none"]
    if len(conn):
        print("  connections:")
        print(conn[["pre_cluster", "post_cluster", "class", "conn", "z", "q_exc", "q_inh",
                    "dist_um", "pre_t2p_ms"]].sort_values("z").to_string(index=False))

    figs = {f"{stem}_examples": fig_examples(op, C, P, cids),
            f"{stem}_by_class": fig_by_class(op, summ),
            f"{stem}_signal_corr": fig_signal_corr(op, summ)}
    for name, fig in figs.items():
        base = results_dir("opponent_barcode") / name
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {base.with_suffix('.png')}")
    return 0


def _row(variant, measure, stat, obs, null, n):
    return {"variant": variant, "measure": measure, "stat": stat, "observed": obs, "n": n,
            "null_mean": null.mean(), "null_lo": np.quantile(null, 0.025),
            "null_hi": np.quantile(null, 0.975),
            "p": (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())))
            / (null.size + 1)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_examples(op, C, P, cids):
    """Strongest co-tuned and competing pairs, each sign."""
    picks = []
    for cls in ("co-tuned", "competing"):
        sub = op[op["class"] == cls]
        if len(sub):
            picks.append((cls, sub.loc[sub.z.idxmax()]))
            picks.append((cls, sub.loc[sub.z.idxmin()]))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.0 * len(picks), 2.8), squeeze=False)
    sel = np.abs(LAGS) <= 0.025
    for ax, (cls, r) in zip(axes.flat, picks):
        # C rows hold t_J - t_I with I < J; pre > post means the reversed direction
        c =C[int(r.pair)][::-1] if r.pre > r.post else C[int(r.pair)]
        pr = P[int(r.pair)][::-1] if r.pre > r.post else P[int(r.pair)]
        x = LAGS[sel] * 1000
        ax.bar(x, c[sel], width=BIN * 1000, color="#9e9e9e", edgecolor="none")
        ax.bar(x[CAUSAL[sel]], c[sel][CAUSAL[sel]], width=BIN * 1000,
               color=CLASS_COLORS[cls], edgecolor="none")
        ax.plot(x, pr[sel], color=INK, lw=1.0)
        ax.fill_between(x, poisson.ppf(0.025, pr[sel]), poisson.ppf(0.975, pr[sel]),
                        color="#4c78a8", alpha=0.2, lw=0)
        ax.axvline(0, color=INK, lw=0.6, ls="--")
        lo = min(c[sel].min(), poisson.ppf(0.025, pr[sel]).min())
        ax.set_ylim(max(0, lo * 0.85), None)
        _style(ax)
        ax.set_xlabel("Lag, post - pre (ms)", fontsize=9)
        ax.set_title(f"{cls}: #{int(r.pre_cluster)} → #{int(r.post_cluster)}\n"
                     f"z = {r.z:+.1f}, q = {min(r.q_exc, r.q_inh):.2g}", fontsize=8.5)
    axes[0, 0].set_ylabel("Spike count (0.5 ms bins)", fontsize=9)
    fig.tight_layout()
    return fig


def fig_by_class(op, summ):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    rng = np.random.default_rng(0)
    ax = axes[0]
    for k, c in enumerate(CLASSES):
        y = op.loc[op["class"] == c, "z"].to_numpy()
        if c == "non x non":
            y = rng.choice(y, min(y.size, 2000), replace=False)
        ax.scatter(k + rng.uniform(-0.25, 0.25, y.size), y, s=3, alpha=0.4,
                   color=CLASS_COLORS[c], edgecolors="none")
        full = op.loc[op["class"] == c, "z"]
        ax.hlines(full.mean(), k - 0.32, k + 0.32, color=INK, lw=1.6)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([c.replace(" x ", "\n× ") for c in CLASSES], fontsize=8)
    ax.set_ylabel("Excess z, +1 to +4 ms", fontsize=9)
    ax.set_ylim(np.quantile(op.z, 0.002), np.quantile(op.z, 0.998))
    _style(ax)
    ax = axes[1]
    w = 0.38
    for k, c in enumerate(CLASSES):
        m = op["class"] == c
        ax.bar(k - w / 2, 100 * (op.conn[m] == "exc").mean(), w, color=SIG_COLOR)
        ax.bar(k + w / 2, 100 * (op.conn[m] == "inh").mean(), w, color=NONSIG_COLOR)
        ax.text(k, 1.0, f"n = {int(m.sum())}", ha="center", va="bottom", fontsize=7,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([c.replace(" x ", "\n× ") for c in CLASSES], fontsize=8)
    ax.set_ylabel("Ordered pairs with BH connection (%)", fontsize=9)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SIG_COLOR),
                       plt.Rectangle((0, 0), 1, 1, color=NONSIG_COLOR)],
              labels=["Excitatory", "Inhibitory"], frameon=False, fontsize=8)
    _style(ax)
    s = summ[(summ.variant == "all") & (summ.measure == "z")].set_index("stat")
    r = s.loc["co-tuned minus competing (pref-label perm)"]
    fig.suptitle(f"Co-tuned − competing excess z = {r.observed:+.2f} "
                 f"(null 95% [{r.null_lo:+.2f}, {r.null_hi:+.2f}], p = {r.p:.2g})",
                 fontsize=9.5)
    fig.tight_layout()
    return fig


def fig_signal_corr(op, summ):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    edges = np.linspace(-1, 1, 11)
    b = np.clip(np.digitize(op.signal_corr, edges) - 1, 0, edges.size - 2)
    mids = (edges[:-1] + edges[1:]) / 2
    m = np.array([op.z[b == k].mean() if (b == k).sum() > 20 else np.nan
                  for k in range(mids.size)])
    se = np.array([op.z[b == k].std() / np.sqrt((b == k).sum()) if (b == k).sum() > 20
                   else np.nan for k in range(mids.size)])
    ax.errorbar(mids, m, se, color=INK, marker="o", ms=4, lw=1, capsize=0)
    ax.axhline(0, color=INK, lw=0.5)
    s = summ[(summ.variant == "all") & (summ.measure == "z")].set_index("stat")
    r = s.loc["spearman(signal corr, coupling) (block label perm)"]
    ax.set_title(f"ρ = {r.observed:+.3f} (null [{r.null_lo:+.3f}, {r.null_hi:+.3f}], "
                 f"p = {r.p:.2g})", fontsize=8.5)
    ax.set_xlabel("Signal correlation (d across opponents)", fontsize=9)
    ax.set_ylabel("Excess z, +1 to +4 ms (bin mean ± SE)", fontsize=9)
    ax.set_ylim(np.nanmin(m - se) - 1, np.nanmax(m + se) + 1)
    _style(ax)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    sys.exit(main())
