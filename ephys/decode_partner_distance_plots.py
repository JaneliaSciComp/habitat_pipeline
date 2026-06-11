"""
Plots for partner-distance decoding (``ephys.decode_partner_distance``).

Every function takes a result dict from ``decode_partner_distance`` /
``_analyze``, guards on ``status``, and returns a Matplotlib ``Figure`` (or
``None`` if the result is not a successful decode). Mirrors the plotting
conventions of ``ephys.decode_location``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _ok(result: Dict) -> bool:
    if result is None or result.get("status") != "success":
        print(f"Cannot plot: {None if result is None else result.get('status')}")
        return False
    return True


def _title(result: Dict) -> str:
    return f"{result.get('focal', '?')} → {result.get('partner', '?')}"


def plot_distance_tuning_curves(result: Dict, n_top: int = 12,
                                figsize: Tuple[float, float] = (12, 8)):
    """Distance tuning curves for the top cells (by ``cell_ranking``)."""
    import matplotlib.pyplot as plt

    if not _ok(result):
        return None
    tuning = result["tuning"]
    centers = result["dist_centers"]
    ranking = result["cell_ranking"]
    units = result["units"]
    r2 = result["r2_per_cell"]
    partial = result.get("r2_partial_per_cell")

    n_top = int(min(n_top, tuning.shape[1]))
    n_cols = 4
    n_rows = int(np.ceil(n_top / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols][i % n_cols]
        if i >= n_top:
            ax.axis("off")
            continue
        cell = int(ranking[i])
        ax.plot(centers, tuning[:, cell], color="tab:blue", lw=1.5)
        ax.fill_between(centers, tuning[:, cell], alpha=0.2, color="tab:blue")
        sub = f"R²={r2[cell]:.2f}"
        if partial is not None:
            sub += f"  (partial {partial[cell]:.2f})"
        ax.set_title(f"cell {cell} · {sub}", fontsize=9)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel(f"distance ({units})")
        if i % n_cols == 0:
            ax.set_ylabel("rate (Hz)")

    fig.suptitle(f"Distance tuning — top {n_top} cells · {_title(result)}",
                 fontsize=12)
    fig.tight_layout()
    return fig


def plot_per_cell_r2_distribution(result: Dict,
                                  figsize: Tuple[float, float] = (8, 5)):
    """Histogram of per-cell raw vs partial R², with the null reference line."""
    import matplotlib.pyplot as plt

    if not _ok(result):
        return None
    r2 = np.asarray(result["r2_per_cell"], dtype=np.float64)
    r2 = r2[np.isfinite(r2)]
    partial = result.get("r2_partial_per_cell")

    fig, ax = plt.subplots(figsize=figsize)
    bins = np.linspace(min(-0.2, r2.min() if r2.size else -0.2),
                       max(0.6, r2.max() if r2.size else 0.6), 30)
    ax.hist(r2, bins=bins, alpha=0.6, color="tab:blue", label="raw R²")
    if partial is not None:
        p = np.asarray(partial, dtype=np.float64)
        p = p[np.isfinite(p)]
        ax.hist(p, bins=bins, alpha=0.6, color="tab:orange",
                label="partial R² (beyond self-motion)")

    null_r2 = result.get("null_r2")
    if null_r2 is not None and np.isfinite(null_r2):
        ax.axvline(null_r2, color="tab:gray", ls="--",
                   label=f"null R² = {null_r2:.3f}")
    ax.axvline(0.0, color="black", lw=0.7)
    ax.set_xlabel("cross-validated R²")
    ax.set_ylabel("# cells")
    ax.set_title(f"Per-cell distance decoding · {_title(result)}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual_scatter(result: Dict,
                                     figsize: Tuple[float, float] = (6, 6)):
    """Population predicted-vs-actual distance scatter with identity line."""
    import matplotlib.pyplot as plt

    if not _ok(result):
        return None
    y_true = np.asarray(result["y_true"], dtype=np.float64)
    y_pred = np.asarray(result["y_pred"], dtype=np.float64)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    units = result["units"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, s=4, alpha=0.3, color="tab:red")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="identity")
    ax.set_xlabel(f"actual distance ({units})")
    ax.set_ylabel(f"decoded distance ({units})")
    ax.set_aspect("equal")
    txt = f"CV R² = {result['cv_r2']:.3f}\nRMSE = {result['rmse']:.2f} {units}"
    if result.get("cv_r2_partial") is not None:
        txt += f"\npartial R² = {result['cv_r2_partial']:.3f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_title(f"Population distance decoding · {_title(result)}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual_timeseries(result: Dict,
                                        figsize: Tuple[float, float] = (12, 4)):
    """Actual vs decoded distance over time."""
    import matplotlib.pyplot as plt

    if not _ok(result):
        return None
    t = np.asarray(result["bin_centers"], dtype=np.float64)
    y_true = np.asarray(result["y_true"], dtype=np.float64)
    y_pred = np.asarray(result["y_pred"], dtype=np.float64)
    units = result["units"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, y_true, color="gray", lw=1.0, label="actual")
    ax.plot(t, y_pred, color="tab:red", lw=1.0, alpha=0.8, label="decoded")
    ax.set_xlabel("time (s, ephys)")
    ax.set_ylabel(f"distance ({units})")
    ax.set_title(f"Decoded distance over time · {_title(result)} "
                 f"(CV R² = {result['cv_r2']:.3f})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_partner_distance_summary(result: Dict,
                                  figsize: Tuple[float, float] = (15, 9)):
    """Combined overview: tuning curves, R² distribution, scatter, time series."""
    import matplotlib.pyplot as plt

    if not _ok(result):
        return None
    units = result["units"]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3)

    # (0,0:2) top-6 tuning curves stacked
    ax_tun = fig.add_subplot(gs[0, 0])
    tuning, centers, ranking = result["tuning"], result["dist_centers"], result["cell_ranking"]
    for i in range(int(min(6, tuning.shape[1]))):
        c = int(ranking[i])
        ax_tun.plot(centers, tuning[:, c], lw=1.2, label=f"cell {c}")
    ax_tun.set_xlabel(f"distance ({units})")
    ax_tun.set_ylabel("rate (Hz)")
    ax_tun.set_title("Top-6 distance tuning")
    ax_tun.legend(fontsize=7)

    # (0,1) per-cell R² histogram
    ax_hist = fig.add_subplot(gs[0, 1])
    r2 = np.asarray(result["r2_per_cell"], dtype=np.float64)
    r2 = r2[np.isfinite(r2)]
    ax_hist.hist(r2, bins=25, color="tab:blue", alpha=0.7)
    partial = result.get("r2_partial_per_cell")
    if partial is not None:
        p = np.asarray(partial, dtype=np.float64)
        ax_hist.hist(p[np.isfinite(p)], bins=25, color="tab:orange", alpha=0.6)
    if result.get("null_r2") is not None and np.isfinite(result["null_r2"]):
        ax_hist.axvline(result["null_r2"], color="tab:gray", ls="--")
    ax_hist.axvline(0.0, color="black", lw=0.7)
    ax_hist.set_xlabel("per-cell CV R²")
    ax_hist.set_ylabel("# cells")
    ax_hist.set_title("Single-cell R² (blue=raw, orange=partial)")

    # (0,2) scatter
    ax_sc = fig.add_subplot(gs[0, 2])
    y_true = np.asarray(result["y_true"], dtype=np.float64)
    y_pred = np.asarray(result["y_pred"], dtype=np.float64)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    ax_sc.scatter(y_true[m], y_pred[m], s=3, alpha=0.25, color="tab:red")
    lo = float(min(y_true[m].min(), y_pred[m].min()))
    hi = float(max(y_true[m].max(), y_pred[m].max()))
    ax_sc.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax_sc.set_xlabel(f"actual ({units})")
    ax_sc.set_ylabel(f"decoded ({units})")
    ax_sc.set_title(f"Population: R²={result['cv_r2']:.3f}, "
                    f"RMSE={result['rmse']:.1f}{units}")

    # (1,:) time series
    ax_ts = fig.add_subplot(gs[1, :])
    t = np.asarray(result["bin_centers"], dtype=np.float64)
    ax_ts.plot(t, y_true, color="gray", lw=0.9, label="actual")
    ax_ts.plot(t, y_pred, color="tab:red", lw=0.9, alpha=0.8, label="decoded")
    ax_ts.set_xlabel("time (s, ephys)")
    ax_ts.set_ylabel(f"distance ({units})")
    ax_ts.set_title("Decoded vs actual distance over time")
    ax_ts.legend()

    null_txt = ""
    if result.get("null_r2") is not None:
        null_txt = f"  ·  null R² = {result['null_r2']:.3f} ± {result['null_r2_std']:.3f}"
    fig.suptitle(
        f"Partner-distance decoding · {_title(result)}  "
        f"({result['n_cells']} cells, {result['n_bins']} bins){null_txt}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig
