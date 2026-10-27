"""
Plotting helpers for inter-brain neural dynamics.

Mirrors the style of :mod:`ephys.decoding_plots`: titles and axis labels
are driven by ``fit.parameters['analysis_title']`` and
``fit.parameters['class_label']`` so the same figures work for any
future variant of the shared-subspace fit without code changes here.

All functions return ``matplotlib.figure.Figure`` (or ``None`` when the
input does not provide the data they need); none of them call
``plt.show()`` themselves.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ephys.inter_brain_dynamics import SharedSubspaceFit


__all__ = [
    "plot_canonical_correlations",
    "plot_variance_partition",
    "plot_shared_dimensions",
    "plot_cross_animal_correlation",
    "plot_time_lagged_cca",
    "plot_shared_vs_behavior",
    "plot_inter_brain_summary",
]


# ---------------------------------------------------------------------------
# Local title / label helpers (mirrors ephys/decoding_plots.py:34-39)
# ---------------------------------------------------------------------------

def _analysis_title(
    fit: SharedSubspaceFit, default: str = "Inter-brain shared subspace",
) -> str:
    return fit.parameters.get("analysis_title", default)


def _class_label(fit: SharedSubspaceFit, default: str = "Shared dim") -> str:
    return fit.parameters.get("class_label", default)


def _animal_ids(fit: SharedSubspaceFit) -> Tuple[str, str]:
    aids = fit.parameters.get("animal_ids")
    if aids is None or len(aids) != 2:
        return ("A", "B")
    return (str(aids[0]), str(aids[1]))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 2:
        return float("nan")
    a = a[valid] - a[valid].mean()
    b = b[valid] - b[valid].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


# ---------------------------------------------------------------------------
# Canonical correlations
# ---------------------------------------------------------------------------

def plot_canonical_correlations(
    fit: SharedSubspaceFit,
    shuffle_null: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (8, 5),
    null_percentile: float = 95.0,
) -> plt.Figure:
    """Bar chart of per-K canonical correlations with optional null band.

    Train CCs (solid) and CV-mean CCs (hatched) are shown side-by-side.
    If ``shuffle_null`` is provided, the per-K ``null_percentile``-th
    percentile is overlaid as a step line.
    """
    train = np.asarray(fit.canonical_correlations["train"], dtype=np.float64)
    cv_mean = np.asarray(fit.canonical_correlations["cv_mean"], dtype=np.float64)
    K = len(train)
    ks = np.arange(1, K + 1)
    width = 0.4

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(ks - width / 2, train, width=width, color="steelblue",
           label="Train")
    ax.bar(ks + width / 2, cv_mean, width=width, color="darkorange",
           label="CV mean", alpha=0.85)

    if shuffle_null is not None:
        null = np.asarray(shuffle_null, dtype=np.float64)
        if null.ndim != 2 or null.shape[1] != K:
            raise ValueError(
                f"shuffle_null must have shape (n_shuffles, K={K}); got {null.shape}"
            )
        p = np.nanpercentile(null, null_percentile, axis=0)
        ax.step(
            np.concatenate(([ks[0] - 0.5], ks + 0.5)),
            np.concatenate(([p[0]], p)),
            where="pre", color="black", linestyle="--",
            label=f"Null p{int(null_percentile)}",
        )

    ax.set_xticks(ks)
    ax.set_xlabel("Canonical component")
    ax.set_ylabel("Canonical correlation")
    ax.set_ylim(min(0.0, float(np.nanmin([train.min(), cv_mean.min(), 0.0]))) - 0.05,
                1.05)
    ax.set_title(f"Canonical correlations — {_analysis_title(fit)}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Variance partition
# ---------------------------------------------------------------------------

def plot_variance_partition(
    fit: SharedSubspaceFit,
    figsize: Tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Stacked bars per animal: shared vs unique variance (z-scored)."""
    v = fit.variance_partition
    aid_A, aid_B = _animal_ids(fit)
    labels = [aid_A, aid_B]
    shared = np.array([v.get("shared_var_A_z", v.get("shared_var_A", 0.0)),
                       v.get("shared_var_B_z", v.get("shared_var_B", 0.0))])
    unique = 1.0 - shared

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(2)
    ax.bar(x, shared, color="steelblue", label="Shared")
    ax.bar(x, unique, bottom=shared, color="lightgray", label="Unique")
    for xi, s, u in zip(x, shared, unique):
        ax.text(xi, s / 2, f"{s:.1%}", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        ax.text(xi, s + u / 2, f"{u:.1%}", ha="center", va="center",
                color="black", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of per-animal variance (z-scored)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        f"Variance partition — {_analysis_title(fit)}\n"
        f"K = {fit.n_components}"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Shared dimensions
# ---------------------------------------------------------------------------

def plot_shared_dimensions(
    fit: SharedSubspaceFit,
    t_bins: Optional[np.ndarray] = None,
    k_dims: Sequence[int] = (0, 1, 2),
    figsize: Tuple[float, float] = (11, 7),
) -> plt.Figure:
    """Overlaid time courses of selected shared dimensions for A vs B.

    Pearson correlation between S_A[:, k] and S_B[:, k] is annotated in
    each subplot title.
    """
    S_A = fit.S_A
    S_B = fit.S_B
    K = S_A.shape[1]
    k_dims = [k for k in k_dims if 0 <= k < K]
    if not k_dims:
        raise ValueError(f"No valid dims in k_dims={k_dims!r} for K={K}")

    if t_bins is None:
        t = np.arange(S_A.shape[0])
        x_label = "Bin index"
    else:
        t = np.asarray(t_bins, dtype=np.float64)
        if len(t) != S_A.shape[0]:
            raise ValueError(
                f"len(t_bins)={len(t)} does not match S_A length {S_A.shape[0]}"
            )
        x_label = "Time (ephys s)"

    aid_A, aid_B = _animal_ids(fit)
    fig, axes = plt.subplots(
        len(k_dims), 1, figsize=figsize, sharex=True, squeeze=False,
    )
    axes = axes[:, 0]

    for ax, k in zip(axes, k_dims):
        r = _pearson(S_A[:, k], S_B[:, k])
        ax.plot(t, S_A[:, k], color="steelblue", linewidth=1.3,
                label=f"{aid_A}")
        ax.plot(t, S_B[:, k], color="darkorange", linewidth=1.3,
                label=f"{aid_B}", alpha=0.85)
        ax.set_ylabel(f"{_class_label(fit)} {k + 1}")
        ax.set_title(f"k = {k + 1}    r = {r:+.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel(x_label)
    fig.suptitle(_analysis_title(fit))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# Cross-animal correlation heatmap
# ---------------------------------------------------------------------------

def plot_cross_animal_correlation(
    corr_matrix: np.ndarray,
    figsize: Tuple[float, float] = (7, 6),
    cluster: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of cell-pair Pearson cross-correlations.

    With ``cluster=True`` (default) rows and columns are reordered by
    hierarchical clustering (average linkage on Euclidean distance over
    rows/columns of the matrix).
    """
    C = np.asarray(corr_matrix, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("corr_matrix must be 2-D")

    if cluster:
        row_order = _cluster_order(C)
        col_order = _cluster_order(C.T)
        C_show = C[np.ix_(row_order, col_order)]
    else:
        C_show = C

    vmax = float(np.nanmax(np.abs(C_show))) if C_show.size > 0 else 1.0
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        C_show, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )
    ax.set_xlabel("Animal B cells")
    ax.set_ylabel("Animal A cells")
    ax.set_title(title or "Cross-animal cell-pair correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    return fig


def _cluster_order(matrix: np.ndarray) -> np.ndarray:
    """Return a row-reordering via average-linkage hierarchical clustering."""
    n = matrix.shape[0]
    if n < 2:
        return np.arange(n)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
    except ImportError:
        return np.arange(n)
    finite_rows = np.all(np.isfinite(matrix), axis=1)
    if finite_rows.sum() < 2:
        return np.arange(n)
    try:
        Z = linkage(matrix[finite_rows], method="average", metric="euclidean")
        ordered = leaves_list(Z)
    except Exception:
        return np.arange(n)
    out = np.arange(n)
    finite_idx = np.flatnonzero(finite_rows)
    out[: len(ordered)] = finite_idx[ordered]
    if finite_rows.sum() < n:
        out[len(ordered):] = np.flatnonzero(~finite_rows)
    return out


# ---------------------------------------------------------------------------
# Time-lagged CCA
# ---------------------------------------------------------------------------

def plot_time_lagged_cca(
    lags: np.ndarray,
    ccs: np.ndarray,
    bin_size_sec: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """Canonical correlations vs integer lag for K dims.

    Positive lag means animal B leads animal A by that many bins. Peak
    of the top-CC curve is marked with a vertical line.
    """
    lags = np.asarray(lags)
    ccs = np.asarray(ccs, dtype=np.float64)
    if ccs.ndim != 2 or len(lags) != ccs.shape[0]:
        raise ValueError(
            f"Expected ccs of shape (n_lags, K); got lags={lags.shape}, ccs={ccs.shape}"
        )
    K = ccs.shape[1]

    if bin_size_sec is not None:
        x = lags * float(bin_size_sec)
        x_label = "Lag (s; positive = B leads A)"
    else:
        x = lags
        x_label = "Lag (bins; positive = B leads A)"

    fig, ax = plt.subplots(figsize=figsize)
    for k in range(K):
        ax.plot(x, ccs[:, k], linewidth=1.5, label=f"k = {k + 1}")

    peak_idx = int(np.nanargmax(ccs[:, 0]))
    ax.axvline(x[peak_idx], color="black", linestyle="--", alpha=0.6,
               label=f"Peak (k=1) @ lag={x[peak_idx]:g}")
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.6)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Canonical correlation")
    ax.set_title(title or "Time-lagged CCA")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Shared vs behavior (regression R²)
# ---------------------------------------------------------------------------

def plot_shared_vs_behavior(
    fit: SharedSubspaceFit,
    regression_results: Dict,
    figsize: Tuple[float, float] = (11, 6),
) -> plt.Figure:
    """Bar chart of R² (self / partner / both) per shared dim, per animal."""
    aid_A, aid_B = _animal_ids(fit)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, aid in zip(axes, (aid_A, aid_B)):
        if aid not in regression_results:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No regression results for {aid}",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        per_dim = regression_results[aid]
        ks = sorted(per_dim.keys())
        r2_self = [per_dim[k]["R2_self"] for k in ks]
        r2_partner = [per_dim[k]["R2_partner"] for k in ks]
        r2_both = [per_dim[k]["R2_both"] for k in ks]

        x = np.arange(len(ks))
        width = 0.28
        ax.bar(x - width, r2_self, width=width, label="Self",
               color="steelblue")
        ax.bar(x, r2_partner, width=width, label="Partner",
               color="darkorange")
        ax.bar(x + width, r2_both, width=width, label="Both",
               color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k + 1}" for k in ks])
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_xlabel(_class_label(fit))
        ax.set_title(f"Animal {aid}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("CV R²")
    fig.suptitle(f"Shared dims vs behavior — {_analysis_title(fit)}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------

def plot_inter_brain_summary(
    fit: SharedSubspaceFit,
    shuffle_null: Optional[np.ndarray] = None,
    t_bins: Optional[np.ndarray] = None,
    cross_corr: Optional[np.ndarray] = None,
    time_lagged: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    regression_results: Optional[Dict] = None,
    k_dims: Sequence[int] = (0, 1, 2),
    bin_size_sec: Optional[float] = None,
    figsize: Tuple[float, float] = (16, 10),
) -> plt.Figure:
    """Six-panel dashboard combining the individual plots.

    Panels:

    1. Canonical correlations with optional null band
    2. Variance partition (shared vs unique per animal)
    3. Cross-animal cell-pair correlation heatmap (if ``cross_corr``)
    4. Top-K shared dimension time courses (overlaid A vs B)
    5. Time-lagged CCA leader/follower profile (if ``time_lagged``)
    6. Per-dim R² self vs partner vs both (if ``regression_results``)
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # --- Panel 1: canonical correlations -------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_canonical_correlations(ax1, fit, shuffle_null)

    # --- Panel 2: variance partition -----------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_variance_partition(ax2, fit)

    # --- Panel 3: cross-animal corr heatmap ----------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    if cross_corr is not None:
        _draw_cross_correlation(fig, ax3, cross_corr)
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5,
                 "Pass cross_corr=cross_animal_correlation_matrix(...)",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=9)

    # --- Panel 4: shared dim time courses ------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    _draw_shared_dimensions_one_axis(ax4, fit, t_bins, k_dims)

    # --- Panel 5: time-lagged CCA --------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    if time_lagged is not None:
        _draw_time_lagged(ax5, time_lagged[0], time_lagged[1], bin_size_sec)
    else:
        ax5.axis("off")
        ax5.text(0.5, 0.5,
                 "Pass time_lagged=time_lagged_cca(...) output",
                 ha="center", va="center", transform=ax5.transAxes,
                 fontsize=9)

    # --- Panel 6: regression R² ----------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    if regression_results is not None:
        _draw_regression_r2(ax6, fit, regression_results)
    else:
        ax6.axis("off")
        ax6.text(0.5, 0.5,
                 "Pass regression_results=regress_shared_on_behavior(...) output",
                 ha="center", va="center", transform=ax6.transAxes,
                 fontsize=9)

    aid_A, aid_B = _animal_ids(fit)
    fig.suptitle(
        f"{_analysis_title(fit)} — {aid_A} vs {aid_B} "
        f"(K = {fit.n_components}, T = {fit.parameters.get('T_valid', 'n/a')})",
        fontsize=14,
    )
    return fig


# ---------------------------------------------------------------------------
# Single-axis drawers used by the summary
# ---------------------------------------------------------------------------

def _draw_canonical_correlations(ax, fit, shuffle_null):
    train = np.asarray(fit.canonical_correlations["train"], dtype=np.float64)
    cv_mean = np.asarray(fit.canonical_correlations["cv_mean"], dtype=np.float64)
    K = len(train)
    ks = np.arange(1, K + 1)
    width = 0.4
    ax.bar(ks - width / 2, train, width=width, color="steelblue", label="Train")
    ax.bar(ks + width / 2, cv_mean, width=width, color="darkorange",
           alpha=0.85, label="CV mean")
    if shuffle_null is not None:
        null = np.asarray(shuffle_null, dtype=np.float64)
        if null.ndim == 2 and null.shape[1] == K:
            p = np.nanpercentile(null, 95, axis=0)
            ax.step(
                np.concatenate(([ks[0] - 0.5], ks + 0.5)),
                np.concatenate(([p[0]], p)),
                where="pre", color="black", linestyle="--",
                label="Null p95",
            )
    ax.set_xticks(ks)
    ax.set_xlabel("Canonical component")
    ax.set_ylabel("Canonical correlation")
    ax.set_title("Canonical correlations")
    ax.set_ylim(min(0.0, float(np.nanmin([train.min(), cv_mean.min(), 0.0]))) - 0.05,
                1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def _draw_variance_partition(ax, fit):
    v = fit.variance_partition
    aid_A, aid_B = _animal_ids(fit)
    labels = [aid_A, aid_B]
    shared = np.array([
        v.get("shared_var_A_z", v.get("shared_var_A", 0.0)),
        v.get("shared_var_B_z", v.get("shared_var_B", 0.0)),
    ])
    unique = 1.0 - shared
    x = np.arange(2)
    ax.bar(x, shared, color="steelblue", label="Shared")
    ax.bar(x, unique, bottom=shared, color="lightgray", label="Unique")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of variance")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Variance partition")
    ax.legend(loc="upper right", fontsize=8)


def _draw_cross_correlation(fig, ax, corr_matrix):
    C = np.asarray(corr_matrix, dtype=np.float64)
    row_order = _cluster_order(C)
    col_order = _cluster_order(C.T)
    C_show = C[np.ix_(row_order, col_order)]
    vmax = float(np.nanmax(np.abs(C_show))) if C_show.size > 0 else 1.0
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0
    im = ax.imshow(
        C_show, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )
    ax.set_xlabel("Animal B cells")
    ax.set_ylabel("Animal A cells")
    ax.set_title("Cross-animal correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="r")


def _draw_shared_dimensions_one_axis(ax, fit, t_bins, k_dims):
    S_A = fit.S_A
    S_B = fit.S_B
    K = S_A.shape[1]
    k_dims = [k for k in k_dims if 0 <= k < K]
    if not k_dims:
        ax.axis("off")
        return
    if t_bins is None:
        t = np.arange(S_A.shape[0])
        x_label = "Bin index"
    else:
        t = np.asarray(t_bins, dtype=np.float64)
        x_label = "Time (s)"
    aid_A, aid_B = _animal_ids(fit)
    # Plot first k_dim on the axis (subsequent dims offset for clarity).
    offsets = np.arange(len(k_dims))[::-1] * 3.0
    for i, k in enumerate(k_dims):
        sa = S_A[:, k] + offsets[i]
        sb = S_B[:, k] + offsets[i]
        ax.plot(t, sa, color="steelblue", linewidth=1.0,
                label=aid_A if i == 0 else None)
        ax.plot(t, sb, color="darkorange", linewidth=1.0, alpha=0.85,
                label=aid_B if i == 0 else None)
        r = _pearson(S_A[:, k], S_B[:, k])
        ax.text(t[0], offsets[i] + 1.5,
                f"k={k + 1}  r={r:+.2f}",
                fontsize=8, va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    ax.set_xlabel(x_label)
    ax.set_yticks([])
    ax.set_title("Shared dimensions (offset for clarity)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def _draw_time_lagged(ax, lags, ccs, bin_size_sec):
    lags = np.asarray(lags)
    ccs = np.asarray(ccs, dtype=np.float64)
    if bin_size_sec is not None:
        x = lags * float(bin_size_sec)
        x_label = "Lag (s)"
    else:
        x = lags
        x_label = "Lag (bins)"
    K = ccs.shape[1]
    for k in range(K):
        ax.plot(x, ccs[:, k], linewidth=1.3, label=f"k={k + 1}")
    peak = int(np.nanargmax(ccs[:, 0]))
    ax.axvline(x[peak], color="black", linestyle="--", alpha=0.6)
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Canonical correlation")
    ax.set_title("Time-lagged CCA")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def _draw_regression_r2(ax, fit, regression_results):
    aid_A, _ = _animal_ids(fit)
    # Only the focal animal's R² in the summary panel; full plot has both.
    if aid_A not in regression_results:
        ax.axis("off")
        return
    per_dim = regression_results[aid_A]
    ks = sorted(per_dim.keys())
    r2_self = [per_dim[k]["R2_self"] for k in ks]
    r2_partner = [per_dim[k]["R2_partner"] for k in ks]
    r2_both = [per_dim[k]["R2_both"] for k in ks]
    x = np.arange(len(ks))
    width = 0.28
    ax.bar(x - width, r2_self, width=width, label="Self", color="steelblue")
    ax.bar(x, r2_partner, width=width, label="Partner", color="darkorange")
    ax.bar(x + width, r2_both, width=width, label="Both", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k + 1}" for k in ks])
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_ylabel("CV R²")
    ax.set_title(f"Shared dims vs behavior ({aid_A})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
