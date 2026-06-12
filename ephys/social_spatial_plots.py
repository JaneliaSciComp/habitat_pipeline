"""
Plots for the social place fields module (ephys/social_spatial_fields.py).

Driven by ``SocialFieldResults.parameters['analysis_title']`` and
``['class_label']``, mirroring the convention in ``ephys/decoding_plots.py``:
the figures read cosmetic strings off the result dataclass rather than branching
on the analysis type. NaN rate-map bins (occupancy below threshold) are masked
light gray throughout.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ephys.social_spatial_fields import (
    SocialFieldResults,
    field_similarity_across_targets,
)

_CATEGORY_ORDER = ["self_only", "partner_only", "conjunctive", "broadcast", "none"]
_CATEGORY_COLORS = {
    "self_only": "#1f77b4",
    "partner_only": "#ff7f0e",
    "conjunctive": "#2ca02c",
    "broadcast": "#d62728",
    "none": "#999999",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _analysis_title(results: SocialFieldResults, default: str = "Social Place Fields") -> str:
    return results.parameters.get("analysis_title", default)


def _focal(results: SocialFieldResults) -> str:
    return results.parameters.get("focal_animal", "")


def _targets(results: SocialFieldResults) -> List[str]:
    return list(results.parameters.get("target_animals", list(results.rate_maps.keys())))


def _masked_cmap():
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")
    return cmap


def _draw_rate_map(ax, rm, title: Optional[str] = None, vmax: Optional[float] = None):
    """Draw one RateMap on ``ax`` with NaN bins masked. Returns the image."""
    masked = np.ma.masked_invalid(rm.rates)
    extent = [rm.x_edges[0], rm.x_edges[-1], rm.y_edges[0], rm.y_edges[-1]]
    im = ax.imshow(masked, origin="lower", extent=extent, aspect="auto",
                   cmap=_masked_cmap(), vmin=0.0, vmax=vmax)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    if title:
        ax.set_title(title, fontsize=9)
    return im


def _maybe_save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Per-cluster grid
# ---------------------------------------------------------------------------

def plot_rate_maps_grid(results: SocialFieldResults, cluster_id: int,
                        save_path=None, figsize: Optional[Tuple[int, int]] = None):
    """One rate-map panel per target animal for a single cluster (shared colorbar)."""
    targets = _targets(results)
    maps = [results.rate_maps[t][cluster_id] for t in targets if cluster_id in results.rate_maps[t]]
    used = [t for t in targets if cluster_id in results.rate_maps[t]]
    if not maps:
        raise ValueError(f"No rate maps for cluster {cluster_id}.")
    
    vmax = np.nanmax([np.nanmax(m.rates) if np.isfinite(m.rates).any() else 0.0 for m in maps])
    vmax = float(vmax) if vmax > 0 else None

    n = len(maps)
    if figsize is None:
        figsize = (3.2 * n + 1.2, 3.4)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    im = None
    for ax, t, m in zip(axes, used, maps):
        fs = results.stats[t][cluster_id]
        sig = results.signif[t][cluster_id]
        p_txt = "n/a" if not np.isfinite(sig.p_skaggs) else f"{sig.p_skaggs:.3g}"
        self_tag = " (self)" if t == _focal(results) else ""
        title = (f"target {t}{self_tag}\n"
                 f"bits/spk={fs.skaggs_bits_per_spike:.2f}  p={p_txt}")
        im = _draw_rate_map(ax, m, title=title, vmax=vmax)

    if im is not None:
        fig.colorbar(im, ax=list(axes), shrink=0.8, label="Hz")
    fig.suptitle(f"{_analysis_title(results)} — focal {_focal(results)}, cluster {cluster_id}",
                 fontsize=11)
    plt.show()
    return _maybe_save(fig, save_path)


def plot_field_similarity_grid(results: SocialFieldResults, cluster_id: int,
                               save_path=None, figsize=(5.5, 4.5)):
    """Heatmap of pairwise rate-map correlations across targets for one cell."""
    targets = _targets(results)
    rms = {t: results.rate_maps[t][cluster_id] for t in targets
           if cluster_id in results.rate_maps[t]}
    sim = field_similarity_across_targets(rms)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sim)))
    ax.set_yticks(range(len(sim)))
    ax.set_xticklabels(sim.columns, rotation=45, ha="right")
    ax.set_yticklabels(sim.index)
    for i in range(len(sim)):
        for j in range(len(sim)):
            v = sim.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Field similarity across targets — cluster {cluster_id}")
    return _maybe_save(fig, save_path)


# ---------------------------------------------------------------------------
# Aggregate panels
# ---------------------------------------------------------------------------

def plot_cell_classification_summary(results: SocialFieldResults,
                                     save_path=None, figsize=(11, 4.5)):
    """Stacked category-count bar + self-vs-best-partner Skaggs scatter."""
    df = results.cell_classification
    focal = _focal(results)
    targets = _targets(results)
    partners = [t for t in targets if t != focal]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    counts = df["category"].value_counts()
    cats = [c for c in _CATEGORY_ORDER if c in counts.index]
    bar_vals = [counts[c] for c in cats]
    ax1.bar(cats, bar_vals, color=[_CATEGORY_COLORS[c] for c in cats])
    ax1.set_ylabel("n cells")
    ax1.set_title("Cell classification")
    ax1.tick_params(axis="x", rotation=30)

    self_col = f"bits_per_spike_{focal}"
    if self_col in df.columns and partners:
        partner_cols = [f"bits_per_spike_{p}" for p in partners if f"bits_per_spike_{p}" in df.columns]
        self_bits = df[self_col].to_numpy(dtype=float)
        max_partner = df[partner_cols].max(axis=1).to_numpy(dtype=float) if partner_cols else np.zeros(len(df))
        colors = [_CATEGORY_COLORS.get(c, "#999999") for c in df["category"]]
        ax2.scatter(self_bits, max_partner, c=colors, s=25, alpha=0.8)
        lim = np.nanmax([np.nanmax(self_bits) if len(self_bits) else 0,
                         np.nanmax(max_partner) if len(max_partner) else 0, 0.1])
        ax2.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        ax2.set_xlabel(f"bits/spike — self ({focal})")
        ax2.set_ylabel("bits/spike — best partner")
        ax2.set_title("Self vs partner tuning")
    else:
        ax2.text(0.5, 0.5, "No partner targets", ha="center", va="center")
        ax2.axis("off")

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=_CATEGORY_COLORS[c], label=c, markersize=8)
               for c in cats]
    if handles:
        ax2.legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle(f"{_analysis_title(results)} — focal {focal}", fontsize=11)
    return _maybe_save(fig, save_path)


def plot_population_field_similarity(results: SocialFieldResults, target_pair: str,
                                     save_path=None, figsize=(6, 5)):
    """(n_cells, n_cells) rate-map correlation heatmap for a (focal, partner) pair.

    Cells are ordered by hierarchical clustering of the self-target maps when
    SciPy is available; otherwise left in natural order.
    """
    pair = results.population_field_similarity.get(target_pair)
    if pair is None:
        raise ValueError(
            f"No population similarity for {target_pair!r}; "
            f"have {list(results.population_field_similarity.keys())}."
        )
    mat = pair["similarity_matrix"]
    order = np.arange(mat.shape[0])
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        finite = np.where(np.isfinite(mat), mat, 0.0)
        sym = 0.5 * (finite + finite.T)
        dist = 1.0 - sym
        np.fill_diagonal(dist, 0.0)
        dist[dist < 0] = 0.0
        if mat.shape[0] > 2:
            order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    except Exception:
        pass

    ordered = mat[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.ma.masked_invalid(ordered), cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xlabel("cell (partner map)")
    ax.set_ylabel("cell (self map)")
    ax.set_title(f"Population field similarity — {target_pair}")
    return _maybe_save(fig, save_path)


def plot_skaggs_vs_shuffle(results: SocialFieldResults, target_animal: str,
                           top_k: int = 20, save_path=None, figsize=(11, 4.5)):
    """For the top-K cells (by true Skaggs), true value vs shuffle null."""
    stats = results.stats[target_animal]
    signif = results.signif[target_animal]
    ranked = sorted(stats.keys(),
                    key=lambda c: stats[c].skaggs_bits_per_spike, reverse=True)[:top_k]

    fig, ax = plt.subplots(figsize=figsize)
    for i, cid in enumerate(ranked):
        sh = signif[cid].shuffle_skaggs
        if sh is not None and len(sh):
            shv = sh[np.isfinite(sh)]
            if shv.size:
                ax.scatter(np.full(shv.size, i), shv, s=4, color="0.7", alpha=0.4)
                lo, hi = np.percentile(shv, [2.5, 97.5])
                ax.plot([i, i], [lo, hi], color="0.4", lw=1.0)
        ax.scatter([i], [stats[cid].skaggs_bits_per_spike], color="crimson", s=30, zorder=3)
    ax.set_xticks(range(len(ranked)))
    ax.set_xticklabels(ranked, rotation=90, fontsize=7)
    ax.set_xlabel("cluster")
    ax.set_ylabel("Skaggs bits/spike")
    ax.set_title(f"True (red) vs shuffle null — target {target_animal}")
    return _maybe_save(fig, save_path)


def plot_field_stability(results: SocialFieldResults, save_path=None, figsize=(9, 4.5)):
    """Split-half stability per target as violins, with canonical threshold stripes."""
    targets = _targets(results)
    data = []
    labels = []
    for t in targets:
        vals = [fs.split_half_corr for fs in results.stats[t].values()
                if np.isfinite(fs.split_half_corr)]
        if vals:
            data.append(vals)
            labels.append(t + (" (self)" if t == _focal(results) else ""))

    fig, ax = plt.subplots(figsize=figsize)
    if data:
        ax.violinplot(data, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
    for y in (0.3, 0.5):
        ax.axhline(y, ls="--", color="0.5", lw=0.8)
    ax.set_ylabel("split-half correlation")
    ax.set_title(f"{_analysis_title(results)} — field stability")
    return _maybe_save(fig, save_path)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def plot_social_place_summary(results: SocialFieldResults, cluster_id: Optional[int] = None,
                              save_path=None, figsize=(15, 9)):
    """Six-panel dashboard: example grid, classification, stability, Skaggs-vs-shuffle.

    Analog of ``ephys.decoding_plots.plot_decoding_summary``. When ``cluster_id``
    is None, the top cell by max Skaggs across targets is used for the example
    rate-map row.
    """
    targets = _targets(results)
    focal = _focal(results)
    df = results.cell_classification

    if cluster_id is None and not df.empty:
        bits_cols = [f"bits_per_spike_{t}" for t in targets if f"bits_per_spike_{t}" in df.columns]
        if bits_cols:
            idx = df[bits_cols].max(axis=1).idxmax()
            cluster_id = int(df.loc[idx, "cluster_id"])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, max(len(targets), 3), height_ratios=[1.1, 1.0])

    # Top row: example cluster's rate maps per target.
    vmax = None
    if cluster_id is not None:
        ms = [results.rate_maps[t][cluster_id] for t in targets if cluster_id in results.rate_maps[t]]
        finite = [np.nanmax(m.rates) for m in ms if np.isfinite(m.rates).any()]
        vmax = float(np.nanmax(finite)) if finite else None
    for j, t in enumerate(targets):
        ax = fig.add_subplot(gs[0, j])
        if cluster_id is not None and cluster_id in results.rate_maps[t]:
            fs = results.stats[t][cluster_id]
            self_tag = " (self)" if t == focal else ""
            _draw_rate_map(ax, results.rate_maps[t][cluster_id],
                           title=f"{t}{self_tag}  {fs.skaggs_bits_per_spike:.2f} b/s", vmax=vmax)
        else:
            ax.axis("off")

    # Bottom row: classification bar, self/partner scatter, stability.
    ax_bar = fig.add_subplot(gs[1, 0])
    counts = df["category"].value_counts() if not df.empty else {}
    cats = [c for c in _CATEGORY_ORDER if c in getattr(counts, "index", [])]
    ax_bar.bar(cats, [counts[c] for c in cats],
               color=[_CATEGORY_COLORS[c] for c in cats])
    ax_bar.set_title("classification")
    ax_bar.tick_params(axis="x", rotation=45)

    ax_sc = fig.add_subplot(gs[1, 1])
    self_col = f"bits_per_spike_{focal}"
    partners = [t for t in targets if t != focal]
    partner_cols = [f"bits_per_spike_{p}" for p in partners if f"bits_per_spike_{p}" in df.columns]
    if self_col in df.columns and partner_cols:
        self_bits = df[self_col].to_numpy(dtype=float)
        max_partner = df[partner_cols].max(axis=1).to_numpy(dtype=float)
        ax_sc.scatter(self_bits, max_partner,
                      c=[_CATEGORY_COLORS.get(c, "#999") for c in df["category"]], s=20, alpha=0.8)
        ax_sc.set_xlabel("self bits/spk")
        ax_sc.set_ylabel("best partner bits/spk")
    else:
        ax_sc.axis("off")
    ax_sc.set_title("self vs partner")

    ax_stab = fig.add_subplot(gs[1, 2]) if len(targets) >= 3 else fig.add_subplot(gs[1, -1])
    data, labels = [], []
    for t in targets:
        vals = [fs.split_half_corr for fs in results.stats[t].values()
                if np.isfinite(fs.split_half_corr)]
        if vals:
            data.append(vals)
            labels.append(t)
    if data:
        ax_stab.violinplot(data, showmeans=True)
        ax_stab.set_xticks(range(1, len(labels) + 1))
        ax_stab.set_xticklabels(labels, rotation=45)
    for y in (0.3, 0.5):
        ax_stab.axhline(y, ls="--", color="0.5", lw=0.8)
    ax_stab.set_title("split-half stability")

    fig.suptitle(f"{_analysis_title(results)} — focal {focal} "
                 f"(example cluster {cluster_id})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _maybe_save(fig, save_path)
