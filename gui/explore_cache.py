"""Disk cache of what the explore view draws, so a revisited session opens fast.

The explore view (``gui/interactive_app.py``) needs, per ``(session, animal)``:

- the rastermap image and its ``[t0, t1]`` span,
- the PCA trajectory (scores, bin centres, explained variance) — the fit
  depends only on the spikes and the bin size, never on the behaviour type,
- the behavioural events, already synchronised to the ephys clock.

Producing those takes the full pipeline (path resolution, Kilosort load, DIO
sync, quality filter, rastermap fit, PCA). This module computes them in one
place and pickles them to ``.gui_cache/explore/``, **only at the default bin
sizes** (one entry per session/animal, bounded disk use). The app shows a cache
hit immediately and then loads the full data in the background; the background
load compares :func:`source_signature` against the cached one and recomputes
when the Kilosort output or the event files changed.

Nothing here imports Panel.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

EXPLORE_CACHE_VERSION = 1
CACHE_DIR = Path(__file__).resolve().parent.parent / ".gui_cache" / "explore"

#: The bin sizes the explore sliders start at; only these are cached.
DEFAULT_RASTER_BIN = 1.0
DEFAULT_PCA_BIN = 0.5

#: Kilosort outputs whose change invalidates a cached view. ``kilosort_*.pkl``
#: is itself a cache of these, so it is deliberately not part of the signature.
_KILOSORT_SOURCES = ("spike_times.npy", "spike_clusters.npy", "cluster_info.tsv",
                     "cluster_group.tsv", "cluster_KSLabel.tsv")


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def quality_indices(ks_data) -> list:
    """Indices into ``spike_times_by_cell`` of cells passing the default quality filter."""
    passed = set(ks_data.filter_cells_by_firing_patterns()["passed_clusters"])
    return [i for i, cid in enumerate(ks_data.ks_ids) if cid in passed]


def compute_spike_matrix(ks_data, indices, t0, t1, bin_size_s):
    """``(len(indices), n_bins)`` firing-rate matrix for the given cells."""
    edges = np.arange(t0, t1 + bin_size_s, bin_size_s)
    mat = np.zeros((len(indices), len(edges) - 1), dtype=np.float64)
    for row, ci in enumerate(indices):
        counts, _ = np.histogram(ks_data.spike_times_by_cell[ci], bins=edges)
        mat[row] = counts / bin_size_s
    return mat


def fit_rastermap(fr_matrix):
    """Rastermap display image (float64, C-order — what Bokeh's image glyph needs)."""
    from rastermap import Rastermap
    n_cells = fr_matrix.shape[0]
    model = Rastermap(
        n_PCs=min(200, n_cells - 1),
        n_clusters=min(100, max(4, n_cells // 4)),
        normalize=True,
        mean_time=True,
        verbose=False,
        verbose_sorting=False,
    )
    model.fit(fr_matrix)
    # [::-1] gives negative strides — Bokeh image glyph requires C-contiguous float64
    return np.ascontiguousarray(
        np.nan_to_num(model.X_embedding[::-1, :], nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float64,
    )


def fit_pca_trajectory(ks_data, pca_bin, t0, t1) -> Optional[Dict[str, np.ndarray]]:
    """3-component PCA of z-scored quality-cell rates over the whole recording.

    Behaviour-independent, so it is computed once per (session, animal, bin)
    and the event markers are laid on top later. ``None`` for < 3 cells.
    """
    spks, bin_centers = ks_data.bin_spike_times(
        bin_size_sec=pca_bin, t_start=t0, t_end=t1, filtered_only=True,
    )
    if spks.shape[0] < 3:
        return None
    X = np.nan_to_num(zscore(spks, axis=1), nan=0.0)
    pca = PCA(n_components=3)
    return {
        "scores": pca.fit_transform(X.T),          # (n_bins, 3)
        "bin_centers": bin_centers,
        "var_explained": pca.explained_variance_ratio_,
        "pca_bin": float(pca_bin),
    }


def compute_view_products(ks_data, raster_bin: float, pca_bin: float) -> Dict[str, Any]:
    """Rastermap image, its span and the PCA trajectory for one loaded session."""
    indices = quality_indices(ks_data)
    if not indices:
        raise ValueError("no cells pass the quality filter")
    spikes = np.concatenate([ks_data.spike_times_by_cell[i] for i in indices])
    t0, t1 = float(spikes.min()), float(spikes.max())
    raster = fit_rastermap(compute_spike_matrix(ks_data, indices, t0, t1, raster_bin))
    return {
        "raster_img": raster,
        "raster_bin": float(raster_bin),
        "t0": t0,
        "t1": t1,
        "pca": fit_pca_trajectory(ks_data, pca_bin, t0, t1),
    }


# ---------------------------------------------------------------------------
# Signature + disk cache
# ---------------------------------------------------------------------------

def _stat(path: Path):
    try:
        st = path.stat()
        return [path.name, int(st.st_size), float(st.st_mtime)]
    except OSError:
        return [path.name, None, None]


def source_signature(kilosort_path: Optional[Path], event_files: Sequence[Path]) -> str:
    """Hash of the inputs a cached view was computed from (stats only, no reads)."""
    ks = Path(kilosort_path) if kilosort_path else None
    payload = {
        "v": EXPLORE_CACHE_VERSION,
        "kilosort": [_stat(ks / name) for name in _KILOSORT_SOURCES] if ks else None,
        "events": sorted(_stat(Path(f)) for f in event_files),
    }
    return hashlib.sha256(json.dumps(payload, default=str).encode()).hexdigest()[:16]


def _safe(part: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(part))


def cache_path(cohort: str, session_id: str, animal_id: str) -> Path:
    return CACHE_DIR / f"{_safe(cohort)}__{_safe(session_id)}__{_safe(animal_id)}.pkl"


def is_default_bins(raster_bin: float, pca_bin: float) -> bool:
    return (abs(raster_bin - DEFAULT_RASTER_BIN) < 1e-9
            and abs(pca_bin - DEFAULT_PCA_BIN) < 1e-9)


def save_view_cache(cohort: str, session_id: str, animal_id: str,
                    products: Dict[str, Any], events, signature: str) -> Optional[Path]:
    """Persist a computed view. Silently skips non-default bin sizes."""
    pca = products.get("pca")
    if not is_default_bins(products["raster_bin"], pca["pca_bin"] if pca else DEFAULT_PCA_BIN):
        return None
    entry = {
        "version": EXPLORE_CACHE_VERSION,
        "signature": signature,
        "saved_at": time.time(),
        # float32 halves the file; the app converts back to float64 for Bokeh.
        "raster_img": products["raster_img"].astype(np.float32),
        "raster_bin": products["raster_bin"],
        "t0": products["t0"],
        "t1": products["t1"],
        "pca": pca,
        "events": events,
    }
    path = cache_path(cohort, session_id, animal_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(entry, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load_view_cache(cohort: str, session_id: str, animal_id: str) -> Optional[Dict[str, Any]]:
    """The cached view, or ``None`` if absent, unreadable or from another version."""
    path = cache_path(cohort, session_id, animal_id)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            entry = pickle.load(fh)
    except Exception as exc:
        logger.warning("ignoring unreadable explore cache %s: %s", path, exc)
        return None
    if entry.get("version") != EXPLORE_CACHE_VERSION:
        return None
    entry["raster_img"] = np.ascontiguousarray(entry["raster_img"], dtype=np.float64)
    return entry
