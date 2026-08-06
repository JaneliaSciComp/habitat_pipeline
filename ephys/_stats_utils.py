"""
Small shared statistics helpers for the ephys rigor layer.

Kept dependency-free (no ``statsmodels``, which is not installed on the
Janelia workstation this pipeline runs on) so decoding modules can apply
multiple-comparison correction without a new hard dependency.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values). NaNs map to 1.0.

    Mirrors ``ephys.social_spatial_fields._benjamini_hochberg`` (same
    algorithm, kept as a small independent copy here rather than importing
    across modules).
    """
    p = np.asarray(pvals, dtype=np.float64).copy()
    nan_mask = ~np.isfinite(p)
    p[nan_mask] = 1.0
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=np.float64)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q
