"""
Small shared statistics helpers for the ephys rigor layer.

Kept dependency-free (no ``statsmodels``, which is not installed on the
Janelia workstation this pipeline runs on) so decoding modules can apply
multiple-comparison correction without a new hard dependency.
"""

from __future__ import annotations

import math
from typing import Dict

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


def _best_achievable_q(n_tests: int, n_shuffles: int) -> float:
    """BH-adjusted p-value a single strongest-possible test can reach.

    Single-sourced so ``fdr_resolution``'s ``resolvable`` verdict and its
    ``recommended_n_shuffles`` search always agree.
    """
    return min(1.0, n_tests / (n_shuffles + 1))


def fdr_resolution(n_tests: int, n_shuffles: int, alpha: float = 0.05) -> Dict:
    """Can a permutation budget of ``n_shuffles`` resolve BH-FDR significance
    across ``n_tests`` simultaneous tests at level ``alpha``?

    A permutation test with ``n_shuffles`` draws cannot produce a p-value
    below ``p_floor = 1 / (n_shuffles + 1)`` (using the standard add-one
    estimator). Benjamini-Hochberg multiplies the smallest p-value by
    ``n_tests`` in the worst case, so a *single* strongly-tuned test can only
    reach ``q = n_tests * p_floor``. If that exceeds ``alpha``, no lone test
    can ever be called significant no matter how large its true effect —
    a null result is then a statement about the permutation budget, not the
    data.

    Two ways out, both reported here: raise ``n_shuffles`` to
    ``recommended_n_shuffles``, or rely on ``min_tests_at_floor`` — the
    number of tests that must *simultaneously* hit ``p_floor`` before BH's
    rank denominator lets any of them through. (Pooling nulls across tests,
    as ``null_mode='pooled'`` does in
    ``ephys._lda_decoding.compute_population_significance``, lowers
    ``p_floor`` instead, at the cost of assuming a shared null.)

    Returns a dict with ``p_floor``, ``best_achievable_q`` (for one lone
    strong test), ``min_tests_at_floor``, ``recommended_n_shuffles``,
    ``resolvable`` (bool: can a lone test reach ``q < alpha``), plus the
    ``n_tests``/``n_shuffles``/``alpha`` that were asked about.
    """
    n_tests = int(n_tests)
    n_shuffles = int(n_shuffles)
    if n_tests < 1 or n_shuffles < 1:
        raise ValueError(
            f"n_tests and n_shuffles must be >= 1, got {n_tests}, {n_shuffles}"
        )

    alpha = float(alpha)
    p_floor = 1.0 / (n_shuffles + 1)
    best_achievable_q = _best_achievable_q(n_tests, n_shuffles)

    # Smallest budget for which a lone strong test clears alpha strictly.
    # Searched with the *same* predicate used for `resolvable`, so the two can
    # never disagree at a floating-point boundary (e.g. 149/2980, which is
    # 0.05 in exact arithmetic but 0.049999999999999996 as a double).
    recommended = max(1, math.floor(n_tests / alpha) - 1)
    while _best_achievable_q(n_tests, recommended) >= alpha:
        recommended += 1

    return {
        'n_tests': n_tests,
        'n_shuffles': n_shuffles,
        'alpha': float(alpha),
        'p_floor': p_floor,
        'best_achievable_q': best_achievable_q,
        'min_tests_at_floor': min(n_tests, math.ceil(n_tests * p_floor / alpha)),
        'recommended_n_shuffles': recommended,
        'resolvable': bool(best_achievable_q < alpha),
    }


def empirical_p_value(observed: float, null_draws: np.ndarray) -> float:
    """One-tailed permutation p-value, ``P(null >= observed)``, add-one form.

    ``(1 + #exceedances) / (1 + n_valid_draws)``. The add-one ("plus-one")
    estimator is used throughout this repo because a finite permutation test
    cannot justify ``p == 0``: the plain ``k/n`` form returns exactly 0.0 when
    no draw beats the observed value, which then survives Benjamini-Hochberg
    as ``q == 0`` and reads as infinite confidence.

    Non-finite draws are **excluded from both numerator and denominator**.
    This matters for correctness, not tidiness: a degenerate cross-validation
    fold (common when events are few) yields a NaN score, and ``nan >= x``
    evaluates False, so naively counting exceedances would silently treat NaN
    draws as "did not exceed" and bias the p-value downward.

    Returns ``nan`` if ``observed`` is non-finite or no valid draws remain.
    """
    draws = np.asarray(null_draws, dtype=np.float64)
    valid = draws[np.isfinite(draws)]
    if not np.isfinite(observed) or valid.size == 0:
        return float('nan')
    return float((1 + int(np.sum(valid >= observed))) / (valid.size + 1))


def majority_class_baseline(labels: np.ndarray) -> float:
    """Accuracy achieved by always guessing the most common class.

    The honest baseline for a classifier scored with plain (unweighted)
    accuracy on imbalanced labels. ``1 / n_classes`` — used elsewhere in
    this repo as "chance" — understates it whenever classes are unbalanced:
    for a 12/7 split, always guessing the majority already scores 63.2%,
    not 50%.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return float('nan')
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())
