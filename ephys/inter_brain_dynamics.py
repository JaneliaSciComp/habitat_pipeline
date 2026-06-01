"""
Inter-brain neural dynamics — shared subspace via canonical correlation.

Given two simultaneously-recorded firing-rate time series ``X_A`` (T × N_A)
and ``X_B`` (T × N_B), find a K-dimensional shared subspace whose dynamics
correlate across brains, and the orthogonal "unique" subspace in each
animal's cell space. The implementation defaults to a ridge-whitened SVD
of the cross-covariance ("regularized") because RatCity recordings
routinely have N comparable to or larger than T after quality filtering,
where ``sklearn.cross_decomposition.CCA`` is numerically unstable.

Reference: Zhang, Phi, Li et al., "Inter-brain neural dynamics in
biological and artificial intelligence systems." Nature 645, 991–1001
(2025). DOI: 10.1038/s41586-025-09196-4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


__all__ = ["SharedSubspaceFit", "fit_shared_subspace"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SharedSubspaceFit:
    """Result of fitting a shared subspace via (regularized) CCA.

    Attributes
    ----------
    U_A, U_B : np.ndarray
        Cell-space → shared-subspace projection matrices, shapes (N_A, K)
        and (N_B, K). ``S_A = X_A_z @ U_A`` reproduces the shared time
        courses on z-scored inputs.
    S_A, S_B : np.ndarray
        Shared-dim time courses, shape (T_valid, K) each, computed from
        the z-scored data on the full (NaN-dropped) sample.
    V_A_unique, V_B_unique : np.ndarray
        Orthonormal bases of each animal's unique subspace in cell space.
        Shapes (N_A, N_A - K) and (N_B, N_B - K) when rank-full; fewer
        columns when T_valid < N - K.
    canonical_correlations : dict
        ``{'train': (K,), 'cv': (n_folds, K), 'cv_mean': (K,),
        'cv_std': (K,)}``. Pearson correlations between matched shared
        dims on the full sample and per held-out CV fold.
    variance_partition : dict
        ``shared_var_A_z`` / ``unique_var_A_z`` (and equivalents for B
        and for raw, mean-centered rates). The z-scored partition sums
        to 1 by construction; ``shared_var_A`` etc. alias the z-scored
        entries for the headline numbers.
    parameters : dict
        Echoes all inputs plus ``class_label`` and ``analysis_title`` so
        plotting helpers can drive titles off the dataclass (same
        convention as ``_lda_decoding.py``).
    valid_mask : np.ndarray
        Boolean mask over input bins; True for bins kept after NaN
        filtering. Shape (T_input,).
    """

    U_A: np.ndarray
    U_B: np.ndarray
    S_A: np.ndarray
    S_B: np.ndarray
    V_A_unique: np.ndarray
    V_B_unique: np.ndarray
    canonical_correlations: Dict[str, np.ndarray]
    variance_partition: Dict[str, float]
    parameters: Dict
    valid_mask: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.U_A.shape[1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_shared_subspace(
    X_A: np.ndarray,
    X_B: np.ndarray,
    n_components: Optional[int] = None,
    reg: float = 1e-3,
    method: str = "regularized",
    cv_folds: int = 5,
    animal_ids: Optional[Tuple[str, str]] = None,
    t_window: Optional[Tuple[float, float]] = None,
    bin_size_sec: Optional[float] = None,
    smoothing_sigma_sec: Optional[float] = None,
) -> SharedSubspaceFit:
    """Fit a shared subspace between two simultaneously-recorded animals.

    Parameters
    ----------
    X_A, X_B : np.ndarray
        Firing-rate time series of shape (T, N_A) and (T, N_B). Must share
        the same time dimension. Z-scoring is applied internally (and on
        training folds only inside CV); do not pre-z-score.
    n_components : int, optional
        Shared subspace dimensionality. If None, defaults to
        ``max(1, min(N_A, N_B, T // 4, 10))``. For principled selection
        use :func:`choose_n_components` (forthcoming).
    reg : float
        Ridge added to per-animal cell-covariance before whitening (the
        "regularized" method only). Stabilizes inversion when N ≳ T.
    method : {"regularized", "cca", "pls"}
        Fit method. "regularized" (default) uses a ridge-whitened SVD of
        the cross-covariance and is robust when N is comparable to or
        larger than T. "cca" and "pls" use sklearn's ``CCA`` and
        ``PLSCanonical``; both can produce NaN loadings when N > T.
    cv_folds : int
        Number of contiguous time-series CV folds (no shuffling). With
        autocorrelated rates, shuffled K-fold would leak across folds.
    animal_ids, t_window, bin_size_sec, smoothing_sigma_sec
        Echoed into ``parameters`` for plotting and provenance.

    Returns
    -------
    SharedSubspaceFit
    """
    X_A = np.asarray(X_A, dtype=np.float64)
    X_B = np.asarray(X_B, dtype=np.float64)
    if X_A.ndim != 2 or X_B.ndim != 2:
        raise ValueError("X_A and X_B must be 2-D (T, N)")
    if X_A.shape[0] != X_B.shape[0]:
        raise ValueError(
            f"Time dimension mismatch: X_A T={X_A.shape[0]}, X_B T={X_B.shape[0]}"
        )

    T_input, N_A = X_A.shape
    _, N_B = X_B.shape

    valid_mask = ~(np.isnan(X_A).any(axis=1) | np.isnan(X_B).any(axis=1))
    if not valid_mask.all():
        logger.warning(
            "Dropping %d/%d bins with NaNs before fitting",
            int((~valid_mask).sum()), T_input,
        )
    X_A_clean = X_A[valid_mask]
    X_B_clean = X_B[valid_mask]
    T = X_A_clean.shape[0]
    if T < cv_folds + 1:
        raise ValueError(
            f"Not enough valid bins ({T}) for cv_folds={cv_folds}"
        )

    if n_components is None:
        n_components = max(1, min(N_A, N_B, T // 4, 10))
    elif n_components < 1:
        raise ValueError("n_components must be ≥ 1")
    elif n_components > min(N_A, N_B):
        raise ValueError(
            f"n_components={n_components} exceeds min(N_A, N_B)={min(N_A, N_B)}"
        )

    X_A_z, _, _ = _zscore(X_A_clean)
    X_B_z, _, _ = _zscore(X_B_clean)

    U_A, U_B, _ = _fit_method(X_A_z, X_B_z, n_components, method, reg)

    S_A = X_A_z @ U_A
    S_B = X_B_z @ U_B
    train_ccs = _pearson_per_column(S_A, S_B)

    cv_ccs = _cross_validate(X_A_clean, X_B_clean, n_components, method, reg, cv_folds)

    var_partition = _compute_variance_partition(
        X_A_z, X_B_z, X_A_clean, X_B_clean, U_A, U_B,
    )

    Q_A, _ = np.linalg.qr(U_A)
    Q_B, _ = np.linalg.qr(U_B)
    V_A_unique = _unique_basis(X_A_z, Q_A)
    V_B_unique = _unique_basis(X_B_z, Q_B)

    parameters = {
        "n_components": int(n_components),
        "reg": reg,
        "method": method,
        "cv_folds": cv_folds,
        "bin_size_sec": bin_size_sec,
        "smoothing_sigma_sec": smoothing_sigma_sec,
        "t_window": t_window,
        "animal_ids": tuple(animal_ids) if animal_ids is not None else None,
        "N_A": int(N_A),
        "N_B": int(N_B),
        "T_input": int(T_input),
        "T_valid": int(T),
        "class_label": "Shared dim",
        "analysis_title": "Inter-brain shared subspace",
    }

    return SharedSubspaceFit(
        U_A=U_A, U_B=U_B,
        S_A=S_A, S_B=S_B,
        V_A_unique=V_A_unique, V_B_unique=V_B_unique,
        canonical_correlations={
            "train": train_ccs,
            "cv": cv_ccs,
            "cv_mean": np.nanmean(cv_ccs, axis=0),
            "cv_std": np.nanstd(cv_ccs, axis=0),
        },
        variance_partition=var_partition,
        parameters=parameters,
        valid_mask=valid_mask,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _zscore(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, ddof=1, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1.0)
    return (X - mu) / sigma, mu, sigma


def _pearson_per_column(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation between two (T, K) matrices."""
    A_c = A - A.mean(axis=0, keepdims=True)
    B_c = B - B.mean(axis=0, keepdims=True)
    num = (A_c * B_c).sum(axis=0)
    den = np.sqrt((A_c ** 2).sum(axis=0) * (B_c ** 2).sum(axis=0))
    den = np.where(den > 0, den, 1.0)
    return num / den


def _inv_sqrt_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute M^(-1/2) for a symmetric PSD matrix via eigendecomposition."""
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def _fit_regularized_cca(
    X_A: np.ndarray, X_B: np.ndarray, K: int, reg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge-whitened SVD of the cross-covariance.

    Returns ``(U_A, U_B, sigmas)`` where ``U_A`` has shape (N_A, K),
    ``U_B`` has shape (N_B, K), and ``sigmas`` are the top-K singular
    values of the whitened cross-covariance (≈ canonical correlations
    when reg is small relative to data variance).
    """
    T = X_A.shape[0]
    denom = max(T - 1, 1)
    C_AA = (X_A.T @ X_A) / denom
    C_BB = (X_B.T @ X_B) / denom
    C_AB = (X_A.T @ X_B) / denom
    W_A = _inv_sqrt_psd(C_AA + reg * np.eye(C_AA.shape[0]))
    W_B = _inv_sqrt_psd(C_BB + reg * np.eye(C_BB.shape[0]))
    M = W_A @ C_AB @ W_B
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    U_A = W_A @ U[:, :K]
    U_B = W_B @ Vt[:K, :].T
    return U_A, U_B, np.clip(S[:K], 0.0, 1.0)


def _fit_method(
    X_A: np.ndarray, X_B: np.ndarray, K: int, method: str, reg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "regularized":
        return _fit_regularized_cca(X_A, X_B, K, reg)
    if method == "cca":
        from sklearn.cross_decomposition import CCA
        m = CCA(n_components=K, max_iter=2000)
        m.fit(X_A, X_B)
        U_A = np.asarray(m.x_rotations_)
        U_B = np.asarray(m.y_rotations_)
        return U_A, U_B, _pearson_per_column(X_A @ U_A, X_B @ U_B)
    if method == "pls":
        from sklearn.cross_decomposition import PLSCanonical
        m = PLSCanonical(n_components=K, max_iter=2000)
        m.fit(X_A, X_B)
        U_A = np.asarray(m.x_rotations_)
        U_B = np.asarray(m.y_rotations_)
        return U_A, U_B, _pearson_per_column(X_A @ U_A, X_B @ U_B)
    if method == "gfa":
        raise NotImplementedError(
            "GFA / probabilistic CCA is a stretch goal — use 'regularized' or 'cca'."
        )
    raise ValueError(f"Unknown method: {method!r}")


def _cross_validate(
    X_A: np.ndarray, X_B: np.ndarray, K: int, method: str, reg: float, n_folds: int,
) -> np.ndarray:
    """Contiguous-block time-series CV (no shuffling).

    For each fold, z-score on the training portion only, fit, then project
    the held-out test bins and compute per-column Pearson correlation of
    matched shared dims. Returns an (n_folds, K) array.
    """
    T = X_A.shape[0]
    fold_size = T // n_folds
    cv_ccs = np.full((n_folds, K), np.nan)
    for f in range(n_folds):
        test_start = f * fold_size
        test_end = (f + 1) * fold_size if f < n_folds - 1 else T
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate(
            (np.arange(0, test_start), np.arange(test_end, T))
        )
        if len(train_idx) < K + 2 or len(test_idx) < 2:
            continue

        mu_A = X_A[train_idx].mean(axis=0)
        sd_A = X_A[train_idx].std(axis=0, ddof=1)
        sd_A = np.where(sd_A > 0, sd_A, 1.0)
        mu_B = X_B[train_idx].mean(axis=0)
        sd_B = X_B[train_idx].std(axis=0, ddof=1)
        sd_B = np.where(sd_B > 0, sd_B, 1.0)

        X_A_tr_z = (X_A[train_idx] - mu_A) / sd_A
        X_B_tr_z = (X_B[train_idx] - mu_B) / sd_B
        X_A_te_z = (X_A[test_idx] - mu_A) / sd_A
        X_B_te_z = (X_B[test_idx] - mu_B) / sd_B

        try:
            U_A, U_B, _ = _fit_method(X_A_tr_z, X_B_tr_z, K, method, reg)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("CV fold %d failed: %s", f, e)
            continue
        cv_ccs[f] = _pearson_per_column(X_A_te_z @ U_A, X_B_te_z @ U_B)
    return cv_ccs


def _compute_variance_partition(
    X_A_z: np.ndarray, X_B_z: np.ndarray,
    X_A_raw: np.ndarray, X_B_raw: np.ndarray,
    U_A: np.ndarray, U_B: np.ndarray,
) -> Dict[str, float]:
    """Variance partition using QR-orthonormalized loadings.

    CCA loadings are not column-orthonormal in cell space, so a literal
    ``U_A U_A^T`` is not the orthogonal projector onto span(U_A); take
    ``Q_A`` from QR of ``U_A`` and use ``Q_A Q_A^T`` instead. With the
    orthonormal projector the shared and unique fractions sum to 1 by
    construction.
    """
    Q_A, _ = np.linalg.qr(U_A)
    Q_B, _ = np.linalg.qr(U_B)

    def _frac(X: np.ndarray, Q: np.ndarray) -> float:
        total = float((X ** 2).sum())
        if total == 0:
            return 0.0
        proj = X @ Q
        return float((proj ** 2).sum() / total)

    X_A_c = X_A_raw - X_A_raw.mean(axis=0, keepdims=True)
    X_B_c = X_B_raw - X_B_raw.mean(axis=0, keepdims=True)

    out = {
        "shared_var_A_z": _frac(X_A_z, Q_A),
        "shared_var_B_z": _frac(X_B_z, Q_B),
        "shared_var_A_raw": _frac(X_A_c, Q_A),
        "shared_var_B_raw": _frac(X_B_c, Q_B),
    }
    out["unique_var_A_z"] = 1.0 - out["shared_var_A_z"]
    out["unique_var_B_z"] = 1.0 - out["shared_var_B_z"]
    out["unique_var_A_raw"] = 1.0 - out["shared_var_A_raw"]
    out["unique_var_B_raw"] = 1.0 - out["shared_var_B_raw"]
    out["shared_var_A"] = out["shared_var_A_z"]
    out["shared_var_B"] = out["shared_var_B_z"]
    out["unique_var_A"] = out["unique_var_A_z"]
    out["unique_var_B"] = out["unique_var_B_z"]
    return out


def _unique_basis(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the unique subspace in cell space.

    Residualize ``X`` against the shared subspace projector ``Q Q^T``, then
    take the right singular vectors of the residual. The first ``N - K``
    are an orthonormal basis of the orthogonal complement of ``Q`` (up to
    rank ``min(T, N - K)``).
    """
    N, K = Q.shape
    P_perp = np.eye(N) - Q @ Q.T
    R = X @ P_perp
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    rank = min(R.shape[0], N - K)
    return Vt[:rank].T
