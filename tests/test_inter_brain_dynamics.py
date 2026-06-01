"""
Tests for ephys/inter_brain_dynamics.py.

Covers synthetic recovery of a planted shared subspace, shape contracts,
variance-partition algebra, NaN dropping, the T < N regime that requires
regularization, the sklearn-CCA path for well-conditioned cases, and the
``parameters`` dict carrying ``class_label`` / ``analysis_title`` per the
LDA-decoder convention.
"""

import numpy as np
import pytest

from ephys.inter_brain_dynamics import SharedSubspaceFit, fit_shared_subspace


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _generate_shared_data(T, N_A, N_B, K_true, noise_scale, seed):
    """Two firing-rate-like matrices sharing K_true latent factors plus noise."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((T, K_true))
    W_A = rng.standard_normal((K_true, N_A))
    W_B = rng.standard_normal((K_true, N_B))
    X_A = Z @ W_A + noise_scale * rng.standard_normal((T, N_A))
    X_B = Z @ W_B + noise_scale * rng.standard_normal((T, N_B))
    return X_A, X_B, W_A, W_B


def _subspace_cosines(U_est, U_true):
    """Cosines of principal angles between two subspace bases.

    Returns the K singular values of ``Q_est.T @ Q_true``, in decreasing
    order. All values in [0, 1]; min == 1 iff the spans coincide.
    """
    Q_est, _ = np.linalg.qr(U_est)
    Q_true, _ = np.linalg.qr(U_true)
    return np.linalg.svd(Q_est.T @ Q_true, compute_uv=False)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_recovers_planted_subspace_regularized(self):
        X_A, X_B, W_A, W_B = _generate_shared_data(
            T=2000, N_A=50, N_B=50, K_true=3, noise_scale=0.5, seed=0,
        )
        fit = fit_shared_subspace(X_A, X_B, n_components=3, method="regularized")
        assert _subspace_cosines(fit.U_A, W_A.T).min() > 0.9
        assert _subspace_cosines(fit.U_B, W_B.T).min() > 0.9

    def test_canonical_correlations_high_with_planted_structure(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=2000, N_A=50, N_B=50, K_true=3, noise_scale=0.5, seed=1,
        )
        fit = fit_shared_subspace(X_A, X_B, n_components=3, method="regularized")
        assert (fit.canonical_correlations["train"] > 0.85).all()
        assert (fit.canonical_correlations["cv_mean"] > 0.7).all()

    def test_no_shared_structure_yields_low_canonical_correlations(self):
        rng = np.random.default_rng(2)
        X_A = rng.standard_normal((2000, 50))
        X_B = rng.standard_normal((2000, 50))
        fit = fit_shared_subspace(X_A, X_B, n_components=3, method="regularized")
        # Train CCs may inflate from chance maximization but stay well below 1.
        assert (fit.canonical_correlations["train"] < 0.5).all()
        # Held-out CV should be near zero.
        assert (np.abs(fit.canonical_correlations["cv_mean"]) < 0.3).all()


class TestSklearnPaths:
    def test_sklearn_cca_recovers_planted_subspace(self):
        # sklearn CCA is well-conditioned only when T > N.
        X_A, X_B, W_A, _ = _generate_shared_data(
            T=2000, N_A=20, N_B=20, K_true=3, noise_scale=0.5, seed=12,
        )
        fit = fit_shared_subspace(X_A, X_B, n_components=3, method="cca")
        assert _subspace_cosines(fit.U_A, W_A.T).min() > 0.85

    def test_pls_runs(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=1000, N_A=20, N_B=20, K_true=3, noise_scale=0.5, seed=13,
        )
        fit = fit_shared_subspace(X_A, X_B, n_components=3, method="pls")
        assert fit.U_A.shape == (20, 3)
        # PLS maximizes covariance, not correlation, but the recovered
        # subspace should still correlate strongly with itself.
        assert (fit.canonical_correlations["train"] > 0.7).all()

    def test_gfa_raises_notimplemented(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=200, N_A=10, N_B=10, K_true=2, noise_scale=0.5, seed=14,
        )
        with pytest.raises(NotImplementedError):
            fit_shared_subspace(X_A, X_B, n_components=2, method="gfa")


# ---------------------------------------------------------------------------
# Shape / contract tests
# ---------------------------------------------------------------------------

class TestShapes:
    def test_output_shapes(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=200, N_A=20, N_B=15, K_true=3, noise_scale=0.5, seed=3,
        )
        fit = fit_shared_subspace(X_A, X_B, n_components=3, method="regularized")
        assert isinstance(fit, SharedSubspaceFit)
        assert fit.U_A.shape == (20, 3)
        assert fit.U_B.shape == (15, 3)
        assert fit.S_A.shape == (200, 3)
        assert fit.S_B.shape == (200, 3)
        assert fit.V_A_unique.shape == (20, 17)
        assert fit.V_B_unique.shape == (15, 12)
        assert fit.canonical_correlations["train"].shape == (3,)
        assert fit.canonical_correlations["cv"].shape == (5, 3)
        assert fit.canonical_correlations["cv_mean"].shape == (3,)
        assert fit.n_components == 3
        assert fit.valid_mask.shape == (200,)
        assert fit.valid_mask.all()


# ---------------------------------------------------------------------------
# Variance partition
# ---------------------------------------------------------------------------

class TestVariancePartition:
    def test_sums_to_one_zscored(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=500, N_A=20, N_B=20, K_true=3, noise_scale=0.5, seed=4,
        )
        fit = fit_shared_subspace(X_A, X_B, n_components=3, method="regularized")
        v = fit.variance_partition
        assert v["shared_var_A_z"] + v["unique_var_A_z"] == pytest.approx(1.0, abs=1e-9)
        assert v["shared_var_B_z"] + v["unique_var_B_z"] == pytest.approx(1.0, abs=1e-9)
        assert 0.0 < v["shared_var_A_z"] < 1.0
        assert 0.0 < v["shared_var_B_z"] < 1.0
        # Headline aliases exist.
        assert v["shared_var_A"] == v["shared_var_A_z"]
        assert v["unique_var_A"] == v["unique_var_A_z"]


# ---------------------------------------------------------------------------
# Unique subspace
# ---------------------------------------------------------------------------

class TestUniqueSubspace:
    def test_unique_basis_orthogonal_to_shared(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=200, N_A=20, N_B=20, K_true=3, noise_scale=0.5, seed=5,
        )
        fit = fit_shared_subspace(X_A, X_B, n_components=3, method="regularized")
        Q_A, _ = np.linalg.qr(fit.U_A)
        np.testing.assert_allclose(Q_A.T @ fit.V_A_unique, 0.0, atol=1e-9)

    def test_unique_basis_columns_are_orthonormal(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=200, N_A=20, N_B=20, K_true=3, noise_scale=0.5, seed=6,
        )
        fit = fit_shared_subspace(X_A, X_B, n_components=3, method="regularized")
        VtV = fit.V_A_unique.T @ fit.V_A_unique
        np.testing.assert_allclose(VtV, np.eye(VtV.shape[0]), atol=1e-9)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_nan_bins_are_dropped(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=200, N_A=20, N_B=20, K_true=3, noise_scale=0.5, seed=7,
        )
        X_A_nan = X_A.copy()
        X_A_nan[10:15] = np.nan
        fit = fit_shared_subspace(X_A_nan, X_B, n_components=3, method="regularized")
        assert fit.parameters["T_input"] == 200
        assert fit.parameters["T_valid"] == 195
        assert (~fit.valid_mask).sum() == 5
        assert fit.S_A.shape == (195, 3)
        assert fit.S_B.shape == (195, 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_T_smaller_than_N_with_regularization(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=80, N_A=100, N_B=100, K_true=2, noise_scale=0.3, seed=8,
        )
        fit = fit_shared_subspace(
            X_A, X_B, n_components=2, method="regularized", reg=1e-2, cv_folds=3,
        )
        assert fit.U_A.shape == (100, 2)
        # Train CCs will be inflated by overfitting but should be high.
        assert (fit.canonical_correlations["train"] > 0.7).all()

    def test_n_components_default_clipped_by_min_N(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=200, N_A=5, N_B=8, K_true=3, noise_scale=0.5, seed=9,
        )
        fit = fit_shared_subspace(X_A, X_B, method="regularized")
        assert fit.n_components == 5

    def test_n_components_default_clipped_by_T(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=20, N_A=50, N_B=50, K_true=2, noise_scale=0.5, seed=10,
        )
        fit = fit_shared_subspace(X_A, X_B, method="regularized", cv_folds=2)
        # T // 4 = 5.
        assert fit.n_components == 5

    def test_n_components_too_large_raises(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=200, N_A=10, N_B=10, K_true=2, noise_scale=0.5, seed=11,
        )
        with pytest.raises(ValueError, match="exceeds"):
            fit_shared_subspace(X_A, X_B, n_components=15, method="regularized")

    def test_mismatched_T_raises(self):
        rng = np.random.default_rng(15)
        X_A = rng.standard_normal((100, 5))
        X_B = rng.standard_normal((200, 5))
        with pytest.raises(ValueError, match="Time dimension"):
            fit_shared_subspace(X_A, X_B, n_components=2)

    def test_one_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            fit_shared_subspace(np.zeros(10), np.zeros(10))


# ---------------------------------------------------------------------------
# Parameters dict — LDA-decoder convention
# ---------------------------------------------------------------------------

class TestParametersDict:
    def test_class_label_and_analysis_title_present(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=200, N_A=10, N_B=10, K_true=2, noise_scale=0.5, seed=16,
        )
        fit = fit_shared_subspace(
            X_A, X_B, n_components=2, method="regularized",
            animal_ids=("631", "632"),
            t_window=(0.0, 100.0),
            bin_size_sec=0.5,
            smoothing_sigma_sec=0.25,
        )
        assert "class_label" in fit.parameters
        assert "analysis_title" in fit.parameters
        assert fit.parameters["analysis_title"] == "Inter-brain shared subspace"
        assert fit.parameters["animal_ids"] == ("631", "632")
        assert fit.parameters["t_window"] == (0.0, 100.0)
        assert fit.parameters["bin_size_sec"] == 0.5
        assert fit.parameters["smoothing_sigma_sec"] == 0.25
        assert fit.parameters["n_components"] == 2
        assert fit.parameters["method"] == "regularized"
