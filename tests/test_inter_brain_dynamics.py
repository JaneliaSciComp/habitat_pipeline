"""
Tests for ephys/inter_brain_dynamics.py.

Covers synthetic recovery of a planted shared subspace, shape contracts,
variance-partition algebra, NaN dropping, the T < N regime that requires
regularization, the sklearn-CCA path for well-conditioned cases, and the
``parameters`` dict carrying ``class_label`` / ``analysis_title`` per the
LDA-decoder convention.
"""

import numpy as np
import pandas as pd
import pytest

from ephys.inter_brain_dynamics import (
    SharedSubspaceFit,
    choose_n_components,
    cross_animal_correlation_matrix,
    fit_shared_subspace,
    project_onto_shared,
    regress_shared_on_behavior,
    shuffle_null_subspace,
    time_lagged_cca,
)


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


# ---------------------------------------------------------------------------
# Shuffle null
# ---------------------------------------------------------------------------

class TestShuffleNull:
    def test_null_shape_and_dtype(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=400, N_A=20, N_B=20, K_true=3, noise_scale=0.5, seed=20,
        )
        null = shuffle_null_subspace(
            X_A, X_B, n_components=3, n_shuffles=15, seed=0,
        )
        assert null.shape == (15, 3)
        assert null.dtype == np.float64

    def test_seed_reproducibility(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=400, N_A=20, N_B=20, K_true=3, noise_scale=0.5, seed=21,
        )
        null1 = shuffle_null_subspace(X_A, X_B, n_components=3, n_shuffles=15, seed=42)
        null2 = shuffle_null_subspace(X_A, X_B, n_components=3, n_shuffles=15, seed=42)
        np.testing.assert_array_equal(null1, null2)

    def test_observed_exceeds_null_with_planted_structure(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=2000, N_A=30, N_B=30, K_true=3, noise_scale=0.5, seed=22,
        )
        fit = fit_shared_subspace(X_A, X_B, n_components=3, method="regularized")
        null = shuffle_null_subspace(
            X_A, X_B, n_components=3, n_shuffles=50, seed=22,
        )
        # Top observed CC should be far above the top null CCs.
        observed_top = fit.canonical_correlations["train"][0]
        null_top = null[:, 0]
        margin = (observed_top - null_top.mean()) / (null_top.std() + 1e-12)
        assert margin > 3.0, f"observed not >3 SD above null (got {margin:.2f})"
        assert observed_top > np.percentile(null_top, 95)

    def test_null_distribution_independent_of_observed_with_no_structure(self):
        rng = np.random.default_rng(23)
        X_A = rng.standard_normal((1500, 20))
        X_B = rng.standard_normal((1500, 20))
        fit = fit_shared_subspace(X_A, X_B, n_components=2, method="regularized")
        null = shuffle_null_subspace(
            X_A, X_B, n_components=2, n_shuffles=50, seed=23,
        )
        observed_top = fit.canonical_correlations["train"][0]
        null_top = null[:, 0]
        # With no real structure, observed should fall well within null bulk.
        assert observed_top < np.percentile(null_top, 99)

    def test_unsupported_kind_raises(self):
        rng = np.random.default_rng(24)
        X_A = rng.standard_normal((200, 5))
        X_B = rng.standard_normal((200, 5))
        with pytest.raises(NotImplementedError):
            shuffle_null_subspace(X_A, X_B, n_components=2, n_shuffles=5,
                                  kind="permute")


# ---------------------------------------------------------------------------
# choose_n_components
# ---------------------------------------------------------------------------

class TestChooseNComponents:
    def test_recovers_planted_K(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=2000, N_A=30, N_B=30, K_true=3, noise_scale=0.3, seed=30,
        )
        result = choose_n_components(
            X_A, X_B, max_K=6, n_shuffles=40, cv_folds=3, seed=30,
        )
        # The prompt's selection rule (train-CC > 95th-pctl null) has a 5%
        # per-dim false-positive rate, so recommended_K can drift above
        # K_true when noise-dim CCs marginally exceed the null. The robust
        # checks are: (a) recommend at least K_true, (b) the first K_true
        # train CCs are clearly large and (c) cv_mean separates signal
        # from noise cleanly.
        assert result["recommended_K"] >= 3
        assert (result["train_ccs"][:3] > 0.9).all()
        assert (result["train_ccs"][:3] > result["shuffle_p95"][:3] + 0.3).all()
        assert (result["cv_mean"][:3] > 0.9).all()
        assert (np.abs(result["cv_mean"][3:]) < 0.3).all()
        assert result["train_ccs"].shape == (6,)
        assert result["shuffle_null"].shape == (40, 6)
        assert result["shuffle_p95"].shape == (6,)
        # CV-based recommendation is the conservative selector and must
        # pin K_true exactly when noise-dim cv_mean is near zero.
        assert result["recommended_K_cv"] == 3

    def test_zero_K_with_no_structure(self):
        rng = np.random.default_rng(31)
        X_A = rng.standard_normal((1000, 20))
        X_B = rng.standard_normal((1000, 20))
        result = choose_n_components(
            X_A, X_B, max_K=5, n_shuffles=40, cv_folds=3, seed=31,
        )
        # With no real structure, both recommendations should be small;
        # the CV-based one should be 0 since cv_mean is near zero.
        assert result["recommended_K"] <= 2
        assert result["recommended_K_cv"] == 0

    def test_parameters_echoed(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=400, N_A=10, N_B=10, K_true=2, noise_scale=0.3, seed=32,
        )
        result = choose_n_components(
            X_A, X_B, max_K=4, n_shuffles=10, cv_folds=2, seed=99, reg=1e-2,
        )
        p = result["parameters"]
        assert p["max_K"] == 4
        assert p["n_shuffles"] == 10
        assert p["cv_folds"] == 2
        assert p["seed"] == 99
        assert p["reg"] == 1e-2


# ---------------------------------------------------------------------------
# project_onto_shared
# ---------------------------------------------------------------------------

class TestProjectOntoShared:
    def test_shape(self):
        X = np.zeros((100, 10))
        U = np.zeros((10, 3))
        S = project_onto_shared(X, U)
        assert S.shape == (100, 3)

    def test_matches_S_A_on_z_scored_input(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=300, N_A=15, N_B=15, K_true=2, noise_scale=0.5, seed=40,
        )
        fit = fit_shared_subspace(X_A, X_B, n_components=2, method="regularized")
        mu = X_A.mean(axis=0, keepdims=True)
        sd = X_A.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd > 0, sd, 1.0)
        X_A_z = (X_A - mu) / sd
        S_proj = project_onto_shared(X_A_z, fit.U_A)
        np.testing.assert_allclose(S_proj, fit.S_A, atol=1e-9)


# ---------------------------------------------------------------------------
# time_lagged_cca
# ---------------------------------------------------------------------------

class TestTimeLaggedCCA:
    def test_peak_at_planted_lag(self):
        rng = np.random.default_rng(50)
        T = 800
        delay = 5  # X_A reads Z[t+delay] → X_A leads X_B by `delay`
        N_A = N_B = 20
        K_true = 2
        Z = rng.standard_normal((T + delay + 5, K_true))
        W_A = rng.standard_normal((K_true, N_A))
        W_B = rng.standard_normal((K_true, N_B))
        X_A = Z[delay : delay + T] @ W_A + 0.2 * rng.standard_normal((T, N_A))
        X_B = Z[:T] @ W_B + 0.2 * rng.standard_normal((T, N_B))

        lags, ccs = time_lagged_cca(
            X_A, X_B, max_lag_bins=10, n_components=2, method="regularized",
        )
        # X_A[t] uses Z[t+delay], X_B[t-lag] uses Z[t-lag] → match at lag = -delay.
        peak_lag = lags[np.nanargmax(ccs[:, 0])]
        assert peak_lag == -delay

    def test_output_shapes(self):
        X_A, X_B, _, _ = _generate_shared_data(
            T=300, N_A=10, N_B=10, K_true=2, noise_scale=0.5, seed=51,
        )
        lags, ccs = time_lagged_cca(
            X_A, X_B, max_lag_bins=5, n_components=2, method="regularized",
        )
        assert lags.shape == (11,)
        assert ccs.shape == (11, 2)


# ---------------------------------------------------------------------------
# cross_animal_correlation_matrix
# ---------------------------------------------------------------------------

class TestCrossAnimalCorrelation:
    def test_shape(self):
        rng = np.random.default_rng(60)
        X_A = rng.standard_normal((200, 7))
        X_B = rng.standard_normal((200, 11))
        C = cross_animal_correlation_matrix(X_A, X_B)
        assert C.shape == (7, 11)

    def test_self_correlation_diagonal_is_one(self):
        rng = np.random.default_rng(61)
        X = rng.standard_normal((1000, 5))
        C = cross_animal_correlation_matrix(X, X)
        np.testing.assert_allclose(np.diag(C), np.ones(5), atol=1e-9)

    def test_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(62)
        X_A = rng.standard_normal((500, 4))
        X_B = rng.standard_normal((500, 6))
        C = cross_animal_correlation_matrix(X_A, X_B)
        # Build reference via np.corrcoef on stacked columns.
        ref = np.corrcoef(X_A.T, X_B.T)[:4, 4:]
        np.testing.assert_allclose(C, ref, atol=1e-12)

    def test_mismatched_T_raises(self):
        rng = np.random.default_rng(63)
        X_A = rng.standard_normal((100, 5))
        X_B = rng.standard_normal((200, 5))
        with pytest.raises(ValueError, match="Time dimension"):
            cross_animal_correlation_matrix(X_A, X_B)


# ---------------------------------------------------------------------------
# regress_shared_on_behavior
# ---------------------------------------------------------------------------

def _make_regression_setup(seed: int):
    """Build (S_A, S_B, behavior_A, behavior_B, T) with known structure.

    S_A dim 0 is driven by A's "speed"; dim 1 by B's "speed"; dim 2 by both;
    dim 3 is independent noise. S_B mirrors the dependency on swapped sides.
    """
    rng = np.random.default_rng(seed)
    T = 600
    regs = [rng.standard_normal(T) for _ in range(6)]
    behavior_A = pd.DataFrame({
        "speed": regs[0],
        "angular_speed": regs[1],
        "distance": regs[2],
    })
    behavior_B = pd.DataFrame({
        "speed": regs[3],
        "angular_speed": regs[4],
        "distance": regs[5],
    })
    noise = 0.3
    S_A = np.column_stack([
        regs[0] + noise * rng.standard_normal(T),
        regs[3] + noise * rng.standard_normal(T),
        regs[0] + regs[3] + noise * rng.standard_normal(T),
        rng.standard_normal(T),
    ])
    S_B = np.column_stack([
        regs[3] + noise * rng.standard_normal(T),
        regs[0] + noise * rng.standard_normal(T),
        regs[3] + regs[0] + noise * rng.standard_normal(T),
        rng.standard_normal(T),
    ])
    return S_A, S_B, behavior_A, behavior_B, T


def _make_fit_for_regression(
    S_A: np.ndarray, S_B: np.ndarray, T: int,
    animal_ids=("631", "632"), valid_mask=None,
) -> SharedSubspaceFit:
    if valid_mask is None:
        valid_mask = np.ones(T, dtype=bool)
    K = S_A.shape[1]
    return SharedSubspaceFit(
        U_A=np.zeros((1, K)),
        U_B=np.zeros((1, K)),
        S_A=S_A, S_B=S_B,
        V_A_unique=np.zeros((1, 0)),
        V_B_unique=np.zeros((1, 0)),
        canonical_correlations={
            "train": np.zeros(K), "cv": np.zeros((1, K)),
            "cv_mean": np.zeros(K), "cv_std": np.zeros(K),
        },
        variance_partition={},
        parameters={
            "animal_ids": animal_ids,
            "class_label": "Shared dim",
            "analysis_title": "Test",
        },
        valid_mask=valid_mask,
    )


class TestRegressSharedOnBehavior:
    def test_self_dominated_dim(self):
        S_A, S_B, beh_A, beh_B, T = _make_regression_setup(seed=70)
        fit = _make_fit_for_regression(S_A, S_B, T)
        res = regress_shared_on_behavior(
            fit, {"631": beh_A, "632": beh_B}, alpha=1.0, cv_folds=5,
        )
        d0 = res["631"][0]
        assert d0["R2_self"] > 0.7
        assert d0["R2_partner"] < 0.2
        assert d0["R2_self_unique"] > 0.5
        assert d0["R2_partner_unique"] < 0.2

    def test_partner_dominated_dim(self):
        S_A, S_B, beh_A, beh_B, T = _make_regression_setup(seed=71)
        fit = _make_fit_for_regression(S_A, S_B, T)
        res = regress_shared_on_behavior(
            fit, {"631": beh_A, "632": beh_B}, alpha=1.0, cv_folds=5,
        )
        d1 = res["631"][1]
        assert d1["R2_partner"] > 0.7
        assert d1["R2_self"] < 0.2
        assert d1["R2_partner_unique"] > 0.5
        assert d1["R2_self_unique"] < 0.2

    def test_mixed_dim_both_above_either(self):
        S_A, S_B, beh_A, beh_B, T = _make_regression_setup(seed=72)
        fit = _make_fit_for_regression(S_A, S_B, T)
        res = regress_shared_on_behavior(
            fit, {"631": beh_A, "632": beh_B}, alpha=1.0, cv_folds=5,
        )
        d2 = res["631"][2]
        assert d2["R2_both"] > d2["R2_self"] + 0.1
        assert d2["R2_both"] > d2["R2_partner"] + 0.1
        # Both unique components should be substantial.
        assert d2["R2_partner_unique"] > 0.3
        assert d2["R2_self_unique"] > 0.3

    def test_noise_dim_low_R2(self):
        S_A, S_B, beh_A, beh_B, T = _make_regression_setup(seed=73)
        fit = _make_fit_for_regression(S_A, S_B, T)
        res = regress_shared_on_behavior(
            fit, {"631": beh_A, "632": beh_B}, alpha=1.0, cv_folds=5,
        )
        d3 = res["631"][3]
        assert d3["R2_self"] < 0.1
        assert d3["R2_partner"] < 0.1
        assert d3["R2_both"] < 0.1

    def test_animal_B_results_mirror_A(self):
        S_A, S_B, beh_A, beh_B, T = _make_regression_setup(seed=74)
        fit = _make_fit_for_regression(S_A, S_B, T)
        res = regress_shared_on_behavior(
            fit, {"631": beh_A, "632": beh_B}, alpha=1.0, cv_folds=5,
        )
        # S_B dim 0 is driven by B's own speed (animal 632's "self").
        d0_B = res["632"][0]
        assert d0_B["R2_self"] > 0.7
        assert d0_B["R2_partner"] < 0.2

    def test_missing_animal_raises(self):
        S_A, S_B, beh_A, _, T = _make_regression_setup(seed=75)
        fit = _make_fit_for_regression(S_A, S_B, T)
        with pytest.raises(KeyError, match="632"):
            regress_shared_on_behavior(
                fit, {"631": beh_A}, alpha=1.0, cv_folds=5,
            )

    def test_no_animal_ids_in_fit_raises(self):
        S_A, S_B, beh_A, beh_B, T = _make_regression_setup(seed=76)
        fit = _make_fit_for_regression(S_A, S_B, T, animal_ids=None)
        with pytest.raises(ValueError, match="animal_ids"):
            regress_shared_on_behavior(
                fit, {"631": beh_A, "632": beh_B}, alpha=1.0, cv_folds=5,
            )

    def test_output_dict_structure(self):
        S_A, S_B, beh_A, beh_B, T = _make_regression_setup(seed=77)
        fit = _make_fit_for_regression(S_A, S_B, T)
        res = regress_shared_on_behavior(
            fit, {"631": beh_A, "632": beh_B}, alpha=1.0, cv_folds=5,
        )
        assert set(res.keys()) >= {"631", "632", "parameters", "feature_names"}
        K = S_A.shape[1]
        for aid in ("631", "632"):
            assert set(res[aid].keys()) == set(range(K))
            for k in range(K):
                assert set(res[aid][k].keys()) == {
                    "R2_self", "R2_partner", "R2_both",
                    "R2_partner_unique", "R2_self_unique",
                }
        assert res["parameters"]["alpha"] == 1.0
        assert res["parameters"]["cv_folds"] == 5
        assert res["parameters"]["animal_ids"] == ("631", "632")
        assert res["parameters"]["class_label"] == "Shared dim"
        assert res["feature_names"]["631"]["self"] == list(beh_A.columns)
        assert res["feature_names"]["631"]["partner"] == list(beh_B.columns)
        assert res["feature_names"]["632"]["self"] == list(beh_B.columns)
        assert res["feature_names"]["632"]["partner"] == list(beh_A.columns)

    def test_valid_mask_applied_to_unmasked_behavior(self):
        S_A, S_B, beh_A, beh_B, T = _make_regression_setup(seed=78)
        # Pretend the first 100 bins were dropped during fit; S is shorter.
        valid_mask = np.ones(T, dtype=bool)
        valid_mask[:100] = False
        S_A_short = S_A[100:]
        S_B_short = S_B[100:]
        fit = _make_fit_for_regression(
            S_A_short, S_B_short, T, valid_mask=valid_mask,
        )
        # behavior dfs still have length T; function should apply valid_mask.
        res = regress_shared_on_behavior(
            fit, {"631": beh_A, "632": beh_B}, alpha=1.0, cv_folds=5,
        )
        # Self-dominated dim 0 should still have high R² after dropping bins.
        assert res["631"][0]["R2_self"] > 0.7

    def test_pre_masked_behavior_accepted(self):
        S_A, S_B, beh_A, beh_B, T = _make_regression_setup(seed=79)
        # Caller may pre-mask behavior to match S length.
        valid_mask = np.ones(T, dtype=bool)
        valid_mask[:50] = False
        S_A_short = S_A[50:]
        S_B_short = S_B[50:]
        beh_A_short = beh_A.iloc[50:].reset_index(drop=True)
        beh_B_short = beh_B.iloc[50:].reset_index(drop=True)
        fit = _make_fit_for_regression(
            S_A_short, S_B_short, T, valid_mask=valid_mask,
        )
        res = regress_shared_on_behavior(
            fit, {"631": beh_A_short, "632": beh_B_short},
            alpha=1.0, cv_folds=5,
        )
        assert res["631"][0]["R2_self"] > 0.7

    def test_nan_rows_in_behavior_dropped(self):
        S_A, S_B, beh_A, beh_B, T = _make_regression_setup(seed=80)
        beh_A_nan = beh_A.copy()
        beh_A_nan.iloc[200:220] = np.nan
        fit = _make_fit_for_regression(S_A, S_B, T)
        res = regress_shared_on_behavior(
            fit, {"631": beh_A_nan, "632": beh_B}, alpha=1.0, cv_folds=5,
        )
        # 20 dropped rows out of 600; self-dominated dim 0 should still
        # come through cleanly.
        assert res["631"][0]["R2_self"] > 0.6
