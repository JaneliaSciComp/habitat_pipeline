"""
Smoke tests for ephys/inter_brain_plots.py.

These verify that every plotting function produces a matplotlib Figure
without errors on synthetic inputs. We don't pixel-compare; the goal is
to catch import / API breakage and obvious shape bugs.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow .use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ephys.inter_brain_dynamics import (  # noqa: E402
    SharedSubspaceFit,
    cross_animal_correlation_matrix,
    fit_shared_subspace,
    regress_shared_on_behavior,
    shuffle_null_subspace,
    time_lagged_cca,
)
from ephys.inter_brain_plots import (  # noqa: E402
    plot_canonical_correlations,
    plot_cross_animal_correlation,
    plot_inter_brain_summary,
    plot_shared_dimensions,
    plot_shared_vs_behavior,
    plot_time_lagged_cca,
    plot_variance_partition,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _make_synthetic(seed: int = 0, T: int = 500, N_A: int = 12, N_B: int = 10,
                    K_true: int = 3, noise: float = 0.5):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((T, K_true))
    W_A = rng.standard_normal((K_true, N_A))
    W_B = rng.standard_normal((K_true, N_B))
    X_A = Z @ W_A + noise * rng.standard_normal((T, N_A))
    X_B = Z @ W_B + noise * rng.standard_normal((T, N_B))
    return X_A, X_B


@pytest.fixture
def fit_fixture():
    X_A, X_B = _make_synthetic(seed=0, T=500, N_A=12, N_B=10, K_true=3)
    fit = fit_shared_subspace(
        X_A, X_B, n_components=3, method="regularized",
        animal_ids=("631", "632"),
        bin_size_sec=0.5,
        t_window=(0.0, 250.0),
    )
    yield fit, X_A, X_B
    plt.close("all")


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

class TestPlotCanonicalCorrelations:
    def test_returns_figure(self, fit_fixture):
        fit, _, _ = fit_fixture
        fig = plot_canonical_correlations(fit)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Canonical correlations" in ax.get_title()
        assert ax.get_xlabel() == "Canonical component"

    def test_with_shuffle_null(self, fit_fixture):
        fit, X_A, X_B = fit_fixture
        null = shuffle_null_subspace(X_A, X_B, n_components=3, n_shuffles=5, seed=0)
        fig = plot_canonical_correlations(fit, shuffle_null=null)
        assert isinstance(fig, plt.Figure)

    def test_null_shape_mismatch_raises(self, fit_fixture):
        fit, _, _ = fit_fixture
        bad_null = np.zeros((5, 99))  # wrong K
        with pytest.raises(ValueError, match="shape"):
            plot_canonical_correlations(fit, shuffle_null=bad_null)


class TestPlotVariancePartition:
    def test_returns_figure_and_uses_animal_ids(self, fit_fixture):
        fit, _, _ = fit_fixture
        fig = plot_variance_partition(fit)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["631", "632"]
        assert "Variance partition" in ax.get_title()


class TestPlotSharedDimensions:
    def test_returns_figure(self, fit_fixture):
        fit, _, _ = fit_fixture
        fig = plot_shared_dimensions(fit, k_dims=(0, 1, 2))
        assert isinstance(fig, plt.Figure)
        # One subplot per requested dim.
        assert len(fig.axes) == 3

    def test_with_t_bins(self, fit_fixture):
        fit, _, _ = fit_fixture
        t = np.linspace(0.0, 250.0, fit.S_A.shape[0])
        fig = plot_shared_dimensions(fit, t_bins=t, k_dims=(0,))
        assert isinstance(fig, plt.Figure)
        assert fig.axes[-1].get_xlabel() == "Time (ephys s)"

    def test_t_bins_length_mismatch_raises(self, fit_fixture):
        fit, _, _ = fit_fixture
        with pytest.raises(ValueError, match="t_bins"):
            plot_shared_dimensions(fit, t_bins=np.arange(10), k_dims=(0,))

    def test_invalid_k_dims_raises(self, fit_fixture):
        fit, _, _ = fit_fixture
        with pytest.raises(ValueError, match="k_dims"):
            plot_shared_dimensions(fit, k_dims=(99,))


class TestPlotCrossAnimalCorrelation:
    def test_returns_figure(self):
        rng = np.random.default_rng(0)
        C = rng.standard_normal((12, 10)) * 0.3
        fig = plot_cross_animal_correlation(C)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_cluster(self):
        rng = np.random.default_rng(1)
        C = rng.standard_normal((12, 10)) * 0.3
        fig = plot_cross_animal_correlation(C, cluster=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_one_dim_input_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            plot_cross_animal_correlation(np.zeros(10))

    def test_uses_real_cross_animal_corr(self, fit_fixture):
        fit, X_A, X_B = fit_fixture
        C = cross_animal_correlation_matrix(X_A, X_B)
        fig = plot_cross_animal_correlation(C)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTimeLaggedCCA:
    def test_returns_figure(self, fit_fixture):
        fit, X_A, X_B = fit_fixture
        lags, ccs = time_lagged_cca(X_A, X_B, max_lag_bins=5, n_components=2)
        fig = plot_time_lagged_cca(lags, ccs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_bin_size(self, fit_fixture):
        fit, X_A, X_B = fit_fixture
        lags, ccs = time_lagged_cca(X_A, X_B, max_lag_bins=3, n_components=2)
        fig = plot_time_lagged_cca(lags, ccs, bin_size_sec=0.5)
        assert "Lag (s" in fig.axes[0].get_xlabel()
        plt.close(fig)

    def test_shape_mismatch_raises(self):
        lags = np.arange(-3, 4)
        ccs = np.zeros((6, 2))  # wrong n_lags
        with pytest.raises(ValueError, match="shape"):
            plot_time_lagged_cca(lags, ccs)


class TestPlotSharedVsBehavior:
    def test_returns_figure(self, fit_fixture):
        fit, _, _ = fit_fixture
        T_valid = fit.parameters["T_valid"]
        rng = np.random.default_rng(99)
        behavior_A = pd.DataFrame({
            "speed": rng.standard_normal(T_valid),
            "distance": rng.standard_normal(T_valid),
        })
        behavior_B = pd.DataFrame({
            "speed": rng.standard_normal(T_valid),
            "distance": rng.standard_normal(T_valid),
        })
        res = regress_shared_on_behavior(
            fit, {"631": behavior_A, "632": behavior_B},
            alpha=1.0, cv_folds=3,
        )
        fig = plot_shared_vs_behavior(fit, res)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_missing_animal_in_results_draws_placeholder(self, fit_fixture):
        fit, _, _ = fit_fixture
        # Manufacture a results dict with one animal missing.
        fake = {"631": {0: {"R2_self": 0.5, "R2_partner": 0.1, "R2_both": 0.55,
                             "R2_partner_unique": 0.05, "R2_self_unique": 0.45}}}
        fig = plot_shared_vs_behavior(fit, fake)
        # Both axes drawn; second is the placeholder.
        assert len(fig.axes) == 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------

class TestPlotInterBrainSummary:
    def test_minimal(self, fit_fixture):
        fit, _, _ = fit_fixture
        fig = plot_inter_brain_summary(fit)
        assert isinstance(fig, plt.Figure)
        # 6 panels.
        assert len(fig.axes) >= 6
        plt.close(fig)

    def test_with_all_inputs(self, fit_fixture):
        fit, X_A, X_B = fit_fixture
        null = shuffle_null_subspace(X_A, X_B, n_components=3, n_shuffles=5, seed=0)
        C = cross_animal_correlation_matrix(X_A, X_B)
        lags, ccs = time_lagged_cca(X_A, X_B, max_lag_bins=3, n_components=2)
        T_valid = fit.parameters["T_valid"]
        rng = np.random.default_rng(7)
        beh_A = pd.DataFrame({"speed": rng.standard_normal(T_valid),
                               "distance": rng.standard_normal(T_valid)})
        beh_B = pd.DataFrame({"speed": rng.standard_normal(T_valid),
                               "distance": rng.standard_normal(T_valid)})
        reg = regress_shared_on_behavior(
            fit, {"631": beh_A, "632": beh_B}, alpha=1.0, cv_folds=3,
        )

        fig = plot_inter_brain_summary(
            fit,
            shuffle_null=null,
            t_bins=np.linspace(0.0, 250.0, T_valid),
            cross_corr=C,
            time_lagged=(lags, ccs),
            regression_results=reg,
            bin_size_sec=0.5,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
