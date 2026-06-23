"""
Tests for ephys/decode_partner_distance.py.

Synthetic-recovery style (no disk-backed session): all tests exercise the
array-level functions, which is why ``_analyze`` and the cores take pre-binned
arrays. Covers population recovery + cell ranking, distance-tuning
monotonicity, the circular-shift null sitting below the real R², that
contiguous-fold CV does not manufacture signal from autocorrelation, the
partial-R² confound control (a self-motion cell scores raw>0 but partial≈0),
and the ``insufficient_data`` guard.
"""

import numpy as np
import pytest

from ephys.decode_partner_distance import (
    _analyze,
    _bayesian_decode_1d,
    _cv_bayesian_distance,
    _cv_regress,
    _null_bayesian_distance,
    _null_distance_regression,
    _regression_diagnostics,
    compute_distance_tuning,
    population_distance_regression,
    single_cell_distance_scores,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _autocorr_distance(T, seed, theta=0.02, mu=60.0, sigma=3.0):
    """Strongly autocorrelated but *stationary* distance series (OU process).

    Mean-reverting like a real arena-bounded inter-animal distance — unlike an
    unbounded random walk, contiguous CV folds share the same distribution, so
    a genuine linear encoding is recoverable while autocorrelation (which would
    leak under *shuffled* folds) is still present.
    """
    rng = np.random.default_rng(seed)
    d = np.empty(T, dtype=np.float64)
    d[0] = mu
    for t in range(1, T):
        d[t] = d[t - 1] + theta * (mu - d[t - 1]) + sigma * rng.standard_normal()
    return np.abs(d)


def _planted_population(T=1500, n_noise=20, a=1.0, noise=0.4, seed=0):
    """One cell linear in distance (index 0) + ``n_noise`` pure-noise cells."""
    rng = np.random.default_rng(seed)
    distance = _autocorr_distance(T, seed)
    d_z = (distance - distance.mean()) / distance.std()
    signal = a * d_z + noise * rng.standard_normal(T)
    noise_cells = rng.standard_normal((T, n_noise))
    firing_rates = np.column_stack([signal, noise_cells])
    return firing_rates, distance


def _bump_population(T=2000, n_bumps=4, n_noise=20, width=7.0, peak_hz=12.0,
                     seed=0):
    """Bump-tuned cells (Gaussian-in-distance, non-monotonic) + noise cells.

    Each signal cell fires maximally at a preferred distance — a "place field"
    in distance space — so its *linear* correlation with distance is weak, which
    is exactly the regime where the Bayesian decoder beats linear ridge.
    """
    rng = np.random.default_rng(seed)
    distance = _autocorr_distance(T, seed, sigma=6.0)
    centers = np.linspace(25.0, 95.0, n_bumps)
    lam = np.column_stack([peak_hz * np.exp(-0.5 * ((distance - c) / width) ** 2)
                           for c in centers])
    bumps = rng.poisson(lam * 0.5) / 0.5                 # counts -> Hz (0.5 s bins)
    noise = 5.0 + rng.standard_normal((T, n_noise))
    firing_rates = np.clip(np.column_stack([bumps, noise]), 0.0, None)
    return firing_rates, distance


# ---------------------------------------------------------------------------
# Recovery + ranking
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_population_recovers_distance(self):
        fr, dist = _planted_population(seed=0)
        res = population_distance_regression(fr, dist, alpha=1.0, cv_folds=5)
        assert res["cv_r2"] > 0.4
        assert np.isfinite(res["rmse"]) and res["rmse"] > 0

    def test_planted_cell_tops_ranking(self):
        fr, dist = _planted_population(seed=1)
        sc = single_cell_distance_scores(fr, dist, alpha=1.0, cv_folds=5)
        assert sc["cell_ranking"][0] == 0          # planted cell ranked first
        assert sc["r2_per_cell"][0] > 0.3
        # Noise cells should be near-zero (allow small CV noise).
        assert np.nanmedian(sc["r2_per_cell"][1:]) < 0.05
        assert abs(sc["pearson_r_per_cell"][0]) > 0.5


# ---------------------------------------------------------------------------
# Tuning curves
# ---------------------------------------------------------------------------

class TestTuning:
    def test_monotone_cell_gives_monotone_tuning(self):
        rng = np.random.default_rng(3)
        T = 2000
        dist = _autocorr_distance(T, seed=3)
        rate = dist + 0.2 * rng.standard_normal(T)      # monotone increasing in distance
        out = compute_distance_tuning(rate, dist, n_distance_bins=15, smoothing_sigma=1.0)
        occ = out["occupancy"] > 0
        corr = np.corrcoef(out["dist_centers"][occ], out["tuning"][occ, 0])[0, 1]
        assert corr > 0.8

    def test_tuning_shapes(self):
        fr, dist = _planted_population(T=800, n_noise=5, seed=4)
        out = compute_distance_tuning(fr, dist, n_distance_bins=12)
        assert out["tuning"].shape == (12, 6)
        assert out["occupancy"].shape == (12,)
        assert out["dist_centers"].shape == (12,)
        assert int(out["occupancy"].sum()) == 800


# ---------------------------------------------------------------------------
# Null baseline + contiguous-CV honesty
# ---------------------------------------------------------------------------

class TestNull:
    def test_null_below_real_for_signal(self):
        fr, dist = _planted_population(seed=5)
        real = population_distance_regression(fr, dist, cv_folds=5)["cv_r2"]
        null = _null_distance_regression(fr, dist, cv_folds=5, null="shuffle",
                                         n_shuffles=50, seed=5)
        assert null["null_r2"] < real
        assert null["null_r2"] < 0.1               # shifted distance ⇒ ~no signal

    def test_noise_only_population_not_spuriously_high(self):
        """Contiguous-fold CV must not manufacture R² from autocorrelation."""
        rng = np.random.default_rng(6)
        dist = _autocorr_distance(1500, seed=6)
        fr = rng.standard_normal((1500, 20))        # cells independent of distance
        res = population_distance_regression(fr, dist, cv_folds=5)
        null = _null_distance_regression(fr, dist, cv_folds=5, null="shuffle",
                                         n_shuffles=50, seed=6)
        assert res["cv_r2"] < 0.1
        # Real and null are both ~0 (within a fold of each other) — no leakage.
        assert abs(res["cv_r2"] - null["null_r2"]) < 0.15


# ---------------------------------------------------------------------------
# Confound control (partial R²)
# ---------------------------------------------------------------------------

class TestConfound:
    def test_self_motion_cell_has_zero_partial_r2(self):
        rng = np.random.default_rng(7)
        T = 1500
        dist = _autocorr_distance(T, seed=7)
        d_z = (dist - dist.mean()) / dist.std()
        # Focal self-motion correlated with distance; cell encodes self-motion.
        focal_speed = 0.9 * d_z + 0.3 * rng.standard_normal(T)
        cell = focal_speed + 0.1 * rng.standard_normal(T)
        fr = cell[:, None]
        nuisance = focal_speed[:, None]

        sc = single_cell_distance_scores(fr, dist, cv_folds=5, nuisance=nuisance)
        assert sc["r2_per_cell"][0] > 0.1            # raw: looks distance-tuned
        assert sc["r2_partial_per_cell"][0] < 0.05   # but adds nothing beyond self-motion

    def test_partial_none_without_nuisance(self):
        fr, dist = _planted_population(T=600, n_noise=3, seed=8)
        sc = single_cell_distance_scores(fr, dist, cv_folds=5, nuisance=None)
        assert sc["r2_partial_per_cell"] is None
        pop = population_distance_regression(fr, dist, cv_folds=5, nuisance=None)
        assert pop["cv_r2_partial"] is None


# ---------------------------------------------------------------------------
# _analyze orchestration + guards
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_analyze_success_schema(self):
        fr, dist = _planted_population(seed=9)
        nuis = np.column_stack([np.gradient(dist), dist * 0 + 1.0])  # speed-ish + const
        bc = np.arange(len(dist)) * 0.5
        res = _analyze(fr, dist, nuis, bc, units="cm", focal="A", partner="B",
                       cv_folds=5, n_shuffles=20)
        assert res["status"] == "success"
        for k in ("tuning", "r2_per_cell", "cv_r2", "rmse", "y_pred",
                  "cell_ranking", "null_r2", "cv_r2_partial"):
            assert k in res
        assert res["parameters"]["class_label"] == "partner_distance"
        assert res["units"] == "cm"
        assert res["y_pred"].shape == dist.shape

    def test_insufficient_data_guard(self):
        fr = np.random.default_rng(10).standard_normal((6, 4))
        dist = np.arange(6, dtype=float)
        res = _analyze(fr, dist, None, np.arange(6.0), units="pixels",
                       focal="A", partner="B", cv_folds=5)
        assert res["status"] == "insufficient_data"
        assert np.isnan(res["cv_r2"])

    def test_null_can_be_disabled(self):
        fr, dist = _planted_population(T=600, n_noise=3, seed=11)
        res = _analyze(fr, dist, None, np.arange(len(dist)) * 0.5, units="cm",
                       focal="A", partner="B", cv_folds=5, null=None)
        assert res["status"] == "success"
        assert "null_r2" not in res


# ---------------------------------------------------------------------------
# _cv_regress unit behavior
# ---------------------------------------------------------------------------

class TestCvRegress:
    def test_perfect_linear_fit(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal((500, 1))
        y = 3.0 * x[:, 0] + 1.0
        res = _cv_regress(x, y, alpha=1e-6, cv_folds=5)
        assert res["cv_r2"] > 0.99
        assert res["rmse"] < 0.1

    def test_returns_oof_predictions(self):
        rng = np.random.default_rng(13)
        x = rng.standard_normal((300, 2))
        y = x @ np.array([1.0, -2.0])
        res = _cv_regress(x, y, alpha=1e-6, cv_folds=5)
        assert np.all(np.isfinite(res["y_pred"]))
        assert res["y_pred"].shape == y.shape


# ---------------------------------------------------------------------------
# Diagnostics: pooled R² vs mean-of-fold on non-stationary data
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_pooled_matches_sklearn_r2_score(self):
        from sklearn.metrics import r2_score
        rng = np.random.default_rng(20)
        y = rng.standard_normal(400)
        y_pred = y + 0.3 * rng.standard_normal(400)
        diag = _regression_diagnostics(y, y_pred, [0.1, 0.2, 0.3])
        assert diag["pooled_r2"] == pytest.approx(r2_score(y, y_pred))
        assert diag["r2_fold_mean"] == pytest.approx(0.2)
        assert diag["r2_fold_min"] == pytest.approx(0.1)
        assert diag["r2_fold_max"] == pytest.approx(0.3)

    def test_pooled_beats_mean_fold_on_nonstationary(self):
        """A drifting (non-stationary) target makes per-fold means diverge.

        Contiguous folds then occupy distance regimes absent from training, so
        mean-of-fold R² is dragged far below the pooled out-of-fold R².
        """
        rng = np.random.default_rng(21)
        T = 1500
        # Random-walk distance — drifts across the session (non-stationary).
        distance = 60.0 + np.cumsum(0.5 * rng.standard_normal(T))
        d_z = (distance - distance.mean()) / distance.std()
        fr = (d_z + 0.3 * rng.standard_normal(T))[:, None]
        pop = population_distance_regression(fr, distance, alpha=1.0, cv_folds=5)
        diag = pop["diagnostics"]
        assert diag["pooled_r2"] > pop["cv_r2"]      # pooled is the honest summary
        assert diag["pooled_r2"] > 0.3               # signal is genuinely there


# ---------------------------------------------------------------------------
# Bayesian 1-D decoder
# ---------------------------------------------------------------------------

class TestBayesian:
    def test_decode_1d_recovers_monotone_tuning(self):
        """Noiseless monotone tuning ⇒ small decoding error."""
        rng = np.random.default_rng(22)
        T = 2000
        dist = _autocorr_distance(T, seed=22, sigma=6.0)
        rate = (0.3 * dist)[:, None]                 # monotone, high-rate
        tc = compute_distance_tuning(rate, dist, n_distance_bins=25,
                                     smoothing_sigma=1.0)
        pred, post = _bayesian_decode_1d(
            rate, tc["tuning"], tc["occupancy"], 0.5, tc["dist_centers"])
        assert post.shape == (T, 25)
        assert np.allclose(post.sum(axis=1), 1.0)    # normalised posterior
        # decoded tracks truth
        assert np.corrcoef(pred, dist)[0, 1] > 0.9

    def test_bayesian_beats_ridge_on_bump_tuning(self):
        fr, dist = _bump_population(seed=23)
        ridge = population_distance_regression(fr, dist, alpha=1.0, cv_folds=5)
        bayes = _cv_bayesian_distance(fr, dist, bin_size=0.5, n_distance_bins=25,
                                      cv_folds=5)
        assert bayes["cv_r2"] > ridge["diagnostics"]["pooled_r2"] + 0.1
        assert bayes["cv_r2"] > 0.3
        assert bayes["median_error"] > 0

    def test_bayesian_null_below_real(self):
        fr, dist = _bump_population(seed=24)
        real = _cv_bayesian_distance(fr, dist, bin_size=0.5, n_distance_bins=25,
                                     cv_folds=5)["cv_r2"]
        null = _null_bayesian_distance(fr, dist, bin_size=0.5, n_distance_bins=25,
                                       cv_folds=5, null="shuffle", n_shuffles=30,
                                       seed=24)
        assert null["null_r2"] < real
        assert null["null_r2"] < 0.1


# ---------------------------------------------------------------------------
# _analyze with decoder='both' — schema is additive
# ---------------------------------------------------------------------------

class TestDecoderBoth:
    def test_both_keeps_ridge_keys_and_adds_bayesian(self):
        fr, dist = _bump_population(seed=25)
        bc = np.arange(len(dist)) * 0.5
        ridge_only = _analyze(fr, dist, None, bc, units="cm", focal="A",
                              partner="B", cv_folds=5, n_distance_bins=25,
                              n_shuffles=15, decoder="ridge")
        both = _analyze(fr, dist, None, bc, units="cm", focal="A", partner="B",
                        cv_folds=5, n_distance_bins=25, n_shuffles=15,
                        decoder="both")
        # All ridge-mode top-level keys still present and unchanged in meaning.
        for k in ("cv_r2", "rmse", "y_pred", "cell_ranking", "r2_per_cell",
                  "tuning", "null_r2", "cv_r2_pooled", "diagnostics"):
            assert k in ridge_only and k in both
        assert "bayesian" not in ridge_only
        assert both["cv_r2"] == ridge_only["cv_r2"]          # ridge stays primary
        # Bayesian nested, with its own keys.
        b = both["bayesian"]
        assert {"cv_r2", "median_error", "y_pred", "posterior", "null_r2"} <= set(b)
        assert b["posterior"].shape[0] == len(dist)
        # Contract preserved.
        assert both["parameters"]["class_label"] == "partner_distance"
        assert both["parameters"]["analysis_title"] == "Partner-Distance Decoding"
        assert both["parameters"]["decoder"] == "both"

    def test_null_comparison_scalars(self):
        fr, dist = _planted_population(seed=26)
        bc = np.arange(len(dist)) * 0.5
        res = _analyze(fr, dist, None, bc, units="cm", focal="A", partner="B",
                       cv_folds=5, n_shuffles=30)
        assert np.isfinite(res["cv_r2_vs_null_z"])
        assert res["cv_r2_vs_null_z"] > 0                    # real signal beats null
        assert 0.0 <= res["cv_r2_null_percentile"] <= 100.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
