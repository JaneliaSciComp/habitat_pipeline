"""Tests for ephys._stats_utils.

`TestBenjaminiHochberg` mirrors tests/test_social_spatial_fields.py for the
independent copy of the same algorithm used by the LDA-decoding rigor layer.
`TestFdrResolution` pins the arithmetic behind the Phase 1.5 finding that a
200-shuffle budget cannot resolve FDR significance across ~150 cells.
"""
import numpy as np
import pytest

from ephys._stats_utils import (
    benjamini_hochberg,
    empirical_p_value,
    fdr_resolution,
    majority_class_baseline,
)


class TestBenjaminiHochberg:
    def test_zero_stays_zero_and_bounded(self):
        q = benjamini_hochberg(np.array([0.0, 0.5, 0.9]))
        assert q[0] == 0.0
        assert np.all(q <= 1.0) and np.all(q >= 0.0)

    def test_nan_maps_to_one(self):
        q = benjamini_hochberg(np.array([0.01, np.nan, 0.5]))
        assert q[1] == 1.0

    def test_empty_input(self):
        q = benjamini_hochberg(np.array([]))
        assert q.size == 0

    def test_monotonicity_preserved(self):
        pvals = np.array([0.5, 0.01, 0.3, 0.02, 0.9])
        q = benjamini_hochberg(pvals)
        order = np.argsort(pvals)
        assert np.all(np.diff(q[order]) >= -1e-12)

    def test_all_equal_pvalues(self):
        q = benjamini_hochberg(np.full(10, 0.05))
        assert np.allclose(q, 0.05)


class TestFdrResolution:
    def test_the_real_phase1_run_was_unresolvable(self):
        """149 cells x 200 shuffles — the configuration that produced the
        misleading '0/148 significant' Phase 1 result."""
        r = fdr_resolution(149, 200, 0.05)
        assert r['resolvable'] is False
        assert r['p_floor'] == pytest.approx(1 / 201)
        assert r['best_achievable_q'] == pytest.approx(149 / 201)
        # Brute-force BH agrees: 14 cells at the floor is not enough, 15 is.
        assert r['min_tests_at_floor'] == 15
        assert r['recommended_n_shuffles'] == 2980

    def test_min_tests_at_floor_matches_brute_force_bh(self):
        m, n_shuffles, alpha = 149, 200, 0.05
        threshold = fdr_resolution(m, n_shuffles, alpha)['min_tests_at_floor']
        p_floor = 1 / (n_shuffles + 1)

        def best_q(k):
            p = np.full(m, 0.5)
            p[:k] = p_floor
            return benjamini_hochberg(p).min()

        assert best_q(threshold - 1) >= alpha
        assert best_q(threshold) < alpha

    def test_recommended_budget_is_sufficient_and_minimal(self):
        for n_tests in (1, 10, 149, 500):
            for alpha in (0.05, 0.01):
                rec = fdr_resolution(n_tests, 10, alpha)['recommended_n_shuffles']
                assert fdr_resolution(n_tests, rec, alpha)['resolvable'] is True
                if rec > 1:  # one fewer shuffle must not suffice
                    assert fdr_resolution(n_tests, rec - 1, alpha)['resolvable'] is False

    def test_single_test_is_resolvable_at_modest_budget(self):
        # The population-level test is a single test, hence well resolved.
        assert fdr_resolution(1, 200, 0.05)['resolvable'] is True
        assert fdr_resolution(1, 19, 0.05)['resolvable'] is False

    def test_pooling_style_budget_resolves_many_tests(self):
        # Pooled null across 149 cells => 149*200 effective draws.
        assert fdr_resolution(149, 149 * 200, 0.05)['resolvable'] is True

    def test_rejects_nonpositive_inputs(self):
        with pytest.raises(ValueError):
            fdr_resolution(0, 100)
        with pytest.raises(ValueError):
            fdr_resolution(10, 0)


class TestEmpiricalPValue:
    def test_add_one_form_never_returns_zero(self):
        # No draw beats the observed value -> floor at 1/(n+1), not 0.0.
        p = empirical_p_value(10.0, np.zeros(99))
        assert p == pytest.approx(1 / 100)
        assert p > 0.0

    def test_all_draws_exceed(self):
        assert empirical_p_value(0.0, np.ones(9)) == pytest.approx(10 / 10)

    def test_nan_draws_excluded_from_numerator_and_denominator(self):
        """NaN draws must not be counted as 'did not exceed'.

        `nan >= x` is False, so a naive exceedance count would silently treat
        NaN draws as evidence against the null and bias p downward.
        """
        draws = np.array([5.0, 5.0, np.nan, np.nan])
        # Only 2 valid draws, both >= 1.0 -> (1+2)/(2+1) = 1.0
        assert empirical_p_value(1.0, draws) == pytest.approx(1.0)
        # A naive implementation over all 4 draws would give (1+2)/(4+1) = 0.6
        assert empirical_p_value(1.0, draws) != pytest.approx(0.6)

    def test_nan_observed_is_nan(self):
        assert np.isnan(empirical_p_value(np.nan, np.ones(10)))

    def test_all_nan_draws_is_nan(self):
        assert np.isnan(empirical_p_value(1.0, np.full(10, np.nan)))

    def test_empty_draws_is_nan(self):
        assert np.isnan(empirical_p_value(1.0, np.array([])))

    def test_floor_matches_fdr_resolution_p_floor(self):
        # The two must agree, since fdr_resolution reasons about this floor.
        for n in (20, 200, 500):
            observed_floor = empirical_p_value(1e9, np.zeros(n))
            assert observed_floor == pytest.approx(fdr_resolution(1, n)['p_floor'])


class TestMajorityClassBaseline:
    def test_imbalanced_split(self):
        labels = np.array(['winner'] * 12 + ['loser'] * 7)
        assert majority_class_baseline(labels) == pytest.approx(12 / 19)

    def test_balanced_split_equals_uniform_chance(self):
        assert majority_class_baseline(np.array(['a', 'b', 'a', 'b'])) == pytest.approx(0.5)

    def test_matches_the_probe_formula(self):
        # scripts/phase0_probe.py:202 computes this inline; same answer.
        labels = np.array([0, 1, 1, 1, 0, 1])
        expected = max(np.mean(labels == 0), np.mean(labels == 1))
        assert majority_class_baseline(labels) == pytest.approx(expected)

    def test_empty_is_nan(self):
        assert np.isnan(majority_class_baseline(np.array([])))
