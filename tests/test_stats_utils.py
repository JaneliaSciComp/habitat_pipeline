"""Tests for ephys._stats_utils.benjamini_hochberg.

Mirrors tests/test_social_spatial_fields.py::TestBenjaminiHochberg for the
independent copy of the same algorithm used by the LDA-decoding rigor layer.
"""
import numpy as np

from ephys._stats_utils import benjamini_hochberg


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
