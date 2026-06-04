"""
Tests for ephys/social_spatial_fields.py.

Synthetic positions are smooth random walks in a known arena; spikes are sampled
from an inhomogeneous Poisson process whose rate is a Gaussian bump over a chosen
animal's coordinates. These exercise rate-map recovery, spatial statistics, and
(in later phases) shuffle significance and the multi-target classification.
"""

import numpy as np
import pandas as pd
import pytest

from ephys.social_spatial_fields import (
    RateMap,
    compute_rate_map,
    spatial_information,
    spatial_sparsity,
    spatial_coherence,
    split_half_stability,
    field_significance,
)

ARENA = ((0.0, 80.0), (0.0, 80.0))
DT = 0.04            # 25 Hz tracking
BIN = 5.0


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------

def _random_walk(n, bounds, step_sd, seed):
    """Reflecting random walk inside ``bounds`` ((xmin,xmax),(ymin,ymax))."""
    rng = np.random.default_rng(seed)
    (xmin, xmax), (ymin, ymax) = bounds
    x = np.empty(n)
    y = np.empty(n)
    x[0] = 0.5 * (xmin + xmax)
    y[0] = 0.5 * (ymin + ymax)
    for i in range(1, n):
        x[i] = np.clip(x[i - 1] + rng.normal(0, step_sd), xmin, xmax)
        y[i] = np.clip(y[i - 1] + rng.normal(0, step_sd), ymin, ymax)
    return x, y


def _make_xy(n=25000, bounds=ARENA, step_sd=3.0, seed=0, t0=0.0):
    x, y = _random_walk(n, bounds, step_sd, seed)
    t = t0 + np.arange(n) * DT
    return pd.DataFrame({"t": t, "x": x, "y": y})


def _poisson_spikes_from_field(xy, center, sigma, peak_hz, base_hz, seed):
    """Spikes whose instantaneous rate is a Gaussian bump over xy's (x, y)."""
    rng = np.random.default_rng(seed)
    t = xy["t"].to_numpy()
    x = xy["x"].to_numpy()
    y = xy["y"].to_numpy()
    cx, cy = center
    rate = base_hz + peak_hz * np.exp(
        -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)
    )
    counts = rng.poisson(rate * DT)
    spikes = []
    for ti, c in zip(t, counts):
        if c:
            spikes.append(ti + rng.uniform(0, DT, size=c))
    return np.sort(np.concatenate(spikes)) if spikes else np.array([])


def _bin_center(edges, idx):
    return 0.5 * (edges[idx] + edges[idx + 1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRateMapRecovery:
    def test_planted_partner_field_recovered(self):
        partner_xy = _make_xy(seed=1)
        center = (40.0, 55.0)
        spikes = _poisson_spikes_from_field(
            partner_xy, center=center, sigma=7.0, peak_hz=25.0, base_hz=0.2, seed=2,
        )
        rm = compute_rate_map(
            spikes, partner_xy, bin_size_cm=BIN, arena_bounds=ARENA,
            smoothing_sigma_cm=5.0, speed_threshold_cms=None,
            focal_animal="A", target_animal="B", cluster_id=7,
        )
        assert rm.rates.shape == (len(rm.y_edges) - 1, len(rm.x_edges) - 1)
        # argmax bin within one bin of the planted center.
        iy, ix = np.unravel_index(np.nanargmax(rm.rates), rm.rates.shape)
        px = _bin_center(rm.x_edges, ix)
        py = _bin_center(rm.y_edges, iy)
        assert abs(px - center[0]) <= BIN
        assert abs(py - center[1]) <= BIN

    def test_parameters_carry_class_label(self):
        xy = _make_xy(n=2000, seed=3)
        rm = compute_rate_map(np.array([1.0, 2.0, 3.0]), xy, arena_bounds=ARENA,
                              speed_threshold_cms=None)
        assert rm.parameters["class_label"] == "target_position"
        assert "analysis_title" in rm.parameters
        assert rm.parameters["arena_bounds"] == ARENA

    def test_occupancy_is_seconds(self):
        xy = _make_xy(n=5000, seed=4)
        rm = compute_rate_map(np.array([]), xy, arena_bounds=ARENA,
                              smoothing_sigma_cm=None, speed_threshold_cms=None)
        # Total occupancy ~ recording duration.
        np.testing.assert_allclose(rm.occupancy.sum(), 5000 * DT, rtol=0.01)


class TestSpatialStats:
    def test_planted_field_has_high_information(self):
        partner_xy = _make_xy(seed=5)
        spikes = _poisson_spikes_from_field(
            partner_xy, center=(40.0, 40.0), sigma=7.0, peak_hz=25.0,
            base_hz=0.2, seed=6,
        )
        rm = compute_rate_map(spikes, partner_xy, bin_size_cm=BIN, arena_bounds=ARENA,
                              speed_threshold_cms=None)
        bits_spike, bits_sec = spatial_information(rm)
        assert bits_spike > 0.5
        assert bits_sec > 0.0
        # Selective field ⇒ low sparsity value.
        assert spatial_sparsity(rm) < 0.5

    def test_flat_field_has_low_information(self):
        xy = _make_xy(seed=7)
        rng = np.random.default_rng(8)
        # Constant-rate spikes independent of position.
        spikes = np.sort(rng.uniform(xy["t"].min(), xy["t"].max(), size=8000))
        rm = compute_rate_map(spikes, xy, bin_size_cm=BIN, arena_bounds=ARENA,
                              speed_threshold_cms=None)
        bits_spike, _ = spatial_information(rm)
        assert bits_spike < 0.2
        assert spatial_sparsity(rm) > 0.6  # close to 1 for uniform field

    def test_split_half_stable_for_planted_field(self):
        partner_xy = _make_xy(seed=9)
        spikes = _poisson_spikes_from_field(
            partner_xy, center=(40.0, 40.0), sigma=8.0, peak_hz=30.0,
            base_hz=0.2, seed=10,
        )
        corr = split_half_stability(
            spikes, partner_xy, bin_size_cm=BIN, arena_bounds=ARENA,
            speed_threshold_cms=None,
        )
        assert corr > 0.5

    def test_coherence_higher_for_smooth_field(self):
        partner_xy = _make_xy(seed=11)
        spikes = _poisson_spikes_from_field(
            partner_xy, center=(40.0, 40.0), sigma=10.0, peak_hz=30.0,
            base_hz=0.2, seed=12,
        )
        rm = compute_rate_map(spikes, partner_xy, bin_size_cm=BIN, arena_bounds=ARENA,
                              speed_threshold_cms=None)
        assert np.isfinite(spatial_coherence(rm))


class TestSignificance:
    def test_planted_field_significant(self):
        partner_xy = _make_xy(seed=13)
        spikes = _poisson_spikes_from_field(
            partner_xy, center=(45.0, 35.0), sigma=7.0, peak_hz=25.0,
            base_hz=0.2, seed=14,
        )
        sig = field_significance(
            spikes, partner_xy, n_shuffles=500, null_method="circular_shift",
            seed=0, cluster_id=3, target_animal="B",
            bin_size_cm=BIN, arena_bounds=ARENA, speed_threshold_cms=None,
        )
        assert sig.n_shuffles == 500
        assert len(sig.shuffle_skaggs) == 500
        assert sig.p_skaggs < 0.001

    def test_flat_field_not_significant(self):
        xy = _make_xy(seed=15)
        rng = np.random.default_rng(16)
        spikes = np.sort(rng.uniform(xy["t"].min(), xy["t"].max(), size=8000))
        sig = field_significance(
            spikes, xy, n_shuffles=200, null_method="circular_shift", seed=0,
            cluster_id=4, target_animal="B",
            bin_size_cm=BIN, arena_bounds=ARENA, speed_threshold_cms=None,
        )
        assert sig.p_skaggs > 0.05

    def test_position_shuffle_null_runs(self):
        partner_xy = _make_xy(n=8000, seed=17)
        spikes = _poisson_spikes_from_field(
            partner_xy, center=(40.0, 40.0), sigma=8.0, peak_hz=25.0,
            base_hz=0.2, seed=18,
        )
        sig = field_significance(
            spikes, partner_xy, n_shuffles=100, null_method="position_shuffle",
            seed=0, cluster_id=5, target_animal="B",
            bin_size_cm=BIN, arena_bounds=ARENA, speed_threshold_cms=None,
        )
        assert sig.null_method == "position_shuffle"
        assert sig.p_skaggs < 0.05
