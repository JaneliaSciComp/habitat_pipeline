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

from types import SimpleNamespace

from ephys.social_spatial_fields import (
    RateMap,
    compute_rate_map,
    spatial_information,
    spatial_sparsity,
    spatial_coherence,
    split_half_stability,
    field_significance,
    compute_social_place_fields,
    field_similarity_across_targets,
    _benjamini_hochberg,
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
    # 'speed' column mirrors what get_tracking_on_ephys_clock returns.
    speed = np.sqrt(np.gradient(x, t) ** 2 + np.gradient(y, t) ** 2)
    return pd.DataFrame({"t": t, "x": x, "y": y, "speed": speed})


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


def _conjunctive_spikes(xy_self, xy_partner, c_self, c_partner, sigma,
                        peak_hz, base_hz, seed):
    """Spikes whose rate is the product of a self-bump and a partner-bump."""
    rng = np.random.default_rng(seed)
    t = xy_self["t"].to_numpy()
    gs = np.exp(-(((xy_self["x"] - c_self[0]) ** 2 + (xy_self["y"] - c_self[1]) ** 2)
                  / (2 * sigma ** 2)).to_numpy())
    gp = np.exp(-(((xy_partner["x"] - c_partner[0]) ** 2 + (xy_partner["y"] - c_partner[1]) ** 2)
                  / (2 * sigma ** 2)).to_numpy())
    rate = base_hz + peak_hz * gs * gp
    counts = rng.poisson(rate * DT)
    spikes = [ti + rng.uniform(0, DT, size=c) for ti, c in zip(t, counts) if c]
    return np.sort(np.concatenate(spikes)) if spikes else np.array([])


class _StubMAS:
    """Minimal MultiAnimalSession stand-in for compute_social_place_fields."""

    def __init__(self, tracking, session_id="20251216"):
        self._tracking = tracking
        self.animal_ids = list(tracking.keys())
        self.session_id = session_id

    def get_tracking_on_ephys_clock(self, t_start_ephys=None, t_end_ephys=None):
        out = {}
        for aid, df in self._tracking.items():
            d = df
            if t_start_ephys is not None:
                d = d[d["t"] >= t_start_ephys]
            if t_end_ephys is not None:
                d = d[d["t"] <= t_end_ephys]
            out[aid] = d.reset_index(drop=True)
        return out


def _make_ks(spike_lists):
    """SimpleNamespace standing in for KilosortData (use_quality_cells=False path)."""
    return SimpleNamespace(
        ks_ids=list(range(len(spike_lists))),
        spike_times_by_cell=list(spike_lists),
    )


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


def _three_animal_tracking(n=12000):
    return {
        "A": _make_xy(n=n, seed=100),
        "B": _make_xy(n=n, seed=200),
        "C": _make_xy(n=n, seed=300),
    }


def _sweep(ks, tracking, focal="A", **kw):
    mas = _StubMAS(tracking)
    defaults = dict(
        target_animals=list(tracking.keys()),
        bin_size_cm=BIN, smoothing_sigma_cm=5.0,
        speed_filter_subject="none", n_shuffles=100, min_n_spikes=50,
        use_quality_cells=False, arena_bounds=ARENA, seed=0,
    )
    defaults.update(kw)
    return compute_social_place_fields(ks, mas, focal_animal=focal, **defaults)


class TestBenjaminiHochberg:
    def test_zero_stays_zero_and_bounded(self):
        q = _benjamini_hochberg(np.array([0.0, 0.5, 0.9]))
        assert q[0] == 0.0
        assert np.all(q <= 1.0) and np.all(q >= 0.0)

    def test_nan_maps_to_one(self):
        q = _benjamini_hochberg(np.array([np.nan, 0.01]))
        assert q[0] == 1.0


class TestMultiTargetSweep:
    def test_self_only_classification(self):
        tr = _three_animal_tracking()
        spikes = _poisson_spikes_from_field(
            tr["A"], center=(40.0, 40.0), sigma=7.0, peak_hz=25.0, base_hz=0.2, seed=400)
        res = _sweep(_make_ks([spikes]), tr, focal="A")
        row = res.cell_classification.iloc[0]
        assert row["category"] == "self_only"
        assert row["dominant_target"] == "A"

    def test_partner_only_classification(self):
        tr = _three_animal_tracking()
        spikes = _poisson_spikes_from_field(
            tr["B"], center=(40.0, 40.0), sigma=7.0, peak_hz=25.0, base_hz=0.2, seed=401)
        res = _sweep(_make_ks([spikes]), tr, focal="A")
        row = res.cell_classification.iloc[0]
        assert row["category"] == "partner_only"
        assert row["dominant_target"] == "B"

    def test_flat_cell_classified_none(self):
        tr = _three_animal_tracking()
        rng = np.random.default_rng(402)
        spikes = np.sort(rng.uniform(tr["A"]["t"].min(), tr["A"]["t"].max(), size=6000))
        res = _sweep(_make_ks([spikes]), tr, focal="A")
        assert res.cell_classification.iloc[0]["category"] == "none"

    def test_conjunctive_classification(self):
        tr = _three_animal_tracking(n=16000)
        spikes = _conjunctive_spikes(
            tr["A"], tr["B"], c_self=(40.0, 40.0), c_partner=(40.0, 40.0),
            sigma=11.0, peak_hz=80.0, base_hz=0.1, seed=403)
        res = _sweep(_make_ks([spikes]), tr, focal="A")
        row = res.cell_classification.iloc[0]
        assert row["category"] == "conjunctive"

    def test_low_spike_cell_flagged(self):
        tr = _three_animal_tracking(n=4000)
        spikes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # < min_n_spikes
        res = _sweep(_make_ks([spikes]), tr, focal="A", min_n_spikes=50)
        sig = res.signif["A"][0]
        assert sig.n_shuffles == 0
        assert 0 in res.stats["A"]  # stats still present

    def test_speed_filter_removes_everything(self):
        tr = _three_animal_tracking(n=4000)
        spikes = _poisson_spikes_from_field(
            tr["A"], center=(40.0, 40.0), sigma=7.0, peak_hz=25.0, base_hz=0.2, seed=404)
        res = _sweep(_make_ks([spikes]), tr, focal="A",
                     speed_filter_subject="target", speed_threshold_cms=1e9)
        # Zero occupancy ⇒ all-NaN maps, nothing significant, runs gracefully.
        assert np.all(np.isnan(res.rate_maps["A"][0].rates))
        assert res.cell_classification.iloc[0]["category"] == "none"

    def test_similarity_helpers(self):
        tr = _three_animal_tracking(n=6000)
        spikes = _poisson_spikes_from_field(
            tr["A"], center=(40.0, 40.0), sigma=8.0, peak_hz=25.0, base_hz=0.2, seed=405)
        res = _sweep(_make_ks([spikes]), tr, focal="A")
        sim = field_similarity_across_targets(
            {t: res.rate_maps[t][0] for t in ["A", "B", "C"]})
        assert sim.shape == (3, 3)
        assert sim.loc["A", "A"] == 1.0
        # Population similarity keyed by (focal, partner).
        assert set(res.population_field_similarity.keys()) == {"A__B", "A__C"}
        m = res.population_field_similarity["A__B"]["similarity_matrix"]
        assert m.shape == (1, 1)

    def test_parameters_carry_contract(self):
        tr = _three_animal_tracking(n=3000)
        spikes = np.sort(np.random.default_rng(406).uniform(
            tr["A"]["t"].min(), tr["A"]["t"].max(), size=3000))
        res = _sweep(_make_ks([spikes]), tr, focal="A", n_shuffles=10)
        assert res.parameters["class_label"] == "target_position"
        assert res.parameters["analysis_title"]
        assert res.parameters["focal_animal"] == "A"
