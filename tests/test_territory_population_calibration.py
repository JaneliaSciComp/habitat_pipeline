"""Calibration and power for ephys.territory_population (three population tests).

Each test must (a) hold its false-positive rate under scenarios that fool naive
versions - slow drift, place-field density that peaks at the boundary, neural
information about partner distance - and (b) detect a real effect injected into a
fraction of cells. Judged over independent sessions against the 99.9% binomial
bound. One test is documented as known-limited (xfail), not hidden.
"""

import numpy as np
import pytest
from scipy import stats

from ephys import territory_population as tp
from tests._territory_sim import make_session, simulate_population

N_CELLS = 40
BOUND_Q = 0.999


def _bound(n, alpha):
    return stats.binom.ppf(BOUND_Q, n, alpha) / n


def _session(i, nbr_spread=None):
    kw = {} if nbr_spread is None else {"nbr_spread": nbr_spread}
    return make_session(seed=5000 + i, **kw)


def _straddle(ses, counts, **kw):
    return tp.boundary_straddle_test(counts / ses["dt"], ses["x"], ses["y"],
                                     ses["territory_map"], "focal", dt=ses["dt"], **kw)


def _decoder(ses, counts, **kw):
    return tp.territory_decoder_gain(counts / ses["dt"], ses["x"], ses["y"], ses["speed"],
                                     ses["territory_map"], "focal", dt=ses["dt"], **kw)


def _presence(ses, counts, **kw):
    return tp.owner_presence_decoding(counts / ses["dt"], ses["x"], ses["y"], ses["owner"],
                                      ses["owner_present"], ses["owner_dist"], dt=ses["dt"], **kw)


# ---------------------------------------------------------------------------
# Fast structural tests
# ---------------------------------------------------------------------------

class TestDisplacedMasks:
    def test_excludes_overlapping_translations_and_the_identity(self):
        m = np.zeros((20, 20), bool)
        m[2:10, 5:15] = True
        out = tp.displaced_masks(m, max_jaccard=0.5)
        assert out
        for dx, dy, s in out:
            assert (dx, dy) != (0, 0)
            assert np.sum(s & m) / np.sum(s | m) <= 0.5
            assert s.sum() >= 0.6 * m.sum()

    def test_shift_does_not_wrap(self):
        m = np.zeros((10, 10), bool)
        m[6:10, :] = True                       # 4 rows touching the far edge
        s = tp._shifted(m, 2, 0)
        assert s.sum() == 2 * 10                # two rows stay on the grid, two fall off
        assert s[8:].all() and not s[:8].any()  # nothing re-appears at the near edge
        assert tp._shifted(m, 0, 0).sum() == m.sum()

    def test_max_n_subsamples_deterministically(self):
        m = np.zeros((20, 20), bool)
        m[5:12, 5:12] = True
        a = tp.displaced_masks(m, max_n=20, seed=1)
        b = tp.displaced_masks(m, max_n=20, seed=1)
        assert len(a) == 20 and [(d[0], d[1]) for d in a] == [(d[0], d[1]) for d in b]


class TestPreprocess:
    def test_removes_slow_trend_and_zscores(self):
        t = np.arange(3600) * 0.5
        trend = 5 + 4 * np.sin(2 * np.pi * t / 1800)
        rng = np.random.default_rng(0)
        r = rng.poisson(trend[:, None] * np.ones((1, 3))).astype(float)
        z = tp.preprocess_rates(r, 0.5, detrend_sec=120.0)
        assert np.allclose(z.mean(axis=0), 0, atol=1e-9) and np.allclose(z.std(axis=0), 1)
        # slow component largely gone
        assert abs(np.corrcoef(z[:, 0], trend)[0, 1]) < 0.15


class TestStatuses:
    @pytest.fixture(scope="class")
    def ses(self):
        return _session(0, nbr_spread=26.0)

    def test_straddle_output_contract(self, ses):
        c = simulate_population(ses, np.random.default_rng(0), N_CELLS)
        r = _straddle(ses, c)
        assert r["status"] == "ok"
        assert 0 < r["p_value"] <= 1 and r["p_value"] >= r["p_floor"] > 0
        assert r["n_displacements"] >= 5 and len(r["null"]) == r["n_displacements"]
        assert r["n_pairs_straddle"] > 0 and r["n_pairs_same_side"] > 0

    def test_straddle_is_deterministic(self, ses):
        c = simulate_population(ses, np.random.default_rng(0), N_CELLS)
        assert _straddle(ses, c)["p_value"] == _straddle(ses, c)["p_value"]

    def test_straddle_reports_instead_of_guessing_when_data_are_sparse(self, ses):
        c = simulate_population(ses, np.random.default_rng(0), N_CELLS)
        r = tp.boundary_straddle_test(c[:150] / ses["dt"], ses["x"][:150], ses["y"][:150],
                                      ses["territory_map"], "focal", dt=ses["dt"])
        assert r["status"] != "ok" and np.isnan(r["p_value"])

    def test_straddle_rejects_mismatched_inputs(self, ses):
        c = simulate_population(ses, np.random.default_rng(0), N_CELLS)
        with pytest.raises(ValueError):
            tp.boundary_straddle_test(c, ses["x"][:-1], ses["y"], ses["territory_map"], "focal")

    def test_presence_needs_both_classes(self, ses):
        c = simulate_population(ses, np.random.default_rng(0), N_CELLS)
        always_home = np.where(np.isnan(ses["owner_present"]), np.nan, 1.0)
        r = tp.owner_presence_decoding(c / ses["dt"], ses["x"], ses["y"], ses["owner"],
                                       always_home, ses["owner_dist"], dt=ses["dt"])
        assert r["status"] == "insufficient_occupancy" and np.isnan(r["p_value"])

    def test_presence_p_is_add_one(self, ses):
        c = simulate_population(ses, np.random.default_rng(0), N_CELLS, frac_presence=1.0,
                                presence_ratio=3.0)
        r = _presence(ses, c, n_shifts=29)
        assert r["status"] == "ok"
        assert r["p_value"] >= 1 / 30 - 1e-12 and r["p_value"] > 0

    def test_decoder_gain_output_contract(self, ses):
        c = simulate_population(ses, np.random.default_rng(0), N_CELLS)
        r = _decoder(ses, c, n_displace_max=10)
        assert r["status"] == "ok"
        assert 0 <= r["ba_position"] <= 1 and 0 <= r["ba_position_plus_neural"] <= 1
        assert r["gain"] == pytest.approx(r["ba_position_plus_neural"] - r["ba_position"])


# ---------------------------------------------------------------------------
# Calibration and power (slow)
# ---------------------------------------------------------------------------

def _rates(fn, scenario_kw, n_sessions, *, nbr_spread=None, seed0=0, call_kw=None):
    ps = []
    for i in range(n_sessions):
        ses = _session(i + seed0, nbr_spread)
        rng = np.random.default_rng(seed0 + i)
        c = simulate_population(ses, rng, N_CELLS, **scenario_kw)
        r = fn(ses, c, **(call_kw or {}))
        if r["status"] == "ok":
            ps.append(r["p_value"])
    return np.asarray(ps)


def _assert_calibrated(ps, n_expected, label):
    assert len(ps) >= 0.8 * n_expected, f"{label}: only {len(ps)}/{n_expected} sessions testable"
    for alpha in (0.05, 0.01):
        fpr = np.mean(ps <= alpha)
        assert fpr <= _bound(len(ps), alpha), (
            f"{label}: FPR@{alpha}={fpr:.3f} > bound {_bound(len(ps), alpha):.3f} (n={len(ps)})")


SCENARIOS = {
    "uniform": {},
    "drift": {"drift_sd": 1.0},
    "boundary_dense": {"density": "boundary_dense"},
    "boundary_dense_drift": {"density": "boundary_dense", "drift_sd": 1.0},
}
STEP = {"frac_step": 0.4, "step_ratio": 1.6}


@pytest.mark.slow
class TestStraddleCalibration:
    N = 80

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_false_positives_within_bound(self, name):
        ps = _rates(_straddle, SCENARIOS[name], self.N)
        _assert_calibrated(ps, self.N, f"straddle/{name}")

    def test_not_wildly_conservative(self):
        ps = _rates(_straddle, SCENARIOS["uniform"], self.N)
        assert 0.3 < np.median(ps) < 0.7

    def test_detects_a_step_in_40pct_of_cells(self):
        ps = _rates(_straddle, STEP, 40)
        assert np.mean(ps <= 0.05) >= 0.7

    def test_detects_a_step_under_drift(self):
        ps = _rates(_straddle, {**STEP, "drift_sd": 1.0}, 40)
        assert np.mean(ps <= 0.05) >= 0.5

    def test_power_grows_with_the_fraction_of_modulated_cells(self):
        rates = [np.mean(_rates(_straddle, {"frac_step": f, "step_ratio": 1.6}, 30) <= 0.05)
                 for f in (0.0, 0.2, 0.5)]
        assert rates[0] < rates[2] - 0.3 and rates[1] <= rates[2] + 0.05


@pytest.mark.slow
class TestDecoderGainCalibration:
    N = 40
    KW = {"n_displace_max": 60}

    @pytest.mark.parametrize("name", ["uniform", "drift"])
    def test_false_positives_within_bound(self, name):
        ps = _rates(_decoder, SCENARIOS[name], self.N, call_kw=self.KW)
        _assert_calibrated(ps, self.N, f"decoder/{name}")

    def test_detects_a_step_in_40pct_of_cells(self):
        ps = _rates(_decoder, STEP, 30, call_kw=self.KW)
        assert np.mean(ps <= 0.05) >= 0.6

    @pytest.mark.xfail(strict=False, reason=(
        "documented limit: place-field density that is higher at the true boundary makes the "
        "real boundary look special to a displaced-boundary null; test 1 controls for this by "
        "matching pairs within the band, this decoder test does not"))
    def test_boundary_dense_place_fields_do_not_fool_it(self):
        ps = _rates(_decoder, SCENARIOS["boundary_dense"], self.N, call_kw=self.KW)
        _assert_calibrated(ps, self.N, "decoder/boundary_dense")


@pytest.mark.slow
class TestPresenceCalibration:
    N = 40
    SPREAD = 26.0
    KW = {"n_shifts": 99}

    def test_false_positives_within_bound_with_distance_information(self):
        # every cell carries information about focal-owner distance; none about presence
        ps = _rates(_presence, {"dist_gain": 0.6}, self.N, nbr_spread=self.SPREAD, call_kw=self.KW)
        _assert_calibrated(ps, self.N, "presence/dist_info")

    def test_false_positives_within_bound_under_drift(self):
        ps = _rates(_presence, {"drift_sd": 1.0, "dist_gain": 0.3}, self.N,
                    nbr_spread=self.SPREAD, call_kw=self.KW)
        _assert_calibrated(ps, self.N, "presence/drift")

    def test_detects_presence_coding_in_half_the_cells(self):
        ps = _rates(_presence, {"frac_presence": 0.5, "presence_ratio": 1.6}, 30,
                    nbr_spread=self.SPREAD, call_kw=self.KW)
        assert np.mean(ps <= 0.05) >= 0.7

    def test_distance_coding_alone_is_not_mistaken_for_presence(self):
        strong = _rates(_presence, {"dist_gain": 1.5}, 30, nbr_spread=self.SPREAD, call_kw=self.KW)
        assert np.mean(strong <= 0.05) <= _bound(len(strong), 0.05)
