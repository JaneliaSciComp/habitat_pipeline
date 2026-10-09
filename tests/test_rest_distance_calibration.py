"""Calibration and power for ephys.rest_distance_coding (partner distance at a fixed resting spot).

Judged over independent synthetic sessions against the 99.9% binomial bound, one cell per session for the
single-cell test (cells of a session share the same distance trace, so their p-values are not independent).
The naive parametric correlation test is run alongside as the harness 'teeth' check: it must be inflated in the
scenarios the real test survives. One scenario is documented as a known limit, not hidden: when arousal is
measured with noise, the cell's arousal dependence leaks into the distance test.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ephys import rest_distance_coding as rd
from tests._rest_sim import DT, RestSpec, make_rest_session


def _bound(n, p, z=3.09):
    return n * p + z * np.sqrt(n * p * (1 - p))


def _pvals(spec, n_sess, n_shifts=199, pop=False, naive=False, seed0=9000):
    cell, popp, nv = [], [], []
    for s in range(n_sess):
        S = make_rest_session(seed0 + s, spec)
        P = rd.prepare(S["counts"], S["d"], S["nuis"], S["rest"])
        cell.append(rd.single_cell_tests(P, n_shifts=n_shifts, seed=s).p_value.iloc[0])
        if pop:
            popp.append(rd.population_test(P, n_shifts=n_shifts, seed=s)["p_value"])
        if naive:
            nv.append(stats.pearsonr(P["Y"][:, 0], P["d"])[1])
    return np.array(cell), np.array(popp), np.array(nv)


BASE = RestSpec(n_cells=12)


# ---- unit behaviour ---------------------------------------------------------

def test_rest_segments_drops_short_runs_and_numbers_the_rest():
    rest = np.r_[np.ones(200, bool), np.zeros(5, bool), np.ones(50, bool), np.zeros(5, bool), np.ones(300, bool)]
    seg = rd.rest_segments(rest, dt=DT, min_seg_sec=60.0)
    assert list(np.unique(seg)) == [-1, 0, 1]
    assert (seg[205:255] == -1).all()                  # 25 s run is dropped
    assert (seg[:200] == 0).all() and (seg[260:] == 1).all()


def test_highpass_removes_a_slow_trend_but_keeps_fast_structure():
    t = np.arange(1200) * DT
    slow, fast = 5.0 * np.sin(2 * np.pi * t / 600.0), np.sin(2 * np.pi * t / 10.0)
    seg = np.zeros(len(t), int)
    out = rd.highpass(slow + fast, seg, dt=DT, window_sec=60.0)
    mid = slice(200, 1000)                              # away from the segment edges
    assert np.corrcoef(out[mid], fast[mid])[0, 1] > 0.95
    assert abs(np.corrcoef(out[mid], slow[mid])[0, 1]) < 0.3


def test_shifts_stay_inside_their_segment():
    d = np.r_[np.arange(300.0), 1000 + np.arange(300.0)]
    seg = np.r_[np.zeros(300, int), np.ones(300, int)]
    out = rd._shifted(d, seg, 20, np.random.default_rng(0))
    assert (out[:300] < 300).all() and (out[300:] >= 1000).all()


def test_too_little_rest_returns_nan_not_a_p_value():
    S = make_rest_session(1, RestSpec(n_seg=1, seg_sec=60.0))
    P = rd.prepare(S["counts"], S["d"], S["nuis"], S["rest"])
    assert P["n"] < 200
    assert rd.population_test(P)["status"] == "insufficient_rest"


# ---- calibration ------------------------------------------------------------

@pytest.mark.slow
class TestTypeI:
    N = 400

    def test_harness_has_teeth(self):
        # a correlation test that ignores autocorrelation must be badly inflated under slow drift
        _, _, nv = _pvals(RestSpec(slow_sd=2.0, drift_sd=1.0), 200, naive=True)
        assert np.mean(nv <= 0.05) > 0.2

    @pytest.mark.parametrize("name,spec", [
        ("clean", BASE),
        ("slow_drift", RestSpec(slow_sd=2.0, drift_sd=1.0)),
        ("arousal_measured_exactly", RestSpec(rho=0.6, arousal_gain=0.5)),
    ])
    def test_single_cell_false_positives(self, name, spec):
        cell, _, _ = _pvals(spec, self.N)
        assert np.sum(cell <= 0.05) <= _bound(self.N, 0.05), (name, np.mean(cell <= 0.05))

    @pytest.mark.parametrize("spec", [BASE, RestSpec(slow_sd=2.0, drift_sd=1.0)])
    def test_population_false_positives(self, spec):
        _, popp, _ = _pvals(spec, 200, pop=True)
        assert np.sum(popp <= 0.05) <= _bound(200, 0.05)

    def test_known_limit_noisy_arousal_leaks_into_the_distance_test(self):
        # Not a bug to fix in the test: if arousal drives cells and the nuisance measures it with noise,
        # distance (correlated with arousal) inherits the dependence. Pin the size of the problem.
        cell, _, _ = _pvals(RestSpec(rho=0.6, arousal_gain=0.5, nuis_noise=0.7), 150)
        assert np.mean(cell <= 0.05) > 0.3


# ---- power ------------------------------------------------------------------

@pytest.mark.slow
class TestPower:
    def test_single_cell_power_by_effect_size(self):
        pw = {b: np.mean(_pvals(RestSpec(beta=b, n_cells=2), 150)[0] <= 0.05) for b in (0.05, 0.1, 0.2)}
        assert pw[0.1] >= 0.6 and pw[0.2] >= 0.95 and pw[0.05] < pw[0.1] < pw[0.2], pw

    def test_negative_and_positive_effects_equally_detected(self):
        up = np.mean(_pvals(RestSpec(beta=0.1, n_cells=2), 150)[0] <= 0.05)
        dn = np.mean(_pvals(RestSpec(beta=-0.1, n_cells=2), 150)[0] <= 0.05)
        assert abs(up - dn) < 0.15, (up, dn)

    def test_population_detects_a_diffuse_effect(self):
        popp = _pvals(RestSpec(beta=0.05, n_cells=12), 100, pop=True)[1]
        assert np.mean(popp <= 0.05) >= 0.6
