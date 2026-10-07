"""Type-I calibration and power contract for ``ephys.territory_encoding``.

Written BEFORE the implementation. The module is skipped until
``ephys/territory_encoding.py`` exists; when it does, these tests define what
"calibrated" means and must pass before any real-data run.

Contract (the implementation must provide):

    blocked_folds(n, n_folds, purge_bins) -> iterator of (train_idx, test_idx)

    fit_territory_effect(counts, covariates, territory, *, effect="step",
                         dt=0.5, n_boot=199, n_folds=5, purge_bins=20,
                         band=None, seed=0) -> dict

  ``covariates``: dict of equal-length arrays ``x, y, speed, partner_dist``.
  ``territory`` : dict with ``own`` (bool) and ``signed_dist`` (float).
  ``effect``    : ``"step"`` (own vs not-own) or ``"gradient"`` (signed distance).
  ``band``      : if set, restrict to ``|signed_dist| < band``.
  Returned keys: ``status`` ("ok" | "insufficient_spikes" | ...), ``p_value``
  (add-one: >= 1/(n_boot+1), never 0, NaN when not ok), ``lrt_stat``,
  ``delta_ll_cv_per_spike``, ``effect_estimate`` (log rate ratio for the step,
  log-rate per unit distance for the gradient), ``n_spikes``, ``n_boot``.

Scenarios (see tests/_territory_sim.py):
  smooth    place field only
  straddle  strong place field centred ON the boundary, no territory effect
  drift     place field + slow gain drift, no territory effect
  step/grad genuine territory effect (with and without drift)

Calibration is judged over many independent sessions (the focal trajectory, and
therefore the territory covariate, is part of the randomness), against the
upper 99.9% binomial bound for the observed false-positive count.
"""

import numpy as np
import pytest
from scipy import stats

te = pytest.importorskip(
    "ephys.territory_encoding",
    reason="ephys/territory_encoding.py not written yet; these tests are its acceptance contract",
)

from tests._territory_sim import (  # noqa: E402
    CellSpec,
    boundary_point,
    covariates,
    make_session,
    simulate_counts,
    territory_inputs,
)

# Sizes are deliberately modest so the slow suite finishes in minutes; raise
# N_SESSIONS before trusting a borderline result.
N_SESSIONS = 30
CELLS_PER_SESSION = 6
N_BOOT = 199
ALPHAS = (0.05, 0.01)


def _fit(counts, ses, **kw):
    kw.setdefault("n_boot", N_BOOT)
    return te.fit_territory_effect(counts, covariates(ses), territory_inputs(ses),
                                   dt=ses["dt"], **kw)


def _scenario_spec(kind, ses, rng):
    center = (rng.uniform(10, 90), rng.uniform(10, 90))
    if kind == "smooth":
        return CellSpec(field_center=center)
    if kind == "drift":
        return CellSpec(field_center=center, drift_sd=1.0, drift_tau_sec=400.0)
    if kind == "straddle":
        return CellSpec(field_center=boundary_point(ses, rng), field_gain=1.8, field_sigma=12.0)
    if kind == "straddle_drift":
        return CellSpec(field_center=boundary_point(ses, rng), field_gain=1.8, field_sigma=12.0,
                        drift_sd=1.0, drift_tau_sec=400.0)
    raise ValueError(kind)


def _run(kind, *, effect="step", seed0=1000, n_sessions=N_SESSIONS, per=CELLS_PER_SESSION,
         spec_fn=None, fit_kw=None):
    ps, ests = [], []
    for i in range(n_sessions):
        ses = make_session(seed=seed0 + i)
        rng = np.random.default_rng(seed0 + i)
        for _ in range(per):
            spec = spec_fn(ses, rng) if spec_fn else _scenario_spec(kind, ses, rng)
            counts = simulate_counts(ses, spec, rng)
            if counts.sum() < 50:
                continue
            r = _fit(counts, ses, effect=effect, seed=i, **(fit_kw or {}))
            if r["status"] == "ok":
                ps.append(r["p_value"])
                ests.append(r["effect_estimate"])
    return np.asarray(ps), np.asarray(ests)


# ---------------------------------------------------------------------------
# Fast structural tests
# ---------------------------------------------------------------------------

class TestBlockedFolds:
    @pytest.mark.parametrize("n,k,purge", [(3600, 5, 20), (1000, 4, 0), (500, 5, 50)])
    def test_no_train_sample_within_purge_of_a_test_block(self, n, k, purge):
        for train, test in te.blocked_folds(n, k, purge):
            assert len(np.intersect1d(train, test)) == 0
            lo, hi = test.min(), test.max()
            assert not np.any((train >= lo - purge) & (train <= hi + purge))

    def test_test_blocks_are_contiguous_and_cover_everything_once(self):
        tests = [t for _, t in te.blocked_folds(1000, 5, 10)]
        for t in tests:
            assert np.array_equal(t, np.arange(t.min(), t.max() + 1))
        assert np.array_equal(np.sort(np.concatenate(tests)), np.arange(1000))

    def test_not_shuffled(self):
        tests = [t.min() for _, t in te.blocked_folds(1000, 5, 10)]
        assert tests == sorted(tests)


class TestOutputContract:
    @pytest.fixture(scope="class")
    def ses(self):
        return make_session(seed=7)

    def test_keys_and_types(self, ses):
        rng = np.random.default_rng(0)
        r = _fit(simulate_counts(ses, CellSpec(), rng), ses, n_boot=49)
        for k in ("status", "p_value", "lrt_stat", "delta_ll_cv_per_spike",
                  "effect_estimate", "n_spikes", "n_boot"):
            assert k in r, k
        assert r["status"] == "ok" and r["n_boot"] == 49

    def test_p_is_add_one_and_never_zero(self, ses):
        rng = np.random.default_rng(0)
        strong = CellSpec(step_log=np.log(4.0), base_hz=4.0)
        r = _fit(simulate_counts(ses, strong, rng), ses, n_boot=49)
        assert r["p_value"] >= 1 / 50 - 1e-12
        assert r["p_value"] > 0

    def test_too_few_spikes_is_not_reported_as_a_result(self, ses):
        rng = np.random.default_rng(0)
        r = _fit(simulate_counts(ses, CellSpec(base_hz=0.005, field_gain=0.0), rng), ses,
                 n_boot=49)
        assert r["status"] == "insufficient_spikes"
        assert np.isnan(r["p_value"])         # not 1.0 and not 0.0

    def test_deterministic_for_a_seed(self, ses):
        c = simulate_counts(ses, CellSpec(), np.random.default_rng(1))
        assert _fit(c, ses, n_boot=49, seed=3)["p_value"] == _fit(c, ses, n_boot=49, seed=3)["p_value"]

    def test_exclusivity_effect_runs_and_recovers_sign(self, ses):
        c = simulate_counts(ses, CellSpec(excl_slope=1.5, base_hz=3.0), np.random.default_rng(2))
        r = _fit(c, ses, effect="exclusivity", n_boot=99)
        assert r["status"] == "ok" and r["effect_estimate"] > 0

    def test_exclusivity_needs_the_covariate(self, ses):
        c = simulate_counts(ses, CellSpec(), np.random.default_rng(2))
        with pytest.raises(ValueError, match="exclusivity"):
            te.fit_territory_effect(c, covariates(ses), {"own": ses["own"], "signed_dist": ses["signed_dist"]},
                                    effect="exclusivity", dt=ses["dt"], n_boot=9)

    def test_territory_labels_are_actually_used(self, ses):
        c = simulate_counts(ses, CellSpec(step_log=np.log(3.0), base_hz=3.0),
                            np.random.default_rng(2))
        real = _fit(c, ses, n_boot=99)
        flipped = te.fit_territory_effect(
            c, covariates(ses), {"own": ~ses["own"], "signed_dist": -ses["signed_dist"]},
            dt=ses["dt"], n_boot=99)
        assert real["effect_estimate"] > 0 > flipped["effect_estimate"]


# ---------------------------------------------------------------------------
# Type-I calibration
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTypeICalibration:
    @pytest.mark.parametrize("kind", ["smooth", "straddle", "drift", "straddle_drift"])
    @pytest.mark.parametrize("effect", ["step", "gradient"])
    def test_false_positive_rate_within_binomial_bound(self, kind, effect):
        ps, _ = _run(kind, effect=effect)
        n = len(ps)
        assert n >= 0.8 * N_SESSIONS * CELLS_PER_SESSION, "too many cells dropped"
        for alpha in ALPHAS:
            upper = stats.binom.ppf(0.999, n, alpha) / n
            assert np.mean(ps <= alpha) <= upper, (
                f"{kind}/{effect}: FPR@{alpha}={np.mean(ps <= alpha):.3f} > bound {upper:.3f} (n={n})")

    @pytest.mark.parametrize("kind", ["smooth", "drift"])
    def test_p_values_not_wildly_conservative_or_skewed(self, kind):
        # A test that is calibrated only because it never rejects would pass the
        # bound above; guard against that with a loose uniformity check.
        ps, _ = _run(kind)
        assert 0.35 < np.median(ps) < 0.65
        assert stats.kstest(ps, "uniform").pvalue > 1e-4

    def test_boundary_band_restriction_stays_calibrated(self):
        ps, _ = _run("straddle_drift", fit_kw={"band": 10.0})
        n = len(ps)
        assert n >= 0.5 * N_SESSIONS * CELLS_PER_SESSION
        assert np.mean(ps <= 0.05) <= stats.binom.ppf(0.999, n, 0.05) / n


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

def _step_spec(ratio, drift=False, straddle=False):
    def fn(ses, rng):
        center = boundary_point(ses, rng) if straddle else (rng.uniform(10, 90), rng.uniform(10, 90))
        return CellSpec(field_center=center, base_hz=3.0, step_log=np.log(ratio),
                        drift_sd=1.0 if drift else 0.0, drift_tau_sec=400.0)
    return fn


@pytest.mark.slow
class TestPower:
    def test_detects_a_1p5x_step(self):
        ps, _ = _run(None, spec_fn=_step_spec(1.5))
        assert np.mean(ps <= 0.05) >= 0.8

    def test_detects_a_step_that_is_a_decrease(self):
        ps, ests = _run(None, spec_fn=_step_spec(1 / 1.5))
        assert np.mean(ps <= 0.05) >= 0.8
        assert np.median(ests) < 0

    def test_still_detects_a_step_on_top_of_a_boundary_straddling_field(self):
        ps, _ = _run(None, spec_fn=_step_spec(1.6, straddle=True))
        assert np.mean(ps <= 0.05) >= 0.6

    def test_still_detects_a_step_under_drift(self):
        ps, _ = _run(None, spec_fn=_step_spec(1.7, drift=True))
        assert np.mean(ps <= 0.05) >= 0.5

    def test_detects_a_gradient(self):
        spec = lambda ses, rng: CellSpec(field_center=(rng.uniform(10, 90), rng.uniform(10, 90)),
                                         base_hz=3.0, grad_per_unit=0.03)
        ps, ests = _run(None, effect="gradient", spec_fn=spec)
        assert np.mean(ps <= 0.05) >= 0.8
        assert np.median(ests) == pytest.approx(0.03, rel=0.35)

    def test_step_estimate_recovers_the_injected_log_ratio(self):
        _, ests = _run(None, spec_fn=_step_spec(1.5), n_sessions=15)
        assert np.median(ests) == pytest.approx(np.log(1.5), abs=0.12)

    def test_power_increases_with_effect_size(self):
        rates = [np.mean(_run(None, spec_fn=_step_spec(r), n_sessions=15)[0] <= 0.05)
                 for r in (1.0, 1.25, 1.6)]
        assert rates[0] < rates[1] <= rates[2] + 0.05
        assert rates[2] > rates[0] + 0.3

    def test_misaligned_territory_labels_lose_the_effect(self):
        # same spikes, labels circularly shifted by an hour of bins: detection
        # must collapse, i.e. the test is sensitive to the true labelling and not
        # just to any slow covariate.
        hits, n = 0, 0
        for i in range(15):
            ses = make_session(seed=3000 + i)
            rng = np.random.default_rng(i)
            for _ in range(CELLS_PER_SESSION):
                c = simulate_counts(ses, _step_spec(1.8)(ses, rng), rng)
                shift = len(c) // 2 + int(rng.integers(0, len(c) // 4))
                bad = {"own": np.roll(ses["own"], shift), "signed_dist": np.roll(ses["signed_dist"], shift)}
                r = te.fit_territory_effect(c, covariates(ses), bad, dt=ses["dt"],
                                            n_boot=N_BOOT, seed=i)
                if r["status"] == "ok":
                    n += 1
                    hits += r["p_value"] <= 0.05
        assert hits / n < 0.2


# ---------------------------------------------------------------------------
# Continuous exclusivity covariate
# ---------------------------------------------------------------------------
# Exclusivity (the focal's share of everyone's occupancy) is smooth in position, so
# the place model absorbs most of it. Measured behaviour (30 sessions x 6 cells per
# null scenario): 0/540 false positives at alpha 0.05 and a null median p of ~0.73,
# i.e. conservative rather than uniform - so there is no uniformity check here, and
# the power tests below are what stop "never rejects" from passing.

def _excl_spec(slope, drift=False):
    # mean exclusivity in the simulator is ~0.75, so exp(slope * 0.75) changes the mean rate; keep the
    # mean rate (hence the spike count, hence the power) equal across slopes so +/- are comparable
    def fn(ses, rng):
        return CellSpec(field_center=(rng.uniform(10, 90), rng.uniform(10, 90)),
                        base_hz=3.0 * np.exp(-slope * float(np.nanmean(ses["exclusivity"]))),
                        excl_slope=slope, drift_sd=1.0 if drift else 0.0, drift_tau_sec=400.0)
    return fn


@pytest.mark.slow
class TestExclusivityEffect:
    @pytest.mark.parametrize("kind", ["smooth", "straddle", "drift", "straddle_drift"])
    def test_false_positive_rate_within_binomial_bound(self, kind):
        ps, _ = _run(kind, effect="exclusivity")
        n = len(ps)
        assert n >= 0.8 * N_SESSIONS * CELLS_PER_SESSION
        for alpha in ALPHAS:
            assert np.mean(ps <= alpha) <= stats.binom.ppf(0.999, n, alpha) / n, (kind, alpha)

    # Power is asymmetric in the sign of the slope even at matched spike counts (measured, 15 sessions
    # x 6 cells, ~8k spikes: +1.0 -> 0.51, -1.0 -> 0.87, +1.5 -> 0.96, -1.5 -> 1.0). Exclusivity is
    # skewed (most bins near 1, a rare low tail); a positive slope makes the rare low bins fire
    # *less*, so they carry fewer spikes and less information about the contrast. Thresholds below
    # are set from the weaker direction.
    def test_detects_a_positive_slope_of_one_half_the_time(self):
        ps, ests = _run(None, effect="exclusivity", spec_fn=_excl_spec(1.0), n_sessions=15)
        assert np.mean(ps <= 0.05) >= 0.35
        assert np.median(ests) == pytest.approx(1.0, abs=0.25)

    @pytest.mark.parametrize("slope", [1.5, -1.5])
    def test_detects_a_slope_of_one_and_a_half(self, slope):
        ps, ests = _run(None, effect="exclusivity", spec_fn=_excl_spec(slope), n_sessions=15)
        assert np.mean(ps <= 0.05) >= 0.85
        assert np.median(ests) == pytest.approx(slope, abs=0.3)

    def test_detects_a_slope_under_drift(self):
        ps, _ = _run(None, effect="exclusivity", spec_fn=_excl_spec(1.5, drift=True), n_sessions=15)
        assert np.mean(ps <= 0.05) >= 0.5

    def test_power_increases_with_slope(self):
        rates = [np.mean(_run(None, effect="exclusivity", spec_fn=_excl_spec(s), n_sessions=12)[0] <= 0.05)
                 for s in (0.0, 0.5, 1.5)]
        assert rates[0] < rates[1] < rates[2] + 1e-9
        assert rates[2] > 0.9 and rates[0] < 0.1
