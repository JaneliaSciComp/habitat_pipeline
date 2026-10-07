"""Checks on the territory simulator and on the harness itself.

These run now (no GLM needed). The last class matters most: it shows that the
simulated scenarios *can* expose a mis-calibrated test, by running a plain
Poisson LRT that is known to be naive about slow drift. If this class ever
stops failing the naive tester, the calibration tests for the real analysis
have lost their teeth and would pass vacuously.
"""

import numpy as np
import pytest
from scipy import stats

from tests._territory_sim import (
    CellSpec,
    boundary_point,
    cell_rate,
    make_session,
    naive_poisson_lrt,
    simulate_counts,
)


@pytest.fixture(scope="module")
def session():
    return make_session(seed=1)


class TestSession:
    def test_focal_spends_time_on_both_sides_of_the_boundary(self, session):
        own = session["own"].mean()
        band = (np.abs(session["signed_dist"]) < 10).mean()
        assert 0.3 < own < 0.85
        assert band > 0.10      # enough boundary-adjacent time to test a step locally

    def test_territory_map_never_saw_the_test_chunk(self, session):
        p = session["territory_map"].parameters
        assert p["test_chunk"] == "test" and "test" not in p["train_chunks"]

    def test_signed_distance_agrees_with_own(self, session):
        assert (session["signed_dist"][session["own"]] > 0).all()
        assert (session["signed_dist"][~session["own"]] < 0).all()

    def test_deterministic_for_a_seed(self):
        a, b = make_session(seed=5), make_session(seed=5)
        assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["own"], b["own"])

    def test_boundary_point_is_on_the_boundary(self, session):
        bx, by = boundary_point(session)
        i = np.argmin(np.hypot(session["x"] - bx, session["y"] - by))
        assert abs(session["signed_dist"][i]) < 4.0


class TestCells:
    def test_step_scales_rate_inside_own_territory_only(self, session):
        rng = np.random.default_rng(0)
        flat = CellSpec(field_gain=0.0, step_log=0.0)
        step = CellSpec(field_gain=0.0, step_log=np.log(2.0))
        r0 = cell_rate(session, flat, rng)
        r1 = cell_rate(session, step, rng)
        assert np.allclose(r1[session["own"]] / r0[session["own"]], 2.0)
        assert np.allclose(r1[~session["own"]], r0[~session["own"]])

    def test_null_cell_rate_ignores_territory(self, session):
        rng = np.random.default_rng(0)
        r = cell_rate(session, CellSpec(field_gain=0.0), rng)
        assert np.ptp(r) == pytest.approx(0.0)

    def test_gradient_is_monotone_in_signed_distance(self, session):
        rng = np.random.default_rng(0)
        r = cell_rate(session, CellSpec(field_gain=0.0, grad_per_unit=0.05), rng)
        d = np.clip(session["signed_dist"], -30, 30)
        assert stats.spearmanr(d, r).statistic > 0.99

    def test_drift_is_slow_and_mean_one_in_log(self, session):
        rng = np.random.default_rng(3)
        r = cell_rate(session, CellSpec(field_gain=0.0, drift_sd=0.5), rng)
        lr = np.log(r) - np.log(2.0)
        assert 0.1 < lr.std() < 1.0
        ac1 = np.corrcoef(lr[:-1], lr[1:])[0, 1]
        assert ac1 > 0.99                      # slow relative to the 0.5 s bin

    def test_counts_are_poisson_with_the_right_mean(self, session):
        rng = np.random.default_rng(4)
        spec = CellSpec(field_gain=0.0, base_hz=3.0)
        c = simulate_counts(session, spec, rng)
        assert c.mean() == pytest.approx(3.0 * session["dt"], rel=0.05)
        assert c.var() == pytest.approx(c.mean(), rel=0.15)


class TestHarnessHasTeeth:
    """A naive Poisson LRT must be visibly mis-calibrated under slow drift."""

    N_SESSIONS, PER_SESSION = 25, 6

    def _fpr(self, make_spec, alpha):
        ps = []
        for i in range(self.N_SESSIONS):
            ses = make_session(seed=200 + i)
            rng = np.random.default_rng(i)
            for _ in range(self.PER_SESSION):
                counts = simulate_counts(ses, make_spec(ses, rng), rng)
                if counts.sum() >= 50:
                    ps.append(naive_poisson_lrt(counts, ses))
        ps = np.asarray(ps)
        return len(ps), float(np.mean(ps <= alpha))

    @staticmethod
    def _rand_center(rng):
        return (rng.uniform(10, 90), rng.uniform(10, 90))

    @pytest.mark.slow
    def test_naive_lrt_is_inflated_under_drift(self):
        n, fpr = self._fpr(lambda s, r: CellSpec(field_center=self._rand_center(r),
                                                 drift_sd=1.0, drift_tau_sec=400.0), 0.01)
        upper = stats.binom.ppf(0.999, n, 0.01) / n
        assert fpr > upper, f"naive LRT FPR@0.01={fpr:.3f} not above binomial bound {upper:.3f}"

    @pytest.mark.slow
    def test_naive_lrt_is_not_broken_without_drift(self):
        # sanity: the simulator itself does not manufacture false positives, so
        # the inflation above is attributable to drift and not to the generator.
        n, fpr = self._fpr(lambda s, r: CellSpec(field_center=self._rand_center(r)), 0.05)
        upper = stats.binom.ppf(0.999, n, 0.05) / n
        assert fpr <= upper, f"FPR@0.05={fpr:.3f} above {upper:.3f} even without drift"
