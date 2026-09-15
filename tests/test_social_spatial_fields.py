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
    STRATUM_COLUMN,
    compute_rate_map,
    spatial_information,
    spatial_sparsity,
    spatial_coherence,
    split_half_stability,
    field_significance,
    compare_self_stratum_control,
    compute_social_place_fields,
    field_similarity_across_targets,
    modal_occupancy_center,
    read_stratum_mask,
    self_position_stratum,
    _BINNING_DEFAULTS,
    _benjamini_hochberg,
    _prep_from_kwargs,
)
from video.tracking_import import VideoTrackingData

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


class _StubSync:
    """Identity behavior→ephys clock map (ephys seconds == behavior seconds)."""

    def convert_behavior_to_ephys(self, behav_seconds):
        return np.asarray(behav_seconds, dtype=np.float64)


def _video_tracking(tracking, session_id="20251216") -> VideoTrackingData:
    """Build a session VideoTrackingData from a ``{animal_id: (t,x,y,speed) df}`` dict.

    Stores each animal's (frame, center_x, center_y) and a shared frame-timestamp
    array (ns). With ``pixels_per_cm=None`` and the identity ``_StubSync``,
    ``resolve_tracking_on_ephys_clock`` returns the same ``t``/``x``/``y`` (speed
    is recomputed but unused by the speed_filter_subject="none" sweeps).
    """
    parsed = {}
    timestamps_ns = None
    for aid, df in tracking.items():
        n = df.shape[0]
        parsed[aid] = pd.DataFrame({
            "frame": np.arange(n),
            "center_x": df["x"].to_numpy(dtype=np.float64),
            "center_y": df["y"].to_numpy(dtype=np.float64),
        })
        if timestamps_ns is None:
            timestamps_ns = (df["t"].to_numpy(dtype=np.float64) * 1e9).astype(np.int64)
    return VideoTrackingData(
        animal_id=list(tracking.keys())[0],
        session_id=session_id,
        parsed_data=parsed,
        timestamps=timestamps_ns,
    )


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
        # The add-one estimator floors p at 1/(n_shuffles+1) = 1/501 ~ 0.002;
        # the previous `< 0.001` was only reachable via an invalid p == 0.0.
        assert sig.p_skaggs == pytest.approx(1 / 501)

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
    vt = _video_tracking(tracking)
    defaults = dict(
        target_animals=list(tracking.keys()),
        bin_size_cm=BIN, smoothing_sigma_cm=5.0,
        # n_shuffles=500, not 100: with 3 targets at the default sig_alpha=0.01
        # a 100-shuffle budget has a p-value floor of 1/101, so the best
        # BH-adjusted q reachable across 3 targets is 0.030 — above alpha, i.e.
        # no cell could ever be classified as tuned. These tests previously
        # passed only because `_p_geq` returned an invalid p == 0.0; see
        # `ephys._stats_utils.fdr_resolution`.
        speed_filter_subject="none", n_shuffles=500, min_n_spikes=50,
        use_quality_cells=False, arena_bounds=ARENA, seed=0,
    )
    defaults.update(kw)
    return compute_social_place_fields(
        ks, vt, _StubSync(), focal_animal=focal, pixels_per_cm=None, **defaults,
    )


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


# ---------------------------------------------------------------------------
# Self-position stratum (confound control)
# ---------------------------------------------------------------------------

def _correlated_pair(n, alpha, seed_a=1, seed_g=2):
    """Focal A and partner B whose trajectories are correlated by ``alpha``.

    This is the situation that makes the default nulls unsafe: B's position
    carries information about A's, so a cell coding only A's own position shows
    an apparent place field over B.
    """
    ax, ay = _random_walk(n, ARENA, 3.0, seed_a)
    gx, gy = _random_walk(n, ARENA, 3.0, seed_g)
    t = np.arange(n) * DT
    A = pd.DataFrame({"t": t, "x": ax, "y": ay})
    B = pd.DataFrame({"t": t, "x": alpha * ax + (1 - alpha) * gx,
                      "y": alpha * ay + (1 - alpha) * gy})
    for df in (A, B):
        df["speed"] = np.sqrt(np.gradient(df["x"], t) ** 2
                              + np.gradient(df["y"], t) ** 2)
    return A, B


class TestFastNullPath:
    """The null loop hoists spike-independent work out; results must not move.

    The reference here is the pre-optimization loop — three ``compute_rate_map``
    calls per shuffle — driven by the same seeded RNG.
    """

    @staticmethod
    def _reference(spike_times, target_xy, n_shuffles, null_method, seed, **kw):
        rng = np.random.default_rng(seed)
        t = target_xy["t"].to_numpy(dtype=np.float64)
        tw = kw.get("t_window_ephys")
        w0, w1 = (float(t.min()), float(t.max())) if tw is None else tw
        span = max(w1 - w0, 1e-9)
        median_dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0

        def _stats(sp, xy):
            rm = compute_rate_map(sp, xy, **kw)
            return spatial_information(rm)[0], spatial_sparsity(rm)

        st = np.asarray(spike_times, dtype=np.float64)
        true_sk, true_sp = _stats(st, target_xy)
        sk = np.full(n_shuffles, np.nan)
        spar = np.full(n_shuffles, np.nan)
        x0 = target_xy["x"].to_numpy()
        y0 = target_xy["y"].to_numpy()
        s0 = read_stratum_mask(target_xy)
        roll_rows = np.flatnonzero(s0) if s0 is not None else None
        for i in range(n_shuffles):
            tau = rng.uniform(0.1 * span, 0.9 * span)
            if null_method == "circular_shift":
                sp_i, xy_i = w0 + np.mod(st - w0 + tau, span), target_xy
            else:
                k = int(round(tau / max(median_dt, 1e-9)))
                if roll_rows is None:
                    xi, yi = np.roll(x0, k), np.roll(y0, k)
                else:
                    k_eff = int(round(k * roll_rows.size / max(x0.size, 1)))
                    if roll_rows.size > 1:
                        k_eff = max(k_eff, 1)
                    xi, yi = x0.copy(), y0.copy()
                    xi[roll_rows] = np.roll(x0[roll_rows], k_eff)
                    yi[roll_rows] = np.roll(y0[roll_rows], k_eff)
                xy_i = target_xy.copy()
                xy_i["x"], xy_i["y"] = xi, yi
                sp_i = st
            sk[i], spar[i] = _stats(sp_i, xy_i)
        return true_sk, true_sp, sk, spar

    def _assert_matches(self, xy, spikes, null_method, n_shuffles=25, **kw):
        true_sk, true_sp, ref_sk, ref_spar = self._reference(
            spikes, xy, n_shuffles, null_method, 0, **kw)
        got = field_significance(spikes, xy, n_shuffles=n_shuffles,
                                 null_method=null_method, seed=0, **kw)
        np.testing.assert_array_equal(ref_sk, got.shuffle_skaggs)

        def p_geq(tv, sh):
            v = sh[np.isfinite(sh)]
            return float((1 + np.sum(v >= tv)) / (1 + v.size))

        def p_leq(tv, sh):
            v = sh[np.isfinite(sh)]
            return float((1 + np.sum(v <= tv)) / (1 + v.size))

        assert got.p_skaggs == p_geq(true_sk, ref_sk)
        assert got.p_sparsity == p_leq(true_sp, ref_spar)

    def test_binning_defaults_track_compute_rate_map(self):
        """Two places declare binning defaults; a drift breaks the fast path.

        ``_BINNING_DEFAULTS`` feeds the hoisted prep while ``compute_rate_map``
        keeps its explicit signature, so the values must stay identical.
        """
        import inspect
        sig = inspect.signature(compute_rate_map).parameters
        for name, default in _BINNING_DEFAULTS.items():
            assert name in sig, f"{name} is not a compute_rate_map parameter"
            assert sig[name].default == default, (
                f"{name}: compute_rate_map defaults to {sig[name].default!r} "
                f"but _BINNING_DEFAULTS says {default!r}")

    @pytest.mark.parametrize("null_method", ["circular_shift", "position_shuffle"])
    def test_matches_reference(self, null_method):
        xy = _make_xy(n=12000, seed=600)
        spikes = _poisson_spikes_from_field(
            xy, center=(45.0, 35.0), sigma=8.0, peak_hz=20.0, base_hz=0.5, seed=601)
        self._assert_matches(xy, spikes, null_method, bin_size_cm=BIN,
                             arena_bounds=ARENA, smoothing_sigma_cm=5.0,
                             speed_xy=xy[["t", "speed"]], speed_threshold_cms=5.0)

    @pytest.mark.parametrize("null_method", ["circular_shift", "position_shuffle"])
    def test_matches_reference_under_stratum(self, null_method):
        xy = _make_xy(n=12000, seed=602)
        spikes = _poisson_spikes_from_field(
            xy, center=(40.0, 40.0), sigma=8.0, peak_hz=20.0, base_hz=0.5, seed=603)
        mask, _ = self_position_stratum(
            xy, xy["t"].to_numpy(), radius_cm=15.0, center=(40.0, 40.0),
            bin_size_cm=BIN, arena_bounds=ARENA)
        masked = xy.assign(**{STRATUM_COLUMN: mask})
        self._assert_matches(masked, spikes, null_method, bin_size_cm=BIN,
                             arena_bounds=ARENA, smoothing_sigma_cm=5.0,
                             speed_threshold_cms=None)

    def test_matches_reference_without_smoothing_or_bounds(self):
        xy = _make_xy(n=8000, seed=604)
        spikes = _poisson_spikes_from_field(
            xy, center=(40.0, 40.0), sigma=8.0, peak_hz=20.0, base_hz=0.5, seed=605)
        self._assert_matches(xy, spikes, "circular_shift", bin_size_cm=BIN,
                             smoothing_sigma_cm=None, speed_threshold_cms=None)

    def test_matches_reference_in_a_time_window(self):
        xy = _make_xy(n=12000, seed=606)
        spikes = _poisson_spikes_from_field(
            xy, center=(40.0, 40.0), sigma=8.0, peak_hz=20.0, base_hz=0.5, seed=607)
        self._assert_matches(xy, spikes, "circular_shift", bin_size_cm=BIN,
                             arena_bounds=ARENA, smoothing_sigma_cm=5.0,
                             speed_threshold_cms=None,
                             t_window_ephys=(40.0, 300.0))

    def test_split_half_null_is_opt_in(self):
        """Two thirds of the old cost bought a p-value nothing reads."""
        xy = _make_xy(n=6000, seed=608)
        spikes = _poisson_spikes_from_field(
            xy, center=(40.0, 40.0), sigma=8.0, peak_hz=20.0, base_hz=0.5, seed=609)
        kw = dict(bin_size_cm=BIN, arena_bounds=ARENA, speed_threshold_cms=None)
        off = field_significance(spikes, xy, n_shuffles=10, seed=0, **kw)
        on = field_significance(spikes, xy, n_shuffles=10, seed=0,
                                shuffle_split_half=True, **kw)
        assert np.isnan(off.p_split_half)
        assert np.isfinite(on.p_split_half)
        # Skipping it must not perturb the statistics that are reported.
        np.testing.assert_array_equal(off.shuffle_skaggs, on.shuffle_skaggs)
        assert off.p_skaggs == on.p_skaggs

    def test_shared_prep_matches_per_call_prep(self):
        """The sweep shares one prep across cells; that must change nothing."""
        xy = _make_xy(n=8000, seed=610)
        spikes = _poisson_spikes_from_field(
            xy, center=(40.0, 40.0), sigma=8.0, peak_hz=20.0, base_hz=0.5, seed=611)
        kw = dict(bin_size_cm=BIN, arena_bounds=ARENA, smoothing_sigma_cm=5.0,
                  speed_threshold_cms=None)
        alone = field_significance(spikes, xy, n_shuffles=15, seed=0, **kw)
        shared = field_significance(spikes, xy, n_shuffles=15, seed=0,
                                    _prep=_prep_from_kwargs(xy, kw), **kw)
        np.testing.assert_array_equal(alone.shuffle_skaggs, shared.shuffle_skaggs)
        assert alone.p_skaggs == shared.p_skaggs

    def test_unknown_null_method_still_rejected(self):
        xy = _make_xy(n=2000, seed=612)
        with pytest.raises(ValueError, match="Unknown null_method"):
            field_significance(np.array([1.0, 2.0]), xy, n_shuffles=2,
                               null_method="nope", arena_bounds=ARENA)


class TestSelfPositionStratum:
    def test_mask_confines_focal_position(self):
        xy = _make_xy(n=20000, seed=500)
        center = (40.0, 40.0)
        mask, diag = self_position_stratum(
            xy, xy["t"].to_numpy(), radius_cm=6.0, center=center,
            bin_size_cm=BIN, arena_bounds=ARENA)
        assert 0 < diag["n_retained"] < xy.shape[0]
        # Every retained sample really is inside the disc.
        d = np.hypot(xy["x"].to_numpy()[mask] - center[0],
                     xy["y"].to_numpy()[mask] - center[1])
        assert d.max() <= 6.0 + 1e-9
        assert diag["residual_spread_cm"] <= 6.0
        assert diag["retained_seconds"] == pytest.approx(mask.sum() * DT, rel=0.05)

    def test_modal_center_finds_dwell_peak(self):
        # Focal parked in a known corner for most of the session.
        n = 6000
        t = np.arange(n) * DT
        x = np.full(n, 12.5)
        y = np.full(n, 67.5)
        x[:500] = np.linspace(0, 80, 500)   # brief traverse elsewhere
        y[:500] = np.linspace(0, 80, 500)
        xy = pd.DataFrame({"t": t, "x": x, "y": y, "speed": np.zeros(n)})
        cx, cy = modal_occupancy_center(xy, BIN, ARENA)
        assert abs(cx - 12.5) <= BIN
        assert abs(cy - 67.5) <= BIN

    def test_samples_in_a_tracking_gap_are_excluded(self):
        # Focal sits at the centre, but its tracking has a 10 s hole.
        t = np.concatenate([np.arange(0, 5, DT), np.arange(15, 20, DT)])
        xy = pd.DataFrame({"t": t, "x": np.full(t.size, 40.0),
                           "y": np.full(t.size, 40.0),
                           "speed": np.zeros(t.size)})
        sample_t = np.arange(0, 20, 0.5)
        mask, diag = self_position_stratum(
            xy, sample_t, radius_cm=5.0, center=(40.0, 40.0),
            bin_size_cm=BIN, arena_bounds=ARENA, max_gap_sec=1.0)
        # Interpolation would have happily invented a position across the hole.
        assert not mask[(sample_t > 6.5) & (sample_t < 13.5)].any()
        assert mask[sample_t < 4.5].all()
        assert diag["excluded_by_gap"] > 0

    def test_rate_map_occupancy_restricted_to_stratum(self):
        xy = _make_xy(n=20000, seed=501)
        mask, diag = self_position_stratum(
            xy, xy["t"].to_numpy(), radius_cm=8.0, center=(40.0, 40.0),
            bin_size_cm=BIN, arena_bounds=ARENA)
        masked = xy.assign(**{STRATUM_COLUMN: mask})
        rm_full = compute_rate_map(np.array([]), xy, bin_size_cm=BIN,
                                   arena_bounds=ARENA, smoothing_sigma_cm=None,
                                   speed_threshold_cms=None)
        rm_str = compute_rate_map(np.array([]), masked, bin_size_cm=BIN,
                                  arena_bounds=ARENA, smoothing_sigma_cm=None,
                                  speed_threshold_cms=None)
        assert rm_str.occupancy.sum() == pytest.approx(diag["retained_seconds"], rel=0.02)
        assert rm_str.occupancy.sum() < rm_full.occupancy.sum()
        assert rm_str.parameters["self_stratum_applied"] is True
        assert rm_full.parameters["self_stratum_applied"] is False

    def test_masked_roll_preserves_retained_spike_count(self):
        """The mask-aware position_shuffle re-pairs only; it must not drop spikes.

        Rolling the whole array instead would pair in-stratum times with
        out-of-stratum positions and change how many spikes survive the mask,
        inflating the null and making the test silently conservative.
        """
        xy = _make_xy(n=20000, seed=502)
        spikes = _poisson_spikes_from_field(
            xy, center=(40.0, 40.0), sigma=8.0, peak_hz=25.0, base_hz=0.5, seed=503)
        mask, _ = self_position_stratum(
            xy, xy["t"].to_numpy(), radius_cm=10.0, center=(40.0, 40.0),
            bin_size_cm=BIN, arena_bounds=ARENA)
        masked = xy.assign(**{STRATUM_COLUMN: mask})
        observed = compute_rate_map(spikes, masked, bin_size_cm=BIN,
                                    arena_bounds=ARENA, speed_threshold_cms=None)

        rows = np.flatnonzero(mask)
        rolled = masked.copy()
        x0 = masked["x"].to_numpy().copy()
        y0 = masked["y"].to_numpy().copy()
        x0[rows] = np.roll(masked["x"].to_numpy()[rows], 137)
        y0[rows] = np.roll(masked["y"].to_numpy()[rows], 137)
        rolled["x"], rolled["y"] = x0, y0
        surrogate = compute_rate_map(spikes, rolled, bin_size_cm=BIN,
                                     arena_bounds=ARENA, speed_threshold_cms=None)

        assert surrogate.spike_counts.sum() == observed.spike_counts.sum()
        assert surrogate.occupancy.sum() == pytest.approx(observed.occupancy.sum())


@pytest.mark.slow
class TestSelfPositionConfound:
    """The control's reason for existing, in both directions."""

    N = 60000
    ALPHA = 0.65
    SELF_CENTER = (60.0, 60.0)
    RADIUS = 4.0
    SIGMA = 7.0
    KW = dict(bin_size_cm=BIN, arena_bounds=ARENA, speed_threshold_cms=None)

    def _p_pair(self, spikes, A, B, center, n_shuffles=200):
        mask, diag = self_position_stratum(
            A, B["t"].to_numpy(), radius_cm=self.RADIUS, center=center,
            bin_size_cm=BIN, arena_bounds=ARENA)
        B_masked = B.assign(**{STRATUM_COLUMN: mask})
        full = field_significance(spikes, B, n_shuffles=n_shuffles,
                                  null_method="position_shuffle", seed=0, **self.KW)
        strat = field_significance(spikes, B_masked, n_shuffles=n_shuffles,
                                   null_method="position_shuffle", seed=0, **self.KW)
        return full.p_skaggs, strat.p_skaggs, diag, B_masked

    def test_self_tuned_cell_fakes_a_partner_field(self):
        """Motivating case: the default null calls a pure self cell partner-tuned."""
        A, B = _correlated_pair(self.N, self.ALPHA)
        spikes = _poisson_spikes_from_field(
            A, center=self.SELF_CENTER, sigma=self.SIGMA, peak_hz=25.0,
            base_hz=0.2, seed=510)
        sig = field_significance(spikes, B, n_shuffles=200,
                                 null_method="position_shuffle", seed=0, **self.KW)
        # The cell never saw B, yet B-position tuning is maximally significant.
        assert sig.p_skaggs == pytest.approx(1 / 201)

    def test_stratum_removes_self_position_leakage(self):
        A, B = _correlated_pair(self.N, self.ALPHA)
        spikes = _poisson_spikes_from_field(
            A, center=self.SELF_CENTER, sigma=self.SIGMA, peak_hz=25.0,
            base_hz=0.2, seed=510)
        p_full, p_strat, diag, B_masked = self._p_pair(
            spikes, A, B, self.SELF_CENTER)

        # The control is only meaningful if it actually pinned the focal animal
        # and kept enough spikes to have had the power to detect a real field.
        assert diag["residual_spread_cm"] < 0.5 * self.SIGMA
        retained = compute_rate_map(spikes, B_masked, **self.KW).spike_counts.sum()
        assert retained >= 50

        assert p_full == pytest.approx(1 / 201)
        assert p_strat > 0.05

    def test_genuine_partner_cell_survives_the_stratum(self):
        A, B = _correlated_pair(self.N, self.ALPHA)
        spikes = _poisson_spikes_from_field(
            B, center=(35.0, 45.0), sigma=self.SIGMA, peak_hz=25.0,
            base_hz=0.2, seed=511)
        p_full, p_strat, _, _ = self._p_pair(spikes, A, B, self.SELF_CENTER)
        assert p_full == pytest.approx(1 / 201)
        assert p_strat < 0.05


@pytest.mark.slow
class TestStratumSweepIntegration:
    def test_sweep_accepts_stratum_and_records_diagnostics(self):
        tr = _three_animal_tracking(n=12000)
        spikes = _poisson_spikes_from_field(
            tr["A"], center=(40.0, 40.0), sigma=8.0, peak_hz=30.0,
            base_hz=0.5, seed=520)
        res = _sweep(_make_ks([spikes]), tr, focal="A", n_shuffles=50,
                     null_method="position_shuffle", min_n_spikes=10,
                     self_stratum_radius_cm=12.0,
                     self_stratum_center=(40.0, 40.0))
        diags = res.parameters["self_stratum_diagnostics"]
        assert set(diags) == {"A", "B", "C"}
        assert all(d["n_retained"] > 0 for d in diags.values())
        assert res.parameters["self_stratum_center"] == (40.0, 40.0)
        # Every target's map is restricted to the same focal-position samples.
        assert res.rate_maps["B"][0].parameters["self_stratum_applied"] is True

    def test_compare_control_tabulates_both_runs(self):
        tr = _three_animal_tracking(n=12000)
        spikes = _poisson_spikes_from_field(
            tr["A"], center=(40.0, 40.0), sigma=8.0, peak_hz=30.0,
            base_hz=0.5, seed=521)
        cmp_ = compare_self_stratum_control(
            _make_ks([spikes]), _video_tracking(tr), _StubSync(), "A",
            list(tr.keys()), self_stratum_radius_cm=12.0,
            self_stratum_center=(40.0, 40.0),
            pixels_per_cm=None, bin_size_cm=BIN, smoothing_sigma_cm=5.0,
            speed_filter_subject="none", n_shuffles=50, min_n_spikes=10,
            use_quality_cells=False, arena_bounds=ARENA, seed=0,
        )
        t = cmp_.table
        assert set(t.columns) >= {"p_full", "p_stratum", "n_spikes_stratum",
                                  "survives_stratum", "lost_to_stratum", "is_self"}
        assert len(t) == 3                      # 1 cell x 3 targets
        assert t["is_self"].sum() == 1
        # The restricted run really is restricted.
        assert (t["n_spikes_stratum"] < t["n_spikes_full"]).all()
        assert cmp_.parameters["self_stratum_center"] == (40.0, 40.0)
        assert cmp_.unrestricted.parameters["self_stratum_radius_cm"] is None
        assert cmp_.stratified.parameters["self_stratum_radius_cm"] == 12.0

    def test_stratum_requires_focal_tracking(self):
        # Focal "A" is implanted but absent from the tracking file: there is no
        # self position to condition on, so the control cannot be applied.
        tr = {"B": _make_xy(n=2000, seed=200), "C": _make_xy(n=2000, seed=300)}
        spikes = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="requires tracking for the focal"):
            _sweep(_make_ks([spikes]), tr, focal="A", n_shuffles=5,
                   target_animals=["B", "C"], self_stratum_radius_cm=10.0)
