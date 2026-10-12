"""
Tests for video/conflict_territory.py and ephys/decode_conflict_territory.py.

Synthetic, I/O-free (pattern of tests/test_decode_partner_distance.py): plain
numpy trajectories, conflict layouts and populations, no share access.

The key scientific test is ``TestNullsSeparateTerritoryFromPlace``: a pure
place-cell population decodes the conflict-territory label (it is a function of
position), so the time-shift null rejects; the map-specificity null must not.
``TestMapGeometryConfound`` documents where that stops being true - wins and
losses spatially segregated - and that ``diagnostics['position_map_null']``
flags it (``HZ-STAT-017``). ``TestGainNullCalibration`` (slow) pins the primary
``gain_null``'s false-positive rate on pure place populations.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ephys.decode_conflict_territory import (
    _analyze,
    _cv_predict,
    build_territory_binned_data,
    plot_conflict_territory,
    segments_from_mask,
    summarize,
)
from ephys.territory_population import _cv_balanced_accuracy, preprocess_rates
from tests.test_social_spatial_fields import _make_ks
from video.behavioral_events import BehavioralEventsData
from video.conflict_territory import (
    OTHER,
    OWN,
    localize_events,
    outcome_is_win,
    territory_grid,
    voronoi_labels,
)

DT = 0.5
FAST = dict(n_map_shuffles=49, n_time_shifts=49)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _walk(T, seed, L=100.0, step=2.5):
    """Smooth reflected random walk in [0, L]^2 (velocity AR(1)); crosses the arena many times."""
    rng = np.random.default_rng(seed)
    p = np.empty((T, 2))
    p[0] = L / 2
    v = np.zeros(2)
    for t in range(1, T):
        v = 0.9 * v + rng.normal(0, step * 0.45, 2)
        q = p[t - 1] + v
        for k in range(2):
            if q[k] < 0:
                q[k], v[k] = -q[k], -v[k]
            if q[k] > L:
                q[k], v[k] = 2 * L - q[k], -v[k]
        p[t] = q
    return p[:, 0], p[:, 1]


def _place_population(x, y, seed, n=36, sigma=15.0, peak=8.0):
    """Gaussian place cells tiling the arena; no territory term at all."""
    rng = np.random.default_rng(seed + 100)
    g = np.linspace(10, 90, int(np.sqrt(n)))
    cx, cy = (a.ravel() for a in np.meshgrid(g, g))
    lam = 0.5 + peak * np.exp(-0.5 * ((x[:, None] - cx) ** 2 + (y[:, None] - cy) ** 2) / sigma ** 2)
    return rng.poisson(lam * DT) / DT


def _territory_population(x, y, conflicts, seed, n=20, gain=1.5):
    """Cells whose rate steps with the *real* own/other label; no place tuning."""
    rng = np.random.default_rng(seed + 200)
    own = voronoi_labels(x, y, conflicts.x, conflicts.y, conflicts.outcome)["own"]
    w = rng.normal(0, 1, n)
    lam = 4.0 * np.exp(gain * np.outer(own - 0.5, w))
    return rng.poisson(lam * DT) / DT


def _layout(kind, seed, n_w=8, n_l=6):
    """Conflict points: 'interleaved' = outcomes random over the arena; 'clustered' = wins left, losses right."""
    rng = np.random.default_rng(seed + 7)
    if kind == "clustered":
        pts = np.vstack([np.column_stack([rng.uniform(5, 45, n_w), rng.uniform(5, 95, n_w)]),
                         np.column_stack([rng.uniform(55, 95, n_l), rng.uniform(5, 95, n_l)])])
        out = ["winner"] * n_w + ["loser"] * n_l
    else:
        pts = rng.uniform(5, 95, (n_w + n_l, 2))
        out = list(rng.permutation(["winner"] * n_w + ["loser"] * n_l))
    return pd.DataFrame(dict(x=pts[:, 0], y=pts[:, 1], outcome=out))


def _session(kind="interleaved", pop="place", T=2000, seed=0):
    x, y = _walk(T, seed)
    c = _layout(kind, seed)
    rates = _place_population(x, y, seed) if pop == "place" else _territory_population(x, y, c, seed)
    return rates, x, y, np.zeros(T, dtype=int), c


def _focal_df(t, x, y):
    return pd.DataFrame(dict(t=t, x=x, y=y, speed=np.zeros_like(t)))


# ---------------------------------------------------------------------------
# video.conflict_territory
# ---------------------------------------------------------------------------

class TestVoronoiLabels:
    def test_hand_built_layout(self):
        # W at (0,0) and (0,10); L at (10,0).
        cx, cy = [0.0, 0.0, 10.0], [0.0, 10.0, 0.0]
        out = ["winner", "winner", "loser"]
        px = np.array([2.0, 8.0, 1.0, np.nan])
        py = np.array([0.0, 1.0, 9.0, 0.0])
        v = voronoi_labels(px, py, cx, cy, out)
        assert list(v["label"]) == [OWN, OTHER, OWN, None]
        assert list(v["own"]) == [1, 0, 1, -1]
        assert list(v["nearest"]) == [0, 2, 1, -1]
        np.testing.assert_allclose(v["dist"][:3], [2.0, np.hypot(2, 1), np.hypot(1, 1)])
        # margin = distance to the nearest opposite-class point minus distance to the nearest point
        np.testing.assert_allclose(v["margin"][0], 8.0 - 2.0)
        np.testing.assert_allclose(v["margin"][1], np.hypot(8, 1) - np.hypot(2, 1))
        assert np.isnan(v["dist"][3]) and np.isnan(v["margin"][3])

    def test_single_class_margin_is_inf(self):
        v = voronoi_labels([1.0], [1.0], [0.0, 5.0], [0.0, 5.0], ["winner", "winner"])
        assert v["own"][0] == 1 and np.isinf(v["margin"][0])

    def test_grid_matches_pointwise_labels(self):
        g = territory_grid([0.0, 10.0], [0.0, 0.0], ["winner", "loser"], bounds=(0, 10, -5, 5), bins=10)
        assert g["own"].shape == (10, 10)                     # rows = y
        assert np.all(g["own"][:, :5] == 1) and np.all(g["own"][:, 5:] == 0)

    def test_unknown_outcome_raises(self):
        with pytest.raises(ValueError):
            outcome_is_win(["winner", "draw"])


class TestLocalizeEvents:
    def test_events_without_position_are_dropped(self):
        t = np.concatenate([np.arange(0, 50, 0.1), np.arange(80, 100, 0.1)])   # gap 50-80 s
        df = _focal_df(t, t, np.zeros_like(t))
        times = np.array([10.0, 60.0, 90.0, 150.0])                            # 60: in gap, 150: past end
        conf, n_drop = localize_events(times, ["winner", "loser", "loser", "winner"], df, max_gap=1.0,
                                       end_times=times + 2)
        assert n_drop == 2
        assert list(conf["t"]) == [10.0, 90.0]
        np.testing.assert_allclose(conf["x"], [10.0, 90.0], atol=1e-9)
        assert list(conf["outcome"]) == ["winner", "loser"]
        assert list(conf["t_end"]) == [12.0, 92.0]


class TestOutcomeGuardrail:
    """Self/outcome swap guard (cf. TestMultiTargetSweep in test_social_spatial_fields).

    Goes through the real ``extract_outcome_labels`` so the focal-view convention
    is checked end to end: where rat631 WON must be 'own', where it LOST 'other'.
    Fails if winner/loser were swapped anywhere between the events table and the map.
    """

    def _events(self):
        rows = []
        for ts in (30.0, 45.0, 60.0):          # rat631 wins, focal on the left
            rows.append(dict(type="F", initiator="rat631", victim="rat615", winner="rat631", loser="rat615",
                             ts_start_ephys=ts, ts_end_ephys=ts + 1))
        for ts in (240.0, 255.0, 270.0):       # rat631 loses, focal on the right
            rows.append(dict(type="F", initiator="rat615", victim="rat631", winner="rat615", loser="rat631",
                             ts_start_ephys=ts, ts_end_ephys=ts + 1))
        df = pd.DataFrame(rows)
        df["ts_start"], df["ts_end"] = df["ts_start_ephys"] * 1e9, df["ts_end_ephys"] * 1e9
        return BehavioralEventsData(session_id="20251216", events_data=df, synchronized=True)

    def _focal(self):
        t = np.arange(0, 300, 0.1)
        return _focal_df(t, t / 3.0, np.full_like(t, 50.0))   # x sweeps 0 -> 100

    @pytest.mark.parametrize("who", ["rat631", "631"])
    def test_wins_map_to_own_losses_to_other(self, who):
        starts, ends, labels = self._events().extract_outcome_labels(who, None, 3)
        conf, n_drop = localize_events(starts, labels, self._focal(), end_times=ends)
        assert n_drop == 0
        win = conf["outcome"] == "winner"
        assert np.all(conf.loc[win, "x"] < 25) and np.all(conf.loc[~win, "x"] > 75)
        v = voronoi_labels([5.0, 95.0], [50.0, 50.0], conf.x, conf.y, conf.outcome)
        assert list(v["label"]) == [OWN, OTHER]

    def test_opponent_view_is_the_mirror_image(self):
        starts, ends, labels = self._events().extract_outcome_labels("rat615", None, 3)
        conf, _ = localize_events(starts, labels, self._focal(), end_times=ends)
        v = voronoi_labels([5.0, 95.0], [50.0, 50.0], conf.x, conf.y, conf.outcome)
        assert list(v["label"]) == [OTHER, OWN]

    def test_analyze_labels_bins_near_wins_as_own(self):
        rates, x, y, seg, c = _session("clustered", "place", T=800, seed=3)
        res = _analyze(rates, x, y, seg, c, n_map_shuffles=0, n_time_shifts=0, position_map_null=False)
        assert res["status"] == "success"
        left, right = x < 20, x > 80          # wins all in x<45, losses all in x>55
        assert np.all(res["y_true"][left] == 1) and np.all(res["y_true"][right] == 0)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

class TestBinning:
    def _ks(self, spikes):
        ks = _make_ks(spikes)
        ks.calls = []

        def get_filtered_cells_spike_times(**kw):
            ks.calls.append(kw)
            return list(ks.ks_ids), list(ks.spike_times_by_cell)
        ks.get_filtered_cells_spike_times = get_filtered_cells_spike_times
        return ks

    def test_bins_near_events_and_untracked_bins_are_excluded(self):
        t = np.concatenate([np.arange(0, 40, 0.05), np.arange(60, 100, 0.05)])   # untracked 40-60 s
        df = _focal_df(t, t, np.zeros_like(t))
        spikes = [np.sort(np.random.default_rng(0).uniform(0, 100, 500)), np.array([10.1, 10.2, 70.3])]
        ks = self._ks(spikes)
        conflicts = pd.DataFrame(dict(t=[20.0], t_end=[22.0], x=[20.0], y=[0.0], outcome=["winner"]))
        d = build_territory_binned_data(ks, df, conflicts, bin_size=DT, window=(0.0, 100.0),
                                        event_exclusion_sec=5.0, max_gap=1.0)
        from ingestion.kilosort_data_import import _DEFAULT_QUALITY_THRESHOLDS
        assert ks.calls == [dict(_DEFAULT_QUALITY_THRESHOLDS)]      # quality filter, not bin_spike_times
        bc = d["bin_centers"]
        assert not np.any((bc >= 15.0) & (bc <= 27.0))              # [t-5, t_end+5]
        assert not np.any((bc > 41.0) & (bc < 59.0))                # no tracked sample within 1 s
        assert d["n_bins_near_event"] > 0 and d["n_bins_no_position"] > 0
        assert d["n_bins_kept"] == len(bc) == len(d["x"]) == d["rates"].shape[0]
        # three contiguous runs: [0,15), (27,40], [60,100)
        assert len(np.unique(d["segment_id"])) == 3
        # rates are counts / bin_size on the right bins
        i = np.argmin(np.abs(bc - 10.25))
        assert d["rates"][i, 1] == pytest.approx(2 / DT)

    def test_exclusion_intervals_override_cover_dropped_events(self):
        t = np.arange(0, 100, 0.05)
        df = _focal_df(t, t, np.zeros_like(t))
        ks = self._ks([np.array([1.0, 50.0])])
        conflicts = pd.DataFrame(dict(t=[20.0], x=[20.0], y=[0.0], outcome=["winner"]))
        d = build_territory_binned_data(ks, df, conflicts, window=(0.0, 100.0), event_exclusion_sec=2.0,
                                        exclusion_intervals=np.array([[20.0, 20.0], [70.0, 71.0]]))
        bc = d["bin_centers"]
        assert not np.any((bc >= 18) & (bc <= 22)) and not np.any((bc >= 68) & (bc <= 73))

    def test_segments_from_mask(self):
        keep = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
        np.testing.assert_array_equal(segments_from_mask(keep), [0, 0, 1, 2, 2, 2])


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

class TestCore:
    def test_cv_predict_matches_shared_cv(self):
        rates, x, y, seg, c = _session("interleaved", "place", T=1000, seed=1)
        lab = voronoi_labels(x, y, c.x, c.y, c.outcome)["own"]
        Z = preprocess_rates(rates, DT)
        preds, _ = _cv_predict(Z, lab, n_folds=5, purge=20)
        from ephys.territory_population import _balanced_accuracy
        assert _balanced_accuracy(lab, preds) == pytest.approx(
            _cv_balanced_accuracy(Z, lab, n_folds=5, purge=20))

    def test_result_contract_and_p_never_zero(self):
        rates, x, y, seg, c = _session("interleaved", "territory", T=1500, seed=0)
        res = _analyze(rates, x, y, seg, c, **FAST)
        assert res["status"] == "success"
        p = res["parameters"]
        assert p["class_label"] == "territory (own vs other)"
        assert p["analysis_title"] == "Conflict-defined territory decoding"
        for k in ("balanced_accuracy", "baseline_accuracy", "ba_per_fold", "y_true", "y_pred", "bin_centers",
                  "n_bins", "n_cells", "conflicts", "map_null", "shift_null", "diagnostics", "units"):
            assert k in res, k
        assert len(res["ba_per_fold"]) == 5
        for null in (res["map_null"], res["shift_null"]):
            assert null["p_value"] > 0                               # add-one form
            assert null["p_value"] >= null["p_floor"] == pytest.approx(1 / (null["n_valid"] + 1))
        assert res["map_null"]["n_possible_maps"] == 3003            # C(14, 8)
        d = res["diagnostics"]
        assert 0.1 <= d["own_fraction"] <= 0.9
        assert d["bouts"]["n_own"] >= 1 and d["bouts"]["n_other"] >= 1
        assert d["position_only_ba_ceiling"] > 0.8
        assert len(d["fold_class_counts"]) == 5
        summarize(res)   # JSON-able headline must not raise

    def test_seed_is_threaded(self):
        rates, x, y, seg, c = _session("interleaved", "place", T=800, seed=2)
        kw = dict(n_map_shuffles=5, n_time_shifts=5, position_map_null=False)
        a = _analyze(rates, x, y, seg, c, seed=1, **kw)
        b = _analyze(rates, x, y, seg, c, seed=1, **kw)
        z = _analyze(rates, x, y, seg, c, seed=2, **kw)
        np.testing.assert_array_equal(a["map_null"]["draws"], b["map_null"]["draws"])
        assert not np.array_equal(a["map_null"]["draws"], z["map_null"]["draws"])
        assert a["parameters"]["seed"] == 1


class TestNullsSeparateTerritoryFromPlace:
    def test_planted_territory_population_gives_small_map_p(self):
        rates, x, y, seg, c = _session("interleaved", "territory", T=1500, seed=0)
        res = _analyze(rates, x, y, seg, c, **FAST)
        assert res["balanced_accuracy"] > 0.8
        assert res["map_null"]["p_value"] <= 0.05

    def test_place_population_map_null_not_significant_but_shift_null_is(self):
        """KEY: the label is a function of position, so place coding alone beats the time-shift null;
        the map-specificity null must not be fooled when outcomes are spatially interleaved."""
        rates, x, y, seg, c = _session("interleaved", "place", T=2000, seed=0)
        res = _analyze(rates, x, y, seg, c, **FAST)
        assert res["status"] == "success"
        assert res["balanced_accuracy"] > 0.6                        # place cells DO decode the label
        assert res["shift_null"]["p_value"] <= 0.05
        assert res["map_null"]["p_value"] > 0.05


class TestMapGeometryConfound:
    """HZ-STAT-017: wins/losses spatially segregated -> the real map is simpler than a typical
    reassignment, and a place code alone beats the map null. The position-only map null must flag it."""

    def test_clustered_outcomes_fool_the_map_null_and_the_diagnostic_flags_it(self):
        rates, x, y, seg, c = _session("clustered", "place", T=2000, seed=0)
        res = _analyze(rates, x, y, seg, c, **FAST)
        assert res["map_null"]["p_value"] <= 0.05                    # the confound: place cells, small p
        assert res["diagnostics"]["position_map_null"]["p_value"] <= 0.05   # ...and the flag
        assert "gain_null" in res

    def test_interleaved_outcomes_do_not_raise_the_flag(self):
        rates, x, y, seg, c = _session("interleaved", "place", T=2000, seed=0)
        res = _analyze(rates, x, y, seg, c, **FAST)
        assert res["diagnostics"]["position_map_null"]["p_value"] > 0.05


class TestInsufficientData:
    def test_too_few_points_in_a_class(self):
        rates, x, y, seg, _ = _session("interleaved", "place", T=400, seed=0)
        c = pd.DataFrame(dict(x=[10, 20, 30, 80, 90], y=[50] * 5,
                              outcome=["winner", "winner", "winner", "loser", "loser"]))
        res = _analyze(rates, x, y, seg, c, **FAST)
        assert res["status"] == "insufficient_data" and "2 loss" in res["reason"]
        assert res["parameters"]["class_label"] == "territory (own vs other)"

    def test_degenerate_observed_map(self):
        rates, x, y, seg, _ = _session("interleaved", "place", T=400, seed=0)
        # losses far outside the arena: every visited point is nearest to a win
        c = pd.DataFrame(dict(x=[20, 50, 80, 500, 510, 520], y=[50, 50, 50, 500, 500, 500],
                              outcome=["winner"] * 3 + ["loser"] * 3))
        res = _analyze(rates, x, y, seg, c, **FAST)
        assert res["status"] == "insufficient_data" and "degenerate" in res["reason"]

    def test_single_class_training_fold(self):
        T = 1000
        x = np.where(np.arange(T) < 0.8 * T, 20.0, 80.0) + np.random.default_rng(0).normal(0, 1, T)
        y = np.full(T, 50.0)
        c = pd.DataFrame(dict(x=[10, 20, 30, 70, 80, 90], y=[50] * 6, outcome=["winner"] * 3 + ["loser"] * 3))
        rates = _place_population(x, y, 0)
        res = _analyze(rates, x, y, np.zeros(T, int), c, **FAST)
        assert res["status"] == "insufficient_data" and "single class" in res["reason"]

    def test_nonfinite_positions_rejected(self):
        rates, x, y, seg, c = _session("interleaved", "place", T=400, seed=0)
        x = x.copy()
        x[3] = np.nan
        with pytest.raises(ValueError):
            _analyze(rates, x, y, seg, c, **FAST)


def test_figure_smoke(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rates, x, y, seg, c = _session("interleaved", "territory", T=600, seed=0)
    res = _analyze(rates, x, y, seg, c, n_map_shuffles=5, n_time_shifts=5)
    out = tmp_path / "fig.png"
    plot_conflict_territory(res, str(out))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_place_population_map_null_across_seeds(seed):
    rates, x, y, seg, c = _session("interleaved", "place", T=2000, seed=seed)
    res = _analyze(rates, x, y, seg, c, n_map_shuffles=99, n_time_shifts=99, seed=seed)
    assert res["shift_null"]["p_value"] <= 0.05
    assert res["map_null"]["p_value"] > 0.05


@pytest.mark.slow
class TestGainNullCalibration:
    """Primary test (gain over position vs reassigned maps) on pure place populations.

    The 2026-10-12 sweep (99 maps/session) gave 0/20 and 1/20 rejections with
    interleaved outcomes and 11/100 (95% CI 5.6-18.8%) for broad place fields
    with clustered outcomes - the known residual inflation (HZ-STAT-017). These
    bounds catch a regression to the plain map null's behaviour (19/20 there).
    """

    @staticmethod
    def _rejections(layout, seeds, pop="place"):
        k = 0
        for s in seeds:
            x, y = _walk(2000, s)
            c = _layout(layout, s)
            rates = (_place_population(x, y, s) if pop == "place"
                     else _territory_population(x, y, c, s))
            res = _analyze(rates, x, y, np.zeros(2000, int), c, n_map_shuffles=99, n_time_shifts=0, seed=s)
            assert res["status"] == "success"
            k += res["gain_null"]["p_value"] <= 0.05
        return k

    def test_interleaved_place_false_positives(self):
        assert self._rejections("interleaved", range(100, 120)) <= 3

    def test_clustered_place_false_positives_bounded(self):
        assert self._rejections("clustered", range(1000, 1040)) <= 8

    def test_power_for_planted_territory(self):
        assert self._rejections("interleaved", range(100, 110), pop="territory") >= 9
