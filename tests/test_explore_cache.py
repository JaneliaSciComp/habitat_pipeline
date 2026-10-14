"""Tests for gui/explore_cache.py — the explore view's on-disk cache.

A stub stands in for ``KilosortData`` (only the four members the cache uses),
and rastermap is patched out: what is under test is the caching contract
(signature, default-bins-only writes, round trip), not rastermap's sorting.
"""
import os
import pickle

import numpy as np
import pytest

import gui.explore_cache as xc


class StubKs:
    """Minimal KilosortData: every cell passes the quality filter except id 99."""

    def __init__(self, n_cells=6, duration=120.0, seed=0):
        rng = np.random.default_rng(seed)
        self.ks_ids = list(range(n_cells)) + [99]
        self.spike_times_by_cell = [np.sort(rng.uniform(5.0, duration, 400))
                                    for _ in range(n_cells)] + [np.array([1.0, 2.0])]

    def filter_cells_by_firing_patterns(self):
        return {"passed_clusters": [c for c in self.ks_ids if c != 99]}

    def bin_spike_times(self, bin_size_sec, t_start, t_end, filtered_only=True):
        cells = [st for cid, st in zip(self.ks_ids, self.spike_times_by_cell) if cid != 99]
        edges = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)
        mat = np.array([np.histogram(st, bins=edges)[0] / bin_size_sec for st in cells])
        return mat, (edges[:-1] + edges[1:]) / 2


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(xc, "CACHE_DIR", tmp_path / "explore")
    # Rastermap's sort is not under test; keep the image's shape contract.
    monkeypatch.setattr(xc, "fit_rastermap",
                        lambda m: np.ascontiguousarray(m[::-1], dtype=np.float64))


@pytest.fixture
def sources(tmp_path):
    ks = tmp_path / "kilosort4"
    ks.mkdir()
    np.save(ks / "spike_times.npy", np.arange(10))
    np.save(ks / "spike_clusters.npy", np.zeros(10))
    ev = tmp_path / "20251216" / "20251216_behavior_event_df.csv"
    ev.parent.mkdir()
    ev.write_text("type,ts_start\nF,1\n")
    return ks, ev


class TestCompute:
    def test_products_use_quality_cells_only(self):
        products = xc.compute_view_products(StubKs(), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        # Cell 99 (spikes at 1-2 s) fails the filter, so the span starts after 5 s.
        assert products["t0"] >= 5.0
        assert products["raster_img"].shape[0] == 6
        assert products["raster_img"].flags["C_CONTIGUOUS"]
        assert products["pca"]["scores"].shape == (len(products["pca"]["bin_centers"]), 3)
        assert products["pca"]["pca_bin"] == xc.DEFAULT_PCA_BIN

    def test_pca_needs_three_cells(self):
        assert xc.fit_pca_trajectory(StubKs(n_cells=2), 0.5, 5.0, 100.0) is None


class TestSignature:
    def test_stable_when_nothing_changes(self, sources):
        ks, ev = sources
        assert xc.source_signature(ks, [ev]) == xc.source_signature(ks, [ev])

    def test_changes_with_spike_file(self, sources):
        ks, ev = sources
        before = xc.source_signature(ks, [ev])
        np.save(ks / "spike_times.npy", np.arange(11))
        assert xc.source_signature(ks, [ev]) != before

    def test_changes_with_event_file(self, sources):
        ks, ev = sources
        before = xc.source_signature(ks, [ev])
        st = ev.stat()
        os.utime(ev, (st.st_atime, st.st_mtime + 5))
        assert xc.source_signature(ks, [ev]) != before

    def test_ignores_kilosort_pickle_cache(self, sources):
        ks, ev = sources
        before = xc.source_signature(ks, [ev])
        (ks / "kilosort_processed_x.pkl").write_bytes(b"cache")
        assert xc.source_signature(ks, [ev]) == before


class TestDiskCache:
    def test_round_trip(self):
        products = xc.compute_view_products(StubKs(), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        events = {"stand-in": "events"}
        assert xc.save_view_cache("cohort7", "20251216_094334", "rat613",
                                  products, events, "sig1") is not None
        entry = xc.load_view_cache("cohort7", "20251216_094334", "rat613")
        assert entry["signature"] == "sig1"
        assert entry["events"] == events
        assert entry["raster_img"].dtype == np.float64
        assert entry["raster_img"].flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(entry["raster_img"], products["raster_img"], rtol=1e-6)
        np.testing.assert_array_equal(entry["pca"]["scores"], products["pca"]["scores"])

    def test_non_default_bins_are_not_written(self):
        products = xc.compute_view_products(StubKs(), 2.0, xc.DEFAULT_PCA_BIN)
        assert xc.save_view_cache("cohort7", "s", "rat613", products, None, "sig") is None
        assert xc.load_view_cache("cohort7", "s", "rat613") is None

    def test_missing_and_other_version(self):
        assert xc.load_view_cache("cohort7", "nothing", "rat1") is None
        path = xc.cache_path("cohort7", "s", "rat613")
        path.parent.mkdir(parents=True)
        with open(path, "wb") as fh:
            pickle.dump({"version": xc.EXPLORE_CACHE_VERSION + 1}, fh)
        assert xc.load_view_cache("cohort7", "s", "rat613") is None

    def test_unreadable_file_is_ignored(self):
        path = xc.cache_path("cohort7", "s", "rat613")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not a pickle")
        assert xc.load_view_cache("cohort7", "s", "rat613") is None


class _FakeCEBRA:
    """Stands in for ``cebra.CEBRA``: records what it was fitted on, embeds by PCA-free slicing."""
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state_dict_ = {"loss": [3.0, 2.0, 1.5]}   # where real CEBRA keeps it

    def fit(self, X, y=None, z=None):          # z: the discrete index of a mixed fit
        type(self).last = {"X": np.array(X), "y": None if y is None else np.array(y),
                           "z": None if z is None else np.array(z), "kwargs": self.kwargs}
        return self

    def transform(self, X):
        return np.asarray(X)[:, : self.kwargs["output_dimension"]]


@pytest.fixture
def fake_cebra(monkeypatch):
    import sys
    import types
    mod = types.ModuleType("cebra")
    mod.CEBRA = _FakeCEBRA
    mod.__version__ = "0.0-test"
    monkeypatch.setitem(sys.modules, "cebra", mod)
    monkeypatch.setattr(xc, "cebra_available", lambda: True)
    _FakeCEBRA.last = None
    return mod


class TestCebra:
    def test_missing_package_raises_install_hint(self, monkeypatch):
        monkeypatch.setattr(xc, "cebra_available", lambda: False)
        with pytest.raises(ImportError, match="pip install cebra"):
            xc.fit_cebra_embedding(StubKs(), 0.5, 5.0, 100.0)

    def test_fit_shape_orientation_and_params(self, fake_cebra):
        fit = xc.fit_cebra_embedding(StubKs(), 0.5, 5.0, 100.0, max_iterations=123, seed=7)
        n_bins = len(fit["bin_centers"])
        assert fit["scores"].shape == (n_bins, 3)
        # CEBRA is fitted on (n_bins, n_cells), z-scored per cell.
        X = _FakeCEBRA.last["X"]
        assert X.shape == (n_bins, 6)
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-9)
        kw = _FakeCEBRA.last["kwargs"]
        assert kw["conditional"] == "time" and kw["max_iterations"] == 123
        assert "random_state" not in kw and "seed" not in kw   # not constructor args
        assert kw["output_dimension"] == 3
        assert fit["params"] == xc.cebra_params(0.5, 123, 7)
        assert fit["loss_final"] == 1.5

    def test_needs_three_cells(self, fake_cebra):
        assert xc.fit_cebra_embedding(StubKs(n_cells=2), 0.5, 5.0, 100.0) is None
        assert _FakeCEBRA.last is None

    def _default_fit(self, fake_cebra):
        return xc.fit_cebra_embedding(
            StubKs(), xc.DEFAULT_CEBRA_BIN, 5.0, 100.0,
            max_iterations=xc.DEFAULT_CEBRA_ITERATIONS, seed=xc.DEFAULT_CEBRA_SEED)

    def test_cache_round_trip(self, fake_cebra):
        products = xc.compute_view_products(StubKs(), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        fit = self._default_fit(fake_cebra)
        assert not xc.update_cached_cebra("c", "s", "rat1", fit, "sig")     # no entry yet
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        assert xc.update_cached_cebra("c", "s", "rat1", fit, "sig")
        entry = xc.load_view_cache("c", "s", "rat1")
        got = xc.cached_cebra(entry, "sig", fit["params"])
        np.testing.assert_array_equal(got["scores"], fit["scores"])
        assert entry["signature"] == "sig"          # the rest of the entry is untouched

    def test_cache_misses_on_other_params_or_sources(self, fake_cebra):
        products = xc.compute_view_products(StubKs(), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        fit = self._default_fit(fake_cebra)
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        xc.update_cached_cebra("c", "s", "rat1", fit, "sig")
        entry = xc.load_view_cache("c", "s", "rat1")
        other = xc.cebra_params(xc.DEFAULT_CEBRA_BIN, xc.DEFAULT_CEBRA_ITERATIONS + 500,
                                xc.DEFAULT_CEBRA_SEED)
        assert xc.cached_cebra(entry, "sig", other) is None
        assert xc.cached_cebra(entry, "changed", fit["params"]) is None

    def test_only_default_fits_are_cached(self, fake_cebra):
        products = xc.compute_view_products(StubKs(), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        fit = xc.fit_cebra_embedding(StubKs(), 0.5, 5.0, 100.0, max_iterations=10)
        assert not xc.update_cached_cebra("c", "s", "rat1", fit, "sig")
        assert "cebra" not in xc.load_view_cache("c", "s", "rat1")

    def test_legacy_entry_without_cebra_is_a_miss(self):
        products = xc.compute_view_products(StubKs(), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        entry = xc.load_view_cache("c", "s", "rat1")
        assert xc.cached_cebra(entry, "sig", xc.cebra_params(0.5, 2000, 0)) is None
        assert xc.cached_cebra(None, "sig", xc.cebra_params(0.5, 2000, 0)) is None

    @pytest.mark.slow
    def test_real_cebra_smoke(self):
        pytest.importorskip("cebra")
        fit = xc.fit_cebra_embedding(StubKs(), 0.5, 5.0, 100.0, max_iterations=20)
        assert fit["scores"].shape == (len(fit["bin_centers"]), 3)
        assert np.isfinite(fit["scores"]).all()
        assert fit["loss_final"] is not None and np.isfinite(fit["loss_final"])


class TestCebraPosition:
    """CEBRA-Behavior supervised by (x, y); the model itself is the stand-in."""

    def _traj(self, t_end=600.0, fmt="apt_tqt", step=0.1):
        t = np.arange(5.0, t_end, step)
        return {"t": t, "x": 1000.0 + 50.0 * np.sin(t / 7), "y": 300.0 + 20.0 * t / t_end,
                "format": fmt}

    def test_labels_are_zscored_xy_on_tracked_bins_only(self, fake_cebra):
        traj = self._traj(t_end=300.0)                       # tracked 5-300 s of 5-600 s
        fit = xc.fit_cebra_position_embedding(StubKs(duration=600.0), traj, 0.5, 5.0, 600.0)
        kw, X, y = _FakeCEBRA.last["kwargs"], _FakeCEBRA.last["X"], _FakeCEBRA.last["y"]
        n_bins = len(fit["bin_centers"])
        assert kw["conditional"] == "time_delta"
        assert fit["scores"].shape == (n_bins, 3)           # every bin is embedded ...
        assert fit["n_tracked_bins"] == len(X) == len(y) < n_bins    # ... few were fitted on
        assert y.shape == (len(X), 2) and y.dtype == np.float32
        np.testing.assert_allclose(y.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(y.std(axis=0), 1.0, atol=1e-5)
        bc = fit["bin_centers"]
        # The tracked bins are the early ones, plus those within max_gap (1 s) of the last frame.
        assert bc[bc <= 299.0].size <= len(X) <= bc[bc <= 301.0].size

    def test_params_name_mode_and_tracking_format(self, fake_cebra):
        fit = xc.fit_cebra_position_embedding(StubKs(duration=600.0), self._traj(fmt="mask_metrics"),
                                              0.5, 5.0, 600.0, max_iterations=77, seed=3)
        assert fit["params"] == xc.cebra_params(0.5, 77, 3, "position", "mask_metrics")
        assert xc.cebra_params(0.5, 77, 3) == {"bin": 0.5, "max_iterations": 77, "seed": 3}

    def test_too_few_tracked_bins_or_no_tracking(self, fake_cebra):
        short = {"t": np.arange(5.0, 6.0, 0.1), "x": np.zeros(10), "y": np.zeros(10),
                 "format": "apt_tqt"}
        assert xc.fit_cebra_position_embedding(StubKs(duration=600.0), short, 0.5, 5.0, 600.0) is None
        assert xc.fit_cebra_position_embedding(StubKs(duration=600.0), None, 0.5, 5.0, 600.0) is None
        assert _FakeCEBRA.last is None

    def test_constant_axis_does_not_produce_nan(self, fake_cebra):
        traj = self._traj()
        traj["y"] = np.full_like(traj["t"], 123.0)
        fit = xc.fit_cebra_position_embedding(StubKs(duration=600.0), traj, 0.5, 5.0, 600.0)
        assert np.isfinite(_FakeCEBRA.last["y"]).all() and np.isfinite(fit["scores"]).all()

    def test_missing_package_raises_install_hint(self, monkeypatch):
        monkeypatch.setattr(xc, "cebra_available", lambda: False)
        with pytest.raises(ImportError, match="pip install cebra"):
            xc.fit_cebra_position_embedding(StubKs(duration=600.0), self._traj(), 0.5, 5.0, 600.0)

    def test_cached_in_its_own_slot_beside_the_time_fit(self, fake_cebra):
        products = xc.compute_view_products(StubKs(duration=600.0), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        t_fit = xc.fit_cebra_embedding(StubKs(duration=600.0), xc.DEFAULT_CEBRA_BIN, 5.0, 600.0)
        p_fit = xc.fit_cebra_position_embedding(StubKs(duration=600.0), self._traj(), xc.DEFAULT_CEBRA_BIN,
                                                5.0, 600.0)
        assert xc.update_cached_cebra("c", "s", "rat1", t_fit, "sig")
        assert xc.update_cached_cebra("c", "s", "rat1", p_fit, "sig|tsig")
        entry = xc.load_view_cache("c", "s", "rat1")
        assert xc.cached_cebra(entry, "sig", t_fit["params"])["params"] == t_fit["params"]
        got = xc.cached_cebra(entry, "sig|tsig", p_fit["params"])
        np.testing.assert_array_equal(got["scores"], p_fit["scores"])
        # Stale tracking, or another tracking format, is a miss; so is crossing the slots.
        assert xc.cached_cebra(entry, "sig|other", p_fit["params"]) is None
        other_fmt = xc.cebra_params(xc.DEFAULT_CEBRA_BIN, xc.DEFAULT_CEBRA_ITERATIONS,
                                    xc.DEFAULT_CEBRA_SEED, "position", "mask_metrics")
        assert xc.cached_cebra(entry, "sig|tsig", other_fmt) is None
        assert xc.cached_cebra(entry, "sig|tsig", t_fit["params"]) is None

    @pytest.mark.slow
    def test_real_cebra_smoke(self):
        pytest.importorskip("cebra")
        fit = xc.fit_cebra_position_embedding(StubKs(duration=600.0), self._traj(), 0.5, 5.0, 600.0,
                                              max_iterations=20)
        assert fit["scores"].shape == (len(fit["bin_centers"]), 3)
        assert np.isfinite(fit["scores"]).all() and np.isfinite(fit["loss_final"])


class _StubEvents:
    """Stands in for ``BehavioralEventsData.extract_opponent_labels`` (filters by min events)."""

    def __init__(self, starts, opponents, fail=False):
        self.starts, self.opponents, self.fail = np.asarray(starts, float), np.asarray(opponents), fail
        self.calls = []

    def extract_opponent_labels(self, animal_of_interest, behavior_type, min_events_per_class):
        self.calls.append((animal_of_interest, behavior_type, min_events_per_class))
        if self.fail:
            raise ValueError("No events")
        names, counts = np.unique(self.opponents, return_counts=True)
        keep = np.isin(self.opponents, names[counts >= min_events_per_class])
        return self.starts[keep], self.starts[keep] + 1.0, self.opponents[keep]


def _events_fixture():
    a = 30.0 + 25.0 * np.arange(12)
    b = 42.0 + 25.0 * np.arange(12)
    c = np.array([400.0, 450.0])
    starts = np.concatenate([a, b, c])
    opp = np.array(["ratA"] * 12 + ["ratB"] * 12 + ["ratC"] * 2)
    return _StubEvents(starts, opp)


class TestCebraEvents:
    """CEBRA-Behavior on opponent labels in a window after each onset; the model is a stand-in."""

    def _fit(self, events=None, **kw):
        kw.setdefault("min_events", 10)
        return xc.fit_cebra_events_embedding(StubKs(duration=600.0), events or _events_fixture(),
                                             "rat1", "EC", 0.5, 5.0, 600.0, **kw)

    def test_bin_labels_cover_the_window_after_onset(self):
        bc = np.arange(0.25, 30.0, 0.5)
        lab = xc.event_bin_labels(bc, np.array([10.0]), np.array([2]), (0.0, 2.0))
        # centres 10.25, 10.75, 11.25, 11.75 are inside [10, 12]; 9.75 and 12.25 are not.
        np.testing.assert_array_equal(np.flatnonzero(lab), [20, 21, 22, 23])
        assert set(lab[lab > 0]) == {3} and lab[lab == 0].size == len(bc) - 4

    def test_overlapping_windows_later_onset_wins(self):
        bc = np.arange(0.25, 30.0, 0.5)
        lab = xc.event_bin_labels(bc, np.array([10.0, 11.0]), np.array([0, 1]), (0.0, 2.0))
        assert lab[20] == 1 and lab[21] == 1          # 10.25, 10.75: only the first event
        assert lab[22] == 2 and lab[25] == 2          # 11.25 on: the later event
        # Same result whatever order the events come in.
        flipped = xc.event_bin_labels(bc, np.array([11.0, 10.0]), np.array([1, 0]), (0.0, 2.0))
        np.testing.assert_array_equal(lab, flipped)

    def test_fits_discrete_opponent_labels_on_window_bins_only(self, fake_cebra):
        events = _events_fixture()
        fit = self._fit(events)
        kw, X, y = _FakeCEBRA.last["kwargs"], _FakeCEBRA.last["X"], _FakeCEBRA.last["y"]
        n_bins = len(fit["bin_centers"])
        assert fit["scores"].shape == (n_bins, 3)         # every bin embedded ...
        assert len(X) == len(y) == fit["n_labeled_bins"] == 4 * 24 < n_bins   # ... 24 events fitted
        assert np.issubdtype(y.dtype, np.integer) and y.ndim == 1
        assert sorted(set(y)) == [0, 1]                   # contiguous classes: ratC (2 events) dropped
        assert fit["opponents"] == ["ratA", "ratB"] and fit["n_events"] == 24
        assert kw["conditional"] is None                  # lets CEBRA pick its discrete sampler
        assert kw["model_architecture"] == "offset1-model"
        assert events.calls == [("rat1", "EC", 10)]

    def test_each_event_contributes_the_bins_after_its_onset(self, fake_cebra):
        fit = self._fit()
        y = _FakeCEBRA.last["y"]
        bc = fit["bin_centers"]
        inside = xc.event_bin_labels(bc, np.array([30.0]), np.array([0]), xc.EVENTS_WINDOW) > 0
        # The earliest event (ratA at 30 s) is the first four fitted bins, in time order.
        assert inside.sum() == 4 and np.flatnonzero(inside)[0] == np.flatnonzero(
            xc.event_bin_labels(bc, np.array([30.0]), np.array([0]), xc.EVENTS_WINDOW))[0]
        assert set(y[:4]) == {0}

    def test_params_name_mode_behavior_window_and_min_events(self, fake_cebra):
        fit = self._fit(max_iterations=77, seed=3, min_events=5)
        assert fit["params"] == xc.cebra_events_params(0.5, 77, 3, "EC", 5)
        assert fit["params"]["mode"] == "events" and fit["params"]["window"] == (0.0, 2.0)
        assert xc.cebra_params(0.5, 77, 3) == {"bin": 0.5, "max_iterations": 77, "seed": 3}

    def test_declines_when_there_is_nothing_to_learn(self, fake_cebra):
        one_opp = _StubEvents(30.0 + 25.0 * np.arange(12), ["ratA"] * 12)
        assert self._fit(one_opp) is None                            # a single class
        assert self._fit(min_events=13) is None                      # nobody has enough events
        assert self._fit(_StubEvents([], [])) is None                # no events at all
        assert self._fit(_StubEvents([], [], fail=True)) is None     # extract_opponent_labels raised
        few = _StubEvents([30.0, 60.0], ["ratA", "ratB"])
        assert self._fit(few, min_events=1) is None                  # 8 labelled bins < MIN_EVENT_BINS
        assert xc.fit_cebra_events_embedding(StubKs(n_cells=2), _events_fixture(), "rat1", "EC",
                                             0.5, 5.0, 100.0) is None
        assert xc.fit_cebra_events_embedding(StubKs(), None, "rat1", "EC", 0.5, 5.0, 100.0) is None
        assert _FakeCEBRA.last is None

    def test_missing_package_raises_install_hint(self, monkeypatch):
        monkeypatch.setattr(xc, "cebra_available", lambda: False)
        with pytest.raises(ImportError, match="pip install cebra"):
            self._fit()

    def test_cached_in_its_own_slot_and_only_at_defaults(self, fake_cebra):
        sig = "sig"
        products = xc.compute_view_products(StubKs(duration=600.0), xc.DEFAULT_RASTER_BIN,
                                            xc.DEFAULT_PCA_BIN)
        xc.save_view_cache("c", "s", "rat1", products, None, sig)
        t_fit = xc.fit_cebra_embedding(StubKs(duration=600.0), xc.DEFAULT_CEBRA_BIN, 5.0, 600.0)
        not_default = self._fit(min_events=5)                          # not the default min events
        assert not xc.update_cached_cebra("c", "s", "rat1", not_default, sig)
        ev_fit = self._fit(min_events=xc.DEFAULT_EVENTS_MIN)           # 12 events each: both qualify
        assert xc.update_cached_cebra("c", "s", "rat1", t_fit, sig)
        assert xc.update_cached_cebra("c", "s", "rat1", ev_fit, sig)
        entry = xc.load_view_cache("c", "s", "rat1")
        got = xc.cached_cebra(entry, sig, ev_fit["params"])
        np.testing.assert_array_equal(got["scores"], ev_fit["scores"])
        assert got["opponents"] == ["ratA", "ratB"]
        assert xc.cached_cebra(entry, sig, t_fit["params"])["params"] == t_fit["params"]
        # Another behaviour type or window, or changed event files, is a miss.
        other = xc.cebra_events_params(xc.DEFAULT_CEBRA_BIN, xc.DEFAULT_CEBRA_ITERATIONS,
                                       xc.DEFAULT_CEBRA_SEED, "F", xc.DEFAULT_EVENTS_MIN)
        assert xc.cached_cebra(entry, sig, other) is None
        assert xc.cached_cebra(entry, "changed", ev_fit["params"]) is None
        wider = xc.cebra_events_params(xc.DEFAULT_CEBRA_BIN, xc.DEFAULT_CEBRA_ITERATIONS,
                                       xc.DEFAULT_CEBRA_SEED, "EC", xc.DEFAULT_EVENTS_MIN,
                                       window=(0.0, 4.0))
        assert xc.cached_cebra(entry, sig, wider) is None

    def test_non_default_window_is_not_cached(self, fake_cebra):
        products = xc.compute_view_products(StubKs(duration=600.0), xc.DEFAULT_RASTER_BIN,
                                            xc.DEFAULT_PCA_BIN)
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        wide = self._fit(min_events=xc.DEFAULT_EVENTS_MIN, window=(0.0, 4.0))
        assert not xc.update_cached_cebra("c", "s", "rat1", wide, "sig")
        assert "cebra_events" not in xc.load_view_cache("c", "s", "rat1")

    @pytest.mark.slow
    def test_real_cebra_smoke(self):
        pytest.importorskip("cebra")
        fit = self._fit(max_iterations=20)
        assert fit["scores"].shape == (len(fit["bin_centers"]), 3)
        assert np.isfinite(fit["scores"]).all() and np.isfinite(fit["loss_final"])


class TestCebraEventsPosition:
    """Opponent (discrete) and (x, y) (continuous) together; the model is a stand-in."""

    def _traj(self, t_end=200.0, fmt="apt_tqt"):
        t = np.arange(5.0, t_end, 0.1)
        return {"t": t, "x": 1000.0 + 50.0 * np.sin(t / 7), "y": 300.0 + 20.0 * t / 600.0,
                "format": fmt}

    def _fit(self, events=None, traj=None, **kw):
        kw.setdefault("min_events", 10)
        return xc.fit_cebra_events_position_embedding(
            StubKs(duration=600.0), events or _events_fixture(),
            traj if traj is not None else self._traj(), "rat1", "EC", 0.5, 5.0, 600.0, **kw)

    def test_fits_zscored_xy_and_opponent_on_tracked_window_bins(self, fake_cebra):
        fit = self._fit()
        last = _FakeCEBRA.last
        kw, X, xy, y = last["kwargs"], last["X"], last["y"], last["z"]
        n_bins = len(fit["bin_centers"])
        assert fit["scores"].shape == (n_bins, 3)
        # Events past the tracked stretch (200 s) contribute nothing: 7 + 7 events x 4 bins.
        assert len(X) == len(xy) == len(y) == fit["n_labeled_bins"] == 4 * 14
        assert fit["n_events"] == 14 and fit["opponents"] == ["ratA", "ratB"]
        assert xy.shape == (len(X), 2) and xy.dtype == np.float32
        np.testing.assert_allclose(xy.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(xy.std(axis=0), 1.0, atol=1e-5)
        assert np.issubdtype(y.dtype, np.integer) and sorted(set(y)) == [0, 1]
        assert kw["conditional"] == "time_delta" and kw["model_architecture"] == "offset1-model"

    def test_labels_and_xy_stay_aligned_row_by_row(self, fake_cebra):
        fit = self._fit()
        xy, y = _FakeCEBRA.last["y"], _FakeCEBRA.last["z"]
        bc, traj, events = fit["bin_centers"], self._traj(), _events_fixture()
        keep = np.isin(events.opponents, ["ratA", "ratB"])           # ratC has < 10 events
        starts, opp = events.starts[keep], events.opponents[keep]
        classes = np.unique(opp, return_inverse=True)[1]
        lab = xc.event_bin_labels(bc, starts, classes, xc.EVENTS_WINDOW)
        tracked = (bc >= traj["t"][0] - 1.0) & (bc <= traj["t"][-1] + 1.0)
        rows = (lab > 0) & tracked
        raw = np.column_stack([np.interp(bc[rows], traj["t"], traj[k]) for k in ("x", "y")])
        np.testing.assert_allclose(xy, (raw - raw.mean(0)) / raw.std(0), atol=1e-4)
        np.testing.assert_array_equal(y, lab[rows] - 1)              # opponent, same rows

    def test_params_name_mode_behavior_window_and_tracking(self, fake_cebra):
        fit = self._fit(traj=self._traj(fmt="mask_metrics"), max_iterations=77, seed=3,
                        min_events=5)
        assert fit["params"] == xc.cebra_events_position_params(0.5, 77, 3, "EC", 5, "mask_metrics")
        assert fit["params"]["mode"] == "events_position" and fit["params"]["window"] == (0.0, 2.0)
        assert fit["params"] != xc.cebra_events_params(0.5, 77, 3, "EC", 5)

    def test_declines_without_tracking_or_enough_tracked_events(self, fake_cebra):
        short = self._traj(t_end=60.0)                    # one or two events: < MIN_EVENT_BINS
        assert self._fit(traj=short) is None
        assert xc.fit_cebra_events_position_embedding(
            StubKs(duration=600.0), _events_fixture(), None, "rat1", "EC", 0.5, 5.0, 600.0) is None
        one_opp = _StubEvents(30.0 + 12.0 * np.arange(12), ["ratA"] * 12)
        assert self._fit(one_opp) is None
        assert self._fit(_StubEvents([], [])) is None
        assert xc.fit_cebra_events_position_embedding(
            StubKs(n_cells=2), _events_fixture(), self._traj(), "rat1", "EC", 0.5, 5.0, 100.0) is None
        assert _FakeCEBRA.last is None

    def test_missing_package_raises_install_hint(self, monkeypatch):
        monkeypatch.setattr(xc, "cebra_available", lambda: False)
        with pytest.raises(ImportError, match="pip install cebra"):
            self._fit()

    def test_cached_in_its_own_slot_beside_the_events_fit(self, fake_cebra):
        products = xc.compute_view_products(StubKs(duration=600.0), xc.DEFAULT_RASTER_BIN,
                                            xc.DEFAULT_PCA_BIN)
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        ev = xc.fit_cebra_events_embedding(
            StubKs(duration=600.0), _events_fixture(), "rat1", "EC", xc.DEFAULT_CEBRA_BIN, 5.0, 600.0)
        evp = self._fit()
        assert xc.update_cached_cebra("c", "s", "rat1", ev, "sig")
        assert xc.update_cached_cebra("c", "s", "rat1", evp, "sig|tsig")
        entry = xc.load_view_cache("c", "s", "rat1")
        got = xc.cached_cebra(entry, "sig|tsig", evp["params"])
        np.testing.assert_array_equal(got["scores"], evp["scores"])
        assert xc.cached_cebra(entry, "sig", ev["params"])["params"] == ev["params"]
        # Stale tracking, another tracking format and crossing the two slots are misses.
        assert xc.cached_cebra(entry, "sig|other", evp["params"]) is None
        other_fmt = xc.cebra_events_position_params(
            xc.DEFAULT_CEBRA_BIN, xc.DEFAULT_CEBRA_ITERATIONS, xc.DEFAULT_CEBRA_SEED, "EC",
            xc.DEFAULT_EVENTS_MIN, "mask_metrics")
        assert xc.cached_cebra(entry, "sig|tsig", other_fmt) is None
        assert xc.cached_cebra(entry, "sig|tsig", ev["params"]) is None

    def test_only_default_settings_are_cached(self, fake_cebra):
        products = xc.compute_view_products(StubKs(duration=600.0), xc.DEFAULT_RASTER_BIN,
                                            xc.DEFAULT_PCA_BIN)
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        assert not xc.update_cached_cebra("c", "s", "rat1", self._fit(min_events=5), "sig|t")
        assert not xc.update_cached_cebra("c", "s", "rat1",
                                          self._fit(window=(0.0, 4.0)), "sig|t")
        assert "cebra_events_position" not in xc.load_view_cache("c", "s", "rat1")

    @pytest.mark.slow
    def test_real_cebra_smoke(self):
        pytest.importorskip("cebra")
        fit = self._fit(max_iterations=20)
        assert fit["scores"].shape == (len(fit["bin_centers"]), 3)
        assert np.isfinite(fit["scores"]).all() and np.isfinite(fit["loss_final"])


class _FakeUMAP:
    """Stands in for ``umap.UMAP``; records its arguments and what it was fitted on."""
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, X):
        type(self).last = {"X": np.array(X), "kwargs": self.kwargs}
        return np.asarray(X)[:, : self.kwargs["n_components"]]


@pytest.fixture
def fake_umap(monkeypatch):
    import sys
    import types
    mod = types.ModuleType("umap")
    mod.UMAP = _FakeUMAP
    mod.__version__ = "0.0-test"
    monkeypatch.setitem(sys.modules, "umap", mod)
    monkeypatch.setattr(xc, "umap_available", lambda: True)
    _FakeUMAP.last = None
    return mod


class TestUmap:
    def test_missing_package_raises_install_hint(self, monkeypatch):
        monkeypatch.setattr(xc, "umap_available", lambda: False)
        with pytest.raises(ImportError, match="pip install umap-learn"):
            xc.fit_umap_embedding(StubKs(), 0.5, 5.0, 100.0)

    def test_fit_shape_input_and_params(self, fake_umap):
        fit = xc.fit_umap_embedding(StubKs(), 0.5, 5.0, 100.0, n_neighbors=15,
                                    min_dist=0.3, seed=4)
        n_bins = len(fit["bin_centers"])
        assert fit["scores"].shape == (n_bins, 3)
        # Same input as PCA/CEBRA: (n_bins, n_cells), z-scored per cell.
        X = _FakeUMAP.last["X"]
        assert X.shape == (n_bins, 6)
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-9)
        kw = _FakeUMAP.last["kwargs"]
        assert (kw["n_components"], kw["n_neighbors"], kw["min_dist"], kw["random_state"])             == (3, 15, 0.3, 4)
        assert fit["params"] == xc.umap_params(0.5, 15, 0.3, 4)

    def test_needs_three_cells_and_more_bins_than_neighbours(self, fake_umap):
        assert xc.fit_umap_embedding(StubKs(n_cells=2), 0.5, 5.0, 100.0) is None
        assert xc.fit_umap_embedding(StubKs(), 0.5, 5.0, 100.0, n_neighbors=500) is None
        assert _FakeUMAP.last is None

    def test_cache_round_trip_and_misses(self, fake_umap):
        products = xc.compute_view_products(StubKs(), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        fit = xc.fit_umap_embedding(StubKs(), xc.DEFAULT_UMAP_BIN, 5.0, 100.0)
        assert fit["params"] == xc.umap_params(xc.DEFAULT_UMAP_BIN, xc.DEFAULT_UMAP_NEIGHBORS,
                                               xc.DEFAULT_UMAP_MIN_DIST, xc.DEFAULT_UMAP_SEED)
        assert xc.update_cached_umap("c", "s", "rat1", fit, "sig")
        entry = xc.load_view_cache("c", "s", "rat1")
        np.testing.assert_array_equal(xc.cached_umap(entry, "sig", fit["params"])["scores"],
                                      fit["scores"])
        assert entry["signature"] == "sig"
        other = xc.umap_params(xc.DEFAULT_UMAP_BIN, 50, xc.DEFAULT_UMAP_MIN_DIST,
                               xc.DEFAULT_UMAP_SEED)
        assert xc.cached_umap(entry, "sig", other) is None          # other parameters
        assert xc.cached_umap(entry, "changed", fit["params"]) is None   # other sources
        assert xc.cached_umap(None, "sig", fit["params"]) is None

    def test_only_default_fits_are_cached(self, fake_umap):
        products = xc.compute_view_products(StubKs(), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        fit = xc.fit_umap_embedding(StubKs(), 0.5, 5.0, 100.0, n_neighbors=10)
        assert not xc.update_cached_umap("c", "s", "rat1", fit, "sig")
        assert "umap" not in xc.load_view_cache("c", "s", "rat1")

    def test_umap_slot_does_not_disturb_cebra_slots(self, fake_umap, fake_cebra):
        products = xc.compute_view_products(StubKs(), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        c_fit = xc.fit_cebra_embedding(StubKs(), xc.DEFAULT_CEBRA_BIN, 5.0, 100.0)
        u_fit = xc.fit_umap_embedding(StubKs(), xc.DEFAULT_UMAP_BIN, 5.0, 100.0)
        assert xc.update_cached_cebra("c", "s", "rat1", c_fit, "sig")
        assert xc.update_cached_umap("c", "s", "rat1", u_fit, "sig")
        entry = xc.load_view_cache("c", "s", "rat1")
        assert xc.cached_cebra(entry, "sig", c_fit["params"]) is not None
        assert xc.cached_umap(entry, "sig", u_fit["params"]) is not None

    @pytest.mark.slow
    def test_real_umap_smoke(self):
        pytest.importorskip("umap")
        fit = xc.fit_umap_embedding(StubKs(duration=300.0), 0.5, 5.0, 300.0, n_neighbors=10)
        assert fit["scores"].shape == (len(fit["bin_centers"]), 3)
        assert np.isfinite(fit["scores"]).all()


class TestTrajectory:
    def _traj(self):
        t = np.arange(0.0, 100.0, 0.1)
        return {"formats": {
            "apt_tqt": {"t": t, "x": 1000 + t, "y": 500 + 0 * t, "n_files": 2},
            "mask_metrics": {"t": t[:300], "x": t[:300], "y": t[:300], "n_files": 1},
        }, "default": "apt_tqt"}

    def test_formats_of_both_shapes(self):
        assert set(xc.trajectory_formats(self._traj())) == {"apt_tqt", "mask_metrics"}
        legacy = {"format": "apt_tqt", "t": np.arange(3), "x": np.arange(3), "y": np.arange(3)}
        assert list(xc.trajectory_formats(legacy)) == ["apt_tqt"]
        assert xc.trajectory_formats(None) == {}

    def test_tracking_signature_tracks_files(self, sources):
        _, ev = sources
        before = xc.tracking_signature([ev])
        st = ev.stat()
        os.utime(ev, (st.st_atime, st.st_mtime + 5))
        assert xc.tracking_signature([ev]) != before

    def test_update_cached_trajectory(self):
        products = xc.compute_view_products(StubKs(), xc.DEFAULT_RASTER_BIN, xc.DEFAULT_PCA_BIN)
        assert not xc.update_cached_trajectory("c", "s", "rat1", self._traj(), "tsig")
        xc.save_view_cache("c", "s", "rat1", products, None, "sig")
        assert xc.update_cached_trajectory("c", "s", "rat1", self._traj(), "tsig")
        entry = xc.load_view_cache("c", "s", "rat1")
        assert entry["tracking_signature"] == "tsig"
        assert entry["trajectory"]["default"] == "apt_tqt"
        assert entry["signature"] == "sig"          # the rest of the entry is untouched

    def test_apt_chunk_span_from_directory(self, tmp_path):
        f = tmp_path / "cohort7_20251216_0959" / "solution" / "TQT_named.csv"
        span = xc._apt_chunk_epoch_span(f)
        assert span is not None and span[1] - span[0] == 30 * 60
        assert xc._apt_chunk_epoch_span(tmp_path / "x" / "TQT_named.csv") is None
