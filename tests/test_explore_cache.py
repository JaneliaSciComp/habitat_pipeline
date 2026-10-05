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
