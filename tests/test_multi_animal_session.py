"""
Tests for MultiAnimalSession (ingestion/multi_animal_session.py).

These bypass DataStorageManager construction (no cohort config, no disk
required) and inject pre-built synthetic ``KilosortData`` instances into
``ks_by_animal`` so the common-grid binner can be exercised in isolation.
"""

import numpy as np
import pytest

from ingestion.kilosort_data_import import KilosortData
from ingestion.multi_animal_session import MultiAnimalSession, _bin_spike_lists

SAMPLE_RATE = 30000.0

LOOSE_FILTER = {
    "min_firing_rate": 0.0,
    "max_firing_rate": 1e6,
    "min_presence_ratio": 0.0,
    "max_cv_isi": float("inf"),
}


# ---------------------------------------------------------------------------
# Synthetic KilosortData
# ---------------------------------------------------------------------------

def _make_synthetic_ks(
    animal_id: str,
    session_id: str,
    n_cells: int,
    duration_sec: float,
    firing_rate_hz: float,
    seed: int,
) -> KilosortData:
    """Build a KilosortData with Poisson spike trains and trivial metadata."""
    rng = np.random.default_rng(seed)
    spike_times_by_cell = []
    for _ in range(n_cells):
        n = int(rng.poisson(duration_sec * firing_rate_hz))
        st = np.sort(rng.uniform(0.0, duration_sec, size=n))
        spike_times_by_cell.append(st)

    n_total = sum(len(st) for st in spike_times_by_cell)
    if n_total > 0:
        all_sec = np.concatenate(spike_times_by_cell)
        spike_samples = (all_sec * SAMPLE_RATE).astype(np.int64)
    else:
        spike_samples = np.array([0, int(duration_sec * SAMPLE_RATE)], dtype=np.int64)
    spike_clusters = np.zeros(len(spike_samples), dtype=np.int64)

    return KilosortData(
        animal_id=animal_id,
        session_id=session_id,
        spike_times=spike_samples,
        spike_clusters=spike_clusters,
        spike_times_by_cell=spike_times_by_cell,
        ks_ids=list(range(n_cells)),
        channel=np.zeros(n_cells, dtype=int),
        amplitude=np.zeros(n_cells),
        fr=np.zeros(n_cells),
        amp=np.zeros(n_cells),
        DV=np.zeros(n_cells),
        XX=np.zeros(n_cells),
        cell_numbers=np.zeros((n_cells, 2), dtype=int),
        to_load=np.ones(n_cells, dtype=bool),
    )


def _make_session(animals, sync_from_animal=None) -> MultiAnimalSession:
    """Build a MultiAnimalSession without touching disk."""
    session_id = next(iter(animals.values())).session_id
    animal_ids = list(animals.keys())
    s = MultiAnimalSession.__new__(MultiAnimalSession)
    s.session_id = session_id
    s.animal_ids = animal_ids
    s.config_path = None
    s.dio_channel = 1
    s.sync_from_animal = sync_from_animal or animal_ids[0]
    s.dsm_by_animal = {aid: None for aid in animal_ids}
    s.ks_by_animal = dict(animals)
    s._sync = None
    s._events = None
    return s


# ---------------------------------------------------------------------------
# _bin_spike_lists
# ---------------------------------------------------------------------------

class TestBinSpikeLists:
    def test_basic_counts_and_centers(self):
        spike_times = [np.array([0.05, 0.15, 0.35]), np.array([0.25])]
        rates, centers = _bin_spike_lists(
            spike_times, bin_size_sec=0.1, t_start=0.0, t_end=0.4,
        )
        assert rates.shape == (2, 4)
        assert centers.shape == (4,)
        np.testing.assert_allclose(centers, [0.05, 0.15, 0.25, 0.35])
        # Cell 0: one spike each in bins [0,0.1), [0.1,0.2), [0.3,0.4]
        np.testing.assert_allclose(rates[0], [10.0, 10.0, 0.0, 10.0])
        np.testing.assert_allclose(rates[1], [0.0, 0.0, 10.0, 0.0])

    def test_empty_cell(self):
        rates, _ = _bin_spike_lists([np.array([])], 0.1, 0.0, 1.0)
        assert rates.shape == (1, 10)
        assert (rates == 0).all()

    def test_identical_grid_across_calls(self):
        _, c1 = _bin_spike_lists([np.array([0.1])], 0.5, 0.0, 10.0)
        _, c2 = _bin_spike_lists([np.array([])], 0.5, 0.0, 10.0)
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# MultiAnimalSession
# ---------------------------------------------------------------------------

class TestMultiAnimalSession:
    def setup_method(self):
        self.ks_a = _make_synthetic_ks(
            "631", "20251216", n_cells=20, duration_sec=200.0,
            firing_rate_hz=5.0, seed=0,
        )
        self.ks_b = _make_synthetic_ks(
            "632", "20251216", n_cells=15, duration_sec=180.0,
            firing_rate_hz=5.0, seed=1,
        )
        self.sess = _make_session({"631": self.ks_a, "632": self.ks_b})

    def test_common_time_window_is_intersection(self):
        t0, t1 = self.sess.get_common_time_window()
        assert t0 == 0.0
        assert t1 == pytest.approx(min(self.ks_a.duration_seconds, self.ks_b.duration_seconds))

    def test_common_binned_rates_identical_bins(self):
        centers, rates = self.sess.get_common_binned_rates(
            bin_size_sec=0.5, filter_kwargs=LOOSE_FILTER, use_cache=False,
        )
        assert set(rates.keys()) == {"631", "632"}
        n_bins_a = rates["631"].shape[1]
        n_bins_b = rates["632"].shape[1]
        assert n_bins_a == n_bins_b == len(centers)
        assert rates["631"].shape[0] == 20
        assert rates["632"].shape[0] == 15
        t0, t1 = self.sess.get_common_time_window()
        # All centers within the common window (allow a half-bin overshoot at the end
        # since np.arange may include an edge at t_end).
        assert centers[0] >= t0
        assert centers[-1] <= t1 + 0.5

    def test_explicit_window(self):
        centers, rates = self.sess.get_common_binned_rates(
            bin_size_sec=0.5, t_start_ephys=10.0, t_end_ephys=50.0,
            filter_kwargs=LOOSE_FILTER, use_cache=False,
        )
        assert rates["631"].shape[1] == 80
        assert rates["632"].shape[1] == 80
        np.testing.assert_allclose(centers[0], 10.25)
        np.testing.assert_allclose(centers[-1], 49.75)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            self.sess.get_common_binned_rates(
                bin_size_sec=0.5, t_start_ephys=10.0, t_end_ephys=5.0,
                filter_kwargs=LOOSE_FILTER, use_cache=False,
            )

    def test_smoothing_changes_output(self):
        kwargs = dict(
            bin_size_sec=0.5,
            t_start_ephys=0.0, t_end_ephys=100.0,
            filter_kwargs=LOOSE_FILTER,
            use_cache=False,
        )
        _, raw = self.sess.get_common_binned_rates(**kwargs)
        _, smoothed = self.sess.get_common_binned_rates(smoothing_sigma_sec=2.0, **kwargs)
        assert raw["631"].shape == smoothed["631"].shape
        assert not np.allclose(raw["631"], smoothed["631"])
        # Smoothing should reduce the per-bin variance for each cell.
        assert smoothed["631"].std(axis=1).mean() < raw["631"].std(axis=1).mean()

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        import ingestion.multi_animal_session as mod
        monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path / "multi_animal")
        kwargs = dict(
            bin_size_sec=0.5,
            t_start_ephys=0.0, t_end_ephys=100.0,
            filter_kwargs=LOOSE_FILTER,
            use_cache=True,
        )
        c1, r1 = self.sess.get_common_binned_rates(**kwargs)
        c2, r2 = self.sess.get_common_binned_rates(**kwargs)
        np.testing.assert_array_equal(c1, c2)
        for k in r1:
            np.testing.assert_array_equal(r1[k], r2[k])
        files = list((tmp_path / "multi_animal").glob("*.pkl"))
        assert len(files) == 1

    def test_cache_key_changes_with_params(self, tmp_path, monkeypatch):
        import ingestion.multi_animal_session as mod
        monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path / "multi_animal")
        common = dict(
            t_start_ephys=0.0, t_end_ephys=100.0,
            filter_kwargs=LOOSE_FILTER, use_cache=True,
        )
        self.sess.get_common_binned_rates(bin_size_sec=0.5, **common)
        self.sess.get_common_binned_rates(bin_size_sec=1.0, **common)
        self.sess.get_common_binned_rates(bin_size_sec=0.5, smoothing_sigma_sec=1.0, **common)
        files = list((tmp_path / "multi_animal").glob("*.pkl"))
        assert len(files) == 3
