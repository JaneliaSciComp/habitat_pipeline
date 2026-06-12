"""
Tests for the focal-only data path of partner-distance decoding.

After the refactor off ``MultiAnimalSession``, ``build_distance_binned_data``
takes a focal :class:`KilosortData`, a session :class:`VideoTrackingData`, and a
:class:`DataSyncManager` — the partner contributes a tracking trajectory only and
needs no ephys. These tests exercise:

* the extracted free function ``video.tracking_import.resolve_tracking_on_ephys_clock``
  directly (no MultiAnimalSession needed), and
* ``ephys.decode_partner_distance.build_distance_binned_data`` end-to-end with a
  stub focal ``KilosortData`` and a synthetic two-animal tracking object.

Stub/synthetic style mirrors ``tests/test_multi_animal_tracking.py``; no disk or
cohort config is touched.
"""

import numpy as np
import pandas as pd
import pytest

from ephys.decode_partner_distance import build_distance_binned_data
from video.tracking_import import VideoTrackingData, resolve_tracking_on_ephys_clock

SESSION = "20251216"
DT_SEC = 0.025          # 40 Hz tracking
N_FRAMES = 900          # covers ~22.5 s of ephys time at slope=1


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubSync:
    """Linear behavior→ephys clock map: ephys = slope * behav + intercept."""

    def __init__(self, slope=1.0, intercept=0.0):
        self.slope = slope
        self.intercept = intercept

    def convert_behavior_to_ephys(self, behav_seconds):
        return self.slope * np.asarray(behav_seconds, dtype=np.float64) + self.intercept


class _StubKilosort:
    """Minimal focal KilosortData stand-in for build_distance_binned_data.

    Exposes the two methods the binner calls: ``filter_cells_by_firing_patterns``
    (records its kwargs) and ``bin_spike_times`` (returns a fixed
    ``(n_cells, n_bins)`` rate matrix + bin centers).
    """

    def __init__(self, rates, bin_centers):
        self._rates = np.asarray(rates, dtype=np.float64)
        self._bin_centers = np.asarray(bin_centers, dtype=np.float64)
        self.filter_calls = []

    def filter_cells_by_firing_patterns(self, **kwargs):
        self.filter_calls.append(kwargs)
        return {"passed_clusters": list(range(self._rates.shape[0]))}

    def bin_spike_times(self, bin_size_sec=1.0, t_start=None, t_end=None,
                        filtered_only=True):
        return self._rates, self._bin_centers


def _make_tracking() -> VideoTrackingData:
    """Two animals; '631' moves at 1 px/frame in x, '632' is stationary."""
    frames = np.arange(N_FRAMES)
    df_631 = pd.DataFrame({
        "frame": frames,
        "center_x": 200.0 + frames,
        "center_y": np.full(N_FRAMES, 50.0),
    })
    df_632 = pd.DataFrame({
        "frame": frames,
        "center_x": np.full(N_FRAMES, 300.0),
        "center_y": np.full(N_FRAMES, 80.0),
    })
    timestamps_ns = (frames * DT_SEC * 1e9).astype(np.int64)
    return VideoTrackingData(
        animal_id="631",
        session_id=SESSION,
        parsed_data={"631": df_631, "632": df_632},
        timestamps=timestamps_ns,
    )


# ---------------------------------------------------------------------------
# resolve_tracking_on_ephys_clock — no MultiAnimalSession required
# ---------------------------------------------------------------------------

class TestResolveTracking:
    def test_returns_per_animal_frames(self):
        out = resolve_tracking_on_ephys_clock(
            _make_tracking(), _StubSync(), ["631", "632"], pixels_per_cm=2.0,
        )
        assert set(out.keys()) == {"631", "632"}
        for df in out.values():
            assert list(df.columns) == ["t", "x", "y", "speed"]
            assert len(df) == N_FRAMES

    def test_pixels_to_cm_and_passthrough(self):
        cm = resolve_tracking_on_ephys_clock(
            _make_tracking(), _StubSync(), ["631"], pixels_per_cm=2.0,
        )["631"]["x"].to_numpy()
        np.testing.assert_allclose(cm, (200.0 + np.arange(N_FRAMES)) / 2.0)

        px = resolve_tracking_on_ephys_clock(
            _make_tracking(), _StubSync(), ["631"], pixels_per_cm=None,
        )["631"]["x"].to_numpy()
        np.testing.assert_allclose(px, 200.0 + np.arange(N_FRAMES))

    def test_windowing(self):
        out = resolve_tracking_on_ephys_clock(
            _make_tracking(), _StubSync(), ["631"],
            t_start_ephys=1.0, t_end_ephys=2.0,
        )["631"]
        t = out["t"].to_numpy()
        assert t.min() >= 1.0 and t.max() <= 2.0 and len(t) > 0

    def test_raises_without_timestamps(self):
        tr = _make_tracking()
        tr.timestamps = None
        with pytest.raises(RuntimeError):
            resolve_tracking_on_ephys_clock(tr, _StubSync(), ["631"])


# ---------------------------------------------------------------------------
# build_distance_binned_data — focal ephys + session tracking, no partner ephys
# ---------------------------------------------------------------------------

class TestBuildBinnedData:
    def _binner_inputs(self, n_cells=3, n_bins=40, bin_size=0.5, seed=0):
        rng = np.random.default_rng(seed)
        bin_centers = (np.arange(n_bins) + 0.5) * bin_size   # 0.25 .. within tracking range
        rates = rng.standard_normal((n_cells, n_bins)) + 5.0
        ks = _StubKilosort(rates, bin_centers)
        tracking = _make_tracking()
        return ks, tracking, bin_size, n_cells, n_bins

    def test_aligned_outputs_and_units(self):
        ks, tracking, bin_size, n_cells, n_bins = self._binner_inputs()
        data = build_distance_binned_data(
            ks, tracking, _StubSync(), "631", "632",
            pixels_per_cm=2.0, bin_size=bin_size,
        )
        n = len(data["distance"])
        assert n == n_bins                              # every bin is within tracked range
        assert data["firing_rates"].shape == (n, n_cells)
        assert data["nuisance"].shape == (n, 3)
        assert data["bin_centers"].shape == (n,)
        assert data["nuisance_names"] == ["focal_speed", "focal_x", "focal_y"]
        assert data["n_cells"] == n_cells
        assert data["units"] == "cm"
        assert np.all(np.isfinite(data["distance"]))
        # Partner stationary at (150, 40) cm, focal at ((200+f)/2, 25) cm ⇒ distance finite & > 0.
        assert np.all(data["distance"] > 0)

    def test_pixels_units_without_calibration(self):
        ks, tracking, bin_size, _, _ = self._binner_inputs()
        data = build_distance_binned_data(
            ks, tracking, _StubSync(), "631", "632",
            pixels_per_cm=None, bin_size=bin_size,
        )
        assert data["units"] == "pixels"

    def test_filter_kwargs_forwarded(self):
        ks, tracking, bin_size, _, _ = self._binner_inputs()
        build_distance_binned_data(
            ks, tracking, _StubSync(), "631", "632",
            pixels_per_cm=2.0, bin_size=bin_size,
            filter_kwargs={"min_firing_rate": 1.0},
        )
        assert ks.filter_calls == [{"min_firing_rate": 1.0}]

    def test_no_filter_call_by_default(self):
        ks, tracking, bin_size, _, _ = self._binner_inputs()
        build_distance_binned_data(
            ks, tracking, _StubSync(), "631", "632",
            pixels_per_cm=2.0, bin_size=bin_size,
        )
        assert ks.filter_calls == []

    def test_missing_partner_raises(self):
        ks, tracking, bin_size, _, _ = self._binner_inputs()
        with pytest.raises(KeyError):
            build_distance_binned_data(
                ks, tracking, _StubSync(), "631", "999_absent",
                pixels_per_cm=2.0, bin_size=bin_size,
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
