"""
Tests for ``MultiAnimalSession.get_tracking_on_ephys_clock``.

These build a synthetic :class:`VideoTrackingData` and a stub sync/DSM, monkeypatch
``load_tracking_data`` so no disk or cohort config is required, and exercise the
ephys-clock conversion, pixels→cm calibration, speed computation, and windowing.
"""

import numpy as np
import pandas as pd
import pytest

import ingestion.multi_animal_session as mas_module
from ingestion.multi_animal_session import MultiAnimalSession
from video.tracking_import import VideoTrackingData

SESSION = "20251216"
DT_SEC = 0.025          # 40 Hz tracking
N_FRAMES = 200


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


class _StubDSM:
    def __init__(self, pixels_per_cm=None):
        self._pixels_per_cm = pixels_per_cm

    def get_pixels_per_cm(self):
        return self._pixels_per_cm


def _make_tracking() -> VideoTrackingData:
    """Two animals; '631' moves at 1 px/frame in x, '632' is stationary."""
    frames = np.arange(N_FRAMES)
    df_631 = pd.DataFrame({
        "frame": frames,
        "center_x": 200.0 + frames,   # 1 px / frame
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


def _make_session(pixels_per_cm=None, slope=1.0, intercept=0.0) -> MultiAnimalSession:
    s = MultiAnimalSession.__new__(MultiAnimalSession)
    s.session_id = SESSION
    s.animal_ids = ["631", "632"]
    s.config_path = None
    s.dio_channel = 1
    s.sync_from_animal = "631"
    stub_dsm = _StubDSM(pixels_per_cm=pixels_per_cm)
    s.dsm_by_animal = {"631": stub_dsm, "632": stub_dsm}
    s.ks_by_animal = {}
    s._sync = _StubSync(slope=slope, intercept=intercept)
    s._events = None
    return s


@pytest.fixture(autouse=True)
def _patch_loader(monkeypatch):
    monkeypatch.setattr(mas_module, "load_tracking_data", lambda dsm: _make_tracking())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrackingOnEphysClock:
    def test_returns_per_animal_frames(self):
        out = _make_session().get_tracking_on_ephys_clock()
        assert set(out.keys()) == {"631", "632"}
        for df in out.values():
            assert list(df.columns) == ["t", "x", "y", "speed"]
            assert len(df) == N_FRAMES

    def test_ephys_time_monotonic(self):
        out = _make_session().get_tracking_on_ephys_clock()
        t = out["631"]["t"].to_numpy()
        assert np.all(np.diff(t) > 0)
        # slope=1, intercept=0 ⇒ ephys seconds == behavior seconds == frame*DT
        np.testing.assert_allclose(t, np.arange(N_FRAMES) * DT_SEC)

    def test_intercept_offset_applied(self):
        out = _make_session(intercept=10.0).get_tracking_on_ephys_clock()
        t = out["631"]["t"].to_numpy()
        np.testing.assert_allclose(t, np.arange(N_FRAMES) * DT_SEC + 10.0)

    def test_pixels_to_cm_conversion(self):
        out = _make_session(pixels_per_cm=2.0).get_tracking_on_ephys_clock()
        x = out["631"]["x"].to_numpy()
        np.testing.assert_allclose(x, (200.0 + np.arange(N_FRAMES)) / 2.0)

    def test_passthrough_when_no_calibration(self):
        out = _make_session(pixels_per_cm=None).get_tracking_on_ephys_clock()
        x = out["631"]["x"].to_numpy()
        np.testing.assert_allclose(x, 200.0 + np.arange(N_FRAMES))

    def test_speed_linear_motion(self):
        # cm: x = (200 + frame)/2 ⇒ 0.5 cm/frame; dt = 0.025 s ⇒ 20 cm/s.
        out = _make_session(pixels_per_cm=2.0).get_tracking_on_ephys_clock()
        speed = out["631"]["speed"].to_numpy()
        # Interior only — Gaussian smoothing (sigma ~4 frames) damps the edges.
        np.testing.assert_allclose(speed[20:-20], 20.0, rtol=1e-3)

    def test_stationary_animal_zero_speed(self):
        out = _make_session(pixels_per_cm=2.0).get_tracking_on_ephys_clock()
        speed = out["632"]["speed"].to_numpy()
        np.testing.assert_allclose(speed[5:-5], 0.0, atol=1e-6)

    def test_windowing(self):
        out = _make_session().get_tracking_on_ephys_clock(
            t_start_ephys=1.0, t_end_ephys=2.0,
        )
        t = out["631"]["t"].to_numpy()
        assert t.min() >= 1.0 and t.max() <= 2.0
        assert len(t) > 0

    def test_raises_without_timestamps(self, monkeypatch):
        tr = _make_tracking()
        tr.timestamps = None
        monkeypatch.setattr(mas_module, "load_tracking_data", lambda dsm: tr)
        with pytest.raises(RuntimeError):
            _make_session().get_tracking_on_ephys_clock()
