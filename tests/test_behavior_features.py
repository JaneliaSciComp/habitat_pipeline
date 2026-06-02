"""
Tests for video/behavior_features.py.

These bypass real DataSyncManager / file I/O by constructing synthetic
VideoTrackingData and BehavioralEventsData dataclasses directly and
stubbing the sync as identity (slope=1, intercept=0).
"""

import numpy as np
import pandas as pd
import pytest

from ingestion.ephys_sync import DataSyncManager
from video.behavior_features import build_behavior_feature_matrix
from video.behavioral_events import BehavioralEventsData
from video.tracking_import import VideoTrackingData


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _identity_sync() -> DataSyncManager:
    s = DataSyncManager.__new__(DataSyncManager)
    s.slope = 1.0
    s.intercept = 0.0
    s.mapping = {"slope": 1.0, "intercept": 0.0, "r_squared": 1.0}
    return s


def _make_tracking_from_arrays(
    t_ephys: np.ndarray,
    objects: dict,
    use_sync_path: bool = False,
) -> VideoTrackingData:
    """Build a VideoTrackingData with synthetic per-object position arrays.

    Each entry in ``objects`` is ``{'x': arr, 'y': arr, 'head_dir': arr?}``.
    ``use_sync_path=True`` populates ``timestamps`` (ns) and leaves
    ``ephys_timestamps`` None, forcing the function to convert via sync.
    """
    parsed = {}
    n = len(t_ephys)
    for name, obj in objects.items():
        df = pd.DataFrame({
            "frame": np.arange(n),
            "center_x": np.asarray(obj["x"], dtype=np.float64),
            "center_y": np.asarray(obj["y"], dtype=np.float64),
        })
        if "head_dir" in obj:
            df["head_dir"] = np.asarray(obj["head_dir"], dtype=np.float64)
        parsed[name] = df

    if use_sync_path:
        return VideoTrackingData(
            animal_id="631", session_id="20251216",
            parsed_data=parsed,
            timestamps=(t_ephys * 1e9).astype(np.int64),
            ephys_timestamps=None,
            synchronized=False,
        )
    return VideoTrackingData(
        animal_id="631", session_id="20251216",
        parsed_data=parsed,
        timestamps=None,
        ephys_timestamps=np.asarray(t_ephys, dtype=np.float64),
        synchronized=True,
    )


def _make_events(rows) -> BehavioralEventsData:
    df = pd.DataFrame(rows)
    return BehavioralEventsData(
        session_id="20251216",
        events_data=df,
        synchronized=True,
    )


def _idx(grid: np.ndarray, val: float) -> int:
    return int(np.argmin(np.abs(grid - val)))


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

class TestKinematics:
    def test_constant_velocity_gives_constant_speed_and_distance(self):
        n = 101
        t = np.linspace(0.0, 10.0, n)
        # Partner placed far enough that focal never reaches it: distance
        # stays monotonically decreasing over t ∈ [0, 10].
        objects = {
            "rat631": {"x": t, "y": np.zeros(n)},                  # 1 m/s in +x
            "rat632": {"x": 100.0 * np.ones(n), "y": np.zeros(n)}, # stationary
        }
        tracking = _make_tracking_from_arrays(t, objects)
        events = _make_events([])
        sync = _identity_sync()
        t_grid = np.linspace(1.0, 9.0, 17)

        df = build_behavior_feature_matrix(
            tracking, events, sync, t_grid, focal="rat631", partner="rat632",
        )

        assert df.shape[0] == 17
        np.testing.assert_allclose(df["speed"].to_numpy(), 1.0, atol=1e-6)
        # distance = 100 - focal_x = 100 - t_grid
        np.testing.assert_allclose(df["distance"].to_numpy(), 100.0 - t_grid, atol=1e-6)
        # relative_speed = d(distance)/dt = -1 (approaching)
        np.testing.assert_allclose(df["relative_speed"].to_numpy(), -1.0, atol=1e-6)

    def test_stationary_focal_gives_zero_speed(self):
        n = 50
        t = np.linspace(0.0, 5.0, n)
        objects = {
            "rat631": {"x": np.zeros(n), "y": np.zeros(n)},
            "rat632": {"x": np.ones(n), "y": np.zeros(n)},
        }
        tracking = _make_tracking_from_arrays(t, objects)
        df = build_behavior_feature_matrix(
            tracking, _make_events([]), _identity_sync(),
            np.linspace(0.5, 4.5, 9), focal="rat631", partner="rat632",
        )
        np.testing.assert_allclose(df["speed"].to_numpy(), 0.0, atol=1e-6)
        np.testing.assert_allclose(df["distance"].to_numpy(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Event indicators
# ---------------------------------------------------------------------------

class TestEventIndicators:
    def test_indicator_one_in_window(self):
        n = 101
        t = np.linspace(0.0, 10.0, n)
        objects = {
            "rat631": {"x": np.zeros(n), "y": np.zeros(n)},
            "rat632": {"x": np.ones(n), "y": np.zeros(n)},
        }
        tracking = _make_tracking_from_arrays(t, objects)
        events = _make_events([
            {"type": "F", "initiator": "rat631", "victim": "rat632",
             "ts_start": 3e9, "ts_end": 3.1e9,
             "ts_start_ephys": 3.0, "ts_end_ephys": 3.1},
        ])
        t_grid = np.arange(0.25, 10.0, 0.5)
        df = build_behavior_feature_matrix(
            tracking, events, _identity_sync(), t_grid,
            focal="rat631", partner="rat632", event_window_sec=1.0,
        )
        # event at 3.0, ±1s → bins at 2.25, 2.75, 3.25, 3.75 are within window.
        for val in (2.25, 2.75, 3.25, 3.75):
            assert df["event_F"].iloc[_idx(t_grid, val)] == 1.0
        # Far away bins are 0.
        assert df["event_F"].iloc[_idx(t_grid, 0.25)] == 0.0
        assert df["event_F"].iloc[_idx(t_grid, 8.25)] == 0.0

    def test_multiple_event_types(self):
        n = 50
        t = np.linspace(0.0, 5.0, n)
        objects = {
            "rat631": {"x": np.zeros(n), "y": np.zeros(n)},
            "rat632": {"x": np.ones(n), "y": np.zeros(n)},
        }
        events = _make_events([
            {"type": "F", "ts_start_ephys": 1.0},
            {"type": "EC", "ts_start_ephys": 2.0},
            {"type": "F", "ts_start_ephys": 4.0},
        ])
        tracking = _make_tracking_from_arrays(t, objects)
        df = build_behavior_feature_matrix(
            tracking, events, _identity_sync(),
            np.linspace(0.5, 4.5, 9),
            focal="rat631", partner="rat632",
        )
        assert "event_F" in df.columns
        assert "event_EC" in df.columns
        assert df["event_F"].sum() > 0
        assert df["event_EC"].sum() > 0

    def test_event_types_filter(self):
        n = 50
        t = np.linspace(0.0, 5.0, n)
        objects = {
            "rat631": {"x": np.zeros(n), "y": np.zeros(n)},
            "rat632": {"x": np.ones(n), "y": np.zeros(n)},
        }
        events = _make_events([
            {"type": "F", "ts_start_ephys": 1.0},
            {"type": "EC", "ts_start_ephys": 2.0},
        ])
        tracking = _make_tracking_from_arrays(t, objects)
        df = build_behavior_feature_matrix(
            tracking, events, _identity_sync(),
            np.linspace(0.5, 4.5, 9),
            focal="rat631", partner="rat632",
            event_types=["F"],
        )
        assert "event_F" in df.columns
        assert "event_EC" not in df.columns

    def test_empty_events_yields_no_event_columns(self):
        n = 50
        t = np.linspace(0.0, 5.0, n)
        objects = {
            "rat631": {"x": np.zeros(n), "y": np.zeros(n)},
            "rat632": {"x": np.ones(n), "y": np.zeros(n)},
        }
        df = build_behavior_feature_matrix(
            _make_tracking_from_arrays(t, objects),
            _make_events([]),
            _identity_sync(),
            np.linspace(0.5, 4.5, 9),
            focal="rat631", partner="rat632",
        )
        assert not any(c.startswith("event_") for c in df.columns)


# ---------------------------------------------------------------------------
# Sync path
# ---------------------------------------------------------------------------

class TestSyncPath:
    def test_uses_sync_when_ephys_timestamps_missing(self):
        n = 51
        t_ephys = np.linspace(0.0, 5.0, n)
        objects = {
            "rat631": {"x": t_ephys, "y": np.zeros(n)},
            "rat632": {"x": 2.0 * np.ones(n), "y": np.zeros(n)},
        }
        tracking = _make_tracking_from_arrays(t_ephys, objects, use_sync_path=True)
        assert tracking.ephys_timestamps is None
        assert tracking.timestamps is not None
        df = build_behavior_feature_matrix(
            tracking, _make_events([]), _identity_sync(),
            np.linspace(1.0, 4.0, 7), focal="rat631", partner="rat632",
        )
        np.testing.assert_allclose(df["speed"].to_numpy(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Head direction
# ---------------------------------------------------------------------------

class TestHeadDir:
    def test_head_dir_passed_through_when_present(self):
        n = 50
        t = np.linspace(0.0, 5.0, n)
        objects = {
            "rat631": {"x": np.zeros(n), "y": np.zeros(n),
                       "head_dir": np.linspace(0.0, 1.0, n)},
            "rat632": {"x": np.ones(n), "y": np.zeros(n)},
        }
        df = build_behavior_feature_matrix(
            _make_tracking_from_arrays(t, objects),
            _make_events([]), _identity_sync(),
            np.linspace(0.5, 4.5, 9),
            focal="rat631", partner="rat632",
        )
        assert "head_dir" in df.columns
        # head_dir is linear from 0 to 1; at t_grid=0.5 expect ~0.1, at t_grid=4.5 expect ~0.9
        assert df["head_dir"].iloc[0] == pytest.approx(0.1, abs=0.02)
        assert df["head_dir"].iloc[-1] == pytest.approx(0.9, abs=0.02)

    def test_head_dir_absent_when_no_column(self):
        n = 50
        t = np.linspace(0.0, 5.0, n)
        objects = {
            "rat631": {"x": np.zeros(n), "y": np.zeros(n)},
            "rat632": {"x": np.ones(n), "y": np.zeros(n)},
        }
        df = build_behavior_feature_matrix(
            _make_tracking_from_arrays(t, objects),
            _make_events([]), _identity_sync(),
            np.linspace(0.5, 4.5, 9),
            focal="rat631", partner="rat632",
        )
        assert "head_dir" not in df.columns


# ---------------------------------------------------------------------------
# Out-of-range handling and errors
# ---------------------------------------------------------------------------

class TestOutOfRange:
    def test_bins_outside_tracking_range_are_nan(self):
        n = 51
        t = np.linspace(2.0, 5.0, n)
        objects = {
            "rat631": {"x": t, "y": np.zeros(n)},
            "rat632": {"x": 2.0 * np.ones(n), "y": np.zeros(n)},
        }
        tracking = _make_tracking_from_arrays(t, objects)
        t_grid = np.linspace(0.0, 7.0, 15)
        df = build_behavior_feature_matrix(
            tracking, _make_events([]), _identity_sync(),
            t_grid, focal="rat631", partner="rat632",
        )
        out_of_range = (t_grid < 2.0) | (t_grid > 5.0)
        assert df.loc[df.index[out_of_range], "speed"].isna().all()
        in_range = ~out_of_range
        np.testing.assert_allclose(
            df.loc[df.index[in_range], "speed"].dropna().to_numpy(), 1.0, atol=1e-3,
        )


class TestErrors:
    def test_unknown_focal_raises(self):
        n = 50
        t = np.linspace(0.0, 5.0, n)
        objects = {
            "rat631": {"x": np.zeros(n), "y": np.zeros(n)},
            "rat632": {"x": np.ones(n), "y": np.zeros(n)},
        }
        tracking = _make_tracking_from_arrays(t, objects)
        with pytest.raises(KeyError, match="not found"):
            build_behavior_feature_matrix(
                tracking, _make_events([]), _identity_sync(),
                np.linspace(0.0, 5.0, 10),
                focal="rat999", partner="rat631",
            )

    def test_no_timestamps_raises(self):
        parsed = {
            "rat631": pd.DataFrame({"frame": np.arange(50),
                                     "center_x": np.zeros(50),
                                     "center_y": np.zeros(50)}),
            "rat632": pd.DataFrame({"frame": np.arange(50),
                                     "center_x": np.ones(50),
                                     "center_y": np.zeros(50)}),
        }
        tracking = VideoTrackingData(
            animal_id="631", session_id="20251216",
            parsed_data=parsed, timestamps=None,
            ephys_timestamps=None, synchronized=False,
        )
        with pytest.raises(ValueError, match="ephys_timestamps"):
            build_behavior_feature_matrix(
                tracking, _make_events([]), _identity_sync(),
                np.linspace(0.0, 5.0, 10),
                focal="rat631", partner="rat632",
            )


# ---------------------------------------------------------------------------
# Index / column contract
# ---------------------------------------------------------------------------

class TestDataFrameContract:
    def test_index_matches_t_grid(self):
        n = 50
        t = np.linspace(0.0, 5.0, n)
        objects = {
            "rat631": {"x": np.zeros(n), "y": np.zeros(n)},
            "rat632": {"x": np.ones(n), "y": np.zeros(n)},
        }
        t_grid = np.linspace(1.0, 4.0, 7)
        df = build_behavior_feature_matrix(
            _make_tracking_from_arrays(t, objects),
            _make_events([]), _identity_sync(),
            t_grid, focal="rat631", partner="rat632",
        )
        np.testing.assert_array_equal(df.index.to_numpy(), t_grid)
        assert df.index.name == "t_ephys"
        for col in ("speed", "angular_speed", "distance",
                    "relative_bearing", "relative_speed"):
            assert col in df.columns
