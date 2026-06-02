"""
Per-bin behavior feature matrix for inter-brain regression.

The downstream regression in ``ephys/inter_brain_dynamics.py``
(``regress_shared_on_behavior``) needs continuous, ephys-aligned
behavioral features at exactly the same bin centers as the firing-rate
matrices produced by ``MultiAnimalSession.get_common_binned_rates``.
This module builds that matrix in a single call.

All timestamps are converted into ephys seconds at the top of the call
(via ``sync.convert_behavior_to_ephys`` if the tracking is not already
synced; events are expected to carry ``ts_start_ephys``), then per-frame
features are computed on the focal's own time grid and linearly
resampled onto the caller-supplied ``t_grid_ephys``.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ingestion.ephys_sync import DataSyncManager
from video.behavioral_events import BehavioralEventsData
from video.tracking_import import VideoTrackingData

logger = logging.getLogger(__name__)


__all__ = ["build_behavior_feature_matrix"]


_HEAD_DIR_COLUMNS = ("head_dir", "heading", "head_direction", "orientation")


def build_behavior_feature_matrix(
    tracking: VideoTrackingData,
    events: BehavioralEventsData,
    sync: DataSyncManager,
    t_grid_ephys: np.ndarray,
    focal: str,
    partner: str,
    event_window_sec: float = 1.0,
    event_types: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Build a per-bin behavior-feature DataFrame on a shared ephys grid.

    Parameters
    ----------
    tracking : VideoTrackingData
        Multi-object tracking. Either ephys-synced (``ephys_timestamps``
        array populated) or with frame-indexed ``timestamps`` in Linux
        nanoseconds.
    events : BehavioralEventsData
        Session-level behavioral events. Must carry ``ts_start_ephys`` and
        ``type`` columns (i.e. ``synchronize_with_ephys`` already run) or
        event indicator columns will be skipped with a warning.
    sync : DataSyncManager
        Used as the fallback for tracking timestamp conversion when
        ``tracking.ephys_timestamps`` is missing — converts behavior
        seconds (``timestamps / 1e9``) to ephys seconds via
        ``convert_behavior_to_ephys``.
    t_grid_ephys : np.ndarray
        Bin centers (or any monotonic grid points) in ephys seconds to
        resample features onto. Typically the ``bin_centers`` returned
        from ``MultiAnimalSession.get_common_binned_rates``.
    focal, partner : str
        Object names from ``tracking.parsed_data``. Substring matching
        is applied via :meth:`VideoTrackingData.get_object_data`.
    event_window_sec : float
        ± window around each ``ts_start_ephys`` for the per-event-type
        indicator columns. Defaults to 1.0 s.
    event_types : iterable of str, optional
        Restrict event indicators to these types. Default: every unique
        ``type`` present in ``events.events_data``.

    Returns
    -------
    pd.DataFrame
        Index ``t_ephys`` equal to ``t_grid_ephys``. Columns include:

        * ``speed`` — focal speed magnitude
        * ``angular_speed`` — derivative of focal heading (rad/s); heading
          is derived from velocity (atan2 of vy, vx)
        * ``head_dir`` — only if a head-direction column is present in
          the focal tracking DataFrame
        * ``distance`` — focal–partner Euclidean distance
        * ``relative_bearing`` — bearing from focal to partner in focal's
          heading frame, wrapped to ``[-π, π]``
        * ``relative_speed`` — ``d(distance) / dt``; positive means
          retreating
        * ``event_<TYPE>`` — one indicator column per event ``type``,
          equal to 1 inside the ±``event_window_sec`` window around
          ``ts_start_ephys`` and 0 elsewhere

        Continuous columns are NaN outside the tracked time range.
    """
    t_grid_ephys = np.asarray(t_grid_ephys, dtype=np.float64)

    ephys_per_frame = _ephys_per_frame(tracking, sync)

    focal_df = tracking.get_object_data(focal)
    if focal_df is None:
        raise KeyError(f"Focal animal {focal!r} not found in tracking.parsed_data")
    partner_df = tracking.get_object_data(partner)
    if partner_df is None:
        raise KeyError(f"Partner animal {partner!r} not found in tracking.parsed_data")

    t_f, x_f, y_f = _position_series(focal_df, ephys_per_frame)
    t_p, x_p, y_p = _position_series(partner_df, ephys_per_frame)
    if len(t_f) < 2:
        raise ValueError(f"Focal {focal!r} has fewer than 2 valid tracked frames")
    if len(t_p) < 2:
        raise ValueError(f"Partner {partner!r} has fewer than 2 valid tracked frames")

    speed_f, heading_f, ang_speed_f = _kinematics(t_f, x_f, y_f)
    px_at_f = _interp(t_p, x_p, t_f)
    py_at_f = _interp(t_p, y_p, t_f)
    dx = px_at_f - x_f
    dy = py_at_f - y_f
    distance = np.hypot(dx, dy)
    abs_bearing = np.arctan2(dy, dx)
    rel_bearing = _wrap_pi(abs_bearing - heading_f)
    rel_speed = _diff_series(t_f, distance)

    out = {
        "speed": _interp(t_f, speed_f, t_grid_ephys),
        "angular_speed": _interp(t_f, ang_speed_f, t_grid_ephys),
        "distance": _interp(t_f, distance, t_grid_ephys),
        "relative_bearing": _interp(t_f, rel_bearing, t_grid_ephys),
        "relative_speed": _interp(t_f, rel_speed, t_grid_ephys),
    }

    head_col = next((c for c in _HEAD_DIR_COLUMNS if c in focal_df.columns), None)
    if head_col is not None:
        h_vals = focal_df[head_col].to_numpy(dtype=np.float64)
        t_h, h_v = _series_from_frames(
            focal_df["frame"].to_numpy(), h_vals, ephys_per_frame,
        )
        if len(t_h) >= 2:
            out["head_dir"] = _interp(t_h, h_v, t_grid_ephys)

    _populate_event_indicators(
        out, events, t_grid_ephys, event_window_sec, event_types,
    )

    return pd.DataFrame(out, index=pd.Index(t_grid_ephys, name="t_ephys"))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _ephys_per_frame(
    tracking: VideoTrackingData, sync: DataSyncManager,
) -> np.ndarray:
    """Return the session-wide ephys-time array indexed by frame number."""
    if tracking.ephys_timestamps is not None:
        return np.asarray(tracking.ephys_timestamps, dtype=np.float64)
    if tracking.timestamps is None:
        raise ValueError(
            "tracking has neither ephys_timestamps nor timestamps — "
            "cannot align with the ephys clock."
        )
    behav_seconds = np.asarray(tracking.timestamps, dtype=np.float64) / 1e9
    return np.asarray(
        sync.convert_behavior_to_ephys(behav_seconds), dtype=np.float64,
    )


def _series_from_frames(
    frames: np.ndarray, values: np.ndarray, ephys_per_frame: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Look up ephys time per frame, drop NaN/dup, return sorted (t, value)."""
    frames = np.asarray(frames)
    n_frames = len(ephys_per_frame)
    valid_frame = (frames >= 0) & (frames < n_frames)
    t = np.full(len(frames), np.nan, dtype=np.float64)
    t[valid_frame] = ephys_per_frame[frames[valid_frame]]
    valid = (~np.isnan(t)) & (~np.isnan(values))
    if not valid.any():
        return np.array([]), np.array([])
    t = t[valid]
    v = values[valid].astype(np.float64)
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], v[keep]


def _position_series(
    obj_df: pd.DataFrame, ephys_per_frame: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, x, y) sorted ephys-time position series for one animal."""
    for col in ("frame", "center_x", "center_y"):
        if col not in obj_df.columns:
            raise KeyError(f"Tracking DataFrame missing required column {col!r}")
    frames = obj_df["frame"].to_numpy()
    x_full = obj_df["center_x"].to_numpy(dtype=np.float64)
    y_full = obj_df["center_y"].to_numpy(dtype=np.float64)
    t_x, x = _series_from_frames(frames, x_full, ephys_per_frame)
    t_y, y = _series_from_frames(frames, y_full, ephys_per_frame)
    if np.array_equal(t_x, t_y):
        return t_x, x, y
    common = np.intersect1d(t_x, t_y)
    x_c = np.interp(common, t_x, x)
    y_c = np.interp(common, t_y, y)
    return common, x_c, y_c


def _kinematics(
    t: np.ndarray, x: np.ndarray, y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(speed, heading, angular_speed)`` on the same ``t`` grid.

    Heading is derived from velocity (``atan2(vy, vx)``); angular speed is
    the time derivative of the unwrapped heading.
    """
    nan = np.full_like(t, np.nan, dtype=np.float64)
    if len(t) < 2:
        return nan, nan, nan
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    speed = np.hypot(vx, vy)
    heading = np.arctan2(vy, vx)
    heading_unwrapped = np.unwrap(heading)
    angular_speed = np.gradient(heading_unwrapped, t)
    return speed, heading, angular_speed


def _wrap_pi(theta: np.ndarray) -> np.ndarray:
    """Wrap angles to ``[-π, π]``."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def _diff_series(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return np.full_like(t, np.nan, dtype=np.float64)
    return np.gradient(y, t)


def _interp(
    t_old: np.ndarray, y: np.ndarray, t_new: np.ndarray,
) -> np.ndarray:
    """Linear interp ``t_old → t_new``; NaN outside ``t_old``'s range."""
    t_new = np.asarray(t_new, dtype=np.float64)
    t_old = np.asarray(t_old, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = (~np.isnan(t_old)) & (~np.isnan(y))
    if valid.sum() < 2:
        return np.full_like(t_new, np.nan, dtype=np.float64)
    t_v = t_old[valid]
    y_v = y[valid]
    order = np.argsort(t_v)
    t_v = t_v[order]
    y_v = y_v[order]
    out = np.interp(t_new, t_v, y_v)
    out[(t_new < t_v[0]) | (t_new > t_v[-1])] = np.nan
    return out


def _populate_event_indicators(
    out: dict,
    events: BehavioralEventsData,
    t_grid_ephys: np.ndarray,
    event_window_sec: float,
    event_types: Optional[Iterable[str]],
) -> None:
    df = events.events_data
    if df is None or len(df) == 0:
        return
    if "ts_start_ephys" not in df.columns or "type" not in df.columns:
        logger.warning(
            "events_data lacks ts_start_ephys and/or type columns — skipping "
            "event indicator columns. Call BehavioralEventsData."
            "synchronize_with_ephys first."
        )
        return
    if event_types is None:
        types = sorted(df["type"].dropna().unique())
    else:
        types = list(event_types)
    for etype in types:
        mask = df["type"] == etype
        starts = df.loc[mask, "ts_start_ephys"].dropna().to_numpy(dtype=np.float64)
        col = f"event_{etype}"
        if len(starts) == 0:
            out[col] = np.zeros_like(t_grid_ephys, dtype=np.float64)
            continue
        diffs = np.abs(t_grid_ephys[:, None] - starts[None, :])
        out[col] = (diffs.min(axis=1) <= event_window_sec).astype(np.float64)
