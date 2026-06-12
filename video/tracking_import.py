"""
Video Tracking Import Module

Provides VideoTrackingData (a dataclass holding parsed multi-object tracking
data and frame timestamps) and load_tracking_data() to load it from a
DataStorageManager or directly from a tracking CSV path.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ingestion.data_paths import DataStorageManager
from ingestion.ephys_sync import DataSyncManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VideoTrackingData dataclass
# ---------------------------------------------------------------------------

@dataclass
class VideoTrackingData:
    """Pure data container for multi-object video tracking results.

    All I/O is handled by the standalone ``load_tracking_data()`` function.
    This class only stores parsed per-object DataFrames plus optional frame
    timestamps and provides analysis helpers that operate on the in-memory
    arrays.
    """

    animal_id: str
    session_id: str
    parsed_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    timestamps: Optional[np.ndarray] = None
    ephys_timestamps: Optional[np.ndarray] = None
    tracking_file: Optional[Path] = None
    synchronized: bool = False

    # --- accessors ---------------------------------------------------------

    def get_object_names(self) -> List[str]:
        """List of all tracked object names."""
        return list(self.parsed_data.keys())

    def get_object_data(self, object_name: str) -> Optional[pd.DataFrame]:
        """Return the DataFrame for *object_name*, with substring fallback.

        If *object_name* is not an exact key, returns the data for the first
        key that contains *object_name* as a substring (or vice-versa).
        """
        if object_name in self.parsed_data:
            return self.parsed_data[object_name]

        matches = [k for k in self.parsed_data if object_name in k or k in object_name]
        if len(matches) == 1:
            return self.parsed_data[matches[0]]
        if len(matches) > 1:
            logger.info("Ambiguous object name '%s', matched %s; returning first.",
                        object_name, matches)
            return self.parsed_data[matches[0]]
        return None

    def get_object_trajectory(self, object_name: str) -> Optional[pd.DataFrame]:
        """Frame-sorted (frame, center_x, center_y, [timestamps], [ephys_timestamps]) view."""
        obj_data = self.get_object_data(object_name)
        if obj_data is None:
            return None

        required = ['frame', 'center_x', 'center_y']
        missing = [c for c in required if c not in obj_data.columns]
        if missing:
            logger.warning("Missing trajectory columns for %s: %s", object_name, missing)
            return None

        traj = obj_data[required].copy()
        frames = traj['frame'].to_numpy()

        if self.timestamps is not None:
            traj['timestamps'] = _index_by_frame(self.timestamps, frames)
        if self.ephys_timestamps is not None:
            traj['ephys_timestamps'] = _index_by_frame(self.ephys_timestamps, frames)

        return traj.sort_values('frame').reset_index(drop=True)

    # --- ephys synchronization --------------------------------------------

    def synchronize_with_ephys(self, sync_manager: DataSyncManager) -> bool:
        """Convert frame timestamps from behavior clock (ns) to ephys clock (s).

        Populates ``self.ephys_timestamps`` and adds an ``ephys_timestamps``
        column (frame-indexed) to every per-object DataFrame in
        ``self.parsed_data``.
        """
        if self.timestamps is None or len(self.timestamps) == 0:
            logger.warning("No timestamps loaded; cannot synchronize.")
            return False

        timestamps_sec = self.timestamps / 1e9
        self.ephys_timestamps = sync_manager.convert_behavior_to_ephys(timestamps_sec)

        for obj_df in self.parsed_data.values():
            if 'frame' in obj_df.columns:
                obj_df['ephys_timestamps'] = _index_by_frame(
                    self.ephys_timestamps, obj_df['frame'].to_numpy()
                )

        self.synchronized = True
        logger.info("Synchronized %d frame timestamps with ephys clock.",
                    len(self.ephys_timestamps))
        return True

    def __repr__(self) -> str:
        return (
            f"VideoTrackingData(animal={self.animal_id}, session={self.session_id}, "
            f"objects={len(self.parsed_data)}, "
            f"frames={len(self.timestamps) if self.timestamps is not None else 0}, "
            f"synchronized={self.synchronized})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _index_by_frame(values: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Look up *values* by frame index, returning NaN for out-of-range frames."""
    n = len(values)
    out = np.full(len(frames), np.nan, dtype=np.float64)
    valid = (frames >= 0) & (frames < n)
    out[valid] = values[frames[valid]]
    return out


def _read_tracking_csv(path: Path) -> pd.DataFrame:
    """Read a tracking CSV/TSV with encoding and separator fallbacks."""
    suffix = path.suffix.lower()
    if suffix == '.csv':
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='latin-1')
        except pd.errors.ParserError:
            return pd.read_csv(path, sep=';')
    if suffix in ('.tsv', '.txt'):
        try:
            return pd.read_csv(path, sep='\t')
        except pd.errors.ParserError:
            try:
                return pd.read_csv(path, sep=' ')
            except pd.errors.ParserError:
                return pd.read_csv(path, sep=',')
    raise ValueError(f"Unsupported tracking file format: {suffix}")


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def parse_tracking(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group a tracking DataFrame by ``object_name``.

    Returns a dict mapping each object name to a DataFrame with the
    ``object_id`` and ``object_name`` columns dropped and the index reset.
    """
    required = ['object_name', 'object_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        return {}

    object_names = [n for n in df['object_name'].unique() if pd.notna(n)]
    if not object_names:
        raise ValueError("No valid object names found in the DataFrame")

    result: Dict[str, pd.DataFrame] = {}
    for name in object_names:
        rows = df[df['object_name'] == name].copy()
        rows = rows.drop(columns=[c for c in required if c in rows.columns])
        result[str(name)] = rows.reset_index(drop=True)
    return result


def load_timestamps(tracking_file_path: Union[str, Path]) -> np.ndarray:
    """Load the ``*_ts.npy`` timestamp file paired with a tracking CSV.

    Looks for a sibling file whose name is the tracking file's stem with
    ``_mask_metrics`` replaced by ``_ts`` (e.g.
    ``RatCity_20251210_1359_40Hz_mask_metrics.csv`` →
    ``RatCity_20251210_1359_40Hz_ts.npy``).
    """
    path = Path(tracking_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tracking file not found: {path}")

    stem = path.stem
    if '_mask_metrics' in stem:
        ts_path = path.parent / f"{stem.replace('_mask_metrics', '_ts')}.npy"
        if ts_path.exists():
            return np.load(ts_path)

    raise FileNotFoundError(
        f"No matching '*_ts.npy' timestamp file found for {path}"
    )


def load_tracking_data(
    source: Union[DataStorageManager, str, Path],
    file_index: int = 0,
    load_ts: bool = True,
) -> VideoTrackingData:
    """Load tracking data into a :class:`VideoTrackingData`.

    Parameters
    ----------
    source : DataStorageManager, str, or Path
        Either a configured ``DataStorageManager`` (in which case
        ``file_index`` selects from ``source.get_tracking_files()``) or a
        direct path to a tracking CSV/TSV file.
    file_index : int
        Index into the manager's tracking-file list. Ignored when *source*
        is a path.
    load_ts : bool
        If True, attempt to load the paired ``*_ts.npy`` timestamp file.
    """
    if isinstance(source, DataStorageManager):
        files = source.get_tracking_files()
        if not files:
            raise FileNotFoundError(
                f"No tracking files found for {source.animal_id}/{source.session_id}"
            )
        if file_index >= len(files):
            raise IndexError(
                f"file_index {file_index} out of range (have {len(files)} tracking files)"
            )
        path = Path(files[file_index])
        animal_id = source.animal_id
        session_id = source.session_id
    else:
        path = Path(source)
        animal_id = "unknown_animal"
        session_id = "unknown_session"

    if not path.exists():
        raise FileNotFoundError(f"Tracking file not found: {path}")

    logger.info("Loading tracking data from %s", path)
    df = _read_tracking_csv(path)
    if df.empty:
        logger.warning("Loaded tracking DataFrame is empty: %s", path)
    parsed = parse_tracking(df)

    timestamps = None
    if load_ts:
        try:
            timestamps = load_timestamps(path)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Could not load timestamps for %s: %s", path.name, e)

    logger.info("Loaded %d objects (%s), %d frames",
                len(parsed), list(parsed.keys()),
                len(timestamps) if timestamps is not None else 0)

    return VideoTrackingData(
        animal_id=animal_id,
        session_id=session_id,
        parsed_data=parsed,
        timestamps=timestamps,
        tracking_file=path,
    )


# ---------------------------------------------------------------------------
# Tracking on the ephys clock (canonical conversion)
# ---------------------------------------------------------------------------

def _compute_speed(t, x, y, smoothing_sec, gaussian_filter1d) -> np.ndarray:
    """Gaussian-smoothed speed (units/s) from a position time series."""
    n = len(t)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    dt = np.diff(t)
    median_dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
    if smoothing_sec > 0 and median_dt > 0:
        sigma_frames = smoothing_sec / median_dt
        if sigma_frames > 0:
            x = gaussian_filter1d(x, sigma=sigma_frames, mode="nearest")
            y = gaussian_filter1d(y, sigma=sigma_frames, mode="nearest")
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    return np.sqrt(vx ** 2 + vy ** 2)


def resolve_tracking_on_ephys_clock(
    tracking: VideoTrackingData,
    sync,
    animal_ids: Sequence[str],
    *,
    pixels_per_cm: Optional[float] = None,
    t_start_ephys: Optional[float] = None,
    t_end_ephys: Optional[float] = None,
    speed_smoothing_sec: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    """Per-animal ``(t, x, y, speed)`` tracking on the shared ephys clock.

    This is the **single** place tracking↔ephys time conversion happens. A
    session tracking file already contains every animal, so the same
    ``VideoTrackingData`` is queried per ``animal_id`` via the substring-fallback
    resolver in :class:`VideoTrackingData`. Used both by
    :meth:`ingestion.multi_animal_session.MultiAnimalSession.get_tracking_on_ephys_clock`
    (which loads the tracking + supplies the session sync/calibration) and by
    single-focal analyses such as ``ephys.decode_partner_distance`` that only have
    ephys for the focal animal but still need a partner's trajectory.

    Returns ``{animal_id: DataFrame}`` where each frame has columns:

    - ``t``     : ephys seconds,
    - ``x``,``y``: position in cm (or pixels if ``pixels_per_cm`` is ``None``),
    - ``speed`` : speed in cm/s (or px/s), the Gaussian-smoothed gradient of
      ``(x, y)`` with respect to ``t`` (sigma ``speed_smoothing_sec``).

    Parameters
    ----------
    tracking : VideoTrackingData
        The session tracking (all animals); not required to be pre-synchronized.
    sync : DataSyncManager
        Behavior↔ephys clock map (anything exposing ``convert_behavior_to_ephys``).
    animal_ids : sequence of str
        Animals to resolve trajectories for.
    pixels_per_cm : float, optional
        Calibration. If ``None``, positions are left in pixels and a single
        warning is logged; downstream ``*_cm`` parameters then refer to pixels.
    t_start_ephys, t_end_ephys : float, optional
        Restrict each returned frame to this ephys-second window.
    speed_smoothing_sec : float
        Gaussian sigma (seconds) for smoothing position before differentiating.
    """
    from scipy.ndimage import gaussian_filter1d

    if not tracking.synchronize_with_ephys(sync):
        raise RuntimeError(
            "Could not synchronize tracking with the ephys clock "
            f"(session {tracking.session_id}); no frame timestamps available."
        )

    if pixels_per_cm is None:
        logger.warning(
            "No 'pixels_per_cm' calibration for session %s; tracking positions "
            "are left in PIXELS. All *_cm parameters in downstream analyses then "
            "refer to pixels.",
            tracking.session_id,
        )
        scale = 1.0
    else:
        scale = 1.0 / float(pixels_per_cm)

    out: Dict[str, pd.DataFrame] = {}
    for aid in animal_ids:
        traj = tracking.get_object_trajectory(aid)
        if traj is None or "ephys_timestamps" not in traj.columns:
            logger.warning(
                "No tracking object resolved for animal %s in session %s; "
                "skipping.", aid, tracking.session_id,
            )
            continue

        t = traj["ephys_timestamps"].to_numpy(dtype=np.float64)
        x = traj["center_x"].to_numpy(dtype=np.float64) * scale
        y = traj["center_y"].to_numpy(dtype=np.float64) * scale

        valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
        t, x, y = t[valid], x[valid], y[valid]
        order = np.argsort(t, kind="stable")
        t, x, y = t[order], x[order], y[order]

        speed = _compute_speed(t, x, y, speed_smoothing_sec, gaussian_filter1d)

        df = pd.DataFrame({"t": t, "x": x, "y": y, "speed": speed})
        if t_start_ephys is not None:
            df = df[df["t"] >= t_start_ephys]
        if t_end_ephys is not None:
            df = df[df["t"] <= t_end_ephys]
        out[aid] = df.reset_index(drop=True)

    return out
