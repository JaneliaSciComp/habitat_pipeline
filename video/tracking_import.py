"""
Video Tracking Import Module

Provides VideoTrackingData (a dataclass holding parsed multi-object tracking
data and frame timestamps) and load_tracking_data() to load it from a
DataStorageManager or directly from a tracking CSV path.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

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
