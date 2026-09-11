"""
Video Tracking Import Module

Provides VideoTrackingData (a dataclass holding parsed multi-object tracking
data and frame timestamps) and load_tracking_data() to load it from a
DataStorageManager or directly from a tracking CSV path.

Two on-disk tracking formats are supported; :func:`detect_tracking_format`
picks between them from the header row alone, and both land in the same
:class:`VideoTrackingData`:

``mask_metrics`` (long)
    One row per ``(frame, object)``, identified by ``object_name`` /
    ``object_id``, with ``center_x`` / ``center_y`` and blob shape statistics.
    Frame timestamps live in a sibling ``*_ts.npy``.

``apt_tqt`` (wide)
    APT/TQT export (``<root>/cohort7_<date>_<HHMM>/solution/TQT_named.csv``).
    One row per *frame*, a ``timestamp`` column in Linux nanoseconds, and one
    column group per animal: ``<name>_center_{x,y}`` plus ``<name>_kp<i>_{x,y}``
    keypoints. Animals are named ``rat4635`` for rat 635 — an extra ``4``
    that :func:`normalize_object_name` strips, so object names match the
    ``rat635`` used everywhere else in the pipeline. Frames where an animal was
    not detected are blank and are dropped from that animal's DataFrame, which
    reproduces the long format's "no row for a missing detection".
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

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


def _read_tracking_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a tracking CSV/TSV with encoding and separator fallbacks.

    Reads as ``utf-8-sig`` so that a byte-order mark — which the APT exports
    carry, and which would otherwise turn the first column name into
    ``'﻿timestamp'`` — is stripped. Plain UTF-8 files are unaffected.
    Extra *kwargs* (``nrows``, ``usecols``, ...) pass through to
    :func:`pandas.read_csv`.
    """
    suffix = path.suffix.lower()
    if suffix == '.csv':
        try:
            return pd.read_csv(path, encoding='utf-8-sig', **kwargs)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='latin-1', **kwargs)
        except pd.errors.ParserError:
            return pd.read_csv(path, sep=';', encoding='utf-8-sig', **kwargs)
    if suffix in ('.tsv', '.txt'):
        try:
            return pd.read_csv(path, sep='\t', encoding='utf-8-sig', **kwargs)
        except pd.errors.ParserError:
            try:
                return pd.read_csv(path, sep=' ', encoding='utf-8-sig', **kwargs)
            except pd.errors.ParserError:
                return pd.read_csv(path, sep=',', encoding='utf-8-sig', **kwargs)
    raise ValueError(f"Unsupported tracking file format: {suffix}")


# ---------------------------------------------------------------------------
# Tracking file formats
# ---------------------------------------------------------------------------

#: Long format: one row per (frame, object), keyed by ``object_name``.
TRACKING_FORMAT_MASK_METRICS = 'mask_metrics'

#: Wide format: one row per frame, one column group per animal (APT/TQT).
TRACKING_FORMAT_APT_TQT = 'apt_tqt'

#: Header matched neither of the above.
TRACKING_FORMAT_UNKNOWN = 'unknown'

#: A wide-format position column: ``rat4635_center_x``, ``rat4635_kp2_y``, ...
#: The object part is greedy, so a name containing ``_center_x`` would still
#: split at the last such suffix; APT names have no underscores in practice.
_APT_COLUMN_RE = re.compile(r'^(?P<obj>.+)_(?P<field>center|kp\d+)_(?P<axis>[xy])$')

#: APT names rat 635 as ``rat4635``. The leading ``4`` is a tracker-side prefix,
#: not part of the animal id, and is stripped so object names match the
#: ``rat635`` / ``635`` used by the configs, kilosort directories and events.
_APT_ANIMAL_RE = re.compile(r'^rat4(\d{3})$', re.IGNORECASE)


def normalize_object_name(name: str) -> str:
    """``'rat4635'`` -> ``'rat635'``; any other name is returned unchanged."""
    match = _APT_ANIMAL_RE.match(str(name).strip())
    return f"rat{match.group(1)}" if match else str(name)


def detect_tracking_format(columns: Iterable) -> str:
    """Identify a tracking file's layout from its column names alone.

    Cheap enough to run on a header-only read, which is why
    :func:`read_tracking_header` exists — the APT exports are ~100 MB.
    """
    cols = {str(c) for c in columns}
    if 'object_name' in cols and 'object_id' in cols:
        return TRACKING_FORMAT_MASK_METRICS
    if 'timestamp' in cols and any(_APT_COLUMN_RE.match(c) for c in cols):
        return TRACKING_FORMAT_APT_TQT
    return TRACKING_FORMAT_UNKNOWN


def read_tracking_header(path: Union[str, Path]) -> List[str]:
    """Column names of a tracking file, without reading any data rows."""
    return [str(c) for c in _read_tracking_csv(Path(path), nrows=0).columns]


def _apt_column_groups(columns: Iterable) -> Dict[str, Dict[str, str]]:
    """Map each wide-format object to ``{'center_x': <source column>, ...}``.

    Fields come out ordered ``center_x, center_y, kp0_x, kp0_y, ...`` so that
    every per-object DataFrame has the same column layout.
    """
    groups: Dict[str, Dict[str, str]] = {}
    for col in columns:
        match = _APT_COLUMN_RE.match(str(col))
        if match:
            field_name = f"{match.group('field')}_{match.group('axis')}"
            groups.setdefault(match.group('obj'), {})[field_name] = str(col)

    def _field_order(field_name: str):
        stem, axis = field_name.rsplit('_', 1)
        return (0, -1, axis) if stem == 'center' else (1, int(stem[2:]), axis)

    return {obj: {f: fields[f] for f in sorted(fields, key=_field_order)}
            for obj, fields in groups.items()}


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def parse_tracking(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split a tracking DataFrame into one DataFrame per tracked object.

    Dispatches on :func:`detect_tracking_format`: wide APT/TQT frames go to
    :func:`parse_apt_tracking`, long mask-metrics frames are grouped by
    ``object_name`` (with ``object_id`` / ``object_name`` dropped and the index
    reset), which is the historical behaviour.
    """
    if detect_tracking_format(df.columns) == TRACKING_FORMAT_APT_TQT:
        return parse_apt_tracking(df)

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


def parse_apt_tracking(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split a wide APT/TQT frame into one DataFrame per animal.

    Each returned DataFrame has columns ``frame``, ``timestamp`` (Linux ns, if
    the source carried one), ``center_x``, ``center_y`` and any
    ``kp<i>_{x,y}`` keypoints, sorted as in :func:`_apt_column_groups`.

    Assumptions:
        - **``frame`` is the 0-based row index of the source file.** The wide
          format has no frame column, and ``VideoTrackingData.timestamps`` is
          indexed by frame through :func:`_index_by_frame`, so numbering rows
          from 0 against a timestamps array built from the same rows makes
          ``timestamps[frame]`` exactly that row's own timestamp. Rows are not
          reordered, and a missing detection removes a row from *one* animal
          without shifting anybody else's frame numbers.
        - **A frame where both centre coordinates are blank is dropped** for
          that animal rather than kept as NaN, which is what the long format
          does for an undetected object. Keypoint-only rows are kept, since the
          centre is what every downstream analysis reads.
        - Object names are normalised by :func:`normalize_object_name`. If two
          raw names collide after normalising (e.g. both ``rat4635`` and
          ``rat635`` present) the second keeps its raw name and a warning is
          logged, because silently dropping one animal's trajectory is worse
          than an oddly-named key.
    """
    groups = _apt_column_groups(df.columns)
    if not groups:
        raise ValueError(
            "No wide-format position columns found (expected e.g. "
            "'rat4635_center_x'); this does not look like an APT/TQT export.")

    n_rows = len(df)
    frames = np.arange(n_rows, dtype=np.int64)
    timestamps = (df['timestamp'].to_numpy() if 'timestamp' in df.columns
                  else None)

    result: Dict[str, pd.DataFrame] = {}
    for raw_name, fields in groups.items():
        name = normalize_object_name(raw_name)
        if name in result:
            # The raw name is the first fallback, but it can collide too (a
            # file holding both 'rat4631' and 'rat631'), so keep suffixing
            # until the key is free. Dropping an animal is never an option.
            candidate = raw_name
            suffix = 2
            while candidate in result:
                candidate = f"{raw_name}__{suffix}"
                suffix += 1
            logger.warning(
                "Object name collision: '%s' normalises to '%s', which is "
                "already present; storing it as '%s' instead.",
                raw_name, name, candidate)
            name = candidate

        columns: Dict[str, np.ndarray] = {'frame': frames}
        if timestamps is not None:
            columns['timestamp'] = timestamps
        for field_name, source_col in fields.items():
            columns[field_name] = pd.to_numeric(
                df[source_col], errors='coerce').to_numpy(dtype=np.float64)

        obj = pd.DataFrame(columns)
        if 'center_x' in obj.columns and 'center_y' in obj.columns:
            detected = obj['center_x'].notna() | obj['center_y'].notna()
            obj = obj[detected]
        result[name] = obj.reset_index(drop=True)

    return result


def read_tracking_centers(path: Union[str, Path]) -> pd.DataFrame:
    """Long ``(frame, object_name, center_x, center_y)`` view of either format.

    Reads only the columns it needs, so it stays cheap on the ~100 MB APT
    exports. Exists so that consumers which want centre positions but not a
    full :class:`VideoTrackingData` — chiefly
    :func:`discovery.manifest_build.probe_tracking` — can treat both formats
    through one shape.
    """
    path = Path(path)
    header = read_tracking_header(path)
    fmt = detect_tracking_format(header)

    if fmt == TRACKING_FORMAT_APT_TQT:
        centre_cols = [c for c in header
                       if _APT_COLUMN_RE.match(c)
                       and _APT_COLUMN_RE.match(c).group('field') == 'center']
        usecols = (['timestamp'] if 'timestamp' in header else []) + centre_cols
        wide = _read_tracking_csv(path, usecols=usecols)
        parsed = parse_apt_tracking(wide)
        if not parsed:
            return pd.DataFrame(columns=['frame', 'object_name',
                                         'center_x', 'center_y'])
        return pd.concat(
            [obj.assign(object_name=name) for name, obj in parsed.items()],
            ignore_index=True,
        )[['frame', 'object_name', 'center_x', 'center_y']]

    wanted = ['frame', 'object_id', 'object_name', 'center_x', 'center_y']
    usecols = [c for c in wanted if c in header]
    return _read_tracking_csv(path, usecols=usecols)


def load_timestamps(tracking_file_path: Union[str, Path]) -> np.ndarray:
    """Frame timestamps (Linux nanoseconds) for a tracking file.

    Long mask-metrics files keep them in a sibling ``*_ts.npy``, named as the
    tracking file's stem with ``_mask_metrics`` replaced by ``_ts`` (e.g.
    ``RatCity_20251210_1359_40Hz_mask_metrics.csv`` →
    ``RatCity_20251210_1359_40Hz_ts.npy``). Wide APT/TQT files carry their own
    ``timestamp`` column, which is read directly.
    """
    path = Path(tracking_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tracking file not found: {path}")

    # Sibling-file lookup first: it costs a stat, where format detection costs
    # a header read over the share.
    stem = path.stem
    if '_mask_metrics' in stem:
        ts_path = path.parent / f"{stem.replace('_mask_metrics', '_ts')}.npy"
        if ts_path.exists():
            return np.load(ts_path)

    try:
        has_timestamp_column = (
            detect_tracking_format(read_tracking_header(path))
            == TRACKING_FORMAT_APT_TQT)
    except Exception as e:          # unreadable header: fall through to raise
        logger.debug("Could not read header of %s: %s", path.name, e)
        has_timestamp_column = False

    if has_timestamp_column:
        return _read_tracking_csv(path, usecols=['timestamp'])['timestamp'].to_numpy()

    raise FileNotFoundError(
        f"No frame timestamps found for {path}: no '*_ts.npy' sibling and no "
        "'timestamp' column"
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
        If True, attempt to load frame timestamps. For long mask-metrics files
        that means the paired ``*_ts.npy``; wide APT/TQT files carry a
        ``timestamp`` column, which is taken from the frame already in memory
        rather than re-read from disk.

    Notes
    -----
    Both on-disk formats are accepted and produce the same object; see the
    module docstring. One call loads **one file**. The APT exports are 30-minute
    chunks, so a date can resolve to several of them — select with *file_index*
    rather than expecting a whole session in one object, and check which
    recording a chunk actually belongs to (``tracking.attachment_status`` in the
    capability manifest) before trusting it on a given recording's clock.
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
    fmt = detect_tracking_format(df.columns)
    parsed = parse_tracking(df)

    timestamps = None
    if load_ts:
        if fmt == TRACKING_FORMAT_APT_TQT and 'timestamp' in df.columns:
            # Already in memory, and guaranteed row-aligned with the frame
            # numbering parse_apt_tracking assigned.
            timestamps = df['timestamp'].to_numpy()
        else:
            try:
                timestamps = load_timestamps(path)
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Could not load timestamps for %s: %s", path.name, e)

    logger.info("Loaded %d objects (%s) in %s format, %d frames",
                len(parsed), list(parsed.keys()), fmt,
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
