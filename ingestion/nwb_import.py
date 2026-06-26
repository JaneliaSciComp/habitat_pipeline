"""
NWB Import Module
=================

Import NWB (Neurodata Without Borders) files into the pipeline's native data
classes so that the existing analysis stack (decoders, plots, social place
fields, ...) can run on NWB-distributed datasets unchanged.

Every analysis module consumes the three dataclasses
(:class:`~ingestion.kilosort_data_import.KilosortData`,
:class:`~video.behavioral_events.BehavioralEventsData`,
:class:`~video.tracking_import.VideoTrackingData`) plus a ``DataSyncManager``-
shaped object -- never file paths. This module produces *identical* dataclasses
from an NWB file, so nothing downstream needs to change.

Two things make NWB simpler than the native Trodes pipeline:

* NWB ``units`` tables store spike times **already in seconds on one clock**,
  so there is no sample->second conversion and no Kilosort folder to locate.
* A self-contained NWB file needs **no DIO/pulse-log synchronization** -- every
  stream already shares the file's clock. The clock map is therefore the
  identity (:class:`IdentitySyncManager`), and behavioral-event timestamps are
  written straight into the ``*_ephys`` columns with ``synchronized=True``.

Entry point
-----------
    from ingestion.nwb_import import load_nwb_session
    session = load_nwb_session("path/to/file.nwb")
    ks, events, tracking, sync = session   # NamedTuple, unpackable

``pynwb`` is imported lazily inside the functions so the rest of the pipeline
never depends on it. Install with ``pip install pynwb``.

Tested against DANDI 001749 (Tye lab reward-competition dataset): mouse mPFC
ephys, one subject per file, ``units`` table + ``cs_onsets``/``us_deliveries``
interval tables. The readers degrade gracefully when a stream is absent (e.g.
no pose estimation / no winner-loser columns).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd

from ingestion.kilosort_data_import import KilosortData
from video.behavioral_events import BehavioralEventsData
from video.tracking_import import VideoTrackingData

logger = logging.getLogger(__name__)

# Behavioral event columns the pipeline understands beyond the timing columns.
# Any of these present in an NWB interval/trials table are passed through so
# that extract_opponent_labels / extract_outcome_labels keep working.
_IDENTITY_COLUMNS = ("initiator", "victim", "winner", "loser")


# ---------------------------------------------------------------------------
# Identity clock map (NWB streams already share one clock)
# ---------------------------------------------------------------------------

class IdentitySyncManager:
    """A no-op ``DataSyncManager`` for self-contained NWB files.

    NWB stores spikes, behavior and tracking on a single clock, so the
    ephys<->behavior map is the identity. This object is interface-compatible
    with :class:`ingestion.ephys_sync.DataSyncManager` (``slope``, ``intercept``,
    ``convert_behavior_to_ephys``, ``convert_ephys_to_behavior``) so any code
    expecting a sync manager works unchanged.
    """

    def __init__(self, session_id: str = "nwb_session"):
        self.session_id = session_id
        self.slope = 1.0
        self.intercept = 0.0
        self.mapping = {
            "slope": 1.0,
            "intercept": 0.0,
            "r_squared": 1.0,
            "n_matches": 0,
            "source": "nwb_identity",
        }

    def convert_ephys_to_behavior(self, ephys_timestamps):
        return np.asarray(ephys_timestamps, dtype=float)

    def convert_behavior_to_ephys(self, behavior_timestamps):
        return np.asarray(behavior_timestamps, dtype=float)

    def __repr__(self) -> str:
        return f"IdentitySyncManager(session={self.session_id}, slope=1.0, intercept=0.0)"


class NwbSession(NamedTuple):
    """Bundle returned by :func:`load_nwb_session`, unpackable as a 4-tuple."""

    ks: KilosortData
    events: BehavioralEventsData
    tracking: VideoTrackingData
    sync: IdentitySyncManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_ids(nwbfile, animal_id: Optional[str], session_id: Optional[str]):
    """Derive (animal_id, session_id) from the NWB file, honoring overrides."""
    subject = getattr(nwbfile, "subject", None)
    sub_id = getattr(subject, "subject_id", None) if subject is not None else None

    if animal_id is None:
        animal_id = str(sub_id) if sub_id else "nwb_subject"

    if session_id is None:
        sid = getattr(nwbfile, "session_id", None)
        if sid:
            session_id = str(sid)
        else:
            start = getattr(nwbfile, "session_start_time", None)
            if start is not None:
                session_id = start.strftime("%Y%m%dT%H%M%S")
            elif sub_id:
                session_id = str(sub_id)
            else:
                session_id = "nwb_session"
    return str(animal_id), str(session_id)


def _column_names(table) -> tuple:
    """Return a table's column names across pynwb versions."""
    cols = getattr(table, "colnames", None)
    return tuple(cols) if cols is not None else tuple()


# ---------------------------------------------------------------------------
# Units  ->  KilosortData
# ---------------------------------------------------------------------------

def nwb_to_kilosort_data(
    nwbfile,
    animal_id: str,
    session_id: str,
    quality_filter: Optional[str] = None,
    source_path: Optional[str] = None,
) -> KilosortData:
    """Build a :class:`KilosortData` from an NWB ``units`` table.

    Only ``spike_times_by_cell`` and ``ks_ids`` are needed by the decoding
    stack (quality-cell selection recomputes firing metrics from spike times),
    so per-cluster geometry fields (channel/DV/XX/amplitude) are populated from
    the ``electrodes`` table when a unit links to it, else left as zero/empty
    stubs.

    Parameters
    ----------
    quality_filter : str, optional
        If given (e.g. ``"good"``), keep only units whose ``quality`` (or
        ``KSLabel``) column equals this value. Default ``None`` keeps every unit.
    """
    units = getattr(nwbfile, "units", None)
    if units is None or len(units.id) == 0:
        raise ValueError("NWB file has no units table (no spike data to import)")

    cols = _column_names(units)
    n_total = len(units.id)

    # cluster ids: prefer an explicit cluster_id column, else the row ids.
    if "cluster_id" in cols:
        all_cluster_ids = [int(c) for c in units["cluster_id"][:]]
    else:
        all_cluster_ids = [int(c) for c in np.asarray(units.id[:])]

    # quality labels (for optional filtering + traceability).
    quality_col = None
    for cand in ("quality", "KSLabel", "group"):
        if cand in cols:
            quality_col = cand
            break
    qualities = (
        [str(q) for q in units[quality_col][:]] if quality_col else [""] * n_total
    )

    # which rows to keep
    if quality_filter is not None:
        keep_idx = [i for i in range(n_total) if qualities[i] == quality_filter]
        if not keep_idx:
            logger.warning(
                "quality_filter=%r matched 0 of %d units (column %r values=%s); "
                "keeping all units instead",
                quality_filter, n_total, quality_col, sorted(set(qualities)),
            )
            keep_idx = list(range(n_total))
    else:
        keep_idx = list(range(n_total))

    # materialize per-unit spike times (seconds) -- this is the core mapping.
    spike_times_by_cell: List[np.ndarray] = []
    for i in keep_idx:
        st = np.asarray(units.get_unit_spike_times(i), dtype=np.float64)
        spike_times_by_cell.append(st)

    ks_ids = [all_cluster_ids[i] for i in keep_idx]
    n = len(ks_ids)
    n_spikes = int(sum(len(st) for st in spike_times_by_cell))

    # per-cluster geometry from the electrodes table, when units link to it.
    channel = np.zeros(n, dtype=int)
    DV = np.zeros(n, dtype=float)
    XX = np.zeros(n, dtype=float)
    if "electrodes" in cols:
        try:
            etable = nwbfile.electrodes
            ecols = _column_names(etable)
            for out_i, i in enumerate(keep_idx):
                region = units["electrodes"][i]
                rows = list(np.atleast_1d(region))
                if not rows:
                    continue
                er = int(rows[0])
                channel[out_i] = er
                if "rel_y" in ecols:
                    DV[out_i] = float(etable["rel_y"][er])
                if "rel_x" in ecols:
                    XX[out_i] = float(etable["rel_x"][er])
        except Exception as exc:  # geometry is optional; never fail the import
            logger.debug("Could not map electrode geometry from NWB: %s", exc)

    cell_numbers = np.column_stack(
        [np.round(XX / 100.0).astype(int), np.arange(n, dtype=int)]
    ) if n else np.zeros((0, 2), dtype=int)

    cluster_info = pd.DataFrame(
        {"cluster_id": ks_ids, "quality": [qualities[i] for i in keep_idx]}
    )

    duration = float(max((st[-1] for st in spike_times_by_cell if len(st)), default=0.0))

    metadata = {
        "data_path": str(source_path) if source_path else "",
        "animal_id": animal_id,
        "session_id": session_id,
        "n_clusters": n,
        "n_spikes": n_spikes,
        "source": "nwb",
        "quality_column": quality_col,
    }

    logger.info(
        "NWB units -> KilosortData: %d units (%d total, filter=%r), %d spikes, ~%.1fs",
        n, n_total, quality_filter, n_spikes, duration,
    )

    return KilosortData(
        animal_id=animal_id,
        session_id=session_id,
        spike_times=np.array([]),          # raw sample stream not stored in NWB
        spike_clusters=np.array([]),
        spike_times_by_cell=spike_times_by_cell,
        ks_ids=ks_ids,
        channel=channel,
        amplitude=np.zeros(n, dtype=float),
        fr=np.array([]),
        amp=np.array([]),
        DV=DV,
        XX=XX,
        cell_numbers=cell_numbers,
        to_load=np.ones(n, dtype=bool),
        cluster_info=cluster_info,
        metadata=metadata,
        _duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Interval / trials tables  ->  BehavioralEventsData
# ---------------------------------------------------------------------------

def _interval_tables(nwbfile) -> Dict[str, object]:
    """Collect all TimeIntervals tables (``intervals`` dict + ``trials``)."""
    tables: Dict[str, object] = {}
    intervals = getattr(nwbfile, "intervals", None)
    if intervals:
        for name, tbl in intervals.items():
            tables[name] = tbl
    trials = getattr(nwbfile, "trials", None)
    if trials is not None and "trials" not in tables:
        tables["trials"] = trials
    return tables


def nwb_to_behavioral_events(
    nwbfile,
    session_id: str,
    source_path: Optional[str] = None,
) -> BehavioralEventsData:
    """Build :class:`BehavioralEventsData` from NWB interval/trials tables.

    Each TimeIntervals table (e.g. ``cs_onsets``, ``us_deliveries``, ``trials``)
    contributes its rows as events, with ``type`` set to the table name and the
    ephys-clock columns populated directly (NWB times are already in seconds on
    one clock), so the returned object is pre-synchronized -- no
    ``synchronize_with_ephys`` call is required.

    Identity columns (initiator/victim/winner/loser) are passed through when
    present, so outcome/opponent decoding works for datasets that score them.
    This dataset (DANDI 001749) only has stimulus/reward intervals, so those
    columns are absent and only time-aligned analyses apply.
    """
    tables = _interval_tables(nwbfile)
    src_name = Path(source_path).name if source_path else "nwb"

    frames: List[pd.DataFrame] = []
    for name, tbl in tables.items():
        try:
            df = tbl.to_dataframe().reset_index(drop=True)
        except Exception as exc:
            logger.warning("Could not read interval table %r: %s", name, exc)
            continue
        if "start_time" not in df.columns:
            continue
        start_s = df["start_time"].to_numpy(dtype=float)
        stop_s = (
            df["stop_time"].to_numpy(dtype=float)
            if "stop_time" in df.columns else start_s
        )

        out = pd.DataFrame({
            "type": name,
            "behavior_full_name": name,
            "ts_start": (start_s * 1e9).astype("int64"),   # ns, format-compat
            "ts_end": (stop_s * 1e9).astype("int64"),
            "ts_start_ephys": start_s,                      # seconds, pre-synced
            "ts_end_ephys": stop_s,
            "source_file": src_name,
        })
        # pass through identity + any extra descriptive columns
        for col in df.columns:
            if col in ("start_time", "stop_time"):
                continue
            out[col] = df[col].to_numpy()
        frames.append(out)

    if frames:
        events_data = pd.concat(frames, ignore_index=True)
        events_data.sort_values("ts_start_ephys", inplace=True, kind="stable")
        events_data.reset_index(drop=True, inplace=True)
    else:
        logger.warning(
            "No interval/trials tables found in NWB file; behavioral events empty"
        )
        events_data = pd.DataFrame(
            columns=["type", "behavior_full_name", "ts_start", "ts_end",
                     "ts_start_ephys", "ts_end_ephys", "source_file"]
        )

    logger.info(
        "NWB intervals -> BehavioralEventsData: %d events across tables %s",
        len(events_data), list(tables.keys()),
    )

    obj = BehavioralEventsData(
        session_id=session_id,
        events_data=events_data,
        event_files=[Path(source_path)] if source_path else [],
        synchronized=True,   # ts_*_ephys populated directly from NWB clock
    )
    return obj


# ---------------------------------------------------------------------------
# Position / pose  ->  VideoTrackingData
# ---------------------------------------------------------------------------

def _spatial_series_to_df(series) -> Optional[pd.DataFrame]:
    """Convert a SpatialSeries-like object to a [frame, center_x, center_y] df."""
    data = np.asarray(series.data[:])
    if data.ndim == 1:
        data = data[:, None]
    n = data.shape[0]
    df = pd.DataFrame({"frame": np.arange(n, dtype=int)})
    df["center_x"] = data[:, 0]
    df["center_y"] = data[:, 1] if data.shape[1] > 1 else np.nan

    # timestamps (seconds) -- explicit array or rate-based
    ts = getattr(series, "timestamps", None)
    if ts is not None:
        sec = np.asarray(ts[:], dtype=float)
    else:
        start = float(getattr(series, "starting_time", 0.0) or 0.0)
        rate = float(getattr(series, "rate", 0.0) or 0.0)
        sec = start + np.arange(n) / rate if rate else np.full(n, np.nan)
    df["ephys_timestamps"] = sec
    return df


def _iter_spatial_series(nwbfile):
    """Yield (name, SpatialSeries) from behavior processing modules / pose."""
    proc = getattr(nwbfile, "processing", {})
    for mod in proc.values():
        interfaces = getattr(mod, "data_interfaces", {})
        for iname, obj in interfaces.items():
            # Position container holds one or more SpatialSeries
            spatial = getattr(obj, "spatial_series", None)
            if spatial:
                for sname, series in spatial.items():
                    yield sname, series
                continue
            # ndx-pose PoseEstimation holds named PoseEstimationSeries
            nodes = getattr(obj, "pose_estimation_series", None)
            if nodes:
                for nname, series in nodes.items():
                    yield nname, series
                continue
            # a bare SpatialSeries / TimeSeries with 2-D data
            if hasattr(obj, "data") and getattr(obj, "data", None) is not None:
                try:
                    if np.asarray(obj.data).ndim == 2:
                        yield iname, obj
                except Exception:
                    pass


def nwb_to_tracking_data(
    nwbfile,
    animal_id: str,
    session_id: str,
    source_path: Optional[str] = None,
) -> VideoTrackingData:
    """Build :class:`VideoTrackingData` from NWB Position/pose data.

    Discovers SpatialSeries (under any ``processing`` module) and ndx-pose
    PoseEstimationSeries, mapping each to a ``[frame, center_x, center_y]``
    DataFrame keyed by series name. Returns an empty (but valid) object when no
    spatial data is present -- as in DANDI 001749, which has video but no pose.
    """
    parsed_data: Dict[str, pd.DataFrame] = {}
    timestamps = None
    ephys_timestamps = None

    for name, series in _iter_spatial_series(nwbfile):
        try:
            df = _spatial_series_to_df(series)
        except Exception as exc:
            logger.warning("Could not convert spatial series %r: %s", name, exc)
            continue
        parsed_data[name] = df
        if ephys_timestamps is None and "ephys_timestamps" in df:
            ephys_timestamps = df["ephys_timestamps"].to_numpy()
            timestamps = ephys_timestamps * 1e9  # ns, for format-compat

    if not parsed_data:
        logger.info("NWB file has no position/pose data; tracking is empty")

    return VideoTrackingData(
        animal_id=animal_id,
        session_id=session_id,
        parsed_data=parsed_data,
        timestamps=timestamps,
        ephys_timestamps=ephys_timestamps,
        tracking_file=Path(source_path) if source_path else None,
        synchronized=bool(parsed_data),  # NWB clock == ephys clock
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def open_nwb(path: Union[str, Path]):
    """Open an NWB file read-only and return ``(nwbfile, io)``.

    The caller must keep ``io`` alive while lazily reading datasets and call
    ``io.close()`` when done. :func:`load_nwb_session` does this for you.
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "pynwb is required to import NWB files. Install with: pip install pynwb"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NWB file not found: {path}")
    io = NWBHDF5IO(str(path), mode="r", load_namespaces=True)
    return io.read(), io


def load_nwb_session(
    path: Union[str, Path],
    animal_id: Optional[str] = None,
    session_id: Optional[str] = None,
    quality_filter: Optional[str] = None,
) -> NwbSession:
    """Load an NWB file into the pipeline's native data classes.

    Returns an :class:`NwbSession` NamedTuple ``(ks, events, tracking, sync)``
    where ``ks``/``events``/``tracking`` are the pipeline's standard dataclasses
    and ``sync`` is an :class:`IdentitySyncManager`. All datasets are fully
    materialized into memory before the underlying file handle is closed.

    Parameters
    ----------
    path : str or Path
        Local path to a ``.nwb`` file.
    animal_id, session_id : str, optional
        Override the ids derived from ``subject.subject_id`` /
        ``session_start_time``.
    quality_filter : str, optional
        Keep only units whose quality label matches (e.g. ``"good"``); default
        keeps all units.
    """
    path = Path(path)
    nwbfile, io = open_nwb(path)
    try:
        animal_id, session_id = _resolve_ids(nwbfile, animal_id, session_id)
        ks = nwb_to_kilosort_data(
            nwbfile, animal_id, session_id,
            quality_filter=quality_filter, source_path=str(path),
        )
        events = nwb_to_behavioral_events(nwbfile, session_id, source_path=str(path))
        tracking = nwb_to_tracking_data(
            nwbfile, animal_id, session_id, source_path=str(path)
        )
    finally:
        io.close()

    sync = IdentitySyncManager(session_id=session_id)
    return NwbSession(ks=ks, events=events, tracking=tracking, sync=sync)
