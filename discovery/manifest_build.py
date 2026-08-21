"""
Builds the capability manifest. This is the **expensive** half of Layer 0.

Everything here touches ``//nearline``, so it must run on a machine where the
share is mounted, and it must run rarely. Consulting the result is a JSON read
and lives in :mod:`discovery.capability_manifest`, which deliberately cannot
import this module's dependencies.

Cost notes
----------
Per session, roughly in order of expense:

``probe_ephys``
    ``load_kilosort_data`` over each animal's directory. The dominant cost on a
    first build; it reuses that function's own pickle cache, so a rebuild is
    much cheaper than the first pass.
``probe_sync``
    Reads a DIO channel and the pulse log. Seconds, but it is what determines
    whether *anything* event- or tracking-aligned is testable at all, so it is
    worth doing up front rather than discovering the failure half an hour into
    an analysis.
``probe_tracking``
    Reads the merged ``*_mask_metrics.csv``. These run to ~90 MB, so this
    deliberately does **not** call ``load_tracking_data`` — it reads five
    columns with ``usecols`` and groups them itself.
``probe_events``
    Trivial: a few hundred rows of pandas filtering.

Assumptions:
    - **Call the same label extractors the analyses call.** The
      ``behavior_type='F'`` yields one usable opponent while ``'EC'`` yields
      eight asymmetry *is* ``BehavioralEventsData.extract_opponent_labels``
      with ``min_events_per_class=5``. Re-implementing that filter here would
      guarantee that the manifest and the analysis eventually disagree, and a
      manifest that disagrees with reality is worse than no manifest. The cost
      is negligible.
    - **``get_animals_and_sessions`` only sees sessions that have ephys**, since
      it walks the configured ephys root for ``*.rec`` directories. Sessions
      with tracking and events but no ephys are invisible to it and are picked
      up by a separate scan, or the manifest silently under-reports the
      dataset.
    - **Metadata, not content, for provenance.** ``mtime`` and ``size`` per
      source file. Hashing gigabytes over SMB to detect drift would be theatre;
      replacement and re-export both change one of those two.
    - **Write incrementally and atomically.** A full pass over ~47
      animal/session pairs is multi-hour, so each session is merged into the
      artifact as it completes via a temp-file replace. A crash costs the
      session in flight, not the run.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import socket
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from discovery.capability_manifest import (
    MANIFEST_SCHEMA_VERSION,
    config_sha256,
)

__all__ = [
    'DEFAULT_QUALITY_THRESHOLDS',
    'enumerate_targets',
    'probe_ephys',
    'probe_sync',
    'probe_tracking',
    'probe_events',
    'derive_analysis_readiness',
    'build_session_record',
    'atomic_write_manifest',
    'merge_session_record',
    'new_manifest',
]

#: Mirrors the thresholds the GUI and decoders use when ``use_quality_cells``
#: is set, so ``n_quality_cells`` in the manifest means the same thing the
#: analyses will see.
DEFAULT_QUALITY_THRESHOLDS = {
    'min_firing_rate': 0.5,
    'min_presence_ratio': 0.8,
    'max_cv_isi': 5.0,
}

_IDENTITY_RE = re.compile(r'^rat\d+$', re.IGNORECASE)
_DATE_RE = re.compile(r'(20\d{6})')

_TRACKING_USECOLS = ['object_name', 'object_id', 'frame', 'center_x', 'center_y']


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _session_date(value: Any) -> Optional[str]:
    match = _DATE_RE.search(str(value or ''))
    return match.group(1) if match else None


def _is_identity_resolved(object_name: str) -> bool:
    """Is this tracked object a named animal rather than an unassigned blob?

    Identity resolution is per-session; some sessions resolve only the focal
    animal and leave the rest as generic detections.
    """
    return bool(_IDENTITY_RE.match(str(object_name).strip()))


def _stat_entry(path: Any) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return {'path': str(path), 'size': None, 'mtime': None, 'missing': True}
    return {'path': str(path), 'size': int(stat.st_size), 'mtime': int(stat.st_mtime)}


@contextlib.contextmanager
def _quiet():
    """Swallow the loaders' ``print``/warning chatter during a sweep.

    Several of the functions called here print progress unconditionally, which
    would bury the per-session PASS/FAIL table this builder is meant to
    produce. Errors still propagate.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield buffer


# ------------------------------------------------------------- enumeration

def enumerate_targets(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Discover sessions and animals for one cohort config.

    Combines the ephys walk with a scan of the configured tracking and event
    roots, because a session with behaviour and video but no ephys does not
    appear in ``get_animals_and_sessions`` at all.
    """
    from ingestion.data_paths import get_animals_and_sessions

    with _quiet():
        frame = get_animals_and_sessions(config_path)

    sessions: Dict[str, Dict[str, Any]] = {}
    if frame is not None and not frame.empty:
        for row in frame.itertuples(index=False):
            session_id = str(getattr(row, 'session'))
            record = sessions.setdefault(session_id, {
                'session_id': session_id,
                'session_date': _session_date(session_id),
                'animals': [],
                'kilosort_paths': {},
                'has_ephys': True,
            })
            animal = str(getattr(row, 'animal'))
            record['animals'].append(animal)
            record['kilosort_paths'][animal] = str(getattr(row, 'kilosort_path', '') or '')

    # Sessions visible only through tracking/events.
    for root_key in ('tracking', 'events'):
        for date in _scan_dates(config_path, root_key):
            if any(record['session_date'] == date for record in sessions.values()):
                continue
            sessions[date] = {
                'session_id': date, 'session_date': date, 'animals': [],
                'kilosort_paths': {}, 'has_ephys': False,
                'discovered_via': root_key,
            }

    for record in sessions.values():
        record['animals'] = sorted(set(record['animals']))
    return sessions


def _scan_dates(config_path: Optional[str], root_key: str) -> List[str]:
    """Dates appearing under a configured non-ephys root."""
    try:
        from ingestion.data_paths import DataStorageManager
        config = DataStorageManager._load_config(config_path) \
            if hasattr(DataStorageManager, '_load_config') else None
    except Exception:
        config = None
    if config is None:
        try:
            with open(config_path or 'config/default_paths.json', encoding='utf-8') as fh:
                config = json.load(fh)
        except Exception:
            return []

    roots = config.get(root_key)
    if isinstance(roots, str):
        roots = [roots]
    if not isinstance(roots, (list, tuple)):
        return []

    found: List[str] = []
    for root in roots:
        try:
            for entry in Path(str(root)).iterdir():
                date = _session_date(entry.name)
                if date:
                    found.append(date)
        except OSError:
            continue
    return sorted(set(found))


# ------------------------------------------------------------------ probes

def probe_ephys(animal_id: str, session_id: str, dsm) -> Dict[str, Any]:
    """Cluster counts, duration and quality-cell count for one animal."""
    from ingestion.kilosort_data_import import load_kilosort_data
    from ingestion.data_paths import verify_kilosort_path

    out: Dict[str, Any] = {
        'kilosort_path': None, 'path_exists': False, 'files_verified': None,
        'n_clusters': None, 'n_quality_cells': None, 'duration_seconds': None,
        'quality_thresholds': dict(DEFAULT_QUALITY_THRESHOLDS),
        'load_error': None,
    }
    try:
        path = dsm.get_kilosort_path()
    except Exception as exc:
        out['load_error'] = f"path resolution failed: {exc}"
        return out
    if path is None:
        out['load_error'] = 'no kilosort path resolved'
        return out

    out['kilosort_path'] = str(path)
    out['path_exists'] = Path(path).exists()
    with contextlib.suppress(Exception):
        out['files_verified'] = bool(verify_kilosort_path(path, check_files=True))

    if not out['path_exists']:
        out['load_error'] = 'kilosort path does not exist'
        return out

    try:
        with _quiet():
            # The path, not the DataStorageManager - passing the DSM here is the
            # stale-API trap recorded as HZ-API-001.
            ks = load_kilosort_data(path)
        out['n_clusters'] = int(len(ks.ks_ids))
        duration = getattr(ks, 'duration_seconds', None)
        if duration is not None:
            out['duration_seconds'] = float(duration)
            out['ephys_window'] = [0.0, float(duration)]
        with contextlib.suppress(Exception):
            with _quiet():
                filtered = ks.get_filtered_cells_spike_times(**DEFAULT_QUALITY_THRESHOLDS)
            out['n_quality_cells'] = int(len(filtered[0]))
    except Exception as exc:
        out['load_error'] = f"{type(exc).__name__}: {exc}"
    return out


def probe_sync(dsm, dio_channel: int = 1) -> Dict[str, Any]:
    """The linear ephys/behaviour clock map, or why there isn't one.

    Worth probing eagerly: a session whose sync fails is untestable for
    everything event- or tracking-aligned, and that is currently discovered
    long after the data has been loaded.
    """
    out: Dict[str, Any] = {'ok': False, 'dio_channel': dio_channel, 'slope': None,
                           'intercept': None, 'error': None}
    try:
        from ingestion.ephys_sync import DataSyncManager
        with _quiet():
            sync = DataSyncManager(dsm, dio_channel=dio_channel)
        out['slope'] = float(getattr(sync, 'slope', float('nan')))
        out['intercept'] = float(getattr(sync, 'intercept', float('nan')))
        out['ok'] = bool(np.isfinite(out['slope']) and out['slope'] != 0.0)
        if not out['ok']:
            out['error'] = f"degenerate sync mapping (slope={out['slope']})"
        return out
    except Exception as exc:
        out['error'] = f"{type(exc).__name__}: {exc}"
        return out


def probe_tracking(dsm, sync=None, *, ephys_duration: Optional[float] = None,
                   pixels_per_cm: Optional[float] = None) -> Dict[str, Any]:
    """Per-object tracking facts, read cheaply.

    Deliberately avoids ``load_tracking_data``: the merged mask-metrics files
    run to tens of megabytes and only five columns are needed.
    """
    from video.tracking_import import _compute_speed, load_timestamps

    out: Dict[str, Any] = {
        'available': False, 'tracking_file': None, 'n_tracking_files': 0,
        'n_frames': None, 'frame_rate_hz': None, 'ephys_window': None,
        'frac_of_ephys_duration_covered': None, 'n_objects': 0,
        'identity_resolved_animals': [], 'n_identity_resolved_animals': 0,
        'unresolved_object_names': [], 'objects': {}, 'error': None,
    }
    try:
        files = dsm.get_tracking_files() or []
    except Exception as exc:
        out['error'] = f"path resolution failed: {exc}"
        return out
    out['n_tracking_files'] = len(files)
    if not files:
        out['error'] = 'no tracking file resolved'
        return out

    path = Path(str(files[0]))
    out['tracking_file'] = str(path)
    try:
        frame = pd.read_csv(path, usecols=_TRACKING_USECOLS)
    except ValueError:
        # Column set differs; fall back to a full read rather than guessing.
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            out['error'] = f"could not read tracking csv: {exc}"
            return out
    except Exception as exc:
        out['error'] = f"could not read tracking csv: {exc}"
        return out

    if 'object_name' not in frame.columns:
        out['error'] = "tracking csv has no 'object_name' column"
        return out

    out['available'] = True
    timestamps = None
    with contextlib.suppress(Exception):
        timestamps = load_timestamps(path)
    if timestamps is not None and len(timestamps):
        out['n_frames'] = int(len(timestamps))
        seconds = np.asarray(timestamps, dtype=np.float64)
        if seconds.max() > 1e12:      # Linux nanoseconds
            seconds = seconds / 1e9
        span = float(seconds[-1] - seconds[0])
        if span > 0:
            out['frame_rate_hz'] = round(float(len(seconds) - 1) / span, 3)
        if sync is not None:
            with contextlib.suppress(Exception):
                start = float(sync.convert_behavior_to_ephys(seconds[0]))
                end = float(sync.convert_behavior_to_ephys(seconds[-1]))
                out['ephys_window'] = [start, end]
                if ephys_duration:
                    covered = max(0.0, min(end, ephys_duration) - max(start, 0.0))
                    out['frac_of_ephys_duration_covered'] = round(
                        covered / float(ephys_duration), 4)

    total_frames = out['n_frames'] or int(frame['frame'].nunique()) \
        if 'frame' in frame.columns else None

    names = [n for n in frame['object_name'].dropna().unique()]
    out['n_objects'] = len(names)
    has_xy = {'center_x', 'center_y'}.issubset(frame.columns)

    for name in sorted(str(n) for n in names):
        group = frame[frame['object_name'] == name]
        record: Dict[str, Any] = {
            'identity_resolved': _is_identity_resolved(name),
            'n_frames_present': int(len(group)),
        }
        if total_frames:
            record['frac_frames_present'] = round(len(group) / float(total_frames), 4)
        if has_xy:
            x = pd.to_numeric(group['center_x'], errors='coerce').to_numpy(dtype=float)
            y = pd.to_numeric(group['center_y'], errors='coerce').to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() >= 2:
                x, y = x[finite], y[finite]
                record['x_std_px'] = round(float(np.std(x)), 3)
                record['y_std_px'] = round(float(np.std(y)), 3)
                record['bbox_px'] = [round(float(x.min()), 1), round(float(x.max()), 1),
                                     round(float(y.min()), 1), round(float(y.max()), 1)]
                if pixels_per_cm:
                    record['x_std_cm'] = round(record['x_std_px'] / pixels_per_cm, 3)
                    record['y_std_cm'] = round(record['y_std_px'] / pixels_per_cm, 3)
                # Same helper the analyses use, so manifest speed and analysis
                # speed cannot diverge.
                with contextlib.suppress(Exception):
                    from scipy.ndimage import gaussian_filter1d
                    t = np.arange(len(x), dtype=np.float64) / (out['frame_rate_hz'] or 40.0)
                    speed = _compute_speed(t, x, y, 0.25, gaussian_filter1d)
                    record['median_speed_px_s'] = round(float(np.median(speed)), 3)
        out['objects'][name] = record

    resolved = sorted(n for n, r in out['objects'].items() if r['identity_resolved'])
    out['identity_resolved_animals'] = resolved
    out['n_identity_resolved_animals'] = len(resolved)
    out['unresolved_object_names'] = sorted(
        n for n, r in out['objects'].items() if not r['identity_resolved'])
    return out


def probe_events(dsm, animal_ids: Sequence[str], sync=None, *,
                 ephys_duration: Optional[float] = None,
                 min_events_per_class: int = 5) -> Dict[str, Any]:
    """Event counts and, per focal animal, the real usable label sets."""
    from video.behavioral_events import load_behavioral_events

    out: Dict[str, Any] = {
        'available': False, 'event_files': [], 'n_events_total': None,
        'by_type': {}, 'animals_seen': [], 'sync_probe_ok': False,
        'frac_events_within_ephys_window': None, 'per_animal': {}, 'error': None,
    }
    try:
        files = dsm.get_behavioral_event_files() or []
    except Exception as exc:
        out['error'] = f"path resolution failed: {exc}"
        return out
    out['event_files'] = [str(f) for f in files]
    if not files:
        out['error'] = 'no behavioural event file resolved'
        return out

    try:
        with _quiet():
            behavior = load_behavioral_events(files, session_id=str(
                getattr(dsm, 'session_id', 'unknown_session')))
    except Exception as exc:
        out['error'] = f"could not load events: {type(exc).__name__}: {exc}"
        return out

    out['available'] = True
    frame = getattr(behavior, 'events_data', None)
    if frame is not None:
        out['n_events_total'] = int(len(frame))
    with contextlib.suppress(Exception):
        out['animals_seen'] = sorted(str(a) for a in behavior.get_available_rats())

    types: List[Optional[str]] = []
    with contextlib.suppress(Exception):
        types = list(behavior.get_available_event_types())
    for event_type in types:
        with contextlib.suppress(Exception):
            subset = behavior.get_events_by_type(event_type)
            out['by_type'][str(event_type)] = int(0 if subset is None else len(subset))

    if sync is not None:
        try:
            with _quiet():
                out['sync_probe_ok'] = bool(
                    behavior.synchronize_with_ephys(sync, create_new_columns=True))
        except Exception as exc:
            out['error'] = f"sync failed: {type(exc).__name__}: {exc}"
        if out['sync_probe_ok'] and ephys_duration:
            with contextlib.suppress(Exception):
                starts = pd.to_numeric(behavior.events_data['ts_start_ephys'],
                                       errors='coerce').to_numpy(dtype=float)
                finite = np.isfinite(starts)
                if finite.any():
                    inside = ((starts >= 0.0) & (starts <= float(ephys_duration))
                              & finite).sum()
                    out['frac_events_within_ephys_window'] = round(
                        float(inside) / float(finite.sum()), 4)

    # Label sets, computed by the very functions the analyses call.
    if not out['sync_probe_ok']:
        out['error'] = (out['error'] or
                        'events not synchronized; label counts unavailable')
        return out

    for animal in animal_ids:
        record: Dict[str, Any] = {'opponent_labels': {}, 'group_labels': {},
                                  'outcome_labels': {}, 'unusable_types': {},
                                  'min_events_per_class': min_events_per_class}
        for event_type in list(types) + [None]:
            key = '__any__' if event_type is None else str(event_type)
            for kind, extractor in (
                    ('opponent_labels', behavior.extract_opponent_labels),
                    ('group_labels', behavior.extract_group_labels),
                    ('outcome_labels', behavior.extract_outcome_labels)):
                try:
                    with _quiet():
                        # All three extractors return
                        # (start_times, end_times, labels) - labels last.
                        _, _, labels = extractor(
                            str(animal), behavior_type=event_type,
                            min_events_per_class=min_events_per_class)
                except Exception as exc:
                    record['unusable_types'].setdefault(
                        key, {})[kind] = f"{type(exc).__name__}: {exc}"
                    continue
                labels = np.asarray(labels)
                if labels.size == 0:
                    record['unusable_types'].setdefault(key, {})[kind] = 'no usable events'
                    continue
                values, counts = np.unique(labels, return_counts=True)
                n_classes = int(len(values))
                entry = {
                    'n_events': int(labels.size),
                    'n_classes_usable': n_classes,
                    'class_counts': {str(v): int(c) for v, c in zip(values, counts)},
                    'majority_baseline': round(float(counts.max() / counts.sum()), 6),
                    'usable': bool(n_classes >= 2),
                }
                if not entry['usable']:
                    entry['reason'] = (
                        f"only {n_classes} class reaches "
                        f"min_events_per_class={min_events_per_class}; LDA needs >= 2")
                record[kind][key] = entry
        out['per_animal'][str(animal)] = record
    return out


# ------------------------------------------------------------- readiness

def derive_analysis_readiness(session_record: Mapping[str, Any]) -> Dict[str, Any]:
    """Precompute the common testability answers.

    A cache over :mod:`discovery.requirements`, so the frequent question is a
    dict lookup. Never trusted by ``check_testable``, which recomputes from the
    underlying facts — a stale readiness block therefore cannot lie, only go
    out of date, and a test asserts the two agree.
    """
    from discovery.requirements import PARAM_SWEEPS, known_analyses, requirements_for
    from discovery.requirements import evaluate_req

    out: Dict[str, Any] = {}
    animals = ((session_record.get('ephys') or {}).get('animals')) or []

    for analysis in known_analyses():
        viable: List[Dict[str, Any]] = []
        notes: List[str] = []
        sweeps = PARAM_SWEEPS.get(analysis) or {}

        for animal in animals:
            base = {'animal_id': animal}
            combos: List[Dict[str, Any]] = [base]
            for name, template in sweeps.items():
                expanded: List[Dict[str, Any]] = []
                for combo in combos:
                    container = _lookup(session_record, template, combo)
                    values = (sorted(container) if isinstance(container, Mapping)
                              else list(container or ()))
                    for value in values:
                        expanded.append(dict(combo, **{name: value}))
                combos = expanded or []
            for combo in combos or [base]:
                unmet = [r for r in requirements_for(analysis)
                         if r.severity == 'blocking'
                         and evaluate_req(r, session_record, combo).satisfied is not True]
                if not unmet:
                    viable.append(combo)
                elif len(combos) == 1:
                    notes.append(f"{combo}: {unmet[0].reason}")

        out[analysis] = {
            'testable': bool(viable),
            'viable_params': viable[:40],
            'n_viable': len(viable),
            'notes': notes[:6],
        }
    return out


def _lookup(record: Mapping[str, Any], template: str, params: Mapping[str, Any]):
    from discovery._predicates import MissingValue, extract_path
    try:
        return extract_path(record, template.format(**dict(params)))
    except (MissingValue, KeyError, IndexError):
        return None


# ------------------------------------------------------------ orchestration

def build_session_record(session_id: str, animals: Sequence[str], *,
                         cohort: str, config_path: Optional[str] = None,
                         probe_level: str = 'full',
                         dio_channel: int = 1) -> Dict[str, Any]:
    """Probe one session. Never raises: failures land in ``build_errors``."""
    from ingestion.data_paths import DataStorageManager

    started = datetime.now(timezone.utc)
    record: Dict[str, Any] = {
        'session_id': session_id,
        'session_date': _session_date(session_id),
        'cohort': cohort,
        'pixels_per_cm': None,
        'aliases': {'numeric_animal_ids': [
            re.sub(r'^rat', '', str(a), flags=re.IGNORECASE) for a in animals]},
        'ephys': {'animals': sorted(str(a) for a in animals),
                  'n_animals_with_ephys': len(animals), 'per_animal': {}},
        'tracking': {'available': False},
        'events': {'available': False},
        'provenance': {'sources': {}, 'probe_level': probe_level, 'errors': []},
    }
    errors: List[Dict[str, Any]] = []
    primary_dsm = None
    ephys_duration = None
    sync = None

    for animal in sorted(str(a) for a in animals):
        try:
            with _quiet():
                dsm = DataStorageManager(animal, session_id, config_path=config_path)
        except Exception as exc:
            errors.append({'animal': animal, 'stage': 'paths',
                           'error': f"{type(exc).__name__}: {exc}"})
            continue
        if primary_dsm is None:
            primary_dsm = dsm
            with contextlib.suppress(Exception):
                record['pixels_per_cm'] = dsm.get_pixels_per_cm()

        if probe_level == 'paths':
            path = None
            with contextlib.suppress(Exception):
                path = dsm.get_kilosort_path()
            record['ephys']['per_animal'][animal] = {
                'kilosort_path': str(path) if path else None,
                'path_exists': bool(path and Path(path).exists()),
                'n_clusters': None, 'n_quality_cells': None,
                'sync': {'ok': None, 'error': 'not probed at --probe-level paths'},
            }
            record['provenance']['sources'].setdefault('kilosort', {})[animal] = \
                _stat_entry(path)
            continue

        block = probe_ephys(animal, session_id, dsm)
        if block.get('load_error'):
            errors.append({'animal': animal, 'stage': 'ephys',
                           'error': block['load_error']})
        if ephys_duration is None and block.get('duration_seconds'):
            ephys_duration = block['duration_seconds']

        sync_block = probe_sync(dsm, dio_channel=dio_channel)
        if sync is None and sync_block.get('ok'):
            sync = _rebuild_sync(dsm, dio_channel)
        block['sync'] = sync_block
        record['ephys']['per_animal'][animal] = block
        record['provenance']['sources'].setdefault('kilosort', {})[animal] = \
            _stat_entry(block.get('kilosort_path'))

    if primary_dsm is not None and probe_level != 'paths':
        record['tracking'] = probe_tracking(
            primary_dsm, sync, ephys_duration=ephys_duration,
            pixels_per_cm=record.get('pixels_per_cm'))
        if record['tracking'].get('error'):
            errors.append({'stage': 'tracking', 'error': record['tracking']['error']})
        record['provenance']['sources']['tracking'] = \
            _stat_entry(record['tracking'].get('tracking_file'))

        record['events'] = probe_events(
            primary_dsm, sorted(record['ephys']['per_animal']), sync,
            ephys_duration=ephys_duration)
        if record['events'].get('error'):
            errors.append({'stage': 'events', 'error': record['events']['error']})
        record['provenance']['sources']['events'] = {
            str(Path(f).name): _stat_entry(f)
            for f in record['events'].get('event_files', [])
        }
    elif primary_dsm is not None:
        with contextlib.suppress(Exception):
            files = primary_dsm.get_tracking_files() or []
            record['tracking'] = {'available': bool(files),
                                  'tracking_file': str(files[0]) if files else None,
                                  'n_tracking_files': len(files)}
            record['provenance']['sources']['tracking'] = \
                _stat_entry(record['tracking'].get('tracking_file'))
        with contextlib.suppress(Exception):
            files = primary_dsm.get_behavioral_event_files() or []
            record['events'] = {'available': bool(files),
                                'event_files': [str(f) for f in files]}

    record['analysis_readiness'] = derive_analysis_readiness(record)
    record['provenance']['errors'] = errors
    record['provenance']['probe_seconds'] = round(
        (datetime.now(timezone.utc) - started).total_seconds(), 2)
    return record


def _rebuild_sync(dsm, dio_channel: int):
    from ingestion.ephys_sync import DataSyncManager
    try:
        with _quiet():
            return DataSyncManager(dsm, dio_channel=dio_channel)
    except Exception:
        return None


def new_manifest(cohorts: Sequence[Mapping[str, Any]], *, probe_level: str,
                 argv: str = '', repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    return {
        'schema_version': MANIFEST_SCHEMA_VERSION,
        'generated_at': _now(),
        'generated_by': {
            'script': 'scripts/build_capability_manifest.py',
            'argv': argv,
            'git_commit': _git_head(root),
            'host': socket.gethostname(),
            'probe_level': probe_level,
        },
        'cohorts': [
            {'name': cohort['name'], 'config_path': cohort['config_path'],
             'config_sha256': config_sha256(root / cohort['config_path']),
             'pixels_per_cm': cohort.get('pixels_per_cm')}
            for cohort in cohorts
        ],
        'sessions': {},
        'build_errors': [],
    }


def _git_head(root: Path) -> Optional[str]:
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(root),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def merge_session_record(manifest: Dict[str, Any], record: Mapping[str, Any]) -> None:
    manifest['sessions'][str(record['session_id'])] = dict(record)
    for error in (record.get('provenance') or {}).get('errors', ()):
        manifest['build_errors'].append(dict(error, session=record['session_id']))


def atomic_write_manifest(manifest: Mapping[str, Any], path: Path) -> Path:
    """Write via a temp file and replace, so a crash never truncates the artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True, default=str)
    os.replace(temp, path)
    return path
