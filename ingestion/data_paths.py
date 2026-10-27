"""
Path resolution for one animal's data in one session.

A cohort config (``config/*.json``) names four roots — ``ephys``, ``video``,
``tracking`` (a path or a list of paths) and ``events`` — plus optional
calibration and sync settings. This module turns ``(animal_id, session_id)``
into concrete files under those roots:

- **ephys** — ``<ephys>/<YYYYmmdd_HHMMSS>.rec/<rat>/<stem>.kilosort/kilosort4``
  plus ``<stem>.DIO/…Din<N>.dat``. A ``.rec`` directory is a *day* and may hold
  several recordings; see :func:`_discover_recordings` and
  :func:`_select_recording` for how one is chosen.
- **video / tracking / events** — found by the session's 8-digit date. On a
  day with several recordings every recording is offered the same files; see
  :meth:`DataStorageManager._warn_if_date_resolved_files_may_not_belong`.
- **pulse log** — one file per ephys root, shared by every session.

:class:`DataStorageManager` bundles all of these for one animal/session and
caches the result as JSON, because each lookup lists directories on the
network share.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, TypeVar

import pandas as pd

from ingestion._conventions import (
    FULL_RECORDING_ID_RE as _FULL_RECORDING_ID_RE,
    RECORDING_STAMP_RE as _RECORDING_STAMP_RE,
    parse_session_date as _parse_session_date,
)

logger = logging.getLogger(__name__)

# Bumped to 2 when recording-level resolution landed: a v1 cache holds a
# ``{session}_merged.*`` path that may not exist on disk at all (cohort 5 writes
# ``_merge``), so those entries must be rebuilt rather than trusted.
# v2 -> v3: tracking discovery gained multi-root support, searches *every*
# date-named subdirectory instead of the first, and picks up date-less files
# such as ``solution/TQT_named.csv``. A v2 entry's ``tracking_files`` is a
# strict subset of what v3 finds, so cached entries must be rebuilt.
# v3 -> v4: the pulse log name became configurable (``"pulse_log"`` key), so a
# v3 entry's ``pulse_log_path`` may name the hard-coded ``pulse_log.txt``.
_CACHE_VERSION = 4
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "data_paths"
_CONFIG_DIR = Path(__file__).parent.parent / "config"

#: Prefix assumed when an animal directory holds no ``*.kilosort`` directory at
#: all. Preserves the pre-recording-discovery behaviour of returning a path
#: that does not exist rather than raising, which callers rely on to report
#: "no ephys here" instead of crashing.
_LEGACY_STEM_SUFFIX = '_merged'

_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
_TRACKING_EXTENSIONS = ['.csv', '.h5', '.hdf5', '.mat', '.pkl', '.npz', '.json', '.txt']

#: Tracking files whose *name* carries no date because their *directory* does.
#: The APT/TQT exports land at
#: ``<root>/cohort7_<date>_<HHMM>/solution/TQT_named.csv``, so the date-glob
#: below would never see them. Only the ``_named`` solution is listed: the
#: sibling ``TQT.csv`` uses numeric tracklet ids instead of animal names, and an
#: analysis cannot tell which rat is which from it.
_DATELESS_TRACKING_FILENAMES = ('TQT_named.csv',)

#: Subdirectories of a date-named tracking directory to look inside for the
#: files above, ``'.'`` meaning the directory itself.
_DATELESS_TRACKING_SUBDIRS = ('.', 'solution')


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    """Read a cohort config.

    ``None`` means ``config/default_paths.json`` (cohort 7). Otherwise a file
    of that *name* inside ``config/`` wins, and ``config_path`` is used as
    given only when no such file exists — so ``'cohort5_paths.json'`` works
    from any working directory.

    Raises:
        FileNotFoundError: the config file does not exist.
        ValueError: the file is not valid JSON.
    """
    if config_path is None:
        path = _CONFIG_DIR / "default_paths.json"
    else:
        candidate = _CONFIG_DIR / Path(config_path).name
        path = candidate if candidate.exists() else Path(config_path)

    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def _root(config: dict, key: str) -> Path:
    """The configured root directory for *key* (``'ephys'``, ``'video'``, ...)."""
    if key not in config:
        raise KeyError(f"'{key}' key not found in configuration file")
    return Path(config[key])


def _find_matching_directories(base_path: Path, partial_name: str, suffix: str = "") -> List[Path]:
    """Directories in *base_path* whose name contains *partial_name* and ends with *suffix*.

    An unreadable or missing *base_path* yields an empty list.
    """
    try:
        return [item for item in base_path.iterdir()
                if item.is_dir() and partial_name in item.name and item.name.endswith(suffix)]
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Ephys: day directories and the recordings inside them
# ---------------------------------------------------------------------------

class Recording(NamedTuple):
    """One continuous recording of one animal, as it exists on disk.

    A ``.rec`` directory is a *day* of acquisition, not a single recording:
    ``20251216_094334.rec/rat613/`` holds ``20251216_094334_merged.*``,
    ``20251216_144334_merged.*`` and ``20251216_194334_merged.*`` — morning,
    afternoon and evening blocks, each with its own kilosort output, DIO,
    LFP and timestampoffset. ``stem`` is the shared on-disk prefix; every
    artifact path for this recording is built from it.
    """

    animal_dir: Path
    stem: str
    recording_id: str
    session_dir_id: str
    is_primary: bool

    @property
    def kilosort_path(self) -> Path:
        return self.animal_dir / f"{self.stem}.kilosort" / "kilosort4"

    def dio_path(self, channel: int) -> Path:
        return self.animal_dir / f"{self.stem}.DIO" / f"{self.stem}.dio_Controller_Din{channel}.dat"


def _discover_recordings(animal_dir: Path, session_dir_id: str) -> List[Recording]:
    """List the recordings under one animal directory, newest naming and old.

    Anchored on ``*.kilosort`` directories because that is the artifact every
    caller ultimately wants, and because the suffix between the timestamp and
    the extension is not fixed: cohort 7 writes ``_merged``, cohort 5 writes
    ``_merge`` (21 of 24 animal directories), one cohort-5 directory has no
    suffix, and one cohort-7 directory prefixes the animal name
    (``rat613_20251209_160716_merge``). Discovering the stem from disk covers
    all four without a precedence rule — verified across both cohorts, no
    animal directory carries two stems for the same timestamp.

    The recording whose timestamp equals the ``.rec`` directory's own is the
    *primary* one; it is what every id resolved to before recording-level
    lookup existed, and what a date-level query still resolves to.
    """
    try:
        entries = sorted(animal_dir.iterdir())
    except OSError:
        return []

    recordings: List[Recording] = []
    for entry in entries:
        if not (entry.is_dir() and entry.name.endswith('.kilosort')):
            continue
        stem = entry.name[:-len('.kilosort')]
        match = _RECORDING_STAMP_RE.search(stem)
        recording_id = match.group(1) if match else session_dir_id
        recordings.append(Recording(
            animal_dir=animal_dir, stem=stem, recording_id=recording_id,
            session_dir_id=session_dir_id,
            is_primary=(recording_id == session_dir_id)))
    return recordings


def _legacy_recording(animal_dir: Path, session_dir_id: str) -> Recording:
    """The recording assumed when an animal directory holds no kilosort output.

    Its paths do not exist; returning them (rather than raising) is what lets
    callers report "no ephys here".
    """
    return Recording(animal_dir=animal_dir, stem=f"{session_dir_id}{_LEGACY_STEM_SUFFIX}",
                     recording_id=session_dir_id, session_dir_id=session_dir_id,
                     is_primary=True)


def _select_recording(recordings: List[Recording], session_id: str,
                      session_dir: Path) -> Recording:
    """Pick the one recording a query names, or fail closed.

    Resolution order, chosen so that every id that resolved before still
    resolves to the same recording:

    1. an exact ``recording_id`` match wins outright;
    2. a query that *is* a full recording stamp and matches nothing raises,
       rather than falling back. Asking for ``'20251216_030303'`` and being
       handed the 09:43 recording would silently answer a different question
       than the one asked;
    3. otherwise partial matches are collected (the module's long-standing
       substring convention, so ``'613'`` matches ``'rat613'``);
    4. if several match and one of them is the primary recording, the primary
       wins — a date-level query such as ``'20251216'`` means the day's first
       recording, as it always has;
    5. if several match and none is primary, raise. Silently picking one of
       three afternoon blocks would attach a result to the wrong recording,
       and nothing downstream could detect it.
    """
    exact = [r for r in recordings if r.recording_id == session_id]
    if len(exact) == 1:
        return exact[0]

    if not exact and _FULL_RECORDING_ID_RE.fullmatch(session_id):
        raise FileNotFoundError(
            f"No recording '{session_id}' in {session_dir.name}; it holds "
            f"{', '.join(sorted(r.recording_id for r in recordings))}")

    partial = exact or [r for r in recordings
                        if session_id in r.recording_id or session_id in r.stem]
    if not partial:
        # The query matched the .rec directory but no recording inside it
        # (e.g. a bare date against a directory named for a later time).
        partial = list(recordings)

    if len(partial) == 1:
        return partial[0]

    primary = [r for r in partial if r.is_primary]
    if len(primary) == 1:
        logger.info(
            "'%s' matches %d recordings in %s (%s); using the primary one "
            "(%s). Pass a full recording id to select another.",
            session_id, len(partial), session_dir.name,
            ', '.join(r.recording_id for r in partial),
            primary[0].recording_id)
        return primary[0]

    raise ValueError(
        f"Session id '{session_id}' matches {len(partial)} recordings in "
        f"{session_dir.name} and none of them is the primary recording: "
        f"{', '.join(sorted(r.recording_id for r in partial))}. "
        "Pass a full recording id (e.g. '20251216_144334').")


def _matching_session_dirs(ephys_base: Path, session_id: str) -> List[Path]:
    """``.rec`` day directories that *session_id* can refer to.

    Falls back to matching on the date alone, because a recording id usually
    names a block nested inside a ``.rec`` directory named for a different
    time of the same day (the afternoon and evening blocks).
    """
    matches = sorted(_find_matching_directories(ephys_base, session_id, ".rec"))
    if matches:
        return matches
    try:
        session_date = _parse_session_date(session_id)
    except ValueError:
        return []
    return sorted(_find_matching_directories(ephys_base, session_date, ".rec"))


class _ResolvedRecording(NamedTuple):
    """A selected recording plus every recording found beside it.

    ``siblings`` is empty when the animal directory has no kilosort output and
    ``recording`` is the :func:`_legacy_recording` placeholder.
    """
    recording: Recording
    siblings: List[Recording]


def _resolve_session_animal(animal_id: str, session_id: str, config: dict) -> List[_ResolvedRecording]:
    """Resolve the recording(s) of *animal_id* that *session_id* names.

    One entry per matching ``.rec`` directory, each listed once, so callers
    needing several files for the same recording pay the network listing
    cost only once. Supports the module's substring convention for both ids
    (``'613'`` matches ``'rat613'``).

    Callers must build paths from ``Recording.stem``, never as
    ``f"{session}_merged"``: that string is wrong for cohort 5 and for every
    non-primary recording.
    """
    ephys_base = _root(config, 'ephys')
    session_matches = _matching_session_dirs(ephys_base, session_id)
    if not session_matches:
        raise FileNotFoundError(f"No session directory found matching '{session_id}' in {ephys_base}")
    only_one_day_dir = len(session_matches) == 1

    results: List[_ResolvedRecording] = []
    for session_dir in session_matches:
        session_dir_id = session_dir.name.replace('.rec', '')
        animal_matches = _find_matching_directories(session_dir, animal_id)
        if not animal_matches:
            if only_one_day_dir:
                raise FileNotFoundError(f"No animal directory found matching '{animal_id}' in {session_dir}")
            continue
        if len(animal_matches) > 1:
            raise ValueError(f"Multiple animal directories found matching '{animal_id}': {animal_matches}")
        animal_dir = animal_matches[0]

        recordings = _discover_recordings(animal_dir, session_dir_id)
        if not recordings:
            results.append(_ResolvedRecording(_legacy_recording(animal_dir, session_dir_id), []))
            continue
        try:
            selected = _select_recording(recordings, session_id, session_dir)
        except (ValueError, FileNotFoundError):
            # With several candidate .rec directories the recording may live
            # in one of the others; with only one there is nowhere left to
            # look and the caller needs the reason.
            if only_one_day_dir:
                raise
            continue
        results.append(_ResolvedRecording(selected, recordings))

    if not results:
        raise FileNotFoundError(f"No animal directory found matching '{animal_id}' in any of the matched sessions")
    return results


def resolve_recordings(animal_id: str, session_id: str,
                       config_path: Optional[str] = None,
                       _config: Optional[dict] = None) -> List[Recording]:
    """Every recording of *animal_id* on the day *session_id* names.

    The inventory behind :func:`get_kilosort_path`'s single answer. Use it to
    find out that a day holds more than one recording — the manifest builder
    enumerates with this, and ``DataStorageManager`` uses it to warn when a
    non-primary recording inherits date-resolved tracking or events.
    """
    config = _config or load_config(config_path)
    found: List[Recording] = []
    for session_dir in _matching_session_dirs(_root(config, 'ephys'), session_id):
        animal_matches = _find_matching_directories(session_dir, animal_id)
        if len(animal_matches) == 1:
            found.extend(_discover_recordings(animal_matches[0], session_dir.name.replace('.rec', '')))
    return found


def get_kilosort_path(animal_id: str, session_id: str, config_path: Optional[str] = None,
                      _config: Optional[dict] = None) -> List[Path]:
    """Kilosort4 output directories for *animal_id* in *session_id*.

    ``session_id`` may be a date (``"20251210"``, resolving to that day's
    primary recording), a full recording id, including a non-primary block
    such as ``"20251216_144334"``, or any substring the resolver accepts;
    ``animal_id`` may be partial (``"613"``). The ``*.kilosort`` directory
    suffix is discovered from disk (``_merged``, ``_merge``, none).

    Returns one path per matching ``.rec`` directory.

    Raises:
        FileNotFoundError: no matching session/animal/recording.
        KeyError: the config has no ``'ephys'`` root.
        ValueError: several animal directories match, or a date matches
            several recordings none of which is primary.

    Example:
        >>> get_kilosort_path("613", "20251210")[0]
    """
    config = _config or load_config(config_path)
    return [r.recording.kilosort_path for r in _resolve_session_animal(animal_id, session_id, config)]


def get_dio_path(animal_id: str, session_id: str, dio_channel: int = 1,
                 config_path: Optional[str] = None, _config: Optional[dict] = None) -> List[Path]:
    """DIO channel files (``…dio_Controller_Din<dio_channel>.dat``) for *animal_id* in *session_id*.

    Resolves exactly like :func:`get_kilosort_path`.
    """
    config = _config or load_config(config_path)
    return [r.recording.dio_path(dio_channel)
            for r in _resolve_session_animal(animal_id, session_id, config)]


def get_pulse_log_path(config_path: Optional[str] = None, _config: Optional[dict] = None) -> Path:
    """The behaviour-side pulse log: one running file per ephys root.

    ``<ephys>/pulse_log.txt`` unless the config sets ``"pulse_log"`` (a
    filename relative to the ephys root, or an absolute path) — cohort 9's
    log is ``timekeeper_pulse_log_cohort9only.txt``.
    """
    config = _config or load_config(config_path)
    # Path join keeps an absolute override as-is.
    return _root(config, 'ephys') / config.get("pulse_log", "pulse_log.txt")


def get_animals_and_sessions(config_path: Optional[str] = None,
                             _config: Optional[dict] = None) -> pd.DataFrame:
    """Every recording of every animal under the ephys root.

    Returns a DataFrame with columns: session, animal, kilosort_path,
    recording_id, stem, session_dir, is_primary.

    One row per *recording*, not per ``.rec`` directory. A day holding morning,
    afternoon and evening blocks contributes three rows per animal, and
    ``session`` carries the recording id, so the value can be handed straight
    back to :class:`DataStorageManager`. For a primary recording the recording
    id equals the ``.rec`` directory name, so every id this function returned
    before it enumerated recordings is still returned, unchanged. An animal
    directory with no kilosort output still gets a row (with the legacy
    path), so "no ephys here" stays visible rather than silently absent.
    """
    config = _config or load_config(config_path)
    ephys_base = _root(config, 'ephys')
    if not ephys_base.exists():
        raise FileNotFoundError(f"Ephys directory not found: {ephys_base}")

    rows = []
    try:
        for session_dir in sorted(ephys_base.iterdir()):
            if not (session_dir.is_dir() and session_dir.name.endswith('.rec')):
                continue
            session_dir_id = session_dir.name.replace('.rec', '')
            for animal_dir in sorted(session_dir.iterdir()):
                if not (animal_dir.is_dir() and animal_dir.name.startswith('rat')):
                    continue
                recordings = (_discover_recordings(animal_dir, session_dir_id)
                              or [_legacy_recording(animal_dir, session_dir_id)])
                for recording in recordings:
                    rows.append({
                        'session': recording.recording_id,
                        'animal': animal_dir.name,
                        'kilosort_path': recording.kilosort_path,
                        'recording_id': recording.recording_id,
                        'stem': recording.stem,
                        'session_dir': session_dir_id,
                        'is_primary': recording.is_primary,
                    })
    except OSError as e:
        raise RuntimeError(f"Error accessing ephys directory {ephys_base}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['session', 'animal']).reset_index(drop=True)
    return df


def verify_kilosort_path(kilosort_path: Path, check_files: bool = True) -> bool:
    """True if *kilosort_path* exists and (with *check_files*) holds the spike files.

    Only ``spike_times.npy`` / ``spike_clusters.npy`` are checked, so True
    does not guarantee :func:`ingestion.kilosort_data_import.load_kilosort_data`
    succeeds: it also needs a ``*.timestamps.dat`` beside ``kilosort4/``.
    """
    if not kilosort_path.exists():
        logger.error("Kilosort directory does not exist: %s", kilosort_path)
        return False

    if check_files:
        required_files = ["spike_times.npy", "spike_clusters.npy"]
        missing_files = [f for f in required_files if not (kilosort_path / f).exists()]
        if missing_files:
            logger.error("Missing required Kilosort files in %s: %s", kilosort_path, ', '.join(missing_files))
            return False

    logger.info("Kilosort path verified: %s", kilosort_path)
    return True


# ---------------------------------------------------------------------------
# Video, tracking and events: resolved by date
# ---------------------------------------------------------------------------

def get_video_files_by_date(session_id: str, config_path: Optional[str] = None,
                            video_extensions: List[str] = None, subfolder: Optional[str] = None,
                            _config: Optional[dict] = None) -> List[Path]:
    """Video files under the video root (or its *subfolder*) whose name contains the session date.

    Returns paths sorted by filename.
    """
    config = _config or load_config(config_path)
    video_base = _root(config, 'video')
    if not video_base.exists():
        raise FileNotFoundError(f"Video directory not found: {video_base}")

    search_directory = video_base if subfolder is None else video_base / subfolder
    if not search_directory.exists():
        raise FileNotFoundError(f"Subfolder not found: {search_directory}")

    session_date = _parse_session_date(session_id)
    matching_videos: List[Path] = []
    try:
        for ext in video_extensions or _VIDEO_EXTENSIONS:
            matching_videos.extend(search_directory.glob(f"*{session_date}*{ext}"))
    except OSError as e:
        raise RuntimeError(f"Error accessing directory {search_directory}: {e}")

    matching_videos.sort(key=lambda x: x.name)
    return matching_videos


def _tracking_roots(config: dict) -> List[Path]:
    """Tracking search roots, in configured precedence order.

    ``config['tracking']`` is either a single path (historical) or a list of
    paths. A list lets one cohort draw on several trackers — manual mask
    metrics and APT/TQT exports live in different trees — and **order is
    precedence**: results from earlier roots come first, so appending a root
    cannot change which file an existing analysis picks up at index 0.
    """
    value = config['tracking']
    entries = [value] if isinstance(value, (str, Path)) else list(value)
    return [Path(entry) for entry in entries]


def _tracking_search_dirs(root: Path, session_id: str, session_date: str,
                          subfolder: Optional[str]) -> List[tuple]:
    """``(directory, date_scoped)`` pairs to search under one *root*.

    *date_scoped* is True when the directory itself is known to belong to this
    session — either because the caller named it or because its own name
    carries the date — which is what makes it safe to pick up files that have
    no date in their name.
    """
    if subfolder is not None:
        search_directory = root / subfolder
        if not search_directory.exists():
            raise FileNotFoundError(f"Subfolder not found: {search_directory}")
        return [(search_directory, True)]

    # Date-named subdirectories, one level deep. A date can hold several of
    # them (the APT tree chunks a day into ``cohort7_<date>_<HHMM>``), so all
    # matches are searched rather than only the first.
    try:
        matched = [(child, True) for child in sorted(root.iterdir(), key=lambda p: p.name)
                   if child.is_dir() and (session_id in child.name or session_date in child.name)]
    except OSError:
        matched = []
    return matched or [(root, False)]


def get_tracking_files_by_date(session_id: str, config_path: Optional[str] = None,
                               tracking_extensions: List[str] = None, subfolder: Optional[str] = None,
                               _config: Optional[dict] = None) -> List[Path]:
    """Tracking files for the session date, across every configured tracking root.

    Two on-disk layouts are found:

    - files whose *name* contains the date (``*_mask_metrics.csv`` and friends),
      matched by glob in the search directory;
    - files whose *directory* carries the date but whose name does not
      (``<root>/cohort7_<date>_<HHMM>/solution/TQT_named.csv``), matched by
      name against :data:`_DATELESS_TRACKING_FILENAMES`.

    Returns paths ordered by configured root first, then by filename, then by
    full path (which is what separates same-named files such as several
    ``TQT_named.csv``).
    """
    config = _config or load_config(config_path)
    if 'tracking' not in config:
        raise KeyError("'tracking' key not found in configuration file")

    roots = _tracking_roots(config)
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        raise FileNotFoundError(
            "Tracking directory not found: " + ", ".join(str(root) for root in roots))
    for root in roots:
        if root not in existing_roots:
            logger.warning("Tracking root does not exist, skipping: %s", root)

    session_date = _parse_session_date(session_id)
    ext_set = {ext.lower() for ext in tracking_extensions or _TRACKING_EXTENSIONS}

    def wanted(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in ext_set

    found: Dict[str, Path] = {}   # keyed by path string; first (highest-precedence) wins
    for root in existing_roots:
        per_root: List[Path] = []
        for search_directory, date_scoped in _tracking_search_dirs(
                root, session_id, session_date, subfolder):
            try:
                per_root.extend(f for f in search_directory.glob(f'*{session_date}*') if wanted(f))
            except OSError as e:
                raise RuntimeError(f"Error accessing directory {search_directory}: {e}")
            if date_scoped:
                for sub in _DATELESS_TRACKING_SUBDIRS:
                    for filename in _DATELESS_TRACKING_FILENAMES:
                        candidate = search_directory / sub / filename
                        if wanted(candidate):
                            per_root.append(candidate)
        for path in sorted(per_root, key=lambda p: (p.name, str(p))):
            found.setdefault(str(path), path)
    return list(found.values())


def _event_search_dirs(events_base: Path, session_id: str, session_date: str) -> List[Path]:
    """Subdirectories of the events root to search: the dated ones, else all of them.

    An unreadable root falls back to searching the root itself.
    """
    try:
        children = [d for d in events_base.iterdir() if d.is_dir()]
    except OSError:
        return [events_base]
    dated = [d for d in children if session_id in d.name or session_date in d.name]
    for d in dated:
        logger.debug("Found matching events subfolder: %s", d.name)
    return dated or children


def get_event_files_by_date(session_id: str, config_path: Optional[str] = None,
                            subfolder: Optional[str] = None,
                            _config: Optional[dict] = None) -> List[Path]:
    """Behavioural-event CSVs whose name contains the session id or date.

    Searched in the events root's date-named subdirectories (all of them if
    none is dated, or only *subfolder* if given) and one level below each.
    Returns paths sorted by filename; logs a warning when nothing is found.
    """
    config = _config or load_config(config_path)
    events_base = _root(config, 'events')
    if not events_base.exists():
        raise FileNotFoundError(f"Events directory not found: {events_base}")

    session_date = _parse_session_date(session_id)
    patterns = (f"*{session_id}*.csv", f"*{session_date}*.csv")

    found: List[Path] = []
    try:
        if subfolder is not None:
            search_dirs = [events_base / subfolder]
            if not search_dirs[0].exists():
                raise FileNotFoundError(f"Subfolder not found: {search_dirs[0]}")
        else:
            search_dirs = _event_search_dirs(events_base, session_id, session_date)

        for search_dir in search_dirs:
            # The directory itself, then each of its subdirectories.
            for directory in [search_dir, *(d for d in search_dir.iterdir() if d.is_dir())]:
                for pattern in patterns:
                    found.extend(f for f in directory.glob(pattern) if f.is_file())
    except OSError as e:
        raise RuntimeError(f"Error accessing events directory {events_base}: {e}")

    matching_event_files = sorted(dict.fromkeys(found), key=lambda x: x.name)
    if not matching_event_files:
        logger.warning("No behavioral event CSV files found for session %s in %s", session_id, events_base)
    return matching_event_files


# ---------------------------------------------------------------------------
# DataStorageManager
# ---------------------------------------------------------------------------

_T = TypeVar('_T')


def _best_effort(label: str, find: Callable[[], _T], default: _T) -> _T:
    """Run *find*; on any error log it and return *default*.

    A session routinely lacks some stream (no video, no tracking root for the
    cohort), and that must not stop the others from resolving.
    """
    try:
        return find()
    except Exception as e:
        logger.warning("Error loading %s: %s", label, e)
        return default


class DataStorageManager:
    """All data file paths for one animal/session.

    Loads the config once and resolves every stream on construction (unless
    ``auto_load=False``). A stream that fails to resolve is left empty
    (``None`` / ``[]``) with a logged warning rather than raising. Results are
    cached as JSON under ``.cache/data_paths/`` because resolution lists
    directories on the network share; :meth:`refresh` rescans.
    """

    def __init__(self, animal_id: str, session_id: str, config_path: Optional[str] = None,
                 auto_load: bool = True, use_cache: bool = True,
                 cache_dir: Optional[Path] = None):
        self.animal_id = animal_id
        self.session_id = session_id
        self.config_path = config_path
        self._config = load_config(config_path)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR

        self.kilosort_path: Optional[Path] = None
        self.dio_paths: Dict[int, Path] = {}
        self.video_files: List[Path] = []
        self.tracking_files: List[Path] = []
        self.behavioral_event_files: List[Path] = []
        self.pulse_log_path: Optional[Path] = None

        # Which recording of the day this manager resolved to. ``None`` until
        # the ephys paths load (or when they fail to resolve at all).
        self.recording_id: Optional[str] = None
        self.recording_stem: Optional[str] = None
        self.is_primary_recording: Optional[bool] = None
        self.recording_ids_on_date: List[str] = []

        if auto_load:
            self.load_all_paths()

    # --- loading -----------------------------------------------------------

    def load_all_paths(self):
        """Load all paths, from the JSON cache when available, else by scanning."""
        if self.use_cache and self._load_cache():
            self._report()
            return
        logger.info("Loading data paths for %s/%s", self.animal_id, self.session_id)
        self._scan()

    def refresh(self):
        """Re-scan all paths from disk and update the cache."""
        logger.info("Refreshing data paths for %s/%s", self.animal_id, self.session_id)
        self._scan()

    def _scan(self):
        """Resolve every stream from disk, report, and save the cache."""
        self._load_ephys_paths()
        self.video_files = _best_effort(
            "video paths", lambda: get_video_files_by_date(
                self.session_id, subfolder="social_videos", _config=self._config), [])
        self.tracking_files = _best_effort(
            "tracking paths", lambda: get_tracking_files_by_date(
                self.session_id, _config=self._config), [])
        self.behavioral_event_files = _best_effort(
            "behavioral event paths", lambda: get_event_files_by_date(
                self.session_id, _config=self._config), [])
        self.pulse_log_path = _best_effort(
            "sync paths", lambda: get_pulse_log_path(_config=self._config), None)
        self._report()
        if self.use_cache:
            self._save_cache()

    def _load_ephys_paths(self):
        resolved = _best_effort(
            "ephys paths", lambda: _resolve_session_animal(
                self.animal_id, self.session_id, self._config)[0], None)
        if resolved is None:
            self.kilosort_path = None
            self.dio_paths = {}
            self.recording_id = None
            self.recording_stem = None
            self.is_primary_recording = None
            self.recording_ids_on_date = []
            return

        recording, siblings = resolved
        self.kilosort_path = recording.kilosort_path
        self.dio_paths = {ch: recording.dio_path(ch) for ch in range(1, 5)}
        self.recording_stem = recording.stem
        match = _RECORDING_STAMP_RE.search(recording.stem)
        self.recording_id = match.group(1) if match else None
        self.recording_ids_on_date = sorted({r.recording_id for r in siblings})
        # None for the legacy placeholder: there is no recording on disk to be primary.
        self.is_primary_recording = recording.is_primary if siblings else None

    def _report(self):
        self._warn_if_date_resolved_files_may_not_belong()
        self._log_availability_summary()

    def _warn_if_date_resolved_files_may_not_belong(self):
        """Flag date-resolved files landing on a non-primary recording.

        Tracking and behavioural events resolve by 8-digit date, so on a day
        with several recordings every one of them inherits the same files.
        That is right for the recording the video actually covers and wrong
        for the others: session ``20251216``'s only tracking file spans
        09:50-12:00, inside the 09:43 recording, while the 14:43 and 19:43
        blocks have no video at all. Mapped onto their clocks the window lands
        outside the recording entirely.

        This manager cannot settle it — deciding needs the sync mapping, hence
        a DIO read, and path resolution has to stay cheap. The overlap is
        computed by the capability-manifest probe, which builds a
        ``DataSyncManager`` anyway; here we only make the inheritance visible
        rather than silent. Guarded as ``HZ-DATA-008``.
        """
        if self.is_primary_recording is False and (
                self.tracking_files or self.behavioral_event_files):
            logger.warning(
                "%s/%s is not the primary recording of its day (%s on this "
                "date), but tracking/events were resolved by date and may "
                "belong to a different recording. Check "
                "tracking.attachment_status in the capability manifest before "
                "using them (HZ-DATA-008).",
                self.animal_id, self.recording_id or self.session_id,
                ', '.join(self.recording_ids_on_date) or 'unknown')

    def _log_availability_summary(self):
        logger.info(
            "Data availability — recording: %s (primary: %s), ephys: %s, "
            "DIO channels: %s, video: %d, tracking: %d, events: %d, "
            "pulse_log: %s",
            self.recording_id or 'unresolved', self.is_primary_recording,
            self.kilosort_path is not None,
            list(self.dio_paths.keys()) or "None",
            len(self.video_files),
            len(self.tracking_files),
            len(self.behavioral_event_files),
            self.pulse_log_path is not None,
        )

    # --- JSON cache --------------------------------------------------------

    def _cache_file(self) -> Path:
        return self.cache_dir / f"{self.animal_id}_{self.session_id}.json"

    def _to_cache_dict(self) -> dict:
        def opt(p):
            return str(p) if p else None

        return {
            "version": _CACHE_VERSION,
            "animal_id": self.animal_id,
            "session_id": self.session_id,
            "kilosort_path": opt(self.kilosort_path),
            "dio_paths": {str(ch): str(p) for ch, p in self.dio_paths.items()},
            "video_files": [str(p) for p in self.video_files],
            "tracking_files": [str(p) for p in self.tracking_files],
            "behavioral_event_files": [str(p) for p in self.behavioral_event_files],
            "pulse_log_path": opt(self.pulse_log_path),
            "recording_id": self.recording_id,
            "recording_stem": self.recording_stem,
            "is_primary_recording": self.is_primary_recording,
            "recording_ids_on_date": list(self.recording_ids_on_date),
        }

    def _apply_cache_dict(self, data: dict) -> None:
        def opt(key):
            return Path(data[key]) if data.get(key) else None

        def paths(key):
            return [Path(p) for p in data.get(key, [])]

        self.kilosort_path = opt("kilosort_path")
        self.dio_paths = {int(ch): Path(p) for ch, p in data.get("dio_paths", {}).items()}
        self.video_files = paths("video_files")
        self.tracking_files = paths("tracking_files")
        self.behavioral_event_files = paths("behavioral_event_files")
        self.pulse_log_path = opt("pulse_log_path")
        self.recording_id = data.get("recording_id")
        self.recording_stem = data.get("recording_stem")
        self.is_primary_recording = data.get("is_primary_recording")
        self.recording_ids_on_date = list(data.get("recording_ids_on_date") or [])

    def _save_cache(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file(), "w") as f:
                json.dump(self._to_cache_dict(), f, indent=2)
            logger.debug("Saved data-paths cache: %s", self._cache_file())
        except OSError as e:
            logger.warning("Could not save data-paths cache: %s", e)

    def _load_cache(self) -> bool:
        cache_file = self._cache_file()
        if not cache_file.exists():
            return False
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read cache %s: %s — rebuilding", cache_file, e)
            return False
        if data.get("version") != _CACHE_VERSION:
            logger.info("Cache version mismatch in %s — rebuilding", cache_file)
            return False
        try:
            self._apply_cache_dict(data)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Malformed cache %s: %s — rebuilding", cache_file, e)
            return False
        logger.info("Loaded data paths from cache: %s", cache_file)
        return True

    def clear_cache(self) -> None:
        """Delete this animal/session's cache file if it exists."""
        cache_file = self._cache_file()
        if cache_file.exists():
            cache_file.unlink()
            logger.info("Removed cache file: %s", cache_file)

    # --- accessors ---------------------------------------------------------

    def get_kilosort_path(self) -> Optional[Path]:
        """Kilosort4 output directory, or None if ephys did not resolve."""
        return self.kilosort_path

    def get_dio_path(self, channel: int = 1) -> Optional[Path]:
        """DIO file for *channel* (1-4), or None."""
        return self.dio_paths.get(channel)

    def get_video_files(self) -> List[Path]:
        return self.video_files.copy()

    def get_tracking_files(self) -> List[Path]:
        return self.tracking_files.copy()

    def get_behavioral_event_files(self) -> List[Path]:
        return self.behavioral_event_files.copy()

    def get_pulse_log_path(self) -> Optional[Path]:
        return self.pulse_log_path

    def get_pixels_per_cm(self) -> Optional[float]:
        """Pixels-per-cm calibration from the cohort config, or None if unset.

        Tracking ``center_x`` / ``center_y`` are stored in pixels. When the
        optional ``"pixels_per_cm"`` key is present (and non-null) in the cohort
        config, callers convert positions to cm by dividing by this value. When
        the key is absent or null, positions stay in pixels. The single place
        this conversion happens is
        :func:`video.tracking_import.resolve_tracking_on_ephys_clock` (to which
        :meth:`ingestion.multi_animal_session.MultiAnimalSession.get_tracking_on_ephys_clock`
        delegates).
        """
        val = self._config.get("pixels_per_cm")
        return None if val is None else float(val)

    @property
    def sync_method(self) -> str:
        """Ephys↔behavior sync method from the cohort config.

        ``'pulses'`` (default): DIO edges matched to the timekeeper pulse log.
        ``'greyware'``: per-recording SysClock fit from the session ``.rec``
        (cohort 9; see ``ingestion/greyware_sync.py``).
        """
        return self._config.get("sync_method", "pulses")

    @property
    def greyware_latency_s(self) -> float:
        """SysClock stamp latency subtracted by the Greyware sync (s)."""
        return float(self._config.get("greyware_latency_s", 0.1185))

    def __repr__(self) -> str:
        return (f"DataStorageManager({self.animal_id}/{self.session_id}, "
                f"ephys:{'Y' if self.kilosort_path else 'N'}, "
                f"video:{len(self.video_files)}, tracking:{len(self.tracking_files)}, "
                f"events:{len(self.behavioral_event_files)})")
