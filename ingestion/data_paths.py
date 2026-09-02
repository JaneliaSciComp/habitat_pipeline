"""
Kilosort Path Management Module

This module provides utilities for constructing file paths to Kilosort data
based on animal_id and session_id, reading configuration from the project's
default_paths.json file.
"""

import json
import logging
import re
from pathlib import Path
from typing import NamedTuple, Optional, List, Dict
import pandas as pd

logger = logging.getLogger(__name__)

# Bumped to 2 when recording-level resolution landed: a v1 cache holds a
# ``{session}_merged.*`` path that may not exist on disk at all (cohort 5 writes
# ``_merge``), so those entries must be rebuilt rather than trusted.
_CACHE_VERSION = 2
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "data_paths"

#: The timestamp embedded in a recording's on-disk prefix. Trodes writes
#: ``<YYYYmmdd>_<HHMMSS>`` and every artifact for one recording shares it.
_RECORDING_STAMP_RE = re.compile(r'(20\d{6}_\d{6})')

#: A query that is exactly a recording stamp names one recording and nothing
#: else, so failing to find it must raise rather than fall back to the day's
#: primary recording.
_FULL_RECORDING_ID_RE = re.compile(r'20\d{6}_\d{6}')

#: Prefix assumed when an animal directory holds no ``*.kilosort`` directory at
#: all. Preserves the pre-recording-discovery behaviour of returning a path
#: that does not exist rather than raising, which callers rely on to report
#: "no ephys here" instead of crashing.
_LEGACY_STEM_SUFFIX = '_merged'


def _load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Optional path to config file. If None, uses default location.
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If the config file is not found
        ValueError: If the JSON is invalid
    """
    # Determine config file path
    if config_path is None:
        # Assume this file is in ingestion/ and config/ is a sibling directory
        current_dir = Path(__file__).parent
        config_path = current_dir.parent / "config" / "default_paths.json"
    else:
        # First try the config/ directory using just the filename, then fall back to the path as-is
        config_dir = Path(__file__).parent.parent / "config"
        candidate = config_dir / Path(config_path).name
        config_path = candidate if candidate.exists() else Path(config_path)
        
    # Read the configuration file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    return config


def _parse_session_date(session_id: str) -> str:
    """
    Extract the 8-digit date prefix from a session_id.

    Args:
        session_id: Session identifier (e.g., "20251210" or "20251210_110059")

    Returns:
        str: 8-character date string (YYYYMMDD)

    Raises:
        ValueError: If session_id is too short or doesn't start with 8 digits
    """
    if len(session_id) < 8:
        raise ValueError(f"Session ID too short to extract date: {session_id}")
    session_date = session_id[:8]
    if not session_date.isdigit():
        raise ValueError(f"Could not extract valid date from session_id: {session_id}")
    return session_date


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
    recordings: List[Recording] = []
    try:
        entries = sorted(animal_dir.iterdir())
    except (PermissionError, OSError):
        return recordings

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
        if len(partial) > 1:
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


def _resolve_session_animal(animal_id: str, session_id: str, config: dict) -> List[tuple]:
    """Resolve [(animal_dir, stem), ...] for matching recordings.

    Centralises the two iterdir() walks shared by get_kilosort_path / get_dio_path
    so callers needing several files under the same animal directory only pay the
    network listing cost once.

    ``stem`` replaced the old ``full_session_id`` when recording-level lookup
    landed. Callers must not rebuild it as ``f"{session}_merged"``: that string
    is wrong for cohort 5 and for every non-primary recording.
    """
    if 'ephys' not in config:
        raise KeyError("'ephys' key not found in configuration file")
    ephys_base = Path(config['ephys'])

    session_matches = sorted(_find_matching_directories(ephys_base, session_id, ".rec"))
    if not session_matches:
        # The id may name a recording nested inside a .rec directory named for
        # a different time of the same day, which is the common case for
        # afternoon and evening blocks.
        session_date = None
        try:
            session_date = _parse_session_date(session_id)
        except ValueError:
            session_date = None
        if session_date:
            session_matches = sorted(
                _find_matching_directories(ephys_base, session_date, ".rec"))
    if not session_matches:
        raise FileNotFoundError(f"No session directory found matching '{session_id}' in {ephys_base}")

    results: List[tuple] = []
    for session_dir in session_matches:
        session_dir_id = session_dir.name.replace('.rec', '')
        animal_matches = _find_matching_directories(session_dir, animal_id)
        if not animal_matches:
            if len(session_matches) == 1:
                raise FileNotFoundError(f"No animal directory found matching '{animal_id}' in {session_dir}")
            continue
        if len(animal_matches) > 1:
            raise ValueError(f"Multiple animal directories found matching '{animal_id}': {animal_matches}")
        animal_dir = animal_matches[0]

        recordings = _discover_recordings(animal_dir, session_dir_id)
        if not recordings:
            # No kilosort output at all. Keep the pre-existing behaviour of
            # handing back a path that does not exist, so callers report "no
            # ephys" rather than crashing.
            results.append((animal_dir, f"{session_dir_id}{_LEGACY_STEM_SUFFIX}"))
            continue
        try:
            selected = _select_recording(recordings, session_id, session_dir)
        except (ValueError, FileNotFoundError):
            # With several candidate .rec directories the recording may live
            # in one of the others; with only one there is nowhere left to
            # look and the caller needs the reason.
            if len(session_matches) == 1:
                raise
            continue
        results.append((animal_dir, selected.stem))

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
    config = _config or _load_config(config_path)
    if 'ephys' not in config:
        raise KeyError("'ephys' key not found in configuration file")
    ephys_base = Path(config['ephys'])

    session_matches = sorted(_find_matching_directories(ephys_base, session_id, ".rec"))
    if not session_matches:
        try:
            session_date = _parse_session_date(session_id)
        except ValueError:
            return []
        session_matches = sorted(
            _find_matching_directories(ephys_base, session_date, ".rec"))

    found: List[Recording] = []
    for session_dir in session_matches:
        session_dir_id = session_dir.name.replace('.rec', '')
        animal_matches = _find_matching_directories(session_dir, animal_id)
        if len(animal_matches) != 1:
            continue
        found.extend(_discover_recordings(animal_matches[0], session_dir_id))
    return found


def get_kilosort_path(animal_id: str, session_id: str, config_path: Optional[str] = None,
                      _config: Optional[dict] = None) -> List[Path]:
    """
    Construct the path to a Kilosort folder based on animal_id and session_id.

    The function reads the ephys base path from the configuration file and constructs
    the full path following the expected directory structure. Supports partial matching
    for both animal_id and session_id (e.g., "613" will match "rat613").

    The directory suffix is discovered from disk rather than assumed: cohort 7
    writes ``_merged``, cohort 5 ``_merge``, and a ``.rec`` directory holds one
    recording per acquisition block. ``session_id`` may therefore name a
    non-primary block (``"20251216_144334"``); a bare date still resolves to
    the day's primary recording, unchanged. See :func:`_discover_recordings`.

    Args:
        animal_id: Full or partial identifier for the animal (e.g., "613" or "rat613")
        session_id: Full or partial identifier for the recording session (e.g.,
            "20251210", "20251210_110059", or a non-primary block id such as
            "20251216_144334")
        config_path: Optional path to config file. If None, uses default location.
        _config: Optional pre-loaded config dict to avoid redundant file I/O.

    Returns:
        List[Path]: List of matching kilosort4 directory paths.

    Raises:
        FileNotFoundError: If the config file is not found or no matching directories found
        KeyError: If 'ephys' key is not found in the config file
        ValueError: If multiple animal directories are found within a single session

    Example:
        >>> paths = get_kilosort_path("613", "20251210")
        >>> print(paths[0])
    """
    config = _config or _load_config(config_path)
    return [animal_dir / f"{stem}.kilosort" / "kilosort4"
            for animal_dir, stem in _resolve_session_animal(animal_id, session_id, config)]

def get_dio_path(animal_id: str, session_id: str, dio_channel: int = 1,
                 config_path: Optional[str] = None, _config: Optional[dict] = None) -> List[Path]:
    """
    Construct the path to a DIO file based on animal_id, session_id, and DIO channel.

    Args:
        animal_id: Full or partial identifier for the animal (e.g., "613" or "rat613")
        session_id: Full or partial identifier for the recording session (e.g., "20251210" or "20251210_110059")
        dio_channel: DIO channel number (e.g., 1 for Controller_Din1, 2 for Controller_Din2, etc.)
        config_path: Optional path to config file. If None, uses default location.
        _config: Optional pre-loaded config dict to avoid redundant file I/O.

    Returns:
        List[Path]: List of matching DIO file paths.

    Raises:
        FileNotFoundError: If the config file is not found or no matching directories found
        KeyError: If 'ephys' key is not found in the config file
        ValueError: If multiple animal directories are found within a single session

    Example:
        >>> paths = get_dio_path("613", "20251210", 1)
        >>> print(paths[0])
    """
    config = _config or _load_config(config_path)
    dio_channel_str = f"Controller_Din{dio_channel}"
    return [animal_dir / f"{stem}.DIO" / f"{stem}.dio_{dio_channel_str}.dat"
            for animal_dir, stem in _resolve_session_animal(animal_id, session_id, config)]

def get_pulse_log_path(config_path: Optional[str] = None, _config: Optional[dict] = None) -> Path:
    """
    Get the path to the pulse log file.

    Args:
        config_path: Optional path to config file. If None, uses default location.
        _config: Optional pre-loaded config dict to avoid redundant file I/O.

    Returns:
        Path: Complete path to the pulse_log.txt file
    """
    config = _config or _load_config(config_path)
    
    # Get the ephys base path
    if 'ephys' not in config:
        raise KeyError("'ephys' key not found in configuration file")
    
    ephys_base = Path(config['ephys'])
    
    # Construct the pulse log path
    pulse_log_path = ephys_base / "pulse_log.txt"
    
    return pulse_log_path


def get_video_files_by_date(session_id: str, config_path: Optional[str] = None,
                           video_extensions: List[str] = None, subfolder: Optional[str] = None,
                           _config: Optional[dict] = None) -> List[Path]:
    """
    Find video files in the video directory that match the date from the session_id.

    Args:
        session_id: Session identifier containing date (e.g., "20251210" or "20251210_110059")
        config_path: Optional path to config file. If None, uses default location.
        video_extensions: List of video file extensions to search for. If None, uses common video formats.
        subfolder: Optional subfolder name within video directory to search in.
        _config: Optional pre-loaded config dict to avoid redundant file I/O.

    Returns:
        List[Path]: List of video file paths that match the session date
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']

    config = _config or _load_config(config_path)

    if 'video' not in config:
        raise KeyError("'video' key not found in configuration file")

    video_base = Path(config['video'])
    if not video_base.exists():
        raise FileNotFoundError(f"Video directory not found: {video_base}")

    if subfolder is not None:
        search_directory = video_base / subfolder
        if not search_directory.exists():
            raise FileNotFoundError(f"Subfolder not found: {search_directory}")
    else:
        search_directory = video_base

    session_date = _parse_session_date(session_id)

    matching_videos = []
    try:
        for ext in video_extensions:
            matching_videos.extend(search_directory.glob(f"*{session_date}*{ext}"))
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"Error accessing directory {search_directory}: {e}")

    matching_videos.sort(key=lambda x: x.name)
    return matching_videos


def get_tracking_files_by_date(session_id: str, config_path: Optional[str] = None,
                              tracking_extensions: List[str] = None, subfolder: Optional[str] = None,
                              _config: Optional[dict] = None) -> List[Path]:
    """
    Find tracking files in the tracking directory that match the date from the session_id.

    Args:
        session_id: Session identifier containing date (e.g., "20251210" or "20251210_110059")
        config_path: Optional path to config file. If None, uses default location.
        tracking_extensions: List of tracking file extensions to search for. If None, uses common tracking formats.
        subfolder: Optional subfolder name within tracking directory to search in.
        _config: Optional pre-loaded config dict to avoid redundant file I/O.

    Returns:
        List[Path]: List of tracking file paths that match the session date
    """
    if tracking_extensions is None:
        tracking_extensions = ['.csv', '.h5', '.hdf5', '.mat', '.pkl', '.npz', '.json', '.txt']

    config = _config or _load_config(config_path)

    if 'tracking' not in config:
        raise KeyError("'tracking' key not found in configuration file")

    tracking_base = Path(config['tracking'])
    if not tracking_base.exists():
        raise FileNotFoundError(f"Tracking directory not found: {tracking_base}")

    session_date = _parse_session_date(session_id)

    if subfolder is not None:
        search_directory = tracking_base / subfolder
        if not search_directory.exists():
            raise FileNotFoundError(f"Subfolder not found: {search_directory}")
    else:
        # Try to find a subfolder containing the session_id or session_date (one level deep)
        search_directory = tracking_base
        try:
            for potential_subfolder in tracking_base.iterdir():
                if potential_subfolder.is_dir() and (session_id in potential_subfolder.name or session_date in potential_subfolder.name):
                    search_directory = potential_subfolder
                    break
        except (PermissionError, OSError):
            pass

    matching_tracking_files = []
    ext_set = {ext.lower() for ext in tracking_extensions}
    try:
        for tracking_file in search_directory.glob(f'*{session_date}*'):
            if tracking_file.is_file() and tracking_file.suffix.lower() in ext_set:
                matching_tracking_files.append(tracking_file)
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"Error accessing directory {search_directory}: {e}")

    matching_tracking_files.sort(key=lambda x: x.name)
    return matching_tracking_files


def get_event_files_by_date(session_id: str, config_path: Optional[str] = None,
                            subfolder: Optional[str] = None,
                            _config: Optional[dict] = None) -> List[Path]:
    """
    Find behavioral event CSV files in the events directory that match the date from the session_id.

    Args:
        session_id: Session identifier containing date (e.g., "20251210" or "20251210_110059")
        config_path: Optional path to config file. If None, uses default location.
        subfolder: Optional subfolder name within events directory to search in. If None, searches all subfolders.
        _config: Optional pre-loaded config dict to avoid redundant file I/O.

    Returns:
        List[Path]: List of behavioral event CSV file paths that match the session date
    """
    config = _config or _load_config(config_path)

    if 'events' not in config:
        raise KeyError("'events' key not found in configuration file")

    events_base = Path(config['events'])
    if not events_base.exists():
        raise FileNotFoundError(f"Events directory not found: {events_base}")

    session_date = _parse_session_date(session_id)

    matching_event_files = []
    try:
        if subfolder is not None:
            search_directory = events_base / subfolder
            if not search_directory.exists():
                raise FileNotFoundError(f"Subfolder not found: {search_directory}")
            search_directories = [search_directory]
        else:
            search_directories = []
            try:
                for potential_subfolder in events_base.iterdir():
                    if potential_subfolder.is_dir():
                        if (session_id in potential_subfolder.name or
                            session_date in potential_subfolder.name):
                            search_directories.append(potential_subfolder)
                            logger.debug("Found matching events subfolder: %s", potential_subfolder.name)
                if not search_directories:
                    search_directories = [d for d in events_base.iterdir() if d.is_dir()]
            except (PermissionError, OSError):
                search_directories = [events_base]

        seen = set()
        for search_dir in search_directories:
            for pattern in [f"*{session_id}*.csv", f"*{session_date}*.csv"]:
                for event_file in search_dir.glob(pattern):
                    if event_file.is_file() and event_file not in seen:
                        seen.add(event_file)
                        matching_event_files.append(event_file)

            # Also search one level deeper
            for subdir in search_dir.iterdir():
                if subdir.is_dir():
                    for pattern in [f"*{session_id}*.csv", f"*{session_date}*.csv"]:
                        for event_file in subdir.glob(pattern):
                            if event_file.is_file() and event_file not in seen:
                                seen.add(event_file)
                                matching_event_files.append(event_file)

    except (PermissionError, OSError) as e:
        raise RuntimeError(f"Error accessing events directory {events_base}: {e}")

    matching_event_files.sort(key=lambda x: x.name)

    if not matching_event_files:
        logger.warning("No behavioral event CSV files found for session %s in %s", session_id, events_base)

    return matching_event_files


def get_animals_and_sessions(config_path: Optional[str] = None,
                             _config: Optional[dict] = None) -> pd.DataFrame:
    """
    Get a DataFrame of all animals and sessions from the ephys folder.

    Args:
        config_path: Optional path to config file. If None, uses default location.
        _config: Optional pre-loaded config dict to avoid redundant file I/O.

    Returns:
        pd.DataFrame: DataFrame with columns: session, animal, kilosort_path,
            recording_id, stem, session_dir, is_primary.

    One row per *recording*, not per ``.rec`` directory. A day holding morning,
    afternoon and evening blocks contributes three rows per animal, and
    ``session`` carries the recording id, so the value can be handed straight
    back to :class:`DataStorageManager`. For a primary recording the recording
    id equals the ``.rec`` directory name, so every id this function returned
    before it enumerated recordings is still returned, unchanged.
    """
    config = _config or _load_config(config_path)

    # Get the ephys base path
    if 'ephys' not in config:
        raise KeyError("'ephys' key not found in configuration file")

    ephys_base = Path(config['ephys'])

    if not ephys_base.exists():
        raise FileNotFoundError(f"Ephys directory not found: {ephys_base}")

    data_rows = []

    try:
        # Iterate through all session directories (*.rec)
        for session_dir in sorted(ephys_base.iterdir()):
            if session_dir.is_dir() and session_dir.name.endswith('.rec'):
                session_dir_id = session_dir.name.replace('.rec', '')

                # Look for animal directories within this session
                for animal_dir in sorted(session_dir.iterdir()):
                    if animal_dir.is_dir() and animal_dir.name.startswith('rat'):
                        animal_id = animal_dir.name

                        recordings = _discover_recordings(animal_dir, session_dir_id)
                        if not recordings:
                            # Keep the animal visible with the path that would
                            # have been built, so "no ephys here" stays
                            # reportable rather than silently absent.
                            recordings = [Recording(
                                animal_dir=animal_dir,
                                stem=f"{session_dir_id}{_LEGACY_STEM_SUFFIX}",
                                recording_id=session_dir_id,
                                session_dir_id=session_dir_id,
                                is_primary=True)]

                        for recording in recordings:
                            data_rows.append({
                                'session': recording.recording_id,
                                'animal': animal_id,
                                'kilosort_path': (animal_dir
                                                  / f"{recording.stem}.kilosort"
                                                  / "kilosort4"),
                                'recording_id': recording.recording_id,
                                'stem': recording.stem,
                                'session_dir': session_dir_id,
                                'is_primary': recording.is_primary,
                            })

    except (PermissionError, OSError) as e:
        raise RuntimeError(f"Error accessing ephys directory {ephys_base}: {e}")
    
    # Create DataFrame and sort by session and animal
    df = pd.DataFrame(data_rows)
    if not df.empty:
        df = df.sort_values(['session', 'animal']).reset_index(drop=True)
    
    return df


def _find_matching_directories(base_path: Path, partial_name: str, suffix: str = "") -> List[Path]:
    """
    Find directories that contain the partial name and optionally end with suffix.
    
    Args:
        base_path: Directory to search in
        partial_name: Partial string to match
        suffix: Optional suffix that directories should end with
        
    Returns:
        List of matching directory paths
    """
    if not base_path.exists():
        return []
    
    matching_dirs = []
    try:
        for item in base_path.iterdir():
            if item.is_dir():
                item_name = item.name
                # Check if partial name is in the directory name
                if partial_name in item_name:
                    # If suffix is specified, check that it ends with suffix
                    if suffix and item_name.endswith(suffix):
                        matching_dirs.append(item)
                    elif not suffix:
                        matching_dirs.append(item)
    except (PermissionError, OSError):
        # Handle cases where we can't read the directory
        pass
    
    return matching_dirs


def verify_kilosort_path(kilosort_path: Path, check_files: bool = True) -> bool:
    """
    Verify that a Kilosort path exists and optionally check for expected files.

    Args:
        kilosort_path: Path to the kilosort directory
        check_files: If True, verify that expected Kilosort files exist

    Returns:
        bool: True if the path (and files) exist, False otherwise
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


class DataStorageManager:
    """
    Data Storage Manager for Habitat Pipeline Sessions.

    Manages all data file paths for a specific animal/session combination.
    Loads the config once and passes it to all path-resolution functions.
    """

    def __init__(self, animal_id: str, session_id: str, config_path: Optional[str] = None,
                 auto_load: bool = True, use_cache: bool = True,
                 cache_dir: Optional[Path] = None):
        self.animal_id = animal_id
        self.session_id = session_id
        self.config_path = config_path
        self._config = _load_config(config_path)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR

        self.kilosort_path = None
        self.dio_paths: Dict[int, Path] = {}
        self.video_files: List[Path] = []
        self.tracking_files: List[Path] = []
        self.behavioral_event_files: List[Path] = []
        self.pulse_log_path = None

        # Which recording of the day this manager resolved to. ``None`` until
        # the ephys paths load (or when they fail to resolve at all).
        self.recording_id: Optional[str] = None
        self.recording_stem: Optional[str] = None
        self.is_primary_recording: Optional[bool] = None
        self.recording_ids_on_date: List[str] = []

        if auto_load:
            self.load_all_paths()

    def load_all_paths(self):
        """Load all available data paths for this animal/session.

        Reads from the JSON cache when available; otherwise scans the data
        directories and writes a fresh cache file.
        """
        if self.use_cache and self._load_cache():
            self._warn_if_date_resolved_files_may_not_belong()
            self._log_availability_summary()
            return
        logger.info("Loading data paths for %s/%s", self.animal_id, self.session_id)
        self._load_ephys_paths()
        self._load_video_paths()
        self._load_tracking_paths()
        self._load_behavioral_events()
        self._load_sync_paths()
        self._warn_if_date_resolved_files_may_not_belong()
        self._log_availability_summary()
        if self.use_cache:
            self._save_cache()

    def refresh(self):
        """Re-scan all paths from disk and update the cache."""
        logger.info("Refreshing data paths for %s/%s", self.animal_id, self.session_id)
        self._load_ephys_paths()
        self._load_video_paths()
        self._load_tracking_paths()
        self._load_behavioral_events()
        self._load_sync_paths()
        self._warn_if_date_resolved_files_may_not_belong()
        self._log_availability_summary()
        if self.use_cache:
            self._save_cache()

    def _cache_file(self) -> Path:
        return self.cache_dir / f"{self.animal_id}_{self.session_id}.json"

    def _to_cache_dict(self) -> dict:
        return {
            "version": _CACHE_VERSION,
            "animal_id": self.animal_id,
            "session_id": self.session_id,
            "kilosort_path": str(self.kilosort_path) if self.kilosort_path else None,
            "dio_paths": {str(ch): str(p) for ch, p in self.dio_paths.items()},
            "video_files": [str(p) for p in self.video_files],
            "tracking_files": [str(p) for p in self.tracking_files],
            "behavioral_event_files": [str(p) for p in self.behavioral_event_files],
            "pulse_log_path": str(self.pulse_log_path) if self.pulse_log_path else None,
            "recording_id": self.recording_id,
            "recording_stem": self.recording_stem,
            "is_primary_recording": self.is_primary_recording,
            "recording_ids_on_date": list(self.recording_ids_on_date),
        }

    def _apply_cache_dict(self, data: dict) -> None:
        self.kilosort_path = Path(data["kilosort_path"]) if data.get("kilosort_path") else None
        self.dio_paths = {int(ch): Path(p) for ch, p in data.get("dio_paths", {}).items()}
        self.video_files = [Path(p) for p in data.get("video_files", [])]
        self.tracking_files = [Path(p) for p in data.get("tracking_files", [])]
        self.behavioral_event_files = [Path(p) for p in data.get("behavioral_event_files", [])]
        self.pulse_log_path = Path(data["pulse_log_path"]) if data.get("pulse_log_path") else None
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
        except (OSError, PermissionError) as e:
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

    def _load_ephys_paths(self):
        try:
            animal_dir, stem = _resolve_session_animal(
                self.animal_id, self.session_id, self._config)[0]
            self.kilosort_path = animal_dir / f"{stem}.kilosort" / "kilosort4"
            dio_dir = animal_dir / f"{stem}.DIO"
            self.dio_paths = {
                ch: dio_dir / f"{stem}.dio_Controller_Din{ch}.dat"
                for ch in range(1, 5)
            }
            self.recording_stem = stem
            match = _RECORDING_STAMP_RE.search(stem)
            self.recording_id = match.group(1) if match else None
            try:
                siblings = _discover_recordings(
                    animal_dir, animal_dir.parent.name.replace('.rec', ''))
            except Exception as e:                   # pragma: no cover - listing
                logger.debug("Could not enumerate sibling recordings: %s", e)
                siblings = []
            self.recording_ids_on_date = sorted({r.recording_id for r in siblings})
            self.is_primary_recording = next(
                (r.is_primary for r in siblings if r.stem == stem), None)
        except Exception as e:
            logger.warning("Error loading ephys paths: %s", e)
            self.kilosort_path = None
            self.dio_paths = {}
            self.recording_id = None
            self.recording_stem = None
            self.is_primary_recording = None
            self.recording_ids_on_date = []

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

    def _load_video_paths(self):
        try:
            self.video_files = get_video_files_by_date(
                self.session_id, subfolder="social_videos", _config=self._config)
        except Exception as e:
            logger.warning("Error loading video paths: %s", e)
            self.video_files = []

    def _load_tracking_paths(self):
        try:
            self.tracking_files = get_tracking_files_by_date(self.session_id, _config=self._config)
        except Exception as e:
            logger.warning("Error loading tracking paths: %s", e)
            self.tracking_files = []

    def _load_behavioral_events(self):
        try:
            self.behavioral_event_files = get_event_files_by_date(self.session_id, _config=self._config)
        except Exception as e:
            logger.warning("Error loading behavioral event paths: %s", e)
            self.behavioral_event_files = []

    def _load_sync_paths(self):
        try:
            self.pulse_log_path = get_pulse_log_path(_config=self._config)
        except Exception as e:
            logger.warning("Error loading sync paths: %s", e)
            self.pulse_log_path = None

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

    def get_kilosort_path(self) -> Optional[Path]:
        """Get Kilosort data path."""
        return self.kilosort_path

    def get_dio_path(self, channel: int = 1) -> Optional[Path]:
        """Get DIO path for specific channel."""
        return self.dio_paths.get(channel)

    def get_video_files(self) -> List[Path]:
        """Get video file paths."""
        return self.video_files.copy()

    def get_tracking_files(self) -> List[Path]:
        """Get tracking file paths."""
        return self.tracking_files.copy()

    def get_behavioral_event_files(self) -> List[Path]:
        """Get behavioral event file paths."""
        return self.behavioral_event_files.copy()

    def get_pulse_log_path(self) -> Optional[Path]:
        """Get pulse log path."""
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
        if val is None:
            return None
        return float(val)

    def __repr__(self) -> str:
        return (f"DataStorageManager({self.animal_id}/{self.session_id}, "
                f"ephys:{'Y' if self.kilosort_path else 'N'}, "
                f"video:{len(self.video_files)}, tracking:{len(self.tracking_files)}, "
                f"events:{len(self.behavioral_event_files)})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    animal_id = "613"
    session_id = "20241210"

    try:
        dm = DataStorageManager(animal_id, session_id)
        print(dm)
        print(f"  Kilosort: {dm.get_kilosort_path()}")
        print(f"  Tracking: {len(dm.get_tracking_files())} files")
        print(f"  Events:   {len(dm.get_behavioral_event_files())} files")
    except Exception as e:
        print(f"Error: {e} (expected if test data doesn't exist)")