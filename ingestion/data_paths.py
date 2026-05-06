"""
Kilosort Path Management Module

This module provides utilities for constructing file paths to Kilosort data
based on animal_id and session_id, reading configuration from the project's
default_paths.json file.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd

logger = logging.getLogger(__name__)


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


def _resolve_session_animal(animal_id: str, session_id: str, config: dict) -> List[tuple]:
    """Resolve [(animal_dir, full_session_id), ...] for matching sessions.

    Centralises the two iterdir() walks shared by get_kilosort_path / get_dio_path
    so callers needing several files under the same animal directory only pay the
    network listing cost once.
    """
    if 'ephys' not in config:
        raise KeyError("'ephys' key not found in configuration file")
    ephys_base = Path(config['ephys'])

    session_matches = _find_matching_directories(ephys_base, session_id, ".rec")
    if not session_matches:
        raise FileNotFoundError(f"No session directory found matching '{session_id}' in {ephys_base}")

    results: List[tuple] = []
    for session_dir in session_matches:
        full_session_id = session_dir.name.replace('.rec', '')
        animal_matches = _find_matching_directories(session_dir, animal_id)
        if not animal_matches:
            if len(session_matches) == 1:
                raise FileNotFoundError(f"No animal directory found matching '{animal_id}' in {session_dir}")
            continue
        if len(animal_matches) > 1:
            raise ValueError(f"Multiple animal directories found matching '{animal_id}': {animal_matches}")
        results.append((animal_matches[0], full_session_id))

    if not results:
        raise FileNotFoundError(f"No animal directory found matching '{animal_id}' in any of the matched sessions")
    return results


def get_kilosort_path(animal_id: str, session_id: str, config_path: Optional[str] = None,
                      _config: Optional[dict] = None) -> List[Path]:
    """
    Construct the path to a Kilosort folder based on animal_id and session_id.

    The function reads the ephys base path from the configuration file and constructs
    the full path following the expected directory structure. Supports partial matching
    for both animal_id and session_id (e.g., "613" will match "rat613").

    Args:
        animal_id: Full or partial identifier for the animal (e.g., "613" or "rat613")
        session_id: Full or partial identifier for the recording session (e.g., "20251210" or "20251210_110059")
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
    return [animal_dir / f"{full_session_id}_merged.kilosort" / "kilosort4"
            for animal_dir, full_session_id in _resolve_session_animal(animal_id, session_id, config)]

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
    return [animal_dir / f"{full_session_id}_merged.DIO"
            / f"{full_session_id}_merged.dio_{dio_channel_str}.dat"
            for animal_dir, full_session_id in _resolve_session_animal(animal_id, session_id, config)]

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
        pd.DataFrame: DataFrame with columns: session, animal, kilosort_path
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
        for session_dir in ephys_base.iterdir():
            if session_dir.is_dir() and session_dir.name.endswith('.rec'):
                session_id = session_dir.name.replace('.rec', '')
                
                # Look for animal directories within this session
                for animal_dir in session_dir.iterdir():
                    if animal_dir.is_dir() and animal_dir.name.startswith('rat'):
                        animal_id = animal_dir.name
                        
                        # Construct kilosort path
                        kilosort_path = animal_dir / f"{session_id}_merged.kilosort" / "kilosort4"
                        
                        # Add row to data
                        data_rows.append({
                            'session': session_id,
                            'animal': animal_id,
                            'kilosort_path': kilosort_path
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
                 auto_load: bool = True):
        self.animal_id = animal_id
        self.session_id = session_id
        self.config_path = config_path
        self._config = _load_config(config_path)

        self.kilosort_path = None
        self.dio_paths: Dict[int, Path] = {}
        self.video_files: List[Path] = []
        self.tracking_files: List[Path] = []
        self.behavioral_event_files: List[Path] = []
        self.pulse_log_path = None

        if auto_load:
            self.load_all_paths()

    def load_all_paths(self):
        """Load all available data paths for this animal/session."""
        logger.info("Loading data paths for %s/%s", self.animal_id, self.session_id)
        self._load_ephys_paths()
        self._load_video_paths()
        self._load_tracking_paths()
        self._load_behavioral_events()
        self._load_sync_paths()
        self._log_availability_summary()

    def _load_ephys_paths(self):
        try:
            animal_dir, full_session_id = _resolve_session_animal(
                self.animal_id, self.session_id, self._config)[0]
            self.kilosort_path = animal_dir / f"{full_session_id}_merged.kilosort" / "kilosort4"
            dio_dir = animal_dir / f"{full_session_id}_merged.DIO"
            self.dio_paths = {
                ch: dio_dir / f"{full_session_id}_merged.dio_Controller_Din{ch}.dat"
                for ch in range(1, 5)
            }
        except Exception as e:
            logger.warning("Error loading ephys paths: %s", e)
            self.kilosort_path = None
            self.dio_paths = {}

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
            "Data availability — ephys: %s, DIO channels: %s, video: %d, "
            "tracking: %d, events: %d, pulse_log: %s",
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