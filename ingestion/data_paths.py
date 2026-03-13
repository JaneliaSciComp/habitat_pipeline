"""
Kilosort Path Management Module

This module provides utilities for constructing file paths to Kilosort data
based on animal_id and session_id, reading configuration from the project's
default_paths.json file.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime
import glob
import pandas as pd


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
        config_path = Path(config_path)
        
    # Read the configuration file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    return config


def get_kilosort_path(animal_id: str, session_id: str, config_path: Optional[str] = None) -> Path:
    """
    Construct the path to a Kilosort folder based on animal_id and session_id.
    
    The function reads the ephys base path from the configuration file and constructs
    the full path following the expected directory structure. Supports partial matching
    for both animal_id and session_id (e.g., "613" will match "rat613").
    
    Args:
        animal_id: Full or partial identifier for the animal (e.g., "613" or "rat613")
        session_id: Full or partial identifier for the recording session (e.g., "20251210" or "20251210_110059")
        config_path: Optional path to config file. If None, uses default location.
        
    Returns:
        Path: Complete path to the kilosort4 directory
        
    Raises:
        FileNotFoundError: If the config file is not found or no matching directories found
        KeyError: If 'ephys' key is not found in the config file
        ValueError: If multiple matches are found for the same partial string
        
    Example:
        >>> path = get_kilosort_path("613", "20251210")
        >>> print(path)
    """
    # Load configuration
    config = _load_config(config_path)
    
    # Get the ephys base path
    if 'ephys' not in config:
        raise KeyError("'ephys' key not found in configuration file")
    
    ephys_base = Path(config['ephys'])
    
    # Find matching session directory (session_id.rec format)
    session_matches = _find_matching_directories(ephys_base, session_id, ".rec")
    if not session_matches:
        raise FileNotFoundError(f"No session directory found matching '{session_id}' in {ephys_base}")
    if len(session_matches) > 1:
        raise ValueError(f"Multiple session directories found matching '{session_id}': {session_matches}")
    
    session_dir = session_matches[0]
    full_session_id = session_dir.name.replace('.rec', '')
    
    # Find matching animal directory within the session directory
    animal_matches = _find_matching_directories(session_dir, animal_id)
    if not animal_matches:
        raise FileNotFoundError(f"No animal directory found matching '{animal_id}' in {session_dir}")
    if len(animal_matches) > 1:
        raise ValueError(f"Multiple animal directories found matching '{animal_id}': {animal_matches}")
    
    animal_dir = animal_matches[0]
    full_animal_id = animal_dir.name
    
    # Construct the kilosort path
    kilosort_path = animal_dir / f"{full_session_id}_merged.kilosort" / "kilosort4"
    
    return kilosort_path

def get_dio_path(animal_id: str, session_id: str, dio_channel: int = 1, config_path: Optional[str] = None) -> Path:
    """
    Construct the path to a DIO file based on animal_id, session_id, and DIO channel.
    
    The function reads the ephys base path from the configuration file and constructs
    the full path following the expected directory structure. Supports partial matching
    for both animal_id and session_id (e.g., "613" will match "rat613").
    
    Args:
        animal_id: Full or partial identifier for the animal (e.g., "613" or "rat613")
        session_id: Full or partial identifier for the recording session (e.g., "20251210" or "20251210_110059")
        dio_channel: DIO channel number (e.g., 1 for Controller_Din1, 2 for Controller_Din2, etc.)
        config_path: Optional path to config file. If None, uses default location.
        
    Returns:
        Path: Complete path to the DIO file
        
    Raises:
        FileNotFoundError: If the config file is not found or no matching directories found
        KeyError: If 'ephys' key is not found in the config file
        ValueError: If multiple matches are found for the same partial string
        
    Example:
        >>> path = get_dio_path("613", "20251210", 1)
        >>> print(path)
    """
    # Load configuration
    config = _load_config(config_path)
    
    # Get the ephys base path
    if 'ephys' not in config:
        raise KeyError("'ephys' key not found in configuration file")
    
    ephys_base = Path(config['ephys'])
    
    # Find matching session directory (session_id.rec format)
    session_matches = _find_matching_directories(ephys_base, session_id, ".rec")
    if not session_matches:
        raise FileNotFoundError(f"No session directory found matching '{session_id}' in {ephys_base}")
    if len(session_matches) > 1:
        raise ValueError(f"Multiple session directories found matching '{session_id}': {session_matches}")
    
    session_dir = session_matches[0]
    full_session_id = session_dir.name.replace('.rec', '')
    
    # Find matching animal directory within the session directory
    animal_matches = _find_matching_directories(session_dir, animal_id)
    if not animal_matches:
        raise FileNotFoundError(f"No animal directory found matching '{animal_id}' in {session_dir}")
    if len(animal_matches) > 1:
        raise ValueError(f"Multiple animal directories found matching '{animal_id}': {animal_matches}")
    
    animal_dir = animal_matches[0]
    
    # Construct the DIO path with channel number
    dio_channel_str = f"Controller_Din{dio_channel}"
    dio_path = animal_dir / f"{full_session_id}_merged.DIO" / f"{full_session_id}_merged.dio_{dio_channel_str}.dat"
    
    return dio_path

def get_pulse_log_path(config_path: Optional[str] = None) -> Path:
    """
    Get the path to the pulse log file.
    
    The function reads the video base path from the configuration file and constructs
    the path to the pulse_log.txt file.
    
    Args:
        config_path: Optional path to config file. If None, uses default location.
        
    Returns:
        Path: Complete path to the pulse_log.txt file
    """
    # Load configuration
    config = _load_config(config_path)
    
    # Get the video base path
    if 'video' not in config:
        raise KeyError("'video' key not found in configuration file")
    
    video_base = Path(config['video'])
    
    # Construct the pulse log path
    pulse_log_path = video_base / "pulse_log.txt"
    
    return pulse_log_path


def get_video_files_by_date(session_id: str, config_path: Optional[str] = None, 
                           video_extensions: List[str] = None, subfolder: Optional[str] = None) -> List[Path]:
    """
    Find video files in the video directory that match the date from the session_id.
    
    The function extracts the date portion from session_id (e.g., "20251210" from "20251210_110059")
    and searches for video files containing this date in their filename.
    
    Args:
        session_id: Session identifier containing date (e.g., "20251210" or "20251210_110059")
        config_path: Optional path to config file. If None, uses default location.
        video_extensions: List of video file extensions to search for. If None, uses common video formats.
        subfolder: Optional subfolder name within video directory to search in. If None, searches entire video directory.
        
    Returns:
        List[Path]: List of video file paths that match the session date
        
    Raises:
        FileNotFoundError: If the config file, video directory, or specified subfolder is not found
        KeyError: If 'video' key is not found in the config file
        
    Example:
        >>> video_files = get_video_files_by_date("20251210_110059")
        >>> print(f"Found {len(video_files)} video files for session date")
        >>> 
        >>> # Search in specific subfolder
        >>> video_files = get_video_files_by_date("20251210_110059", subfolder="raw_videos")
        >>> for vf in video_files:
        >>>     print(vf.name)
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    # Load configuration
    config = _load_config(config_path)
    
    # Get the video base path
    if 'video' not in config:
        raise KeyError("'video' key not found in configuration file")
    
    video_base = Path(config['video'])
    
    if not video_base.exists():
        raise FileNotFoundError(f"Video directory not found: {video_base}")
    
    # Set search directory based on subfolder parameter
    if subfolder is not None:
        search_directory = video_base / subfolder
        if not search_directory.exists():
            raise FileNotFoundError(f"Subfolder not found: {search_directory}")
    else:
        search_directory = video_base
    
    # Extract date from session_id (first 8 characters, assuming YYYYMMDD format)
    # Handle both "20251210" and "20251210_110059" formats
    if len(session_id) >= 8:
        session_date = session_id[:8]
        
        # Validate that it looks like a date (8 digits)
        if not session_date.isdigit():
            raise ValueError(f"Could not extract valid date from session_id: {session_id}")
    else:
        raise ValueError(f"Session ID too short to extract date: {session_id}")
    
    matching_videos = []
    
    try:
    # Search through all files in the search directory and subdirectories
    #     for video_file in search_directory.rglob('*'):
    #         if video_file.is_file():
    #             # Check if file has a video extension
    #             if video_file.suffix.lower() in [ext.lower() for ext in video_extensions]:
    #                 # Check if the session date appears in the filename
    #                 if session_date in video_file.name:
    #                     matching_videos.append(video_file)
        search_pattern = "*"+session_date+"*"+video_extensions[0]
        # print(search_pattern)
        matching_videos = list(search_directory.glob(search_pattern))

    
    
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"Error accessing directory {search_directory}: {e}")
    
    # Sort videos by filename for consistent ordering
    matching_videos.sort(key=lambda x: x.name)
    
    return matching_videos


def get_tracking_files_by_date(session_id: str, config_path: Optional[str] = None, 
                              tracking_extensions: List[str] = None, subfolder: Optional[str] = None) -> List[Path]:
    """
    Find tracking files in the tracking directory that match the date from the session_id.
    
    The function extracts the date portion from session_id (e.g., "20251210" from "20251210_110059")
    and searches for tracking files containing this date in their filename.
    
    Args:
        session_id: Session identifier containing date (e.g., "20251210" or "20251210_110059")
        config_path: Optional path to config file. If None, uses default location.
        tracking_extensions: List of tracking file extensions to search for. If None, uses common tracking formats.
        subfolder: Optional subfolder name within tracking directory to search in. If None, searches entire tracking directory.
        
    Returns:
        List[Path]: List of tracking file paths that match the session date
    """
    if tracking_extensions is None:
        tracking_extensions = ['.csv', '.h5', '.hdf5', '.mat', '.pkl', '.npz', '.json', '.txt']
    
    # Load configuration
    config = _load_config(config_path)
    
    # Get the tracking base path
    if 'tracking' not in config:
        raise KeyError("'tracking' key not found in configuration file")
    
    tracking_base = Path(config['tracking'])
    
    if not tracking_base.exists():
        raise FileNotFoundError(f"Tracking directory not found: {tracking_base}")
    
    # Extract date from session_id (first 8 characters, assuming YYYYMMDD format)
    # Handle both "20251210" and "20251210_110059" formats
    if len(session_id) >= 8:
        session_date = session_id[:8]
        
        # Validate that it looks like a date (8 digits)
        if not session_date.isdigit():
            raise ValueError(f"Could not extract valid date from session_id: {session_id}")
    else:
        raise ValueError(f"Session ID too short to extract date: {session_id}")
    
    # Set search directory based on subfolder parameter
    if subfolder is not None:
        search_directory = tracking_base / subfolder
        if not search_directory.exists():
            raise FileNotFoundError(f"Subfolder not found: {search_directory}")
    else:
        # Try to find a subfolder containing the session_id or session_date (one level deep only)
        search_directory = tracking_base
        try:
            # Only search immediate subdirectories (one level deep)
            for potential_subfolder in tracking_base.iterdir():
                if potential_subfolder.is_dir() and (session_id in potential_subfolder.name or session_date in potential_subfolder.name):
                    search_directory = potential_subfolder
                    print(f"Found matching subfolder: {potential_subfolder.name}")
                    break
        except (PermissionError, OSError):
            # If we can't read the directory, fall back to searching the base directory
            pass
    
    matching_tracking_files = []
    
    print(f"Searching for tracking files in {search_directory} with date '{session_date}' and extensions {tracking_extensions}")
    try:
        # Search through files containing the session date in current directory and 1 level deep
        # Search in current directory
        for tracking_file in search_directory.glob(f'*{session_date}*'):
            if tracking_file.is_file():
                # Check if file has a tracking extension
                if tracking_file.suffix.lower() in [ext.lower() for ext in tracking_extensions]:
                    matching_tracking_files.append(tracking_file)
        
        # Search in 1 level deep subdirectories
        # for tracking_file in search_directory.glob(f'*/*{session_date}*'):
        #     if tracking_file.is_file():
        #         # Check if file has a tracking extension
        #         if tracking_file.suffix.lower() in [ext.lower() for ext in tracking_extensions]:
        #             matching_tracking_files.append(tracking_file)
    
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"Error accessing directory {search_directory}: {e}")
    
    # Sort tracking files by filename for consistent ordering
    matching_tracking_files.sort(key=lambda x: x.name)
    
    return matching_tracking_files


def get_event_files_by_date(session_id: str, config_path: Optional[str] = None, 
                                subfolder: Optional[str] = None) -> List[Path]:
    """
    Find behavioral event CSV files in the events directory that match the date from the session_id.
    
    The function extracts the date portion from session_id (e.g., "20251210" from "20251210_110059")
    and searches for CSV files containing this date in their filename within subfolders.
    
    Args:
        session_id: Session identifier containing date (e.g., "20251210" or "20251210_110059")
        config_path: Optional path to config file. If None, uses default location.
        subfolder: Optional subfolder name within events directory to search in. If None, searches all subfolders.
        
    Returns:
        List[Path]: List of behavioral event CSV file paths that match the session date
    """
    # Load configuration
    config = _load_config(config_path)
    
    # Get the events base path
    if 'events' not in config:
        raise KeyError("'events' key not found in configuration file")
    
    events_base = Path(config['events'])
    
    if not events_base.exists():
        raise FileNotFoundError(f"Events directory not found: {events_base}")
    
    # Extract date from session_id (first 8 characters, assuming YYYYMMDD format)
    # Handle both "20251210" and "20251210_110059" formats
    if len(session_id) >= 8:
        session_date = session_id[:8]
        
        # Validate that it looks like a date (8 digits)
        if not session_date.isdigit():
            raise ValueError(f"Could not extract valid date from session_id: {session_id}")
    else:
        raise ValueError(f"Session ID too short to extract date: {session_id}")
    
    matching_event_files = []
    
    try:
        # Set search directory based on subfolder parameter
        if subfolder is not None:
            search_directory = events_base / subfolder
            if not search_directory.exists():
                raise FileNotFoundError(f"Subfolder not found: {search_directory}")
            search_directories = [search_directory]
        else:
            # Try to find subfolders containing the session_id or session_date
            search_directories = []
            try:
                # Search immediate subdirectories (one level deep)
                for potential_subfolder in events_base.iterdir():
                    if potential_subfolder.is_dir():
                        if (session_id in potential_subfolder.name or 
                            session_date in potential_subfolder.name):
                            search_directories.append(potential_subfolder)
                            print(f"Found matching events subfolder: {potential_subfolder.name}")
                
                # If no matching subfolders found, search all subdirectories
                if not search_directories:
                    search_directories = [d for d in events_base.iterdir() if d.is_dir()]
                    
            except (PermissionError, OSError):
                # If we can't read the directory, fall back to searching the base directory
                search_directories = [events_base]
        
        # Search for CSV files in the identified directories
        for search_dir in search_directories:
            # Search for CSV files containing the session_id or session_date
            for pattern in [f"*{session_id}*.csv", f"*{session_date}*.csv"]:
                for event_file in search_dir.glob(pattern):
                    if event_file.is_file() and event_file not in matching_event_files:
                        matching_event_files.append(event_file)
            
            # Also search one level deeper in case there are nested directories
            for subdir in search_dir.iterdir():
                if subdir.is_dir():
                    for pattern in [f"*{session_id}*.csv", f"*{session_date}*.csv"]:
                        for event_file in subdir.glob(pattern):
                            if event_file.is_file() and event_file not in matching_event_files:
                                matching_event_files.append(event_file)
    
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"Error accessing events directory {events_base}: {e}")
    
    # Sort event files by filename for consistent ordering
    matching_event_files.sort(key=lambda x: x.name)
    
    if not matching_event_files:
        print(f"Warning: No behavioral event CSV files found for session {session_id} in {events_base}")
    else:
        print(f"Found {len(matching_event_files)} behavioral event file(s) for session {session_id}")
    
    return matching_event_files


def get_animals_and_sessions(config_path: Optional[str] = None) -> pd.DataFrame:
    """
    Get a DataFrame of all animals and sessions from the ephys folder.
    
    This function scans the ephys directory specified in the config file and returns
    a DataFrame containing all available animal-session combinations with their kilosort paths.
    
    Args:
        config_path: Optional path to config file. If None, uses default location.
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - session: Session ID
            - animal: Animal ID  
            - kilosort_path: Path to the kilosort4 directory
        
    Raises:
        FileNotFoundError: If the config file or ephys directory is not found
        KeyError: If 'ephys' key is not found in the config file
        
    Example:
        >>> df = get_animals_and_sessions()
        >>> print(f"Found {len(df)} animal-session combinations")
        >>> print(df.head())
    """
    # Load configuration
    config = _load_config(config_path)
    
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
        print(f"ERROR: Kilosort directory does not exist: {kilosort_path}")
        return False
    
    if check_files:
        # Check for essential Kilosort 4 output files
        required_files = [
            "spike_times.npy",
            "spike_clusters.npy", 
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (kilosort_path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"ERROR: Missing required Kilosort files in {kilosort_path}: {', '.join(missing_files)}")
            return False
    
    print(f"SUCCESS: Kilosort path verified successfully: {kilosort_path}")
    return True


def get_kilosort_path_with_validation(animal_id: str, session_id: str, 
                                    config_path: Optional[str] = None,
                                    validate: bool = True) -> Path:
    """
    Get Kilosort path with optional validation.
    
    This is a convenience function that combines path construction and validation.
    
    Args:
        animal_id: Identifier for the animal
        session_id: Identifier for the recording session  
        config_path: Optional path to config file
        validate: If True, verify the path exists before returning
        
    Returns:
        Path: Complete path to the kilosort4 directory
        
    Raises:
        FileNotFoundError: If validation is enabled and the path doesn't exist
    """
    kilosort_path = get_kilosort_path(animal_id, session_id, config_path)
    
    if validate and not verify_kilosort_path(kilosort_path, check_files=False):
        raise FileNotFoundError(f"Kilosort path does not exist: {kilosort_path}")
    
    return kilosort_path


class DataStorageManager:
    """
    Data Storage Manager for Habitat Pipeline Sessions
    
    This class manages all data file paths for a specific animal/session combination
    using the existing data_paths functions. It provides a centralized interface for
    accessing ephys, video, tracking, and synchronization data paths.
    
    Attributes:
        animal_id: Animal identifier
        session_id: Session identifier  
        config_path: Path to configuration file
        kilosort_path: Path to Kilosort ephys data
        dio_paths: Dictionary of DIO channel paths
        video_files: List of video file paths
        tracking_files: List of tracking file paths
        behavioral_event_files: List of behavioral event CSV file paths
        pulse_log_path: Path to pulse log file
        metadata: Dictionary containing session metadata
    """
    
    def __init__(self, animal_id: str, session_id: str, config_path: Optional[str] = None, 
                 auto_load: bool = True):
        """
        Initialize DataPathManager for a specific animal/session.
        
        Args:
            animal_id: Animal identifier (e.g., "613" or "rat613")
            session_id: Session identifier (e.g., "20251210" or "20251210_110059")
            config_path: Optional path to config file
            auto_load: If True, automatically discover and load all data paths
        """
        self.animal_id = animal_id
        self.session_id = session_id
        self.config_path = config_path
        
        # Initialize path attributes
        self.kilosort_path = None
        self.dio_paths = {}
        self.video_files = []
        self.tracking_files = []
        self.behavioral_event_files = []
        self.pulse_log_path = None
        
        # Initialize metadata
        self.metadata = {
            'animal_id': animal_id,
            'session_id': session_id,
            'loaded_at': None,
            'data_availability': {},
            'path_validation': {},
            'discovered_files': {}
        }
        
        # Auto-load paths if requested
        if auto_load:
            self.load_all_paths()
    
    def load_all_paths(self):
        """Load all available data paths for this animal/session."""
        print(f"Loading data paths for {self.animal_id}/{self.session_id}")
        
        try:
            self._load_ephys_paths()
            self._load_video_paths() 
            self._load_tracking_paths()
            self._load_behavioral_events()
            self._load_sync_paths()
            self._update_metadata()
            
            print(f"Successfully loaded data paths:")
            self._print_availability_summary()
            
        except Exception as e:
            print(f"Error loading paths: {e}")
            raise
    
    def _load_ephys_paths(self):
        """Load ephys-related paths (Kilosort and DIO)."""
        try:
            # Load Kilosort path
            self.kilosort_path = get_kilosort_path(
                self.animal_id, self.session_id, self.config_path
            )
            print(f"  ✓ Kilosort path: {self.kilosort_path}")
            
            # Load DIO paths for common channels
            self.dio_paths = {}
            for channel in range(1, 5):  # Try channels 1-4
                try:
                    dio_path = get_dio_path(
                        self.animal_id, self.session_id, channel, self.config_path
                    )
                    self.dio_paths[channel] = dio_path
                    print(f"  ✓ DIO channel {channel}: {dio_path}")
                except (FileNotFoundError, ValueError):
                    # Channel doesn't exist, skip
                    pass
                    
        except Exception as e:
            print(f"  ✗ Error loading ephys paths: {e}")
            self.kilosort_path = None
    
    def _load_video_paths(self):
        """Load video file paths."""
        try:
            self.video_files = get_video_files_by_date(
                self.session_id, self.config_path, subfolder="social_videos"
            )
            print(f"  ✓ Found {len(self.video_files)} video files")
            
        except Exception as e:
            print(f"  ✗ Error loading video paths: {e}")
            self.video_files = []
    
    def _load_tracking_paths(self):
        """Load tracking file paths."""
        try:
            self.tracking_files = get_tracking_files_by_date(
                self.session_id, self.config_path
            )
            print(f"  ✓ Found {len(self.tracking_files)} tracking files")
            
        except Exception as e:
            print(f"  ✗ Error loading tracking paths: {e}")
            self.tracking_files = []
    
    def _load_behavioral_events(self):
        """Load behavioral event file paths."""
        try:
            self.behavioral_event_files = get_event_files_by_date(
                self.session_id, self.config_path
            )
            print(f"  ✓ Found {len(self.behavioral_event_files)} behavioral event files")
            
        except Exception as e:
            print(f"  ✗ Error loading behavioral event paths: {e}")
            self.behavioral_event_files = []
    
    def _load_sync_paths(self):
        """Load synchronization-related paths."""
        try:
            self.pulse_log_path = get_pulse_log_path(self.config_path)
            print(f"  ✓ Pulse log: {self.pulse_log_path}")
            
        except Exception as e:
            print(f"  ✗ Error loading sync paths: {e}")
            self.pulse_log_path = None
    
    def _update_metadata(self):
        """Update metadata with current path information."""
        self.metadata['loaded_at'] = datetime.now().isoformat()
        
        # Data availability
        self.metadata['data_availability'] = {
            'ephys': self.kilosort_path is not None,
            'dio_channels': list(self.dio_paths.keys()),
            'video_files': len(self.video_files),
            'tracking_files': len(self.tracking_files),
            'behavioral_event_files': len(self.behavioral_event_files),
            'pulse_log': self.pulse_log_path is not None
        }
        
        # File counts
        self.metadata['discovered_files'] = {
            'video_count': len(self.video_files),
            'tracking_count': len(self.tracking_files),
            'behavioral_event_count': len(self.behavioral_event_files),
            'dio_channels': len(self.dio_paths)
        }
    
    def _print_availability_summary(self):
        """Print summary of available data."""
        availability = self.metadata['data_availability']
        
        print("Data Availability Summary:")
        print(f"  - Ephys (Kilosort): {'✓' if availability['ephys'] else '✗'}")
        print(f"  - DIO channels: {availability['dio_channels'] if availability['dio_channels'] else 'None'}")
        print(f"  - Video files: {availability['video_files']}")
        print(f"  - Tracking files: {availability['tracking_files']}")
        print(f"  - Behavioral event files: {availability['behavioral_event_files']}")
        print(f"  - Pulse log: {'✓' if availability['pulse_log'] else '✗'}")
    
    def validate_paths(self) -> Dict[str, bool]:
        """
        Validate that all discovered paths actually exist.
        
        Returns:
            Dictionary with validation results for each path type
        """
        validation = {}
        
        # Validate Kilosort path
        if self.kilosort_path:
            validation['kilosort'] = verify_kilosort_path(self.kilosort_path, check_files=True)
        else:
            validation['kilosort'] = False
        
        # Validate DIO paths
        validation['dio_channels'] = {}
        for channel, dio_path in self.dio_paths.items():
            validation['dio_channels'][channel] = dio_path.exists()
        
        # Validate video files
        validation['video_files'] = {
            'total': len(self.video_files),
            'existing': sum(1 for vf in self.video_files if vf.exists())
        }
        
        # Validate tracking files
        validation['tracking_files'] = {
            'total': len(self.tracking_files),
            'existing': sum(1 for tf in self.tracking_files if tf.exists())
        }
        
        # Validate behavioral event files
        validation['behavioral_event_files'] = {
            'total': len(self.behavioral_event_files),
            'existing': sum(1 for bf in self.behavioral_event_files if bf.exists())
        }
        
        # Validate pulse log
        if self.pulse_log_path:
            validation['pulse_log'] = self.pulse_log_path.exists()
        else:
            validation['pulse_log'] = False
        
        # Store in metadata
        self.metadata['path_validation'] = validation
        
        return validation
    
    def get_kilosort_path(self) -> Optional[Path]:
        """Get Kilosort data path."""
        return self.kilosort_path
    
    def get_dio_path(self, channel: int = 1) -> Optional[Path]:
        """
        Get DIO path for specific channel.
        
        Args:
            channel: DIO channel number
            
        Returns:
            Path to DIO file or None if not available
        """
        return self.dio_paths.get(channel)
    
    def get_available_dio_channels(self) -> List[int]:
        """Get list of available DIO channels."""
        return list(self.dio_paths.keys())
    
    def get_video_files(self, extension_filter: Optional[str] = None) -> List[Path]:
        """
        Get video files, optionally filtered by extension.
        
        Args:
            extension_filter: Optional file extension to filter by (e.g., '.mp4')
            
        Returns:
            List of video file paths
        """
        if extension_filter:
            return [vf for vf in self.video_files if vf.suffix.lower() == extension_filter.lower()]
        return self.video_files.copy()
    
    def get_tracking_files(self, extension_filter: Optional[str] = None) -> List[Path]:
        """
        Get tracking files, optionally filtered by extension.
        
        Args:
            extension_filter: Optional file extension to filter by (e.g., '.csv')
            
        Returns:
            List of tracking file paths
        """
        if extension_filter:
            return [tf for tf in self.tracking_files if tf.suffix.lower() == extension_filter.lower()]
        return self.tracking_files.copy()
    
    def get_behavioral_event_files(self, extension_filter: Optional[str] = None) -> List[Path]:
        """
        Get behavioral event files, optionally filtered by extension.
        
        Args:
            extension_filter: Optional file extension to filter by (e.g., '.csv')
            
        Returns:
            List of behavioral event file paths
        """
        if extension_filter:
            return [bf for bf in self.behavioral_event_files if bf.suffix.lower() == extension_filter.lower()]
        return self.behavioral_event_files.copy()
    
    def get_pulse_log_path(self) -> Optional[Path]:
        """Get pulse log path."""
        return self.pulse_log_path
    
    def has_complete_dataset(self, required_types: Optional[List[str]] = None) -> bool:
        """
        Check if session has complete dataset.
        
        Args:
            required_types: List of required data types ['ephys', 'video', 'tracking', 'behavioral_events', 'sync']
                          If None, checks for ['ephys', 'video', 'tracking']
            
        Returns:
            True if all required data types are available
        """
        if required_types is None:
            required_types = ['ephys', 'video', 'tracking']
        
        availability = self.metadata['data_availability']
        
        for data_type in required_types:
            if data_type == 'ephys' and not availability['ephys']:
                return False
            elif data_type == 'video' and availability['video_files'] == 0:
                return False
            elif data_type == 'tracking' and availability['tracking_files'] == 0:
                return False
            elif data_type == 'behavioral_events' and availability['behavioral_event_files'] == 0:
                return False
            elif data_type == 'sync' and not availability['pulse_log']:
                return False
        
        return True
    
    def export_paths_summary(self, output_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Export summary of all paths and metadata.
        
        Args:
            output_path: Optional path to save summary as JSON
            
        Returns:
            Dictionary containing complete paths summary
        """
        summary = {
            'session_info': {
                'animal_id': self.animal_id,
                'session_id': self.session_id,
                'config_path': str(self.config_path) if self.config_path else None
            },
            'paths': {
                'kilosort': str(self.kilosort_path) if self.kilosort_path else None,
                'dio_channels': {ch: str(path) for ch, path in self.dio_paths.items()},
                'video_files': [str(vf) for vf in self.video_files],
                'tracking_files': [str(tf) for tf in self.tracking_files],
                'behavioral_event_files': [str(bf) for bf in self.behavioral_event_files], 
                'pulse_log': str(self.pulse_log_path) if self.pulse_log_path else None
            },
            'metadata': self.metadata
        }
        
        # Save to file if requested
        if output_path is not None:
            output_path = Path(output_path)
            import json
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Path summary exported to: {output_path}")
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of DataPathManager."""
        availability = self.metadata.get('data_availability', {})
        ephys_status = "✓" if availability.get('ephys', False) else "✗"
        video_count = availability.get('video_files', 0)
        tracking_count = availability.get('tracking_files', 0)
        behavioral_event_count = availability.get('behavioral_event_files', 0)
        
        return (f"DataPathManager({self.animal_id}/{self.session_id}, "
                f"ephys:{ephys_status}, video:{video_count}, tracking:{tracking_count}, "
                f"events:{behavioral_event_count})")


if __name__ == "__main__":
    # Example usage of DataPathManager class
    print("=== DataPathManager Class Example ===")
    
    try:
        # Create DataPathManager for a session
        animal_id = "613"
        session_id = "20241210"
        
        print(f"\n1. Creating DataStorageManager for {animal_id}/{session_id}")
        path_manager = DataStorageManager(animal_id, session_id, auto_load=True)
        
        print(f"\n2. DataStorageManager status:")
        print(path_manager)
        
        print(f"\n3. Checking available data:")
        print(f"   - Kilosort path: {path_manager.get_kilosort_path()}")
        print(f"   - Available DIO channels: {path_manager.get_available_dio_channels()}")
        print(f"   - Video files: {len(path_manager.get_video_files())}")
        print(f"   - Tracking files: {len(path_manager.get_tracking_files())}")
        print(f"   - Behavioral event files: {len(path_manager.get_behavioral_event_files())}")
        print(f"   - Pulse log: {path_manager.get_pulse_log_path()}")
        
        print(f"\n4. Validating paths:")
        validation = path_manager.validate_paths()
        print(f"   - Kilosort valid: {validation.get('kilosort', False)}")
        print(f"   - Video files valid: {validation['video_files']['existing']}/{validation['video_files']['total']}")
        print(f"   - Tracking files valid: {validation['tracking_files']['existing']}/{validation['tracking_files']['total']}")
        print(f"   - Behavioral event files valid: {validation['behavioral_event_files']['existing']}/{validation['behavioral_event_files']['total']}")
        
        print(f"\n5. Checking completeness:")
        has_ephys_video = path_manager.has_complete_dataset(['ephys', 'video'])
        has_all_data = path_manager.has_complete_dataset(['ephys', 'video', 'tracking'])
        has_with_events = path_manager.has_complete_dataset(['ephys', 'video', 'tracking', 'behavioral_events'])
        print(f"   - Has ephys + video: {has_ephys_video}")
        print(f"   - Has complete dataset: {has_all_data}")
        print(f"   - Has complete dataset + events: {has_with_events}")
        
        print(f"\n6. Export summary:")
        summary = path_manager.export_paths_summary()
        print(f"   - Summary contains {len(summary)} sections")
        print(f"   - Session info: {summary['session_info']}")
        
    except Exception as e:
        print(f"Error during DataPathManager testing: {e}")
        print("Note: This is expected if test data doesn't exist")
    
    print("\n=== Original Functions (Backward Compatibility) ===")
    
    # Test original functions still work
    try:
        animal_id = "613"
        session_id = "20251210"
        
        print(f"\nTesting original functions with {animal_id}/{session_id}:")
        
        # Test original Kilosort function
        kilosort_path = get_kilosort_path(animal_id, session_id)
        print(f"Original get_kilosort_path: {kilosort_path}")
        
        # Test original video function
        video_files = get_video_files_by_date(session_id)
        print(f"Original get_video_files_by_date: {len(video_files)} files")
        
        # Test tracking function
        tracking_files = get_tracking_files_by_date(session_id)
        print(f"Original get_tracking_files_by_date: {len(tracking_files)} files")
        
        # Test behavioral events function
        try:
            behavioral_event_files = get_event_files_by_date(session_id)
            print(f"Original get_event_files_by_date: {len(behavioral_event_files)} files")
        except Exception as e:
            print(f"Original get_event_files_by_date: Error - {e}")
        
        # Test DIO function
        try:
            dio_path = get_dio_path(animal_id, session_id, channel=1)
            print(f"Original get_dio_path (ch1): {dio_path}")
        except Exception as e:
            print(f"Original get_dio_path (ch1): Error - {e}")
        
        # Test validation function
        if kilosort_path:
            is_valid = verify_kilosort_path(kilosort_path, check_files=False)
            print(f"Original verify_kilosort_path: {is_valid}")
        
    except Exception as e:
        print(f"Error testing original functions: {e}")
        print("Note: This is expected if test data doesn't exist")
    
    print(f"\n=== Original Kilosort Example (Partial Matching) ===")
    
    # Original example with partial matching
    animal_id = "613"  # Partial match for "rat613"
    session_id = "20251210"  # Partial match for "20251210_110059"
    
    try:
        path = get_kilosort_path(animal_id, session_id)
        print(f"Animal ID (partial): {animal_id}")
        print(f"Session ID (partial): {session_id}")
        print(f"Kilosort Path: {path}")
        print(f"Path exists: {verify_kilosort_path(path, check_files=False)}")
        
        # Also show example with full names
        print("\n" + "-" * 40)
        full_animal_id = "rat613"
        full_session_id = "20251210_110059"
        path_full = get_kilosort_path(full_animal_id, full_session_id)
        print(f"Animal ID (full): {full_animal_id}")
        print(f"Session ID (full): {full_session_id}")
        print(f"Kilosort Path: {path_full}")
        print(f"Same result: {path == path_full}")
        
    except Exception as e:
        print(f"Error: {e}")