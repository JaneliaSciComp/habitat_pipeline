"""
Kilosort Path Management Module

This module provides utilities for constructing file paths to Kilosort data
based on animal_id and session_id, reading configuration from the project's
default_paths.json file.
"""

import json
from pathlib import Path
from typing import Optional, List
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
    
    # Set search directory based on subfolder parameter
    if subfolder is not None:
        search_directory = tracking_base / subfolder
        if not search_directory.exists():
            raise FileNotFoundError(f"Subfolder not found: {search_directory}")
    else:
        search_directory = tracking_base
    
    # Extract date from session_id (first 8 characters, assuming YYYYMMDD format)
    # Handle both "20251210" and "20251210_110059" formats
    if len(session_id) >= 8:
        session_date = session_id[:8]
        
        # Validate that it looks like a date (8 digits)
        if not session_date.isdigit():
            raise ValueError(f"Could not extract valid date from session_id: {session_id}")
    else:
        raise ValueError(f"Session ID too short to extract date: {session_id}")
    
    matching_tracking_files = []
    
    try:
        # Search through all files in the search directory and subdirectories
        for tracking_file in search_directory.rglob('*'):
            if tracking_file.is_file():
                # Check if file has a tracking extension
                if tracking_file.suffix.lower() in [ext.lower() for ext in tracking_extensions]:
                    # Check if the session date appears in the filename
                    if session_date in tracking_file.name:
                        matching_tracking_files.append(tracking_file)
    
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"Error accessing directory {search_directory}: {e}")
    
    # Sort tracking files by filename for consistent ordering
    matching_tracking_files.sort(key=lambda x: x.name)
    
    return matching_tracking_files


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


if __name__ == "__main__":
    # Example usage
    print("Kilosort Path Management Example:")
    print("-" * 40)
    
    # Example with partial matching
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