"""
Video Tracking Import Module

This module provides utilities for loading tracking data and associated timestamps
from video analysis outputs. Integrates with the habitat_pipeline path management
system for consistent file discovery.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
from datetime import datetime
import json
import warnings

# Import DataStorageManager for session management
from ingestion.data_paths import DataStorageManager


def load_tracking_data(data_manager: DataStorageManager, file_index: int = 0) -> pd.DataFrame:
    """
    Load tracking data from a DataStorageManager and return it as a DataFrame.
    
    This function loads tracking data from various file formats using a
    DataStorageManager instance to automatically select tracking files.
    
    Args:
        data_manager: DataStorageManager instance containing session paths
        file_index: Index of tracking file to use (default: 0)
        
    Returns:
        pandas.DataFrame: Loaded tracking data
    """
    
    # Get tracking files from DataStorageManager
    tracking_files = data_manager.get_tracking_files()
    if not tracking_files:
        raise FileNotFoundError(f"No tracking files found in DataStorageManager for {data_manager.animal_id}/{data_manager.session_id}")
    
    if file_index >= len(tracking_files):
        raise IndexError(f"File index {file_index} out of range. Available files: {len(tracking_files)}")
    
    file_path = tracking_files[file_index]
    print(f"Using tracking file from DataStorageManager: {file_path}")
    
    # Convert to Path object for easier handling
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Tracking file not found: {file_path}")
    
    # Get file extension
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.csv':
            # Try different separators and encodings
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
            except pd.errors.ParserError:
                # Try with different separator
                df = pd.read_csv(file_path, sep=';')
            
        elif suffix in ['.tsv', '.txt']:
            # Tab-separated or space-separated
            try:
                df = pd.read_csv(file_path, sep='\t')
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(file_path, sep=' ')
                except pd.errors.ParserError:
                    # Try comma separator as fallback
                    df = pd.read_csv(file_path, sep=',')

        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Ensure we have a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Could not convert loaded data to DataFrame")
        
        # Basic validation
        if df.empty:
            warnings.warn(f"Loaded DataFrame is empty: {file_path}")
        
        print(f"Successfully loaded tracking data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to load tracking data from {file_path}: {str(e)}")


def load_timestamps(tracking_file_path: Union[str, Path]) -> np.ndarray:
    """
    Load timestamps from an .npy file in the same folder as the tracking file.
    
    This function looks for a timestamp file with a similar name pattern,
    typically replacing patterns like '_mask_metrics.csv' with '_ts.npy'.
    
    Args:
        tracking_file_path: Path to the tracking data file
        
    Returns:
        numpy.ndarray: Array of timestamps
        
    Raises:
        FileNotFoundError: If no matching timestamp file is found
        ValueError: If the timestamp file cannot be loaded or is empty
        
    Example:
        For tracking file: RatCity_20251210_1359_40Hz_mask_metrics.csv
        Looks for timestamp file: RatCity_20251210_1359_40Hz_ts.npy
    """
    # Convert to Path object
    tracking_file_path = Path(tracking_file_path)
    
    if not tracking_file_path.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_file_path}")
    
    # Get the directory and base name
    directory = tracking_file_path.parent
    base_name = tracking_file_path.stem  # filename without extension
    
    # Try different patterns to find the timestamp file
    # Pattern 1: Replace '_mask_metrics' with '_ts'
    if '_mask_metrics' in base_name:
        ts_base_name = base_name.replace('_mask_metrics', '_ts')
        ts_file_path = directory / f"{ts_base_name}.npy"
        
        if ts_file_path.exists():
            try:
                timestamps = np.load(ts_file_path)
                print(f"Loaded timestamps from: {ts_file_path}")
                print(f"Timestamp array shape: {timestamps.shape}")
                return timestamps
            except Exception as e:
                raise ValueError(f"Failed to load timestamps from {ts_file_path}: {str(e)}")
    
    
    # If no timestamp file found
    raise FileNotFoundError(
        f"No matching timestamp file found for {tracking_file_path}. "
        f"Looked for patterns like '*_ts.npy' in {directory}"
    )


def parse_tracking(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Parse tracking DataFrame and organize by object_name.
    
    Takes a DataFrame with tracking data and groups it by object_name,
    creating separate DataFrames for each object with object_id and 
    object_name columns removed.
    
    Args:
        df: DataFrame with columns including 'object_name', 'object_id', 
            'frame', 'area', 'perimeter', 'circularity', 'orientation',
            'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'center_x', 'center_y'
    
    Returns:
        Dictionary where keys are object names (strings) and values are 
        DataFrames containing all rows for that object with object_id and 
        object_name columns removed.
    """
    
    # Check required columns
    required_cols = ['object_name', 'object_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check if DataFrame is empty
    if df.empty:
        return {}
    
    # Get unique object names
    object_names = df['object_name'].unique()
    
    # Remove NaN object names
    object_names = [name for name in object_names if pd.notna(name)]
    
    if not object_names:
        raise ValueError("No valid object names found in the DataFrame")
    
    # Create dictionary of DataFrames grouped by object_name
    object_dict = {}
    
    for obj_name in object_names:
        # Filter rows for this object
        obj_rows = df[df['object_name'] == obj_name].copy()
        
        # Remove object_id and object_name columns
        columns_to_drop = ['object_id', 'object_name']
        columns_to_drop = [col for col in columns_to_drop if col in obj_rows.columns]
        obj_df = obj_rows.drop(columns=columns_to_drop)
        
        # Reset index for cleaner output
        obj_df = obj_df.reset_index(drop=True)
        
        # Store in dictionary
        object_dict[str(obj_name)] = obj_df
        
        print(f"Parsed object '{obj_name}': {len(obj_df)} rows, {len(obj_df.columns)} columns")
    
    return object_dict


class VideoTrackingData:
    """
    VideoTrackingData class for managing video tracking data.
    
    This class encapsulates raw tracking data, parsed data organized by objects,
    and frame timestamps. Uses a DataStorageManager instance for integrated 
    session management and automatic file discovery.
    
    Attributes:
        data_manager: DataStorageManager instance
        file_path: Path to the tracking data file
        raw_data: Raw tracking DataFrame as loaded from file
        parsed_data: Dictionary of DataFrames organized by object name
        timestamps: Array of frame timestamps (if available)
        metadata: Dictionary containing metadata about the tracking data
    """
    
    def __init__(self, data_manager: DataStorageManager, load_ts: bool = True, file_index: int = 0):
        """
        Initialize VideoTrackingData from a DataStorageManager.
        
        Args:
            data_manager: DataStorageManager instance containing session paths
            load_ts: If True, attempt to load associated timestamp file
            file_index: Index of tracking file to use (default: 0)
        """
        # Initialize attributes
        self.data_manager = data_manager
        self.file_index = file_index
        self.raw_data = None
        self.parsed_data = {}
        self.timestamps = None
        
        # Get tracking file from DataStorageManager
        tracking_files = self.data_manager.get_tracking_files()
        
        if not tracking_files:
            raise FileNotFoundError(f"No tracking files found in DataStorageManager for {self.data_manager.animal_id}/{self.data_manager.session_id}")
        
        if file_index >= len(tracking_files):
            raise IndexError(f"File index {file_index} out of range. Available files: {len(tracking_files)}")
        
        self.file_path = tracking_files[file_index]
        print(f"Using tracking file from DataStorageManager: {self.file_path}")
        
        # Initialize metadata
        self.metadata = {
            'loaded_at': datetime.now().isoformat(),
            'file_path': str(self.file_path),
            'file_index': file_index,
            'animal_id': self.data_manager.animal_id,
            'session_id': self.data_manager.session_id,
            'available_tracking_files': len(self.data_manager.get_tracking_files()),
            'has_timestamps': False,
            'n_objects': 0,
            'n_frames': 0,
            'object_names': []
        }
        
        # Load the data
        self._load_data(load_ts)
    
    def _load_data(self, load_ts: bool = True):
        """Load and process tracking data."""
        try:
            # Load raw tracking data
            print(f"Loading tracking data from: {self.file_path}")
            self.raw_data = load_tracking_data(self.data_manager, self.file_index)
            
            # Parse data by objects
            print("Parsing tracking data by objects...")
            self.parsed_data = parse_tracking(self.raw_data)
            
            # Load timestamps if requested
            if load_ts:
                try:
                    print("Attempting to load timestamps...")
                    self.timestamps = load_timestamps(self.file_path)
                    self.metadata['has_timestamps'] = True
                    print("Successfully loaded timestamps")
                except (FileNotFoundError, ValueError) as e:
                    print(f"Could not load timestamps: {e}")
                    self.timestamps = None
                    self.metadata['has_timestamps'] = False
            
            # Update metadata
            self._update_metadata()
            
            print(f"VideoTrackingData loaded successfully:")
            print(f"  - Raw data shape: {self.raw_data.shape}")
            print(f"  - Objects found: {self.metadata['n_objects']}")
            print(f"  - Object names: {self.metadata['object_names']}")
            print(f"  - Frame count: {self.metadata['n_frames']}")
            print(f"  - Has timestamps: {self.metadata['has_timestamps']}")
            
        except Exception as e:
            raise ValueError(f"Failed to load VideoTrackingData: {e}")
    
    def _update_metadata(self):
        """Update metadata based on loaded data."""
        if self.raw_data is not None:
            self.metadata['n_frames'] = len(self.raw_data['frame'].unique()) if 'frame' in self.raw_data.columns else 0
            
        if self.parsed_data:
            self.metadata['n_objects'] = len(self.parsed_data)
            self.metadata['object_names'] = list(self.parsed_data.keys())
        
        if self.timestamps is not None:
            self.metadata['timestamp_shape'] = self.timestamps.shape
            self.metadata['timestamp_range'] = {
                'min': float(self.timestamps.min()),
                'max': float(self.timestamps.max()),
                'duration_seconds': float(self.timestamps.max() - self.timestamps.min())
            }
    
    def get_object_data(self, object_name: str) -> Optional[pd.DataFrame]:
        """
        Get tracking data for a specific object.
        
        Args:
            object_name: Name of the object to retrieve
            
        Returns:
            DataFrame containing tracking data for the specified object,
            or None if the object is not found
        """
        return self.parsed_data.get(object_name)
    
    def get_object_names(self) -> List[str]:
        """Get list of all tracked object names."""
        return list(self.parsed_data.keys())
    
    def get_frame_range(self) -> Tuple[int, int]:
        """
        Get the range of frame numbers in the tracking data.
        
        Returns:
            Tuple of (min_frame, max_frame)
        """
        if self.raw_data is not None and 'frame' in self.raw_data.columns:
            return (int(self.raw_data['frame'].min()), int(self.raw_data['frame'].max()))
        return (0, 0)
    
    def get_object_trajectory(self, object_name: str) -> Optional[pd.DataFrame]:
        """
        Get trajectory data (center_x, center_y, frame) for a specific object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            DataFrame with columns ['frame', 'center_x', 'center_y', 'timestamps'] if available,
            or None if the object is not found
        """
        obj_data = self.get_object_data(object_name)
        if obj_data is None:
            return None
        
        # Check if trajectory columns exist
        required_cols = ['frame', 'center_x', 'center_y']
        missing_cols = [col for col in required_cols if col not in obj_data.columns]
        
        if missing_cols:
            print(f"Warning: Missing trajectory columns for {object_name}: {missing_cols}")
            return None
        
        # Extract trajectory data
        trajectory = obj_data[required_cols].copy()
        
        # Add timestamps if available
        if self.timestamps is not None and len(self.timestamps) >= len(trajectory):
            trajectory['timestamps'] = self.timestamps[:len(trajectory)]
        
        return trajectory.sort_values('frame').reset_index(drop=True)
    
    def export_summary(self, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Export a summary of the tracking data.
        
        Args:
            output_path: Optional path to save summary as JSON file
            
        Returns:
            Dictionary containing summary information
        """
        summary = {
            'metadata': self.metadata,
            'objects': {}
        }
        
        # Add object-specific information
        for obj_name, obj_data in self.parsed_data.items():
            obj_summary = {
                'n_detections': len(obj_data),
                'frame_range': (int(obj_data['frame'].min()), int(obj_data['frame'].max())) if 'frame' in obj_data.columns else None,
                'columns': list(obj_data.columns)
            }
            
            # Add trajectory statistics if available
            if 'center_x' in obj_data.columns and 'center_y' in obj_data.columns:
                obj_summary['trajectory_stats'] = {
                    'x_range': (float(obj_data['center_x'].min()), float(obj_data['center_x'].max())),
                    'y_range': (float(obj_data['center_y'].min()), float(obj_data['center_y'].max())),
                    'mean_position': (float(obj_data['center_x'].mean()), float(obj_data['center_y'].mean()))
                }
            
            summary['objects'][obj_name] = obj_summary
        
        # Save to file if requested
        if output_path is not None:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary exported to: {output_path}")
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of VideoTrackingData."""
        return (f"VideoTrackingData(DataStorageManager({self.data_manager.animal_id}/{self.data_manager.session_id}), "
                f"objects={self.metadata['n_objects']}, "
                f"frames={self.metadata['n_frames']}, "
                f"timestamps={self.metadata['has_timestamps']})")
    
    def get_alternative_tracking_files(self) -> List[Path]:
        """
        Get list of alternative tracking files from DataStorageManager.
        
        Returns:
            List of available tracking file paths
        """
        return self.data_manager.get_tracking_files()
    
    def switch_tracking_file(self, file_index: int, load_ts: bool = True):
        """
        Switch to a different tracking file using DataStorageManager.
        
        Args:
            file_index: Index of the tracking file to switch to
            load_ts: If True, attempt to load associated timestamp file
        """
        tracking_files = self.data_manager.get_tracking_files()
        if file_index >= len(tracking_files):
            raise IndexError(f"File index {file_index} out of range. Available files: {len(tracking_files)}")
        
        # Update file path and index
        self.file_path = tracking_files[file_index]
        self.file_index = file_index
        
        # Update metadata
        self.metadata['file_path'] = str(self.file_path)
        self.metadata['file_index'] = file_index
        
        # Reload data
        print(f"Switching to tracking file: {self.file_path}")
        self._load_data(load_ts)
    
    def has_data_manager(self) -> bool:
        """Check if this instance is using a DataStorageManager."""
        return True  # Always true since we only accept DataStorageManager


# Convenience factory function
def create_tracking_data_from_manager(data_manager: DataStorageManager, 
                                     file_index: int = 0, 
                                     load_ts: bool = True) -> VideoTrackingData:
    """
    Convenience function to create VideoTrackingData from DataStorageManager.
    
    Args:
        data_manager: DataStorageManager instance
        file_index: Index of tracking file to use (default: 0)
        load_ts: If True, attempt to load associated timestamp file
        
    Returns:
        VideoTrackingData instance
    """
    return VideoTrackingData(data_manager, load_ts=load_ts, file_index=file_index)


if __name__ == "__main__":
    # Example usage
    print("Tracking Import Example:")
    print("-" * 40)
    
    # Example 1: Using individual functions
    print("Example 1: Using individual functions")
    
    # Create example DataFrame with the expected schema
    example_data = {
        'frame': [1, 2, 3, 1, 2, 3, 1, 2],
        'object_id': [1, 1, 1, 2, 2, 2, 3, 3],
        'object_name': ['mouse1', 'mouse1', 'mouse1', 'mouse2', 'mouse2', 'mouse2', 'rat1', 'rat1'],
        'area': [120, 125, 130, 90, 95, 100, 200, 205],
        'perimeter': [40, 42, 44, 35, 36, 38, 60, 62],
        'circularity': [0.8, 0.82, 0.81, 0.75, 0.77, 0.76, 0.85, 0.84],
        'orientation': [45, 48, 50, 30, 32, 35, 60, 62],
        'bbox_x': [10, 12, 14, 50, 52, 54, 80, 82],
        'bbox_y': [20, 22, 24, 30, 32, 34, 40, 42],
        'bbox_width': [15, 16, 17, 12, 13, 14, 20, 21],
        'bbox_height': [18, 19, 20, 15, 16, 17, 25, 26],
        'center_x': [17.5, 20, 22.5, 56, 58.5, 61, 90, 92.5],
        'center_y': [29, 31.5, 34, 37.5, 40, 42.5, 52.5, 55]
    }
    
    df = pd.DataFrame(example_data)
    print("Example DataFrame shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head())
    
    # Parse by object name
    print("\nParsing by object name:")
    objects = parse_tracking(df)
    
    print(f"\nFound {len(objects)} objects: {list(objects.keys())}")
    
    for obj_name, obj_df in objects.items():
        print(f"\n{obj_name}:")
        print(f"  Shape: {obj_df.shape}")
        print(f"  Columns: {list(obj_df.columns)}")
        print("  Data:")
        print(obj_df.head(3).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Example 2: Using VideoTrackingData class")
    print("=" * 60)
    
    # Using DataStorageManager (only method)
    print("Using DataStorageManager for session management:")
    print("  from ingestion.data_paths import DataStorageManager")
    print("  data_manager = DataStorageManager('animal_id', 'session_id', auto_load=True)")
    print("  tracking_data = VideoTrackingData(data_manager)  # Uses first tracking file")
    print("  # or specify which tracking file to use")
    print("  tracking_data = VideoTrackingData(data_manager, file_index=1)")
    print("  # or using convenience function")
    print("  tracking_data = create_tracking_data_from_manager(data_manager, file_index=0)")
    print("")
    
    print("VideoTrackingData class provides:")
    print("  - Automatic loading and parsing of tracking data")
    print("  - Integration with DataStorageManager for session management") 
    print("  - Optional timestamp loading")
    print("  - Easy access to individual object trajectories")
    print("  - Metadata and summary generation")
    print("  - Export capabilities")
    print("  - Ability to switch between tracking files")
    print("")
    
    print("Key methods:")
    print("  - get_object_data(name): Get data for specific object")
    print("  - get_object_trajectory(name): Get trajectory (x, y, frame, timestamps)")
    print("  - get_object_names(): List all tracked objects")
    print("  - switch_tracking_file(index): Change to different tracking file")
    print("  - export_summary(): Generate comprehensive data summary")
    print("")
    
    print("Example workflow with DataStorageManager:")
    print("  data_manager = DataStorageManager('613', '20251216', auto_load=True)")
    print("  tracking = VideoTrackingData(data_manager)")
    print("  objects = tracking.get_object_names()")
    print("  trajectory = tracking.get_object_trajectory(objects[0])")
    print("  summary = tracking.export_summary()")
    
    print("\\n" + "=" * 60)