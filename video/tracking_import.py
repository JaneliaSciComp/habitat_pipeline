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
import json
import warnings

# Import path utilities from ingestion module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ingestion.kilosort_paths import get_tracking_files_by_date


def load_tracking_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load tracking data from a file and return it as a DataFrame.
    
    This function loads tracking data from various file formats and converts
    them to a pandas DataFrame. Supports common tracking data formats.
    
    Args:
        file_path: Path to the tracking data file (string or Path object)
        
    Returns:
        pandas.DataFrame: Loaded tracking data
    """
    
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


if __name__ == "__main__":
    # Example usage
    print("Tracking Import Example:")
    print("-" * 40)
    
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
    objects = parse_tracking_by_object_name(df)
    
    print(f"\nFound {len(objects)} objects: {list(objects.keys())}")
    
    for obj_name, obj_df in objects.items():
        print(f"\n{obj_name}:")
        print(f"  Shape: {obj_df.shape}")
        print(f"  Columns: {list(obj_df.columns)}")
        print("  Data:")
        print(obj_df.head(3).to_string(index=False))