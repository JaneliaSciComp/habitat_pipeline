"""
Behavioral Events Import and Processing Module

This module provides utilities for loading and processing behavioral event CSV files
from the Habitat pipeline. It integrates with the DataStorageManager for unified
path management and provides analysis tools for behavioral event data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import DataStorageManager for path management
from ingestion.data_paths import DataStorageManager
from ingestion.ephys_sync import DataSyncManager


class BehavioralEventsData:
    """
    Behavioral Events Data Manager
    
    This class handles loading, processing, and analysis of behavioral event CSV files
    discovered through DataStorageManager. It provides methods for event filtering,
    time-based analysis, and data export.
    
    Expected CSV columns:
    - type: Abbreviated behavior type (F=fight, CO=confrontation, C=chase, etc.)
    - initiator, victim, winner, loser: Rat identities (e.g., "rat616")
    - ts_start, ts_end: Linux timestamps for behavior start/end times
    
    Attributes:
        data_manager: DataStorageManager instance for path management
        event_files: List of discovered behavioral event CSV files
        events_data: Loaded and combined event data
        metadata: Information about loaded events
        behavior_types: Dictionary mapping abbreviations to full behavior names
    """
    
    # Behavior type abbreviation mapping
    BEHAVIOR_TYPES = {
        'F': 'fight',
        'CO': 'confrontation', 
        'C': 'chase',
        'R': 'rob food',
        'FM': 'food_move',
        'I': 'introduce',
        'UD': 'unlock_door',
        'FD': 'food_delivery',
        'P': 'playful fight', #peaceful fight',
        'RW': 'rob wood block',
        'S': 'share',
        'H': 'harvest food',  # especially for foraging, rather than rob from others
        'SN': 'sniff',  # usually is genital sniffing
        'EC': 'encounter',  # come upon face-to-face, unexpectedly
        'HD': 'huddle',
        'G': 'allogrooming',
        'PO': 'policing',
        'D': 'defense',
        'PS': 'push'
    }
    
    def __init__(self, data_manager: DataStorageManager, auto_load: bool = True):
        """
        Initialize BehavioralEventsData with DataStorageManager.
        
        Args:
            data_manager: DataStorageManager instance for the session
            auto_load: If True, automatically load all discovered event files
        """
        if not isinstance(data_manager, DataStorageManager):
            raise TypeError("data_manager must be a DataStorageManager instance")
        
        self.data_manager = data_manager
        self.event_files = data_manager.get_behavioral_event_files()
        self.events_data = None
        self.behavior_types = self.BEHAVIOR_TYPES.copy()
        self.metadata = {
            'animal_id': data_manager.animal_id,
            'session_id': data_manager.session_id,
            'files_loaded': [],
            'total_events': 0,
            'event_types': [],
            'behavior_abbreviations': [],
            'rat_identities': [],
            'time_range': None,
            'loaded_at': None
        }
        
        if auto_load and self.event_files:
            self.load_events()
    
    def load_events(self, file_indices: Optional[List[int]] = None) -> bool:
        """
        Load behavioral event data from CSV files.
        
        Args:
            file_indices: Optional list of file indices to load. If None, loads all files.
            
        Returns:
            True if events were successfully loaded, False otherwise
        """
        if not self.event_files:
            print("No behavioral event files found for this session")
            return False
        
        # Determine which files to load
        if file_indices is None:
            files_to_load = self.event_files
            indices_to_load = list(range(len(self.event_files)))
        else:
            files_to_load = [self.event_files[i] for i in file_indices if i < len(self.event_files)]
            indices_to_load = file_indices
        
        if not files_to_load:
            print("No valid files to load")
            return False
        
        # print(f"Loading {len(files_to_load)} behavioral event file(s)...")
        
        combined_data = []
        loaded_files = []
        
        for idx, file_path in zip(indices_to_load, files_to_load):
            try:
                # Load CSV file
                df = pd.read_csv(file_path)
                
                # Validate expected columns
                expected_key_columns = ['type', 'initiator', 'victim', 'ts_start', 'ts_end']
                missing_columns = [col for col in expected_key_columns if col not in df.columns]
                if missing_columns:
                    print(f"  ⚠ Warning: Missing expected columns in {file_path.name}: {missing_columns}")
                
                # Add decoded behavior types if 'type' column exists
                if 'type' in df.columns:
                    df['behavior_full_name'] = df['type'].map(self.behavior_types).fillna(df['type'])
                
                # Add file metadata
                df['source_file'] = file_path.name
                df['file_index'] = idx
                
                combined_data.append(df)
                loaded_files.append(file_path.name)
                # print(f"  ✓ Loaded {file_path.name}: {len(df)} events")
                
            except Exception as e:
                print(f"  ✗ Error loading {file_path.name}: {e}")
                continue
        
        if not combined_data:
            print("Failed to load any event files")
            return False
        
        # Combine all data
        self.events_data = pd.concat(combined_data, ignore_index=True)
        
        # Update metadata
        self._update_metadata(loaded_files)
        
        print(f"Successfully loaded {len(combined_data)} file(s) with {len(self.events_data)} total events")
        return True
    
    def _update_metadata(self, loaded_files: List[str]):
        """Update metadata after loading events."""
        self.metadata['loaded_at'] = datetime.now().isoformat()
        self.metadata['files_loaded'] = loaded_files
        self.metadata['total_events'] = len(self.events_data)
        
        # Analyze behavior types (abbreviations and full names)
        if 'type' in self.events_data.columns:
            abbreviations = sorted(self.events_data['type'].unique().tolist())
            self.metadata['behavior_abbreviations'] = abbreviations
            # Map to full names
            full_names = [self.behavior_types.get(abbr, abbr) for abbr in abbreviations]
            self.metadata['event_types'] = sorted(set(full_names))
        else:
            self.metadata['behavior_abbreviations'] = []
            self.metadata['event_types'] = []
        
        # Analyze rat identities
        rat_columns = ['initiator', 'victim', 'winner', 'loser']
        rat_ids = set()
        for col in rat_columns:
            if col in self.events_data.columns:
                # Filter out NaN values and get unique rat IDs
                valid_ids = self.events_data[col].dropna().unique()
                rat_ids.update(valid_ids)
        self.metadata['rat_identities'] = sorted([rid for rid in rat_ids if pd.notna(rid)])
        
        # Analyze time range using timestamp columns
        if 'ts_start' in self.events_data.columns and 'ts_end' in self.events_data.columns:
            ts_start_col = self.events_data['ts_start'].dropna()
            ts_end_col = self.events_data['ts_end'].dropna()
            
            if len(ts_start_col) > 0 and len(ts_end_col) > 0:
                min_time = ts_start_col.min()
                max_time = ts_end_col.max()
                
                # Calculate average event duration
                durations = self.events_data['ts_end'] - self.events_data['ts_start']
                valid_durations = durations.dropna()
                
                self.metadata['time_range'] = {
                    'min_time': float(min_time),
                    'max_time': float(max_time),
                    'total_duration': float(max_time - min_time),
                    'time_column_start': 'ts_start',
                    'time_column_end': 'ts_end',
                    'average_event_duration': float(valid_durations.mean()) if len(valid_durations) > 0 else None,
                    'total_events_with_timestamps': len(valid_durations)
                }
    
    def get_events_by_type(self, event_type: str, use_full_name: bool = True) -> Optional[pd.DataFrame]:
        """
        Get all events of a specific type.
        
        Args:
            event_type: Type of event to filter for (can be abbreviation or full name)
            use_full_name: If True, search by full behavior name; if False, search by abbreviation
            
        Returns:
            DataFrame with events of the specified type, or None if not found
        """
        if self.events_data is None:
            print("No events data loaded. Call load_events() first.")
            return None
        
        if 'type' not in self.events_data.columns:
            print("No 'type' column found in behavioral events data")
            return None
        
        # Determine search strategy
        if use_full_name:
            # Search by full behavior name - need to find corresponding abbreviation
            abbreviation = None
            for abbr, full_name in self.behavior_types.items():
                if full_name.lower() == event_type.lower():
                    abbreviation = abbr
                    break
            
            if abbreviation is None:
                # Try direct match with abbreviation
                if event_type in self.behavior_types:
                    abbreviation = event_type
                else:
                    print(f"Unknown behavior type: {event_type}")
                    print(f"Available types: {list(self.behavior_types.values())}")
                    return None
            
            # Filter by abbreviation
            filtered_events = self.events_data[self.events_data['type'] == abbreviation].copy()
        else:
            # Search directly by abbreviation
            filtered_events = self.events_data[self.events_data['type'] == event_type].copy()
        
        if len(filtered_events) == 0:
            print(f"No events found for type: {event_type}")
            return None
        
        return filtered_events
    
    def get_events_in_time_range(self, start_time: float, end_time: float, 
                                time_column: Optional[str] = None, 
                                overlap_mode: str = 'any') -> Optional[pd.DataFrame]:
        """
        Get events within a specific time range.
        
        Args:
            start_time: Start time for filtering (Linux timestamp)
            end_time: End time for filtering (Linux timestamp)
            time_column: Column name containing timestamps. If None, uses ts_start.
            overlap_mode: 'any' (any overlap), 'start' (event starts in range), 
                         'end' (event ends in range), 'contained' (fully contained)
            
        Returns:
            DataFrame with events in the time range, or None if not found
        """
        if self.events_data is None:
            print("No events data loaded. Call load_events() first.")
            return None
        
        # Use ts_start/ts_end columns if available, otherwise fall back to specified column
        if time_column is None:
            if 'ts_start' in self.events_data.columns:
                start_col = 'ts_start'
                end_col = 'ts_end' if 'ts_end' in self.events_data.columns else 'ts_start'
            else:
                time_columns = [col for col in self.events_data.columns if 'time' in col.lower()]
                if not time_columns:
                    print("No timestamp columns found in data")
                    return None
                start_col = end_col = time_columns[0]
        else:
            start_col = end_col = time_column
        
        if start_col not in self.events_data.columns:
            print(f"Time column '{start_col}' not found in data")
            return None
        
        # Filter by time range based on overlap mode
        if overlap_mode == 'any' and end_col in self.events_data.columns:
            # Any overlap: event_start <= range_end AND event_end >= range_start
            mask = (self.events_data[start_col] <= end_time) & (self.events_data[end_col] >= start_time)
        elif overlap_mode == 'start':
            # Event starts within range
            mask = (self.events_data[start_col] >= start_time) & (self.events_data[start_col] <= end_time)
        elif overlap_mode == 'end' and end_col in self.events_data.columns:
            # Event ends within range
            mask = (self.events_data[end_col] >= start_time) & (self.events_data[end_col] <= end_time)
        elif overlap_mode == 'contained' and end_col in self.events_data.columns:
            # Event fully contained within range
            mask = (self.events_data[start_col] >= start_time) & (self.events_data[end_col] <= end_time)
        else:
            # Default: use start time only
            mask = (self.events_data[start_col] >= start_time) & (self.events_data[start_col] <= end_time)
        
        filtered_events = self.events_data[mask].copy()
        
        print(f"Found {len(filtered_events)} events between {start_time} and {end_time} (mode: {overlap_mode})")
        return filtered_events
    
    def get_events_by_rat(self, rat_id: str, role: str = 'any') -> Optional[pd.DataFrame]:
        """
        Get all events involving a specific rat.
        
        Args:
            rat_id: Rat identifier (e.g., "rat616" or "616")
            role: Role to filter by ('any', 'initiator', 'victim', 'winner', 'loser')
            
        Returns:
            DataFrame with events involving the specified rat, or None if not found
        """
        if self.events_data is None:
            print("No events data loaded. Call load_events() first.")
            return None
        
        # Normalize rat ID (ensure it starts with 'rat')
        if not rat_id.startswith('rat'):
            rat_id = f"rat{rat_id}"
        
        rat_columns = ['initiator', 'victim', 'winner', 'loser']
        available_columns = [col for col in rat_columns if col in self.events_data.columns]
        
        if not available_columns:
            print("No rat identity columns found in data")
            return None
        
        if role == 'any':
            # Look for rat in any role
            masks = []
            for col in available_columns:
                masks.append(self.events_data[col] == rat_id)
            combined_mask = pd.concat(masks, axis=1).any(axis=1)
            filtered_events = self.events_data[combined_mask].copy()
        elif role in available_columns:
            # Filter by specific role
            filtered_events = self.events_data[self.events_data[role] == rat_id].copy()
        else:
            print(f"Role '{role}' not found. Available roles: {available_columns}")
            return None
        
        if len(filtered_events) == 0:
            print(f"No events found for rat {rat_id} in role {role}")
            return None
        
        print(f"Found {len(filtered_events)} events for rat {rat_id} (role: {role})")
        return filtered_events
    
    def get_behavior_type_mapping(self) -> Dict[str, str]:
        """
        Get the mapping of behavior type abbreviations to full names.
        
        Returns:
            Dictionary mapping abbreviations to full behavior names
        """
        return self.behavior_types.copy()
    
    def decode_behavior_type(self, abbreviation: str) -> str:
        """
        Decode a behavior type abbreviation to its full name.
        
        Args:
            abbreviation: Behavior type abbreviation (e.g., 'F')
            
        Returns:
            Full behavior name or original abbreviation if not found
        """
        return self.behavior_types.get(abbreviation, abbreviation)
    
    def get_rat_interaction_summary(self, rat_id: str) -> Dict:
        """
        Get summary of interactions for a specific rat.
        
        Args:
            rat_id: Rat identifier (e.g., "rat616" or "616")
            
        Returns:
            Dictionary with interaction statistics
        """
        if self.events_data is None:
            return {"error": "No events data loaded"}
        
        # Normalize rat ID
        if not rat_id.startswith('rat'):
            rat_id = f"rat{rat_id}"
        
        summary = {
            'rat_id': rat_id,
            'total_events': 0,
            'as_initiator': 0,
            'as_victim': 0,
            'as_winner': 0,
            'as_loser': 0,
            'behavior_types': {},
            'interaction_partners': set()
        }
        
        rat_columns = ['initiator', 'victim', 'winner', 'loser']
        
        for role in rat_columns:
            if role in self.events_data.columns:
                mask = self.events_data[role] == rat_id
                count = mask.sum()
                summary[f'as_{role}'] = int(count)
                summary['total_events'] += count
                
                # Track interaction partners
                if count > 0:
                    role_events = self.events_data[mask]
                    for other_role in rat_columns:
                        if other_role != role and other_role in role_events.columns:
                            partners = role_events[other_role].dropna().unique()
                            summary['interaction_partners'].update(partners)
        
        # Remove self from partners
        summary['interaction_partners'].discard(rat_id)
        summary['interaction_partners'] = list(summary['interaction_partners'])
        
        # Count behavior types for this rat
        if 'type' in self.events_data.columns:
            rat_events = self.get_events_by_rat(rat_id, 'any')
            if rat_events is not None:
                behavior_counts = rat_events['type'].value_counts().to_dict()
                # Decode abbreviations
                summary['behavior_types'] = {
                    self.decode_behavior_type(abbr): count 
                    for abbr, count in behavior_counts.items()
                }
        
        return summary
    
    def get_event_statistics(self) -> Dict:
        """
        Generate statistics about the behavioral events.
        
        Returns:
            Dictionary containing event statistics
        """
        if self.events_data is None:
            return {"error": "No events data loaded"}
        
        stats = {
            'total_events': len(self.events_data),
            'files_processed': len(self.metadata['files_loaded']),
            'columns': list(self.events_data.columns),
            'behavior_abbreviations': self.metadata['behavior_abbreviations'],
            'behavior_types': self.metadata['event_types'],
            'rat_identities': self.metadata['rat_identities'],
            'time_range': self.metadata.get('time_range'),
        }

        # Add behavior type counts (both abbreviations and full names)
        if 'type' in self.events_data.columns:
            abbrev_counts = self.events_data['type'].value_counts().to_dict()
            stats['behavior_abbreviation_counts'] = abbrev_counts

            # Convert to full names
            full_name_counts = {}
            for abbr, count in abbrev_counts.items():
                full_name = self.decode_behavior_type(abbr)
                full_name_counts[full_name] = count
            stats['behavior_type_counts'] = full_name_counts

        # Add rat role statistics
        rat_role_stats = {}
        rat_columns = ['initiator', 'victim', 'winner', 'loser']
        for role in rat_columns:
            if role in self.events_data.columns:
                role_counts = self.events_data[role].value_counts().to_dict()
                rat_role_stats[role] = role_counts
        stats['rat_role_counts'] = rat_role_stats

        # Add temporal statistics using ts_start/ts_end
        if 'ts_start' in self.events_data.columns and 'ts_end' in self.events_data.columns:
            start_times = self.events_data['ts_start'].dropna()
            end_times = self.events_data['ts_end'].dropna()

            if len(start_times) > 0 and len(end_times) > 0:
                # Event durations
                durations = self.events_data['ts_end'] - self.events_data['ts_start']
                valid_durations = durations.dropna()

                # Inter-event intervals (time between event starts)
                start_times_sorted = start_times.sort_values()
                intervals = start_times_sorted.diff().dropna()

                stats['temporal_stats'] = {
                    'mean_event_duration': float(valid_durations.mean()) if len(valid_durations) > 0 else None,
                    'median_event_duration': float(valid_durations.median()) if len(valid_durations) > 0 else None,
                    'min_event_duration': float(valid_durations.min()) if len(valid_durations) > 0 else None,
                    'max_event_duration': float(valid_durations.max()) if len(valid_durations) > 0 else None,
                    'mean_inter_event_interval': float(intervals.mean()) if len(intervals) > 0 else None,
                    'median_inter_event_interval': float(intervals.median()) if len(intervals) > 0 else None,
                    'total_session_duration': float(start_times.max() - start_times.min()) if len(start_times) > 0 else None,
                    'events_with_valid_timestamps': len(valid_durations)
                }
        return stats
    
    def get_event_files_info(self) -> List[Dict]:
        """
        Get information about discovered event files.
        
        Returns:
            List of dictionaries containing file information        """
        file_info = []
        for i, file_path in enumerate(self.event_files):
            info = {
                'index': i,
                'filename': file_path.name,
                'full_path': str(file_path),
                'exists': file_path.exists(),
                'size_bytes': file_path.stat().st_size if file_path.exists() else None
            }
            file_info.append(info)
        
        return file_info
    
    def export_filtered_events(self, output_path: Union[str, Path], 
                             event_type: Optional[str] = None,
                             time_range: Optional[Tuple[float, float]] = None,
                             format: str = 'csv') -> bool:
        """
        Export filtered events to file.
        
        Args:
            output_path: Path to save the exported data
            event_type: Optional event type to filter for
            time_range: Optional (start_time, end_time) tuple for time filtering
            format: Output format ('csv', 'json', 'parquet')
            
        Returns:
            True if export was successful
        """
        if self.events_data is None:
            print("No events data loaded. Call load_events() first.")
            return False
        
        # Start with all data
        filtered_data = self.events_data.copy()
        
        # Apply event type filter
        if event_type:
            type_filtered = self.get_events_by_type(event_type)
            if type_filtered is None:
                return False
            filtered_data = type_filtered
        
        # Apply time range filter
        if time_range:
            start_time, end_time = time_range
            time_filtered = self.get_events_in_time_range(start_time, end_time)
            if time_filtered is None:
                return False
            filtered_data = time_filtered
        
        # Export data
        output_path = Path(output_path)
        try:
            if format.lower() == 'csv':
                filtered_data.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                filtered_data.to_json(output_path, orient='records', indent=2)
            elif format.lower() == 'parquet':
                filtered_data.to_parquet(output_path)
            else:
                print(f"Unsupported format: {format}")
                return False
            
            print(f"Exported {len(filtered_data)} events to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
    
    def export_summary(self, output_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Export comprehensive summary of behavioral events.
        
        Args:
            output_path: Optional path to save summary as JSON
            
        Returns:
            Dictionary containing summary information
        """
        summary = {
            'session_info': {
                'animal_id': self.metadata['animal_id'],
                'session_id': self.metadata['session_id'],
                'files_discovered': len(self.event_files),
                'files_loaded': len(self.metadata['files_loaded'])
            },
            'data_info': self.get_event_statistics(),
            'file_info': self.get_event_files_info(),
            'metadata': self.metadata
        }
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            try:
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"Summary exported to: {output_path}")
            except Exception as e:
                print(f"Error saving summary: {e}")
        
        return summary
    
    def get_available_event_types(self, return_format: str = 'full') -> List[str]:
        """
        Get list of available event types.
        
        Args:
            return_format: 'full' for full names, 'abbreviations' for abbreviations, 'both' for tuples
            
        Returns:
            List of event types in requested format
        """
        if return_format == 'full':
            return self.metadata['event_types']
        elif return_format == 'abbreviations':
            return self.metadata['behavior_abbreviations']
        elif return_format == 'both':
            # Return list of (abbreviation, full_name) tuples
            abbrevs = self.metadata['behavior_abbreviations']
            return [(abbr, self.decode_behavior_type(abbr)) for abbr in abbrevs]
        else:
            raise ValueError("return_format must be 'full', 'abbreviations', or 'both'")
    
    def get_available_rats(self) -> List[str]:
        """
        Get list of rat identities found in the data.
        
        Returns:
            List of rat identifiers
        """
        return self.metadata['rat_identities']
    
    def synchronize_with_ephys(self, sync_manager: 'DataSyncManager', 
                             create_new_columns: bool = True) -> bool:
        """
        Convert event timestamps from behavioral time to ephys time using DataSyncManager.
        
        Args:
            sync_manager: DataSyncManager instance for time synchronization
            create_new_columns: If True, creates new columns (ts_start_ephys, ts_end_ephys).
                              If False, overwrites original timestamp columns.
                              
        Returns:
            True if synchronization was successful, False otherwise
        """
        if self.events_data is None:
            print("No events data loaded. Call load_events() first.")
            return False
        
        # Check if we have timestamp columns
        if 'ts_start' not in self.events_data.columns:
            print("No 'ts_start' column found in behavioral events data")
            return False
        
        # print(f"Synchronizing {len(self.events_data)} behavioral events with ephys timestamps...")
        
        # Convert ts_start timestamps
        try:
            # Get valid start timestamps (drop NaN values)
            valid_start_mask = self.events_data['ts_start'].notna()
            valid_start_data = self.events_data[valid_start_mask].copy()
            
            if len(valid_start_data) == 0:
                print("No valid start timestamps found")
                return False
            
            # Convert from nanoseconds to seconds and sync to ephys time
            start_times_seconds = valid_start_data['ts_start'] / 1e9
            ephys_start_times = []
            
            # print("Converting start timestamps...")
            for i, behav_time in enumerate(start_times_seconds):
                try:
                    ephys_time = sync_manager.convert_behavior_to_ephys(behav_time)
                    ephys_start_times.append(ephys_time)
                except Exception as e:
                    print(f"Warning: Failed to convert start timestamp {behav_time}: {e}")
                    ephys_start_times.append(None)
                
                # Progress indicator for large datasets
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(start_times_seconds)} start timestamps")
            
            # Store ephys start times
            if create_new_columns:
                self.events_data.loc[:, 'ts_start_ephys'] = None  # Initialize column
                self.events_data.loc[valid_start_mask, 'ts_start_ephys'] = ephys_start_times
            else:
                self.events_data.loc[valid_start_mask, 'ts_start'] = ephys_start_times
            
            # Convert ts_end timestamps if available
            if 'ts_end' in self.events_data.columns:
                valid_end_mask = self.events_data['ts_end'].notna()
                valid_end_data = self.events_data[valid_end_mask].copy()
                
                if len(valid_end_data) > 0:
                    end_times_seconds = valid_end_data['ts_end'] / 1e9
                    ephys_end_times = []
                    
                    # print("Converting end timestamps...")
                    for i, behav_time in enumerate(end_times_seconds):
                        try:
                            ephys_time = sync_manager.convert_behavior_to_ephys(behav_time)
                            ephys_end_times.append(ephys_time)
                        except Exception as e:
                            print(f"Warning: Failed to convert end timestamp {behav_time}: {e}")
                            ephys_end_times.append(None)
                        
                        # Progress indicator for large datasets
                        if (i + 1) % 1000 == 0:
                            print(f"  Processed {i + 1}/{len(end_times_seconds)} end timestamps")
                    
                    # Store ephys end times
                    if create_new_columns:
                        self.events_data.loc[:, 'ts_end_ephys'] = None  # Initialize column
                        self.events_data.loc[valid_end_mask, 'ts_end_ephys'] = ephys_end_times
                    else:
                        self.events_data.loc[valid_end_mask, 'ts_end'] = ephys_end_times
            
            # Update metadata to reflect synchronization
            self.metadata['synchronized_with_ephys'] = True
            self.metadata['sync_columns_created'] = create_new_columns
            self.metadata['ephys_sync_timestamp'] = datetime.now().isoformat()
            
            # Count successful conversions
            if create_new_columns:
                valid_ephys_start = self.events_data['ts_start_ephys'].notna().sum()
                valid_ephys_end = (self.events_data['ts_end_ephys'].notna().sum() 
                                 if 'ts_end_ephys' in self.events_data.columns else 0)
            else:
                valid_ephys_start = len([t for t in ephys_start_times if t is not None])
                valid_ephys_end = (len([t for t in ephys_end_times if t is not None]) 
                                 if 'ts_end' in self.events_data.columns and len(ephys_end_times) > 0 else 0)
            
            # print(f"Synchronization complete:")
            # print(f"  ✓ Start timestamps: {valid_ephys_start}/{len(self.events_data)} converted")
            # if 'ts_end' in self.events_data.columns:
            #     print(f"  ✓ End timestamps: {valid_ephys_end}/{len(self.events_data)} converted")
            
            # if create_new_columns:
            #     print(f"  ✓ New columns created: ts_start_ephys, ts_end_ephys")
            # else:
            #     print(f"  ✓ Original timestamp columns updated")
            
            return True
            
        except Exception as e:
            print(f"Error during timestamp synchronization: {e}")
            return False
    
    def plot_rat_interaction_heatmap(self, event_type: Optional[str] = None, 
                                   figsize: Tuple[int, int] = (10, 8), 
                                   save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create a heatmap matrix showing number of events for each pair of rats.
        
        Args:
            event_type: Optional event type to filter for (abbreviation or full name)
            figsize: Figure size as (width, height)
            save_path: Optional path to save the plot
        """
        if self.events_data is None:
            print("No events data loaded. Call load_events() first.")
            return
        
        # Get working data - filter by event type if specified
        if event_type:
            data = self.get_events_by_type(event_type)
            if data is None:
                return
            title_suffix = f" - {self.decode_behavior_type(event_type)}"
        else:
            data = self.events_data
            title_suffix = " - All Events"
        
        # Get all rat identities
        rats = self.get_available_rats()
        if len(rats) < 2:
            print("Need at least 2 rats for interaction heatmap")
            return
        
        # Create interaction matrix
        interaction_matrix = pd.DataFrame(0, index=rats, columns=rats)
        
        # Use only initiator-victim pairs for faster vectorized counting
        if 'initiator' in data.columns and 'victim' in data.columns:
            # Get valid interactions (drop rows with NaN values)
            interactions = data[['initiator', 'victim']].dropna()
            
            # Filter to only include rats that are in our rats list (vectorized filtering)
            valid_mask = (interactions['initiator'].isin(rats)) & (interactions['victim'].isin(rats))
            valid_interactions = interactions[valid_mask]
            
            if len(valid_interactions) > 0:
                # Use pandas crosstab for highly efficient vectorized counting
                # Count initiator → victim interactions
                interaction_counts = pd.crosstab(
                    valid_interactions['initiator'], 
                    valid_interactions['victim'], 
                    dropna=False
                )
                
                # Count victim → initiator interactions (reverse direction for symmetry)
                reverse_interaction_counts = pd.crosstab(
                    valid_interactions['victim'], 
                    valid_interactions['initiator'], 
                    dropna=False
                )
                
                # Reindex both matrices to ensure all rats are represented with proper alignment
                interaction_counts = interaction_counts.reindex(
                    index=rats, 
                    columns=rats, 
                    fill_value=0
                )
                
                reverse_interaction_counts = reverse_interaction_counts.reindex(
                    index=rats, 
                    columns=rats, 
                    fill_value=0
                )
                
                # Add both directions to create symmetric matrix
                symmetric_counts = interaction_counts.add(reverse_interaction_counts, fill_value=0)
                
                # Add to the interaction matrix
                interaction_matrix = interaction_matrix.add(symmetric_counts, fill_value=0)
        
        # Create the heatmap
        plt.figure(figsize=figsize)
        
        # Use a color map that works well for count data
        sns.heatmap(interaction_matrix, 
                    annot=True, 
                    fmt='d', 
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Number of Events'},
                    square=True)
        
        plt.title(f'Rat Interaction Matrix{title_suffix}\nSession: {self.data_manager.session_id}')
        plt.xlabel('Target Rat')
        plt.ylabel('Source Rat')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        plt.show()
    
    def plot_rat_behavior_heatmap(self, rat_id: str, 
                                figsize: Tuple[int, int] = (12, 8),
                                save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create a heatmap showing number of events of each behavior type for a specific rat with other rats.
        
        Args:
            rat_id: Rat identifier (e.g., "rat616" or "616")
            figsize: Figure size as (width, height)
            save_path: Optional path to save the plot
        """
        if self.events_data is None:
            print("No events data loaded. Call load_events() first.")
            return
        
        # Normalize rat ID
        if not rat_id.startswith('rat'):
            rat_id = f"rat{rat_id}"
        
        # Get events involving this rat
        rat_events = self.get_events_by_rat(rat_id, 'any')
        if rat_events is None:
            return
        
        # Get all behavior types and other rats
        behavior_types = self.get_available_event_types('abbreviations')
        other_rats = [r for r in self.get_available_rats() if r != rat_id]
        
        if len(behavior_types) == 0 or len(other_rats) == 0:
            print(f"Insufficient data for rat {rat_id} behavior heatmap")
            return
        
        # Create behavior-rat matrix
        behavior_matrix = pd.DataFrame(0, index=behavior_types, columns=other_rats)
        
        # Fill matrix by counting interactions
        rat_columns = ['initiator', 'victim', 'winner', 'loser']
        
        for _, event in rat_events.iterrows():
            event_type = event.get('type')
            if pd.isna(event_type) or event_type not in behavior_types:
                continue
            
            # Find other rats involved in this event
            involved_rats = set()
            for col in rat_columns:
                if col in event and pd.notna(event[col]) and event[col] != rat_id:
                    involved_rats.add(event[col])
            
            # Increment count for each involved rat
            for other_rat in involved_rats:
                if other_rat in other_rats:
                    behavior_matrix.loc[event_type, other_rat] += 1
        
        # Create the heatmap
        plt.figure(figsize=figsize)
        
        # Create full behavior names for y-axis labels
        behavior_labels = [f"{abbr} ({self.decode_behavior_type(abbr)})" for abbr in behavior_types]
        
        sns.heatmap(behavior_matrix, 
                    annot=True, 
                    fmt='d', 
                    cmap='viridis',
                    cbar_kws={'label': 'Number of Events'},
                    yticklabels=behavior_labels)
        
        plt.title(f'Behavior Pattern for {rat_id}\nSession: {self.data_manager.session_id}')
        plt.xlabel('Interaction Partner')
        plt.ylabel('Behavior Type')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Behavior heatmap saved to: {save_path}")
        
        plt.show()
    
    def __repr__(self) -> str:
        """String representation of BehavioralEventsData."""
        if self.events_data is None:
            return f"BehavioralEventsData({self.data_manager.animal_id}/{self.data_manager.session_id}, not loaded)"
        else:
            return (f"BehavioralEventsData({self.data_manager.animal_id}/{self.data_manager.session_id}, "
                   f"{len(self.events_data)} events, {len(self.metadata['event_types'])} types)")


def load_behavioral_events(data_manager: DataStorageManager, 
                         file_indices: Optional[List[int]] = None) -> Optional[BehavioralEventsData]:
    """
    Convenience function to load behavioral events data.
    
    Args:
        data_manager: DataStorageManager instance for the session
        file_indices: Optional list of file indices to load. If None, loads all files.
        
    Returns:
        BehavioralEventsData instance with loaded data, or None if loading failed
    """
    try:
        events = BehavioralEventsData(data_manager, auto_load=False)
        
        if not events.event_files:
            print(f"No behavioral event files found for session {data_manager.session_id}")
            return None
        
        success = events.load_events(file_indices)
        if not success:
            return None
        
        return events
        
    except Exception as e:
        print(f"Error loading behavioral events: {e}")
        return None


def get_behavioral_events_summary(data_manager: DataStorageManager) -> Optional[Dict]:
    """
    Get a quick summary of behavioral events without loading full data.
    
    Args:
        data_manager: DataStorageManager instance for the session
        
    Returns:
        Dictionary with summary information, or None if no files found
    """
    event_files = data_manager.get_behavioral_event_files()
    
    if not event_files:
        return None
    
    summary = {
        'session_id': data_manager.session_id,
        'animal_id': data_manager.animal_id,
        'files_found': len(event_files),
        'file_names': [f.name for f in event_files],
        'file_sizes': []
    }
    
    # Get file sizes
    for file_path in event_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            summary['file_sizes'].append(f"{size_mb:.2f} MB")
        else:
            summary['file_sizes'].append("File not found")
    
    return summary


if __name__ == "__main__":
    """
    Example usage and testing of BehavioralEventsData class.
    """
    print("=== Behavioral Events Module Example ===")
    
    try:
        # Create DataStorageManager for a test session
        animal_id = "613"
        session_id = "20241210"
        
        print(f"\n1. Creating DataStorageManager for {animal_id}/{session_id}")
        data_manager = DataStorageManager(animal_id, session_id, auto_load=True)
        
        print(f"\n2. Checking for behavioral event files:")
        event_files = data_manager.get_behavioral_event_files()
        print(f"   Found {len(event_files)} behavioral event files")
        for i, file_path in enumerate(event_files):
            print(f"   [{i}] {file_path.name}")
        
        if event_files:
            print(f"\n3. Loading behavioral events data:")
            events = BehavioralEventsData(data_manager, auto_load=True)
            print(f"   {events}")
            
            if events.events_data is not None:
                print(f"\n4. Event statistics:")
                stats = events.get_event_statistics()
                print(f"   - Total events: {stats['total_events']}")
                print(f"   - Behavior abbreviations: {stats['behavior_abbreviations']}")
                print(f"   - Behavior types: {stats['behavior_types'][:5]}...")  # Show first 5
                print(f"   - Rat identities: {stats['rat_identities'][:5]}...")  # Show first 5
                print(f"   - Columns: {len(stats['columns'])} columns")
                
                if stats.get('time_range'):
                    time_info = stats['time_range']
                    print(f"   - Time range: {time_info['min_time']:.0f} to {time_info['max_time']:.0f}")
                    if time_info.get('average_event_duration'):
                        print(f"   - Avg event duration: {time_info['average_event_duration']:.2f}s")
                
                print(f"\n5. Available behavior types:")
                behavior_types = events.get_available_event_types('both')
                for abbr, full_name in behavior_types[:8]:  # Show first 8
                    print(f"   - {abbr}: {full_name}")
                if len(behavior_types) > 8:
                    print(f"   ... and {len(behavior_types) - 8} more")
                
                print(f"\n6. Available rats:")
                rats = events.get_available_rats()
                print(f"   - {len(rats)} rats: {rats[:5]}..." if len(rats) > 5 else f"   - {rats}")
                
                # Test rat-specific filtering
                if rats:
                    test_rat = rats[0]
                    rat_events = events.get_events_by_rat(test_rat, 'any')
                    if rat_events is not None:
                        print(f"\n7. Events for {test_rat}: {len(rat_events)} events")
                        rat_summary = events.get_rat_interaction_summary(test_rat)
                        print(f"   - As initiator: {rat_summary['as_initiator']}")
                        print(f"   - As victim: {rat_summary['as_victim']}")
                
                print(f"\n8. Testing visualization functions:")
                if len(rats) >= 2:
                    print("   - Generating rat interaction heatmap...")
                    # Note: In actual use, this would display the plot
                    # events.plot_rat_interaction_heatmap()
                    
                    print("   - Generating rat behavior heatmap...")
                    # Note: In actual use, this would display the plot  
                    # events.plot_rat_behavior_heatmap(test_rat)
                    
                    print("   ✓ Visualization functions ready (plots would display in interactive environment)")
                else:
                    print("   ⚠ Need at least 2 rats for interaction visualizations")
                
                print(f"\n9. Export summary:")
                summary = events.export_summary()
                print(f"   - Summary generated with {len(summary)} sections")
        else:
            print("   No behavioral event files found - testing file discovery only")
            
            print(f"\n3. Using quick summary function:")
            summary = get_behavioral_events_summary(data_manager)
            if summary:
                print(f"   - Files found: {summary['files_found']}")
                print(f"   - File names: {summary['file_names']}")
            else:
                print("   - No summary available (no files found)")
    
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Note: This is expected if test data doesn't exist")
    
    print("\n" + "=" * 60)
    print("Example workflow with DataStorageManager:")
    print("  data_manager = DataStorageManager('613', '20251216', auto_load=True)")
    print("  events = BehavioralEventsData(data_manager)")
    print("  behavior_types = events.get_available_event_types('both')")
    print("  rats = events.get_available_rats()")
    print("  fight_events = events.get_events_by_type('fight')")
    print("  rat_events = events.get_events_by_rat('rat616', 'any')")
    print("  time_filtered = events.get_events_in_time_range(1000, 2000, overlap_mode='any')")
    print("  rat_summary = events.get_rat_interaction_summary('rat616')")
    print("  summary = events.export_summary('events_summary.json')")
    print("")
    print("Visualization examples:")
    print("  # Overall interaction heatmap")
    print("  events.plot_rat_interaction_heatmap()")
    print("  # Fight-specific interactions")
    print("  events.plot_rat_interaction_heatmap(event_type='F')")
    print("  # Behavior patterns for specific rat")
    print("  events.plot_rat_behavior_heatmap('rat616')")
    print("  # Save visualizations")
    print("  events.plot_rat_interaction_heatmap(save_path='interactions.png')")
    print("")
    print("Ephys synchronization examples:")
    print("  # Create sync manager and synchronize timestamps")
    print("  from ingestion.ephys_sync import DataSyncManager")
    print("  sync = DataSyncManager(data_manager, dio_channel=1)")
    print("  events.synchronize_with_ephys(sync, create_new_columns=True)")
    print("  # Now events have ts_start_ephys and ts_end_ephys columns")
    
    print("\\n" + "=" * 60)