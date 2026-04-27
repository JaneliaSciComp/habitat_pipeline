"""
Behavioral Events Import and Processing Module

Loads behavioral event CSV files discovered through DataStorageManager and
provides filtering, opponent-label extraction, and ephys-time synchronization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ingestion.data_paths import DataStorageManager
from ingestion.ephys_sync import DataSyncManager


@dataclass
class BehavioralEventsData:
    """
    Behavioral events for one session.

    Expected CSV columns: type, initiator, victim, ts_start, ts_end
    (winner/loser optional). Timestamps are Linux nanoseconds; after
    `synchronize_with_ephys` ts_*_ephys columns are added in seconds.
    """

    BEHAVIOR_TYPES: ClassVar[Dict[str, str]] = {
        'F': 'fight',
        'CO': 'confrontation',
        'C': 'chase',
        'R': 'rob food',
        'FM': 'food_move',
        'I': 'introduce',
        'UD': 'unlock_door',
        'FD': 'food_delivery',
        'P': 'playful fight',
        'RW': 'rob wood block',
        'S': 'share',
        'H': 'harvest food',
        'SN': 'sniff',
        'EC': 'encounter',
        'HD': 'huddle',
        'G': 'allogrooming',
        'PO': 'policing',
        'D': 'defense',
        'PS': 'push',
    }

    data_manager: DataStorageManager
    auto_load: bool = True

    event_files: List[Path] = field(default_factory=list, init=False)
    events_data: Optional[pd.DataFrame] = field(default=None, init=False)
    synchronized: bool = field(default=False, init=False)

    def __post_init__(self):
        if not isinstance(self.data_manager, DataStorageManager):
            raise TypeError("data_manager must be a DataStorageManager instance")

        self.event_files = self.data_manager.get_behavioral_event_files()

        if self.auto_load and self.event_files:
            self.load_events()

    def load_events(self) -> bool:
        """Load all discovered CSVs into a single DataFrame."""
        if not self.event_files:
            print("No behavioral event files found for this session")
            return False

        frames = []
        for idx, file_path in enumerate(self.event_files):
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"  ✗ Error loading {file_path.name}: {e}")
                continue

            expected = ['type', 'initiator', 'victim', 'ts_start', 'ts_end']
            missing = [c for c in expected if c not in df.columns]
            if missing:
                print(f"  ⚠ Warning: Missing columns in {file_path.name}: {missing}")

            if 'type' in df.columns:
                df['behavior_full_name'] = df['type'].map(self.BEHAVIOR_TYPES).fillna(df['type'])
            df['source_file'] = file_path.name
            df['file_index'] = idx
            frames.append(df)

        if not frames:
            print("Failed to load any event files")
            return False

        self.events_data = pd.concat(frames, ignore_index=True)
        print(f"Successfully loaded {len(frames)} file(s) with {len(self.events_data)} total events")
        if 'type' in self.events_data.columns:
            print(f"✓ Available behavior types: {dict(self.events_data['type'].value_counts())}")
        return True

    def get_events_by_type(self, event_type: str) -> Optional[pd.DataFrame]:
        """Return events whose `type` column equals the given abbreviation."""
        if self.events_data is None:
            print("No events data loaded. Call load_events() first.")
            return None
        if 'type' not in self.events_data.columns:
            print("No 'type' column found in behavioral events data")
            return None

        filtered = self.events_data[self.events_data['type'] == event_type].copy()
        if len(filtered) == 0:
            print(f"No events found for type: {event_type}")
            return None
        return filtered

    def get_events_by_rat(self, rat_id: str, role: str = 'any') -> Optional[pd.DataFrame]:
        """Return events involving a specific rat in any or a given role."""
        if self.events_data is None:
            print("No events data loaded. Call load_events() first.")
            return None

        if not rat_id.startswith('rat'):
            rat_id = f"rat{rat_id}"

        rat_columns = ['initiator', 'victim', 'winner', 'loser']
        available = [c for c in rat_columns if c in self.events_data.columns]
        if not available:
            print("No rat identity columns found in data")
            return None

        if role == 'any':
            mask = pd.concat(
                [self.events_data[c] == rat_id for c in available], axis=1
            ).any(axis=1)
            filtered = self.events_data[mask].copy()
        elif role in available:
            filtered = self.events_data[self.events_data[role] == rat_id].copy()
        else:
            print(f"Role '{role}' not found. Available roles: {available}")
            return None

        if len(filtered) == 0:
            print(f"No events found for rat {rat_id} in role {role}")
            return None

        print(f"Found {len(filtered)} events for rat {rat_id} (role: {role})")
        return filtered

    def get_behavior_type_mapping(self) -> Dict[str, str]:
        """Return abbreviation → full-name mapping (copy)."""
        return self.BEHAVIOR_TYPES.copy()

    def decode_behavior_type(self, abbreviation: str) -> str:
        """Decode an abbreviation to its full behavior name."""
        return self.BEHAVIOR_TYPES.get(abbreviation, abbreviation)

    def get_available_event_types(self) -> List[str]:
        """Sorted list of behavior abbreviations present in the data."""
        if self.events_data is None or 'type' not in self.events_data.columns:
            return []
        return sorted(self.events_data['type'].dropna().unique().tolist())

    def get_available_rats(self) -> List[str]:
        """Sorted list of rat identifiers seen in any role column."""
        if self.events_data is None:
            return []
        rat_columns = ['initiator', 'victim', 'winner', 'loser']
        ids = set()
        for col in rat_columns:
            if col in self.events_data.columns:
                ids.update(self.events_data[col].dropna().unique())
        return sorted(ids)

    def extract_opponent_labels(
        self,
        animal_of_interest: str,
        behavior_type: Optional[str] = None,
        min_events_per_class: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract (start_times, end_times, opponent_labels) for events involving
        `animal_of_interest`, using ephys-synchronized timestamps.

        Requires `synchronize_with_ephys` to have been called (ts_*_ephys columns).
        """
        if self.events_data is None:
            raise ValueError("No events data loaded. Call load_events() first.")

        df = self.events_data
        if behavior_type is not None:
            df = df[df['type'] == behavior_type]

        if 'initiator' not in df.columns or 'victim' not in df.columns:
            raise ValueError("Behavioral data must have 'initiator' and 'victim' columns")

        if not animal_of_interest.startswith('rat'):
            animal_of_interest = f"rat{animal_of_interest}"

        is_init = df['initiator'] == animal_of_interest
        is_vict = df['victim'] == animal_of_interest
        df = df[is_init | is_vict]

        if len(df) == 0:
            return np.array([]), np.array([]), np.array([])

        if 'ts_start_ephys' not in df.columns:
            raise ValueError("No ephys-synchronized timestamp columns found in behavioral data")

        event_start_times = df['ts_start_ephys'].values
        event_end_times = df['ts_end_ephys'].values
        opponent_labels = np.where(
            df['initiator'].values == animal_of_interest,
            df['victim'].values,
            df['initiator'].values,
        )

        if min_events_per_class > 1:
            unique_opponents, counts = np.unique(opponent_labels, return_counts=True)
            valid_opponents = unique_opponents[counts >= min_events_per_class]
            if len(valid_opponents) == 0:
                return np.array([]), np.array([]), np.array([])
            valid_mask = np.isin(opponent_labels, valid_opponents)
            event_start_times = event_start_times[valid_mask]
            event_end_times = event_end_times[valid_mask]
            opponent_labels = opponent_labels[valid_mask]

        print(f"✓ Found {len(event_start_times)} {behavior_type} events with opponent labels")
        print(f"✓ Unique opponents: {np.unique(opponent_labels)}")
        return event_start_times, event_end_times, opponent_labels

    def synchronize_with_ephys(
        self,
        sync_manager: DataSyncManager,
        create_new_columns: bool = True,
    ) -> bool:
        """
        Convert ts_start / ts_end (Linux ns) to ephys time (seconds).

        With create_new_columns=True (default) writes ts_start_ephys /
        ts_end_ephys; otherwise overwrites the originals.
        """
        if self.events_data is None:
            print("No events data loaded. Call load_events() first.")
            return False
        if 'ts_start' not in self.events_data.columns:
            print("No 'ts_start' column found in behavioral events data")
            return False

        try:
            self._convert_timestamp_column('ts_start', create_new_columns, sync_manager)
            if 'ts_end' in self.events_data.columns:
                self._convert_timestamp_column('ts_end', create_new_columns, sync_manager)
            self.synchronized = True
            return True
        except Exception as e:
            print(f"Error during timestamp synchronization: {e}")
            return False

    def _convert_timestamp_column(
        self,
        col: str,
        create_new_columns: bool,
        sync_manager: DataSyncManager,
    ) -> None:
        valid_mask = self.events_data[col].notna()
        if not valid_mask.any():
            return
        behav_seconds = self.events_data.loc[valid_mask, col].to_numpy() / 1e9
        ephys_times = sync_manager.convert_behavior_to_ephys(behav_seconds)

        target = f"{col}_ephys" if create_new_columns else col
        if create_new_columns and target not in self.events_data.columns:
            self.events_data[target] = np.nan
        self.events_data.loc[valid_mask, target] = ephys_times

    def __repr__(self) -> str:
        loc = f"{self.data_manager.animal_id}/{self.data_manager.session_id}"
        if self.events_data is None:
            return f"BehavioralEventsData({loc}, not loaded)"
        return f"BehavioralEventsData({loc}, {len(self.events_data)} events)"
