"""
Behavioral Events Import and Processing Module

Provides BehavioralEventsData (a dataclass holding behavioral event records) and
load_behavioral_events() to load it from CSV file(s) on disk.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ingestion.ephys_sync import DataSyncManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BehavioralEventsData dataclass
# ---------------------------------------------------------------------------

@dataclass
class BehavioralEventsData:
    """
    Pure data container for behavioral event records.

    All I/O is handled by the standalone ``load_behavioral_events()`` function.
    This class only stores data and provides analysis methods that operate on
    the in-memory ``events_data`` DataFrame.

    Expected CSV columns: type, initiator, victim, ts_start, ts_end
    (winner/loser optional). Timestamps are Linux nanoseconds; after
    ``synchronize_with_ephys`` ts_*_ephys columns are added in seconds.
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

    session_id: str
    events_data: pd.DataFrame
    event_files: List[Path] = field(default_factory=list)
    synchronized: bool = False

    # --- analysis methods (pure computation, no I/O) -----------------------

    def get_events_by_type(self, event_type: str) -> Optional[pd.DataFrame]:
        """Return events whose ``type`` column equals the given abbreviation."""
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
        """Return abbreviation -> full-name mapping (copy)."""
        return self.BEHAVIOR_TYPES.copy()

    def decode_behavior_type(self, abbreviation: str) -> str:
        """Decode an abbreviation to its full behavior name."""
        return self.BEHAVIOR_TYPES.get(abbreviation, abbreviation)

    def get_available_event_types(self) -> List[str]:
        """Sorted list of behavior abbreviations present in the data."""
        if 'type' not in self.events_data.columns:
            return []
        return sorted(self.events_data['type'].dropna().unique().tolist())

    def get_available_rats(self) -> List[str]:
        """Sorted list of rat identifiers seen in any role column."""
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
        ``animal_of_interest``, using ephys-synchronized timestamps.

        Requires ``synchronize_with_ephys`` to have been called (ts_*_ephys columns).
        """
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
        return (
            f"BehavioralEventsData(session={self.session_id}, "
            f"n_events={len(self.events_data)})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _resolve_event_files(files: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
    """Resolve the input into a sorted list of CSV file paths."""
    if isinstance(files, (str, Path)):
        p = Path(files)
        if p.is_dir():
            return sorted(p.glob("*.csv"))
        return [p]
    return [Path(f) for f in files]


def load_behavioral_events(
    files: Union[str, Path, List[Union[str, Path]]],
    session_id: str = "unknown_session",
) -> BehavioralEventsData:
    """Load behavioral events from CSV file(s).

    Behavioral events involve multiple animals (initiator/victim/winner/loser),
    so the returned object is keyed by session, not by a single animal.

    Parameters
    ----------
    files : str, Path, or list of paths
        A single CSV path, a directory containing CSVs, or a list of CSV paths.
    session_id : str
        Session identifier (stored on the returned object).

    Returns
    -------
    BehavioralEventsData
    """
    file_list = _resolve_event_files(files)
    if not file_list:
        raise FileNotFoundError(f"No behavioral event CSV files found at {files}")

    expected = ['type', 'initiator', 'victim', 'ts_start', 'ts_end']
    frames = []
    loaded: List[Path] = []
    for idx, fp in enumerate(file_list):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            logger.warning("Error loading %s: %s", fp.name, e)
            continue

        missing = [c for c in expected if c not in df.columns]
        if missing:
            logger.warning("Missing columns in %s: %s", fp.name, missing)

        if 'type' in df.columns:
            df['behavior_full_name'] = (
                df['type'].map(BehavioralEventsData.BEHAVIOR_TYPES).fillna(df['type'])
            )
        df['source_file'] = fp.name
        df['file_index'] = idx
        frames.append(df)
        loaded.append(fp)

    if not frames:
        raise ValueError(f"Failed to load any of the {len(file_list)} event file(s)")

    events_data = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d file(s) with %d total events (session=%s)",
        len(loaded), len(events_data), session_id,
    )
    if 'type' in events_data.columns:
        logger.info("Available behavior types: %s", dict(events_data['type'].value_counts()))

    return BehavioralEventsData(
        session_id=session_id,
        events_data=events_data,
        event_files=loaded,
    )
