"""
Behavioral Events Import and Processing Module

Provides BehavioralEventsData (a dataclass holding behavioral event records) and
load_behavioral_events() to load it from CSV file(s) on disk.

Every ``extract_*_labels`` method follows the same pipeline: restrict to the
requested behaviour type(s) → keep events where the focal animal fills one of
the relevant roles → keep events inside ``ephys_time_range`` → label each
event → drop classes with too few events. The shared steps live in
:meth:`BehavioralEventsData._focal_events`.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ingestion._conventions import (
    RAT_ID_COLUMNS, canonical_rat_id, normalize_object_name, uses_name_normalization,
)
from ingestion.ephys_sync import DataSyncManager

logger = logging.getLogger(__name__)

#: ``(start_times, end_times, labels)`` as returned by every ``extract_*_labels``.
LabelArrays = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _no_events() -> LabelArrays:
    return np.array([]), np.array([]), np.array([])


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

    # --- lookups -------------------------------------------------------------

    def get_events_by_type(self, event_type: str) -> Optional[pd.DataFrame]:
        """Events whose ``type`` equals *event_type*, or None if there are none."""
        if 'type' not in self.events_data.columns:
            logger.warning("No 'type' column found in behavioral events data")
            return None

        filtered = self.events_data[self.events_data['type'] == event_type].copy()
        if len(filtered) == 0:
            logger.warning("No events found for type: %s", event_type)
            return None
        return filtered

    def get_events_by_rat(self, rat_id: str, role: str = 'any') -> Optional[pd.DataFrame]:
        """Events involving *rat_id* in any role (or the given one), or None if there are none."""
        rat_id = canonical_rat_id(rat_id)
        available = [c for c in RAT_ID_COLUMNS if c in self.events_data.columns]
        if not available:
            logger.warning("No rat identity columns found in data")
            return None

        if role == 'any':
            mask = self.events_data[available].eq(rat_id).any(axis=1)
        elif role in available:
            mask = self.events_data[role] == rat_id
        else:
            logger.warning("Role '%s' not found. Available roles: %s", role, available)
            return None

        filtered = self.events_data[mask].copy()
        if len(filtered) == 0:
            logger.warning("No events found for rat %s in role %s", rat_id, role)
            return None
        logger.info("Found %d events for rat %s (role: %s)", len(filtered), rat_id, role)
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
        ids = set()
        for col in RAT_ID_COLUMNS:
            if col in self.events_data.columns:
                ids.update(self.events_data[col].dropna().unique())
        return sorted(ids)

    def print_summary(self) -> None:
        """Print event counts per behavior type and per animal.

        Per-animal counts are "any role" — one event where a rat appears in
        two role columns at once (e.g. initiator and winner) counts once,
        matching ``get_events_by_rat(rat_id, role='any')``.
        """
        df = self.events_data
        print(f"=== Behavioral Events Summary - session {self.session_id} ===")
        print(f"Total events: {len(df)}")

        print("\nEvents per type:")
        if 'type' in df.columns:
            for btype, count in df['type'].value_counts().items():
                full_name = self.BEHAVIOR_TYPES.get(btype, btype)
                print(f"  {btype} ({full_name}): {count}")
        else:
            print("  No 'type' column found")

        print("\nEvents per animal (any role - initiator/victim/winner/loser):")
        rat_columns = [c for c in RAT_ID_COLUMNS if c in df.columns]
        if rat_columns:
            for rat_id in self.get_available_rats():
                print(f"  {rat_id}: {int(df[rat_columns].eq(rat_id).any(axis=1).sum())}")
        else:
            print("  No rat identity columns found")

    # --- shared filtering steps --------------------------------------------

    @staticmethod
    def _filter_by_behavior_type(
        df: pd.DataFrame,
        behavior_type: Optional[Union[str, Iterable[str]]],
    ) -> pd.DataFrame:
        """Filter ``df['type']`` by one abbreviation, a list of them, or no-op.

        ``None`` keeps every type; a single string matches exactly; any
        other iterable (list, tuple, set, ...) keeps rows matching *any* of
        the given types.
        """
        if behavior_type is None:
            return df
        if isinstance(behavior_type, str):
            return df[df['type'] == behavior_type]
        return df[df['type'].isin(behavior_type)]

    @staticmethod
    def _filter_to_ephys_range(
        df: pd.DataFrame,
        ephys_time_range: Optional[Tuple[float, float]],
    ) -> pd.DataFrame:
        """Drop rows whose ``(ts_start_ephys, ts_end_ephys)`` falls outside
        ``ephys_time_range``. No-op when ``ephys_time_range`` is ``None``.

        An event timestamped past the point where the recorded cells stop
        firing has no real spike data to decode from regardless of its
        label — see the ``duration_disagrees_with_window`` gotcha in
        CLAUDE.md, where a stale cached ``duration_seconds`` masked exactly
        this for rat631/20251216.
        """
        if ephys_time_range is None:
            return df
        lo, hi = ephys_time_range
        in_range = (df['ts_start_ephys'] >= lo) & (df['ts_end_ephys'] <= hi)
        n_dropped = int((~in_range).sum())
        if n_dropped > 0:
            print(f"[OK] Dropping {n_dropped}/{len(df)} events outside "
                  f"ephys_time_range [{lo:.1f}, {hi:.1f}]s")
        return df[in_range]

    @staticmethod
    def _require_columns(df: pd.DataFrame, first: str, second: str) -> None:
        if first not in df.columns or second not in df.columns:
            raise ValueError(f"Behavioral data must have '{first}' and '{second}' columns")

    def _focal_events(
        self,
        df: pd.DataFrame,
        animal: str,
        roles: Sequence[str],
        ephys_time_range: Optional[Tuple[float, float]],
    ) -> pd.DataFrame:
        """Rows of *df* where *animal* fills one of *roles*, inside *ephys_time_range*.

        Requires ``synchronize_with_ephys`` to have run (``ts_*_ephys``
        columns) unless no row involves *animal*.
        """
        df = df[df[list(roles)].eq(animal).any(axis=1)]
        if len(df) == 0:
            return df
        if 'ts_start_ephys' not in df.columns:
            raise ValueError("No ephys-synchronized timestamp columns found in behavioral data")
        return self._filter_to_ephys_range(df, ephys_time_range)

    @staticmethod
    def _opponents(df: pd.DataFrame, animal: str) -> np.ndarray:
        """For each initiator/victim event, the animal on the other side of *animal*."""
        return np.where(df['initiator'].values == animal, df['victim'].values, df['initiator'].values)

    # --- label extraction --------------------------------------------------

    def extract_opponent_labels(
        self,
        animal_of_interest: str,
        behavior_type: Optional[Union[str, Iterable[str]]] = None,
        min_events_per_class: int = 5,
        ephys_time_range: Optional[Tuple[float, float]] = None,
    ) -> LabelArrays:
        """
        Extract (start_times, end_times, opponent_labels) for events involving
        ``animal_of_interest``, using ephys-synchronized timestamps.

        Requires ``synchronize_with_ephys`` to have been called (ts_*_ephys columns).

        ``behavior_type`` may be a single abbreviation (``'EC'``), a list of
        them (``['EC', 'F']`` — any matching event is kept), or ``None`` for
        every type.

        ``ephys_time_range``, if given as ``(lo, hi)`` seconds — e.g. from
        ``KilosortData.quality_ephys_window()`` — drops any event whose
        ``ts_start_ephys``/``ts_end_ephys`` falls outside that range before
        ``min_events_per_class`` is applied, so an opponent with too few
        in-range events is correctly excluded rather than passed through with
        events that have no real spike data behind them (see the
        ``duration_disagrees_with_window`` gotcha in CLAUDE.md).
        """
        df = self._filter_by_behavior_type(self.events_data, behavior_type)
        self._require_columns(df, 'initiator', 'victim')
        animal = canonical_rat_id(animal_of_interest)
        df = self._focal_events(df, animal, ('initiator', 'victim'), ephys_time_range)
        if len(df) == 0:
            return _no_events()

        event_start_times = df['ts_start_ephys'].values
        event_end_times = df['ts_end_ephys'].values
        opponent_labels = self._opponents(df, animal)

        if min_events_per_class > 1:
            unique_opponents, counts = np.unique(opponent_labels, return_counts=True)
            valid_opponents = unique_opponents[counts >= min_events_per_class]
            if len(valid_opponents) == 0:
                return _no_events()
            valid_mask = np.isin(opponent_labels, valid_opponents)
            event_start_times = event_start_times[valid_mask]
            event_end_times = event_end_times[valid_mask]
            opponent_labels = opponent_labels[valid_mask]

        print(f"[OK] Found {len(event_start_times)} {behavior_type} events with opponent labels")
        print(f"[OK] Unique opponents: {np.unique(opponent_labels)}")
        return event_start_times, event_end_times, opponent_labels

    @staticmethod
    def _assign_id_groups(rat_ids: Iterable[str]) -> Dict[str, str]:
        """Map each rat id to ``'low'`` or ``'high'`` by splitting on numeric ID.

        Sorts the provided rat ids by their trailing integer suffix, assigns the
        bottom half to ``'low'`` and the top half to ``'high'``. For odd counts,
        the middle rat joins whichever group's median ID is numerically closer
        to its own ID.
        """
        ids = list(rat_ids)
        if len(ids) < 2:
            raise ValueError(
                f"Need at least 2 rats to form 2 groups, got {len(ids)}"
            )

        nums: List[int] = []
        for rid in ids:
            m = re.search(r"(\d+)$", str(rid))
            if m is None:
                raise ValueError(f"Cannot extract numeric suffix from rat id {rid!r}")
            nums.append(int(m.group(1)))

        order = np.argsort(nums)
        sorted_ids = [ids[i] for i in order]
        sorted_nums = [nums[i] for i in order]

        n = len(sorted_ids)
        half = n // 2

        if n % 2 == 0:
            low_ids = sorted_ids[:half]
            high_ids = sorted_ids[half:]
        else:
            middle_id = sorted_ids[half]
            middle_num = sorted_nums[half]
            low_median = float(np.median(sorted_nums[:half]))
            high_median = float(np.median(sorted_nums[half + 1:]))
            if abs(middle_num - low_median) <= abs(middle_num - high_median):
                low_ids = sorted_ids[:half] + [middle_id]
                high_ids = sorted_ids[half + 1:]
            else:
                low_ids = sorted_ids[:half]
                high_ids = [middle_id] + sorted_ids[half + 1:]

        mapping = {rid: 'low' for rid in low_ids}
        mapping.update({rid: 'high' for rid in high_ids})
        return mapping

    def extract_group_labels(
        self,
        animal_of_interest: str,
        behavior_type: Optional[Union[str, Iterable[str]]] = None,
        min_events_per_class: int = 5,
        ephys_time_range: Optional[Tuple[float, float]] = None,
    ) -> LabelArrays:
        """
        Extract (start_times, end_times, group_labels) where group_labels are
        ``'self'`` / ``'others'`` based on whether the opponent falls on the
        same ID-half as ``animal_of_interest``. The ID-half split (including
        ``animal_of_interest``) follows the rule documented on
        ``_assign_id_groups``.

        Mirrors ``extract_opponent_labels`` but pools opponents into two
        groups relative to ``animal_of_interest``. ``behavior_type`` accepts
        the same single-value / list / ``None`` forms. ``min_events_per_class``
        is applied at the group level: each of ``'self'`` / ``'others'``
        must have at least ``min_events_per_class`` events, otherwise empty
        arrays are returned. ``ephys_time_range`` behaves as in
        ``extract_opponent_labels``, applied before that group-level check.
        """
        df = self._filter_by_behavior_type(self.events_data, behavior_type)
        self._require_columns(df, 'initiator', 'victim')
        animal = canonical_rat_id(animal_of_interest)
        df = self._focal_events(df, animal, ('initiator', 'victim'), ephys_time_range)
        if len(df) == 0:
            return _no_events()

        event_start_times = df['ts_start_ephys'].values
        event_end_times = df['ts_end_ephys'].values
        opponent_ids = self._opponents(df, animal)

        unique_opponents = np.unique(opponent_ids)
        if len(unique_opponents) < 2:
            return _no_events()

        all_rats = sorted(set(unique_opponents.tolist()) | {animal})
        half_map = self._assign_id_groups(all_rats)
        self_half = half_map[animal]
        group_map = {
            rid: ('self' if half == self_half else 'others')
            for rid, half in half_map.items()
        }
        group_labels = np.array([group_map[op] for op in opponent_ids])

        unique_groups, counts = np.unique(group_labels, return_counts=True)
        if len(unique_groups) < 2 or int(np.min(counts)) < min_events_per_class:
            return _no_events()

        composition = {op: group_map[op] for op in unique_opponents}
        print(f"[OK] Found {len(event_start_times)} {behavior_type} events with group labels")
        print(f"[OK] Group composition (relative to {animal}): {composition}")
        return event_start_times, event_end_times, group_labels

    def extract_outcome_labels(
        self,
        animal_of_interest: str,
        behavior_type: Optional[Union[str, Iterable[str]]] = None,
        min_events_per_class: int = 5,
        ephys_time_range: Optional[Tuple[float, float]] = None,
    ) -> LabelArrays:
        """
        Extract (start_times, end_times, outcome_labels) for events where
        ``animal_of_interest`` is either the winner or the loser. Labels are
        ``'winner'`` or ``'loser'``.

        Default ``behavior_type=None`` keeps every row whose ``winner`` and
        ``loser`` columns are populated, regardless of behavior type. Pass
        ``behavior_type='F'`` (or a list, e.g. ``['F', 'CO']``) to restrict
        the scope.

        Mirrors ``extract_opponent_labels`` (including ``ephys_time_range``)
        and requires ``synchronize_with_ephys`` to have been called.
        """
        self._require_columns(self.events_data, 'winner', 'loser')
        df = self._filter_by_behavior_type(self.events_data, behavior_type)

        winner_str = df['winner'].astype('string').str.strip()
        loser_str = df['loser'].astype('string').str.strip()
        has_outcome = winner_str.notna() & loser_str.notna() & (winner_str != '') & (loser_str != '')
        df = df[has_outcome]
        if len(df) == 0:
            return _no_events()

        animal = canonical_rat_id(animal_of_interest)
        df = self._focal_events(df, animal, ('winner', 'loser'), ephys_time_range)
        if len(df) == 0:
            return _no_events()

        event_start_times = df['ts_start_ephys'].values
        event_end_times = df['ts_end_ephys'].values
        outcome_labels = np.where(df['winner'].values == animal, 'winner', 'loser')

        unique_outcomes, counts = np.unique(outcome_labels, return_counts=True)
        if len(unique_outcomes) < 2 or int(np.min(counts)) < min_events_per_class:
            return _no_events()

        scope = behavior_type if behavior_type is not None else 'aggressive'
        print(f"[OK] Found {len(event_start_times)} {scope} events with outcome labels")
        print(f"[OK] Outcome counts: {dict(zip(unique_outcomes, counts))}")
        return event_start_times, event_end_times, outcome_labels

    # --- clock synchronisation ---------------------------------------------

    def synchronize_with_ephys(
        self,
        sync_manager: DataSyncManager,
        create_new_columns: bool = True,
    ) -> bool:
        """
        Convert ts_start / ts_end (Linux ns) to ephys time (seconds).

        With create_new_columns=True (default) writes ts_start_ephys /
        ts_end_ephys; otherwise overwrites the originals. Returns False (and
        logs why) if there is nothing to convert or the conversion fails.
        """
        if 'ts_start' not in self.events_data.columns:
            logger.warning("No 'ts_start' column found in behavioral events data")
            return False

        try:
            self._convert_timestamp_column('ts_start', create_new_columns, sync_manager)
            if 'ts_end' in self.events_data.columns:
                self._convert_timestamp_column('ts_end', create_new_columns, sync_manager)
        except Exception:
            logger.exception("Error during timestamp synchronization")
            return False
        self.synchronized = True
        return True

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
        Every file given is read and concatenated, so pass only the canonical
        ``<date>/<date>_behavior_event_df.csv`` (``HZ-DATA-007``).
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

        # Cohort 9's real ids (rat4678) look like APT's rat4<ddd>, so only
        # older cohorts get the rat4635 -> rat635 rewrite.
        if uses_name_normalization(fp):
            for col in RAT_ID_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].map(
                        lambda v: normalize_object_name(v) if pd.notna(v) else v
                    )

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
