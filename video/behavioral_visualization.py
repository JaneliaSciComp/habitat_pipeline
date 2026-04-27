"""
Behavioral Data Visualization Module

Standalone visualization functions for behavioral event data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from video.behavioral_events import BehavioralEventsData
    from video.tracking_import import VideoTrackingData


def plot_rat_interaction_heatmap(events: "BehavioralEventsData",
                                 event_type: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 8),
                                 save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Create a heatmap matrix showing number of events for each pair of rats.

    Args:
        events: BehavioralEventsData instance
        event_type: Optional event type to filter for (abbreviation or full name)
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
    """
    if events.events_data is None:
        print("No events data loaded. Call load_events() first.")
        return

    # Get working data - filter by event type if specified
    if event_type:
        data = events.get_events_by_type(event_type)
        if data is None:
            return
        title_suffix = f" - {events.decode_behavior_type(event_type)}"
    else:
        data = events.events_data
        title_suffix = " - All Events"

    # Get all rat identities
    rats = events.get_available_rats()
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

    sns.heatmap(interaction_matrix,
                annot=True,
                fmt='d',
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Events'},
                square=True)

    plt.title(f'Rat Interaction Matrix{title_suffix}\nSession: {events.data_manager.session_id}')
    plt.xlabel('Target Rat')
    plt.ylabel('Source Rat')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")

    plt.show()


def plot_rat_behavior_heatmap(events: "BehavioralEventsData",
                              rat_id: str,
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Create a heatmap showing number of events of each behavior type for a specific rat with other rats.

    Args:
        events: BehavioralEventsData instance
        rat_id: Rat identifier (e.g., "rat616" or "616")
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
    """
    if events.events_data is None:
        print("No events data loaded. Call load_events() first.")
        return

    # Normalize rat ID
    if not rat_id.startswith('rat'):
        rat_id = f"rat{rat_id}"

    # Get events involving this rat
    rat_events = events.get_events_by_rat(rat_id, 'any')
    if rat_events is None:
        return

    # Get all behavior types and other rats
    behavior_types = events.get_available_event_types()
    other_rats = [r for r in events.get_available_rats() if r != rat_id]

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
    behavior_labels = [f"{abbr} ({events.decode_behavior_type(abbr)})" for abbr in behavior_types]

    sns.heatmap(behavior_matrix,
                annot=True,
                fmt='d',
                cmap='viridis',
                cbar_kws={'label': 'Number of Events'},
                yticklabels=behavior_labels)

    plt.title(f'Behavior Pattern for {rat_id}\nSession: {events.data_manager.session_id}')
    plt.xlabel('Interaction Partner')
    plt.ylabel('Behavior Type')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Behavior heatmap saved to: {save_path}")

    plt.show()


def plot_behavioral_event_timeline(events: "BehavioralEventsData",
                                   rats: Optional[List[str]] = None,
                                   event_types: Optional[List[str]] = None,
                                   figsize: Tuple[int, int] = (16, 6),
                                   save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot behavioral events as connected pairs of points on a timeline.

    Each event is drawn as two markers (one per animal involved) at the same
    chronological event index on the x-axis, connected by a vertical line.
    Y positions correspond to animal IDs; line/marker color encodes event type.

    Args:
        events: BehavioralEventsData instance
        rats: Optional list of rat IDs to include (default: all rats)
        event_types: Optional list of event type abbreviations to include (default: all)
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
    """
    if events.events_data is None:
        print("No events data loaded. Call load_events() first.")
        return

    data = events.events_data.copy()

    # Require initiator and victim columns
    if 'initiator' not in data.columns or 'victim' not in data.columns:
        print("Events data must have 'initiator' and 'victim' columns.")
        return

    # Drop events with missing animal info
    data = data.dropna(subset=['initiator', 'victim'])

    # Filter by event types if specified
    if event_types is not None and 'type' in data.columns:
        data = data[data['type'].isin(event_types)]

    # Normalize rats argument: accept a single string or a list; add 'rat' prefix if missing
    if isinstance(rats, str):
        rats = [rats]
    if rats is not None:
        rats = [r if r.startswith('rat') else f"rat{r}" for r in rats]

    # Filter to only rows where both animals are in the requested rats list
    all_rats = rats if rats is not None else events.get_available_rats()
    data = data[data['initiator'].isin(all_rats) & data['victim'].isin(all_rats)]

    if data.empty:
        print("No events found for the specified filters.")
        return

    # Reset index to get chronological event indices
    data = data.reset_index(drop=True)

    # Reorder rats: most interactive in the middle, others placed above/below
    # to minimize total connecting-line lengths.
    pair_counts: dict = {}
    for _, row in data.iterrows():
        pair = tuple(sorted([row['initiator'], row['victim']]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    total_interactions = {r: sum(v for (a, b), v in pair_counts.items() if r in (a, b)) for r in all_rats}
    sorted_rats = sorted(all_rats, key=lambda r: total_interactions.get(r, 0), reverse=True)

    if len(sorted_rats) <= 2:
        ordered = sorted_rats
    else:
        center = sorted_rats[0]
        above = []  # rats above center, stored closest-first
        below = []  # rats below center, stored closest-first
        for rat in sorted_rats[1:]:
            top_end = above[-1] if above else center
            bot_end = below[-1] if below else center
            top_affinity = pair_counts.get(tuple(sorted([rat, top_end])), 0)
            bot_affinity = pair_counts.get(tuple(sorted([rat, bot_end])), 0)
            # Prefer higher affinity; use shorter side as tiebreaker
            if top_affinity > bot_affinity or (top_affinity == bot_affinity and len(above) <= len(below)):
                above.append(rat)
            else:
                below.append(rat)
        # Y order (low to high): farthest-below ... center ... farthest-above
        ordered = list(reversed(below)) + [center] + above

    all_rats = ordered

    # Build y-axis mapping: each rat gets an integer position
    rat_positions = {rat: i for i, rat in enumerate(all_rats)}

    # Build color mapping for event types
    all_types = sorted(data['type'].dropna().unique()) if 'type' in data.columns else ['unknown']
    palette = plt.get_cmap('tab10')
    type_colors = {t: palette(i % 10) for i, t in enumerate(all_types)}

    fig, ax = plt.subplots(figsize=figsize)

    for idx, row in data.iterrows():
        initiator = row['initiator']
        victim = row['victim']
        event_type = row.get('type', 'unknown') if 'type' in data.columns else 'unknown'
        color = type_colors.get(event_type, 'gray')

        y_init = rat_positions[initiator]
        y_vic = rat_positions[victim]

        # Draw connecting line
        ax.plot([idx, idx], [y_init, y_vic], color=color, linewidth=1.0, alpha=0.7)

        # Draw markers for both animals
        ax.scatter([idx, idx], [y_init, y_vic], color=color, s=20, zorder=3, alpha=0.9)

    # Y axis: rat labels
    ax.set_yticks(list(rat_positions.values()))
    ax.set_yticklabels(list(rat_positions.keys()))
    ax.set_xlabel('Event Index (chronological)')
    ax.set_ylabel('Animal ID')
    ax.set_title(f'Behavioral Event Timeline\nSession: {events.data_manager.session_id}')

    # Legend for event types
    legend_handles = [
        mlines.Line2D([], [], color=type_colors[t], linewidth=2,
                      label=f"{t} ({events.decode_behavior_type(t)})")
        for t in all_types
    ]
    ax.legend(handles=legend_handles, title='Event Type',
              bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved to: {save_path}")

    plt.show()


def plot_events_on_trajectory(events: "BehavioralEventsData",
                              tracking: "VideoTrackingData",
                              animal_id: str,
                              event_type: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 10),
                              marker_size: int = 80,
                              save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot an animal's trajectory with behavioral event markers colored by opponent.

    Each event involving *animal_id* is placed at the animal's tracked position
    at the time of the event start.  Marker color encodes the opponent identity.

    Args:
        events: BehavioralEventsData instance (must be loaded)
        tracking: VideoTrackingData instance with timestamps loaded
        animal_id: Animal identifier (e.g., '631' or 'rat631')
        event_type: Optional behavior type abbreviation to filter (e.g., 'F')
        figsize: Figure size as (width, height)
        marker_size: Scatter marker size for event points
        save_path: Optional path to save the plot
    """
    if events.events_data is None:
        print("No events data loaded.")
        return

    # Normalize animal_id
    if not animal_id.startswith('rat'):
        animal_id_full = f"rat{animal_id}"
    else:
        animal_id_full = animal_id

    # --- Resolve tracking object name ---
    object_names = tracking.get_object_names()
    # Try exact match, then partial match on the numeric part
    track_name = None
    numeric_part = animal_id_full.replace('rat', '')
    for name in object_names:
        if name == animal_id_full or name == numeric_part:
            track_name = name
            break
    if track_name is None:
        # Fuzzy: look for the numeric id anywhere in the object name
        for name in object_names:
            if numeric_part in name:
                track_name = name
                break
    if track_name is None:
        print(f"Could not find tracking object for '{animal_id}'. "
              f"Available: {object_names}")
        return

    # Get trajectory
    trajectory = tracking.get_object_trajectory(track_name)
    if trajectory is None or 'timestamps' not in trajectory.columns:
        print("Trajectory or timestamps not available for this animal.")
        return

    ts = trajectory['timestamps'].values
    x = trajectory['center_x'].values
    y = trajectory['center_y'].values

    # --- Get events for this animal ---
    rat_events = events.get_events_by_rat(animal_id_full, 'any')
    if rat_events is None or rat_events.empty:
        print(f"No events found for {animal_id_full}")
        return

    if event_type is not None and 'type' in rat_events.columns:
        rat_events = rat_events[rat_events['type'] == event_type]
        if rat_events.empty:
            print(f"No '{event_type}' events found for {animal_id_full}")
            return

    # Determine timestamp column (prefer raw Linux timestamps for matching tracking)
    if 'ts_start' in rat_events.columns:
        event_ts = rat_events['ts_start'].values
    elif 'ts_start_ephys' in rat_events.columns:
        event_ts = rat_events['ts_start_ephys'].values
    else:
        print("No timestamp columns found in events data.")
        return

    # Identify opponent for each event
    opponents = []
    for _, row in rat_events.iterrows():
        if row.get('initiator') == animal_id_full:
            opponents.append(row.get('victim', 'unknown'))
        else:
            opponents.append(row.get('initiator', 'unknown'))
    opponents = np.array(opponents)

    # Map event timestamps to nearest tracking frame
    idx_nearest = np.searchsorted(ts, event_ts, side='left')
    idx_nearest = np.clip(idx_nearest, 0, len(ts) - 1)
    # Refine: check if previous index is closer
    prev_idx = np.clip(idx_nearest - 1, 0, len(ts) - 1)
    closer_prev = np.abs(ts[prev_idx] - event_ts) < np.abs(ts[idx_nearest] - event_ts)
    idx_nearest[closer_prev] = prev_idx[closer_prev]

    event_x = x[idx_nearest]
    event_y = y[idx_nearest]

    # --- Plot ---
    unique_opponents = sorted(set(opponents))
    palette = plt.get_cmap('tab10')
    opp_colors = {opp: palette(i % 10) for i, opp in enumerate(unique_opponents)}

    fig, ax = plt.subplots(figsize=figsize)

    # Trajectory
    ax.plot(x, y, color='gray', linewidth=0.5, alpha=0.4, zorder=1)

    # Event markers
    for opp in unique_opponents:
        mask = opponents == opp
        ax.scatter(event_x[mask], event_y[mask],
                   s=marker_size, color=opp_colors[opp],
                   label=opp, edgecolors='black', linewidths=0.5,
                   zorder=3, alpha=0.85)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    type_label = f" ({events.decode_behavior_type(event_type)})" if event_type else ""
    ax.set_title(f'{animal_id_full} trajectory with behavioral events{type_label}\n'
                 f'Session: {events.data_manager.session_id}')
    ax.legend(title='Opponent', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()
