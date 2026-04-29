"""
Behavioral Data Visualization Module

Standalone visualization functions for behavioral event data. Each function
returns a ``matplotlib.figure.Figure``; use
:func:`video.plot_trajectory.save_visualization` to write one to disk.
"""

import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection

from video._plot_helpers import resolve_palette, setup_spatial_axes

if TYPE_CHECKING:
    from video.behavioral_events import BehavioralEventsData
    from video.tracking_import import VideoTrackingData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize_rat_id(rat_id: str) -> str:
    return rat_id if rat_id.startswith('rat') else f"rat{rat_id}"


def _type_color_map(types) -> dict:
    """Map labels to RGB colors via the shared seaborn-backed palette."""
    types = list(types)
    palette = resolve_palette(None, max(len(types), 1), default="tab10")
    return {t: palette[i] for i, t in enumerate(types)}


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_rat_interaction_heatmap(events: "BehavioralEventsData",
                                 event_type: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 8)
                                 ) -> Optional[plt.Figure]:
    """Heatmap matrix of pairwise event counts between rats.

    Args:
        events: BehavioralEventsData instance.
        event_type: Optional behavior abbreviation to filter for.
        figsize: Figure size as (width, height).
    """
    if event_type:
        data = events.get_events_by_type(event_type)
        if data is None:
            return None
        title_suffix = f" - {events.decode_behavior_type(event_type)}"
    else:
        data = events.events_data
        title_suffix = " - All Events"

    rats = events.get_available_rats()
    if len(rats) < 2:
        logger.warning("Need at least 2 rats for interaction heatmap")
        return None

    interaction_matrix = pd.DataFrame(0, index=rats, columns=rats)

    if 'initiator' in data.columns and 'victim' in data.columns:
        pairs = data[['initiator', 'victim']].dropna()
        pairs = pairs[pairs['initiator'].isin(rats) & pairs['victim'].isin(rats)]

        if len(pairs) > 0:
            forward = pd.crosstab(pairs['initiator'], pairs['victim'], dropna=False) \
                .reindex(index=rats, columns=rats, fill_value=0)
            reverse = pd.crosstab(pairs['victim'], pairs['initiator'], dropna=False) \
                .reindex(index=rats, columns=rats, fill_value=0)
            interaction_matrix = forward.add(reverse, fill_value=0).astype(int)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(interaction_matrix,
                annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Number of Events'},
                square=True, ax=ax)
    ax.set_title(f'Rat Interaction Matrix{title_suffix}\nSession: {events.session_id}')
    ax.set_xlabel('Target Rat')
    ax.set_ylabel('Source Rat')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    return fig


def plot_rat_behavior_heatmap(events: "BehavioralEventsData",
                              rat_id: str,
                              figsize: Tuple[int, int] = (12, 8)
                              ) -> Optional[plt.Figure]:
    """Heatmap of behavior-type counts for a specific rat against every other rat.

    Args:
        events: BehavioralEventsData instance.
        rat_id: Rat identifier (e.g., "rat616" or "616").
        figsize: Figure size as (width, height).
    """
    rat_id = _normalize_rat_id(rat_id)

    rat_events = events.get_events_by_rat(rat_id, 'any')
    if rat_events is None:
        return None

    behavior_types = events.get_available_event_types()
    other_rats = [r for r in events.get_available_rats() if r != rat_id]

    if not behavior_types or not other_rats:
        logger.warning("Insufficient data for rat %s behavior heatmap", rat_id)
        return None

    rat_columns = [c for c in ('initiator', 'victim', 'winner', 'loser')
                   if c in rat_events.columns]

    # Long-form: one row per (event, role) where the role-rat is some "other rat".
    # Dedupe per-event so an event doesn't double-count an opponent that shows up
    # in multiple role columns (e.g. victim==loser).
    indexed = rat_events.assign(_evt=np.arange(len(rat_events)))
    long = (indexed[['_evt', 'type', *rat_columns]]
            .melt(id_vars=['_evt', 'type'], value_vars=rat_columns, value_name='other_rat')
            .dropna(subset=['type', 'other_rat']))
    long = long[(long['other_rat'] != rat_id)
                & long['other_rat'].isin(other_rats)
                & long['type'].isin(behavior_types)]
    long = long.drop_duplicates(subset=['_evt', 'other_rat'])

    behavior_matrix = (pd.crosstab(long['type'], long['other_rat'])
                       .reindex(index=behavior_types, columns=other_rats, fill_value=0))

    fig, ax = plt.subplots(figsize=figsize)
    behavior_labels = [f"{abbr} ({events.decode_behavior_type(abbr)})"
                       for abbr in behavior_types]
    sns.heatmap(behavior_matrix,
                annot=True, fmt='d', cmap='viridis',
                cbar_kws={'label': 'Number of Events'},
                yticklabels=behavior_labels, ax=ax)
    ax.set_title(f'Behavior Pattern for {rat_id}\nSession: {events.session_id}')
    ax.set_xlabel('Interaction Partner')
    ax.set_ylabel('Behavior Type')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    return fig


def plot_behavioral_event_timeline(events: "BehavioralEventsData",
                                   rats: Optional[List[str]] = None,
                                   event_types: Optional[List[str]] = None,
                                   figsize: Tuple[int, int] = (16, 6)
                                   ) -> Optional[plt.Figure]:
    """Plot behavioral events as connected pairs of points on a timeline.

    Each event is drawn as two markers (one per animal involved) at the same
    chronological event index on the x-axis, connected by a vertical line.
    Y positions correspond to animal IDs; line/marker color encodes event type.

    Args:
        events: BehavioralEventsData instance.
        rats: Optional list of rat IDs to include (default: all rats).
        event_types: Optional list of event-type abbreviations (default: all).
        figsize: Figure size as (width, height).
    """
    data = events.events_data

    if 'initiator' not in data.columns or 'victim' not in data.columns:
        logger.warning("Events data must have 'initiator' and 'victim' columns.")
        return None

    data = data.dropna(subset=['initiator', 'victim'])

    if event_types is not None and 'type' in data.columns:
        data = data[data['type'].isin(event_types)]

    if isinstance(rats, str):
        rats = [rats]
    if rats is not None:
        rats = [_normalize_rat_id(r) for r in rats]

    all_rats = rats if rats is not None else events.get_available_rats()
    data = data[data['initiator'].isin(all_rats) & data['victim'].isin(all_rats)]

    if data.empty:
        logger.warning("No events found for the specified filters.")
        return None

    data = data.reset_index(drop=True)

    # Reorder rats: most-interactive at the center, others placed above/below
    # to minimize total connecting-line length.
    pairs = pd.DataFrame({
        'a': data[['initiator', 'victim']].min(axis=1),
        'b': data[['initiator', 'victim']].max(axis=1),
    })
    pair_counts = pairs.groupby(['a', 'b']).size().to_dict()
    pair_counts = {(a, b): n for (a, b), n in pair_counts.items()}

    total_interactions = {
        r: sum(n for (a, b), n in pair_counts.items() if r in (a, b))
        for r in all_rats
    }
    sorted_rats = sorted(all_rats, key=lambda r: total_interactions.get(r, 0), reverse=True)

    if len(sorted_rats) <= 2:
        ordered = sorted_rats
    else:
        center = sorted_rats[0]
        above: List[str] = []
        below: List[str] = []
        for rat in sorted_rats[1:]:
            top_end = above[-1] if above else center
            bot_end = below[-1] if below else center
            top_aff = pair_counts.get(tuple(sorted([rat, top_end])), 0)
            bot_aff = pair_counts.get(tuple(sorted([rat, bot_end])), 0)
            if top_aff > bot_aff or (top_aff == bot_aff and len(above) <= len(below)):
                above.append(rat)
            else:
                below.append(rat)
        ordered = list(reversed(below)) + [center] + above

    all_rats = ordered
    rat_positions = {rat: i for i, rat in enumerate(all_rats)}

    all_types = sorted(data['type'].dropna().unique()) if 'type' in data.columns else ['unknown']
    type_colors = _type_color_map(all_types)

    # Vectorized geometry: one LineCollection + one scatter call.
    xs = np.arange(len(data))
    y_init = data['initiator'].map(rat_positions).to_numpy()
    y_vic = data['victim'].map(rat_positions).to_numpy()
    if 'type' in data.columns:
        ev_types = data['type'].fillna('unknown').to_numpy()
    else:
        ev_types = np.full(len(data), 'unknown')
    colors = np.array([type_colors.get(t, (0.5, 0.5, 0.5)) for t in ev_types])

    fig, ax = plt.subplots(figsize=figsize)

    segments = np.stack([np.column_stack([xs, y_init]),
                         np.column_stack([xs, y_vic])], axis=1)  # (N, 2, 2)
    ax.add_collection(LineCollection(segments, colors=colors,
                                     linewidths=1.0, alpha=0.7))
    ax.scatter(np.concatenate([xs, xs]),
               np.concatenate([y_init, y_vic]),
               c=np.concatenate([colors, colors]),
               s=20, zorder=3, alpha=0.9)
    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.set_ylim(-0.5, len(all_rats) - 0.5)

    ax.set_yticks(list(rat_positions.values()))
    ax.set_yticklabels(list(rat_positions.keys()))
    ax.set_xlabel('Event Index (chronological)')
    ax.set_ylabel('Animal ID')
    ax.set_title(f'Behavioral Event Timeline\nSession: {events.session_id}')

    legend_handles = [
        mlines.Line2D([], [], color=type_colors[t], linewidth=2,
                      label=f"{t} ({events.decode_behavior_type(t)})")
        for t in all_types
    ]
    ax.legend(handles=legend_handles, title='Event Type',
              bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

    fig.tight_layout()
    return fig


def plot_events_on_trajectory(events: "BehavioralEventsData",
                              tracking: "VideoTrackingData",
                              animal_id: str,
                              event_type: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 10),
                              marker_size: int = 80
                              ) -> Optional[plt.Figure]:
    """Plot an animal's trajectory with behavioral-event markers colored by opponent.

    Each event involving *animal_id* is placed at the animal's tracked position
    at the event start. Marker color encodes the opponent identity.

    Args:
        events: BehavioralEventsData instance.
        tracking: VideoTrackingData instance with timestamps loaded.
        animal_id: Animal identifier (e.g., '631' or 'rat631').
        event_type: Optional behavior-type abbreviation to filter (e.g., 'F').
        figsize: Figure size as (width, height).
        marker_size: Scatter marker size for event points.
    """
    animal_id_full = _normalize_rat_id(animal_id)

    # VideoTrackingData.get_object_data already handles substring fallback,
    # so passing 'rat631' or '631' will both resolve correctly when the
    # tracking object is keyed under either form.
    trajectory = tracking.get_object_trajectory(animal_id_full)
    if trajectory is None:
        trajectory = tracking.get_object_trajectory(animal_id_full.replace('rat', ''))
    if trajectory is None or 'timestamps' not in trajectory.columns:
        logger.warning("Could not resolve trajectory for '%s'. Available: %s",
                       animal_id, tracking.get_object_names())
        return None

    ts = trajectory['timestamps'].to_numpy()
    x = trajectory['center_x'].to_numpy()
    y = trajectory['center_y'].to_numpy()

    rat_events = events.get_events_by_rat(animal_id_full, 'any')
    if rat_events is None or rat_events.empty:
        logger.warning("No events found for %s", animal_id_full)
        return None

    if event_type is not None and 'type' in rat_events.columns:
        rat_events = rat_events[rat_events['type'] == event_type]
        if rat_events.empty:
            logger.warning("No '%s' events found for %s", event_type, animal_id_full)
            return None

    if 'ts_start' in rat_events.columns:
        event_ts = rat_events['ts_start'].to_numpy()
    elif 'ts_start_ephys' in rat_events.columns:
        event_ts = rat_events['ts_start_ephys'].to_numpy()
    else:
        logger.warning("No timestamp columns found in events data.")
        return None

    opponents = np.where(
        rat_events['initiator'].to_numpy() == animal_id_full,
        rat_events['victim'].fillna('unknown').to_numpy(),
        rat_events['initiator'].fillna('unknown').to_numpy(),
    )

    # Snap each event timestamp to the nearest tracking frame.
    idx_nearest = np.clip(np.searchsorted(ts, event_ts, side='left'), 0, len(ts) - 1)
    prev_idx = np.clip(idx_nearest - 1, 0, len(ts) - 1)
    closer_prev = np.abs(ts[prev_idx] - event_ts) < np.abs(ts[idx_nearest] - event_ts)
    idx_nearest[closer_prev] = prev_idx[closer_prev]

    event_x = x[idx_nearest]
    event_y = y[idx_nearest]

    unique_opponents = sorted(set(opponents))
    opp_colors = _type_color_map(unique_opponents)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color='gray', linewidth=0.5, alpha=0.4, zorder=1)
    for opp in unique_opponents:
        mask = opponents == opp
        ax.scatter(event_x[mask], event_y[mask],
                   s=marker_size, color=opp_colors[opp],
                   label=opp, edgecolors='black', linewidths=0.5,
                   zorder=3, alpha=0.85)

    type_label = f" ({events.decode_behavior_type(event_type)})" if event_type else ""
    ax.set_title(f'{animal_id_full} trajectory with behavioral events{type_label}\n'
                 f'Session: {events.session_id}')
    ax.legend(title='Opponent', bbox_to_anchor=(1.01, 1), loc='upper left')

    setup_spatial_axes(ax)
    fig.tight_layout()
    return fig
