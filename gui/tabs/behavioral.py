import streamlit as st

from gui.loaders import get_synced_behavior, get_tracking_data
from gui.plotting import show_fig
from gui.state import SessionKey
from gui.widgets import plot_picker
from video.behavioral_visualization import (
    plot_behavioral_event_timeline,
    plot_events_on_trajectory,
    plot_rat_behavior_heatmap,
    plot_rat_interaction_heatmap,
)


PLOT_TYPES = [
    "Interaction Heatmap",
    "Per-Rat Heatmap",
    "Event Timeline",
    "Events on Trajectory",
]


def render(key: SessionKey) -> None:
    events = get_synced_behavior(*key.as_loader_args())
    if events is None:
        st.warning("No behavioral event data available for this session.")
        return

    plot_type = plot_picker("Plot type", PLOT_TYPES, key="behavioral_plot")
    types = list(events.BEHAVIOR_TYPES.keys())

    if plot_type == "Interaction Heatmap":
        etype = st.selectbox("Event type", ["All"] + types)
        fig = plot_rat_interaction_heatmap(
            events, event_type=None if etype == "All" else etype
        )
    elif plot_type == "Per-Rat Heatmap":
        rat = st.selectbox("Rat", events.get_available_rats())
        fig = plot_rat_behavior_heatmap(events, rat_id=rat)
    elif plot_type == "Event Timeline":
        fig = plot_behavioral_event_timeline(events)
    elif plot_type == "Events on Trajectory":
        tracking = get_tracking_data(*key.as_loader_args())
        if tracking is None:
            st.warning("No tracking data available for this session.")
            return
        etype = st.selectbox("Event type", ["All"] + types)
        fig = plot_events_on_trajectory(
            events,
            tracking,
            animal_id=key.animal_id,
            event_type=None if etype == "All" else etype,
        )
    else:
        return

    show_fig(fig)
