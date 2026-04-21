import matplotlib.pyplot as plt
import streamlit as st

from gui.loaders import get_behavior_data, get_sync
from video.behavioral_visualization import (
    plot_behavioral_event_timeline,
    plot_rat_behavior_heatmap,
    plot_rat_interaction_heatmap,
)


def render(animal_id: str, session_id: str, config_path):
    events = get_behavior_data(animal_id, session_id, config_path)
    if events is None:
        st.warning("No behavioral event data available for this session.")
        return

    # Sync once; guard against repeated calls since synchronize_with_ephys mutates in place
    if not getattr(events, "_gui_synced", False):
        sync = get_sync(animal_id, session_id, config_path)
        if sync is not None:
            try:
                events.synchronize_with_ephys(sync, create_new_columns=True)
            except Exception:
                pass
        events._gui_synced = True

    plot_type = st.radio(
        "Plot type",
        ["Interaction Heatmap", "Per-Rat Heatmap", "Event Timeline"],
        horizontal=True,
    )

    TYPES = list(events.BEHAVIOR_TYPES.keys())

    if plot_type == "Interaction Heatmap":
        etype = st.selectbox("Event type", ["All"] + TYPES)
        plot_rat_interaction_heatmap(events, event_type=None if etype == "All" else etype)
        st.pyplot(plt.gcf())
        plt.close()

    elif plot_type == "Per-Rat Heatmap":
        rats = events.get_available_rats()
        rat = st.selectbox("Rat", rats)
        plot_rat_behavior_heatmap(events, rat_id=rat)
        st.pyplot(plt.gcf())
        plt.close()

    elif plot_type == "Event Timeline":
        plot_behavioral_event_timeline(events)
        st.pyplot(plt.gcf())
        plt.close()
