import streamlit as st

from gui.loaders import get_tracking_data
from gui.plotting import show_fig
from gui.state import SessionKey
from gui.widgets import plot_picker
from video import plot_trajectory as pt


PLOT_TYPES = [
    "Individual Path",
    "All Paths",
    "Heatmap",
    "Territorial Occupancy",
    "Voronoi Territories",
    "Proximity Network",
]


def render(key: SessionKey) -> None:
    vt = get_tracking_data(*key.as_loader_args())
    if vt is None:
        st.warning("No tracking data available for this session.")
        return

    td = vt.parsed_data
    names = list(td.keys())

    plot_type = plot_picker("Plot type", PLOT_TYPES, key="tracking_plot")

    if plot_type == "Individual Path":
        sel = st.selectbox("Animal", names, key="tracking_individual")
        fig = pt.plot_animal_path(td[sel], sel)
    elif plot_type == "All Paths":
        fig = pt.plot_multiple_paths(td)
    elif plot_type == "Heatmap":
        sel = st.selectbox("Animal", names, key="tracking_heatmap")
        fig = pt.plot_path_heatmap(td[sel], sel)
    elif plot_type == "Territorial Occupancy":
        fig = pt.plot_territorial_occupancy(td)
    elif plot_type == "Voronoi Territories":
        fig = pt.plot_voronoi_territories(td)
    elif plot_type == "Proximity Network":
        threshold = st.slider("Proximity threshold (px)", 50, 300, 100)
        fig = pt.plot_proximity_network(td, proximity_threshold=threshold)
    else:
        return

    show_fig(fig)
