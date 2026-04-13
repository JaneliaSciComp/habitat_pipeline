import streamlit as st

from gui.loaders import get_data_storage
from video.tracking_import import VideoTrackingData
from video import plot_trajectory as pt


def render(animal_id: str, session_id: str, config_path):
    dsm = get_data_storage(animal_id, session_id, config_path)

    try:
        vt = VideoTrackingData(dsm)
    except Exception as e:
        st.warning(f"No tracking data available for this session. ({e})")
        return

    td = vt.parsed_data  # Dict[str, pd.DataFrame]
    names = list(td.keys())

    plot_type = st.radio(
        "Plot type",
        [
            "Individual Path",
            "All Paths",
            "Heatmap",
            "Territorial Occupancy",
            "Voronoi Territories",
            "Proximity Network",
        ],
        horizontal=True,
    )

    if plot_type == "Individual Path":
        sel = st.selectbox("Animal", names)
        fig = pt.plot_animal_path(td[sel], sel)

    elif plot_type == "All Paths":
        fig = pt.plot_multiple_paths(td)

    elif plot_type == "Heatmap":
        sel = st.selectbox("Animal", names)
        fig = pt.plot_path_heatmap(td[sel], sel)

    elif plot_type == "Territorial Occupancy":
        fig = pt.plot_territorial_occupancy(td)

    elif plot_type == "Voronoi Territories":
        fig = pt.plot_voronoi_territories(td)

    elif plot_type == "Proximity Network":
        threshold = st.slider("Proximity threshold (px)", 50, 300, 100)
        fig = pt.plot_proximity_network(td, proximity_threshold=threshold)

    st.pyplot(fig)
