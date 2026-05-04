# Run from project root: streamlit run gui/app.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import; makes plt.show() a no-op

import streamlit as st

from gui.loaders import (
    get_data_storage,
    get_session_summary,
    get_synced_behavior,
    get_tracking_data,
)
from gui.state import is_session_loaded, set_loaded_session
from gui.tabs import behavioral, decoding, population, tracking
from gui.widgets import (
    analysis_param_widgets,
    cache_controls,
    cohort_picker,
    population_param_widgets,
    session_animal_picker,
    session_info_header,
    status_chips,
)
from ingestion.data_paths import get_animals_and_sessions


st.set_page_config(page_title="Habitat Pipeline", layout="wide")
st.title("RatCity — Habitat Pipeline Explorer")


@st.cache_data(show_spinner="Scanning sessions...")
def _load_manifest(config_path):
    return get_animals_and_sessions(config_path=config_path)


def _preload(key) -> None:
    """Warm up the cheap loaders so first-tab open is instant.

    Spike data is intentionally NOT preloaded — it's slow and only the
    Decoding / Population tabs need it.
    """
    get_data_storage(*key.as_loader_args())
    get_synced_behavior(*key.as_loader_args())
    get_tracking_data(*key.as_loader_args())


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")

    config_path = cohort_picker()

    try:
        manifest = _load_manifest(config_path)
    except Exception as e:
        st.error(f"Cannot read session list: {e}")
        st.stop()

    if manifest.empty:
        st.warning("No sessions found for this cohort.")
        st.stop()

    session_key = session_animal_picker(manifest, config_path)

    run_clicked = st.button("Load Session", type="primary", use_container_width=True)
    if run_clicked:
        set_loaded_session(session_key)
        _preload(session_key)

    loaded = is_session_loaded(session_key)
    summary = (
        get_session_summary(*session_key.as_loader_args()) if loaded else None
    )
    if loaded:
        status_chips(summary)

    # Analysis params — only meaningful once a session is loaded; auto-expand then.
    events_for_widget = (
        get_synced_behavior(*session_key.as_loader_args()) if loaded else None
    )

    with st.expander("Analysis parameters", expanded=loaded):
        analysis_params = analysis_param_widgets(
            events=events_for_widget,
            session_cache_key=session_key.as_cache_key(),
            animal_id=session_key.animal_id,
        )

    with st.expander("Population geometry parameters", expanded=False):
        pop_params = population_param_widgets(analysis_params)

    with st.expander("Cache", expanded=False):
        cache_controls()

# ── Main ──────────────────────────────────────────────────────────────────────
if not loaded:
    st.info(
        "Pick a cohort, session, and animal in the sidebar, then press "
        "**Load Session** to begin."
    )
    st.stop()

session_info_header(summary, session_key)

tab_t, tab_b, tab_d, tab_p = st.tabs(
    ["Tracking & Spatial", "Behavioral Events", "Neural Decoding", "Population Geometry"]
)

with tab_t:
    tracking.render(session_key)

with tab_b:
    behavioral.render(session_key)

with tab_d:
    decoding.render(session_key, analysis_params)

with tab_p:
    population.render(session_key, pop_params)
