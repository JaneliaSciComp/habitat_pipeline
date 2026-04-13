# Run from project root: streamlit run gui/app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import; makes plt.show() a no-op

import streamlit as st

from ingestion.data_paths import get_animals_and_sessions
from video.behavioral_events import BehavioralEventsData
from gui.tabs import tracking, behavioral, decoding, population

st.set_page_config(page_title="Habitat Pipeline", layout="wide")
st.title("RatCity — Habitat Pipeline Explorer")

CONFIG_OPTIONS = {
    "Cohort 7 (default)": None,
    "Cohort 5": "cohort5_paths.json",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Selection")

    cohort_label = st.selectbox("Cohort / Config", list(CONFIG_OPTIONS.keys()))
    config_path = CONFIG_OPTIONS[cohort_label]

    @st.cache_data(show_spinner="Scanning sessions...")
    def load_manifest(cfg):
        return get_animals_and_sessions(config_path=cfg)

    try:
        manifest = load_manifest(config_path)
    except Exception as e:
        st.error(f"Cannot read session list: {e}")
        st.stop()

    if manifest.empty:
        st.warning("No sessions found for this cohort.")
        st.stop()

    session_id = st.selectbox("Session", sorted(manifest["session"].unique()))
    animals = sorted(manifest.loc[manifest["session"] == session_id, "animal"].tolist())
    animal_id = st.selectbox("Animal", animals)

    st.divider()
    st.subheader("Decoding / Geometry Parameters")


    BEHAVIOR_TYPES = BehavioralEventsData.BEHAVIOR_TYPES
    abbrevs = list(BEHAVIOR_TYPES.keys())
    labels = list(BEHAVIOR_TYPES.values())

    beh_idx = st.selectbox(
        "Behavior type",
        range(len(abbrevs)),
        format_func=lambda i: f"{abbrevs[i]} — {labels[i]}",
    )
    behavior_type = abbrevs[beh_idx]

    t_start = st.number_input("Window start (s)", value=-1.0, step=0.25)
    t_end = st.number_input("Window end (s)", value=2.0, step=0.25)
    bin_size = st.number_input("Bin size (s)", value=0.5, step=0.05, min_value=0.01)
    cv_folds = st.slider("CV folds", 2, 10, 5)

    # Compute max events per opponent from cached data if a session is already loaded
    _prev_key = st.session_state.get("loaded_session")
    _cur_key = (animal_id, session_id, str(config_path))
    _max_events = 50  # fallback before any session is loaded
    if _prev_key == _cur_key:
        from gui.loaders import get_behavior_data as _get_beh
        _events = _get_beh(animal_id, session_id, config_path)
        if _events is not None:
            try:
                _, _, _labels = _events.extract_opponent_labels(
                    animal_of_interest=animal_id,
                    behavior_type=behavior_type,
                    min_events_per_class=1,
                )
                if len(_labels) > 0:
                    import numpy as _np
                    _counts = {lbl: int((_labels == lbl).sum()) for lbl in _np.unique(_labels)}
                    _max_events = max(_counts.values())
            except Exception:
                pass

    min_evts = st.slider("Min events / class", 2, max(2, _max_events), min(5, max(2, _max_events)))

    decode_params = dict(
        behavior_type=behavior_type,
        time_window=(t_start, t_end),
        time_bin_size=bin_size,
        cv_folds=cv_folds,
        min_events_per_class=min_evts,
    )

    st.divider()
    run_clicked = st.button("Load & Process Session", type="primary", use_container_width=True)

# ── Track which session has been loaded ───────────────────────────────────────
session_key = (animal_id, session_id, str(config_path))

if run_clicked:
    st.session_state["loaded_session"] = session_key

loaded = st.session_state.get("loaded_session") == session_key

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_t, tab_b, tab_d, tab_p = st.tabs(
    ["Tracking & Spatial", "Behavioral Events", "Neural Decoding", "Population Geometry"]
)

if not loaded:
    with tab_t:
        st.info("Select a session and animal in the sidebar, then press **Load & Process Session**.")
else:
    with tab_t:
        tracking.render(animal_id, session_id, config_path)

    with tab_b:
        behavioral.render(animal_id, session_id, config_path)

    with tab_d:
        decoding.render(animal_id, session_id, config_path, decode_params)

    with tab_p:
        population.render(animal_id, session_id, config_path, decode_params)
