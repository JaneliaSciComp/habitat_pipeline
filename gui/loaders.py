import streamlit as st

from ingestion.data_paths import DataStorageManager
from ingestion.kilosort_data_import import load_kilosort_data
from video.behavioral_events import load_behavioral_events
from ingestion.ephys_sync import DataSyncManager


@st.cache_resource(show_spinner="Loading session paths...")
def get_data_storage(animal_id: str, session_id: str, config_path):
    return DataStorageManager(animal_id, session_id, config_path=config_path, auto_load=True)


@st.cache_resource(show_spinner="Loading spike data (may take a minute)...")
def get_ks_data(animal_id: str, session_id: str, config_path):
    return load_kilosort_data(get_data_storage(animal_id, session_id, config_path))


@st.cache_resource(show_spinner="Loading behavioral events...")
def get_behavior_data(animal_id: str, session_id: str, config_path):
    try:
        dsm = get_data_storage(animal_id, session_id, config_path)
        return load_behavioral_events(
            dsm.get_behavioral_event_files(),
            session_id=dsm.session_id,
        )
    except Exception:
        return None


@st.cache_resource(show_spinner="Syncing clocks...")
def get_sync(animal_id: str, session_id: str, config_path):
    try:
        return DataSyncManager(
            get_data_storage(animal_id, session_id, config_path),
            dio_channel=1,
        )
    except Exception:
        return None
