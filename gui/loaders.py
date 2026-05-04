"""Cached data loaders for the Streamlit GUI.

All loaders are memoized with :func:`st.cache_resource` keyed on
``(animal_id, session_id, config_path)``; the same triple is wrapped by
:class:`gui.state.SessionKey` everywhere else in the GUI.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import streamlit as st

from ephys.rastermap_viz import plot_rastermap, plot_rastermap_with_events
from ingestion.data_paths import DataStorageManager
from ingestion.ephys_sync import DataSyncManager
from ingestion.kilosort_data_import import load_kilosort_data
from video.behavioral_events import BehavioralEventsData, load_behavioral_events
from video.tracking_import import load_tracking_data

log = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading session paths...")
def get_data_storage(animal_id: str, session_id: str, config_path):
    return DataStorageManager(animal_id, session_id, config_path=config_path, auto_load=True)


@st.cache_resource(show_spinner="Loading spike data (may take a minute)...")
def get_ks_data(animal_id: str, session_id: str, config_path):
    dsm = get_data_storage(animal_id, session_id, config_path)
    return load_kilosort_data(dsm.get_kilosort_path())


@st.cache_resource(show_spinner="Loading behavioral events...")
def get_behavior_data(animal_id: str, session_id: str, config_path) -> Optional[BehavioralEventsData]:
    try:
        dsm = get_data_storage(animal_id, session_id, config_path)
        return load_behavioral_events(
            dsm.get_behavioral_event_files(),
            session_id=dsm.session_id,
        )
    except Exception as e:
        log.warning("Could not load behavioral events for %s/%s: %s", animal_id, session_id, e)
        return None


@st.cache_resource(show_spinner="Syncing clocks...")
def get_sync(animal_id: str, session_id: str, config_path) -> Optional[DataSyncManager]:
    try:
        return DataSyncManager(
            get_data_storage(animal_id, session_id, config_path),
            dio_channel=1,
        )
    except Exception as e:
        log.warning("Could not build clock sync for %s/%s: %s", animal_id, session_id, e)
        return None


@st.cache_resource(show_spinner="Loading tracking data...")
def get_tracking_data(animal_id: str, session_id: str, config_path):
    """Tracking is optional — return None if unavailable."""
    try:
        dsm = get_data_storage(animal_id, session_id, config_path)
        return load_tracking_data(dsm)
    except Exception as e:
        log.warning("No tracking data for %s/%s: %s", animal_id, session_id, e)
        return None


@st.cache_resource(show_spinner="Synchronizing behavioral events to ephys clock...")
def get_synced_behavior(animal_id: str, session_id: str, config_path) -> Optional[BehavioralEventsData]:
    """Load behavioral events and sync them to ephys time once.

    Returns the events object with ``ts_*_ephys`` columns populated, or the
    raw events object if sync failed (so the behavioral tab can still render
    plots that don't need ephys time).
    """
    events = get_behavior_data(animal_id, session_id, config_path)
    if events is None:
        return None
    if getattr(events, "synchronized", False):
        return events
    sync = get_sync(animal_id, session_id, config_path)
    if sync is None:
        log.warning("Skipping behavioral sync — no DataSyncManager available.")
        return events
    try:
        events.synchronize_with_ephys(sync, create_new_columns=True)
    except Exception as e:
        log.warning("synchronize_with_ephys failed: %s", e)
    return events


@st.cache_resource(show_spinner="Fitting Rastermap (full session)...")
def get_rastermap_full(animal_id: str, session_id: str, config_path, bin_size: float):
    ks_data = get_ks_data(animal_id, session_id, config_path)
    return plot_rastermap(ks_data, bin_size=bin_size)


@st.cache_resource(show_spinner="Fitting Rastermap (event-aligned)...")
def get_rastermap_events(
    animal_id: str,
    session_id: str,
    config_path,
    behavior_type: str,
    bin_size: float,
    time_window: tuple,
):
    ks_data = get_ks_data(animal_id, session_id, config_path)
    events = get_synced_behavior(animal_id, session_id, config_path)
    if events is None:
        return None, None
    return plot_rastermap_with_events(
        ks_data,
        events,
        animal_of_interest=animal_id,
        behavior_type=behavior_type,
        bin_size=bin_size,
        time_window=tuple(time_window),
    )


@st.cache_data(show_spinner=False)
def count_events_per_opponent(
    _events: BehavioralEventsData,
    animal_id: str,
    behavior_type: str,
    session_cache_key: str,
) -> dict:
    """Number of events per opponent for the loaded animal/behavior_type.

    The leading underscore on ``_events`` tells Streamlit not to hash the
    object (it's already keyed by ``session_cache_key``).
    """
    if _events is None:
        return {}
    try:
        _, _, labels = _events.extract_opponent_labels(
            animal_of_interest=animal_id,
            behavior_type=behavior_type,
            min_events_per_class=1,
        )
    except Exception as e:
        log.warning("count_events_per_opponent failed: %s", e)
        return {}
    if len(labels) == 0:
        return {}
    return {str(lbl): int((labels == lbl).sum()) for lbl in np.unique(labels)}


@st.cache_data(show_spinner=False)
def get_session_summary(
    animal_id: str, session_id: str, config_path,
) -> dict:
    """Lightweight summary used in the post-load header banner.

    Returns a dict with keys: n_cells, duration_min, n_events, opponents,
    has_tracking, has_sync. Missing pieces appear as None.
    """
    summary: dict = {
        "n_cells": None,
        "duration_min": None,
        "n_events": None,
        "opponents": [],
        "has_tracking": False,
        "has_sync": False,
    }

    try:
        ks = get_ks_data(animal_id, session_id, config_path)
        summary["n_cells"] = len(ks.spike_times_by_cell)
        summary["duration_min"] = ks.duration_seconds / 60.0
    except Exception as e:
        log.warning("Could not summarize ks_data: %s", e)

    events = get_synced_behavior(animal_id, session_id, config_path)
    if events is not None:
        try:
            summary["n_events"] = int(len(events.events_data))
            summary["opponents"] = events.get_available_rats()
        except Exception as e:
            log.warning("Could not summarize events: %s", e)

    summary["has_sync"] = get_sync(animal_id, session_id, config_path) is not None
    summary["has_tracking"] = get_tracking_data(animal_id, session_id, config_path) is not None
    return summary
