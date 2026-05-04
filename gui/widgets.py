"""Reusable Streamlit widgets used by the GUI sidebar and tabs."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd
import streamlit as st

from gui.loaders import count_events_per_opponent
from gui.state import AnalysisParams, PopulationParams, SessionKey
from video.behavioral_events import BehavioralEventsData


CONFIG_OPTIONS = {
    "Cohort 7 (default)": None,
    "Cohort 5": "cohort5_paths.json",
}


def cohort_picker() -> Optional[str]:
    """Render the cohort selector and return the chosen config path (or None)."""
    label = st.selectbox("Cohort / Config", list(CONFIG_OPTIONS.keys()))
    return CONFIG_OPTIONS[label]


def session_animal_picker(manifest: pd.DataFrame, config_path: Optional[str]) -> SessionKey:
    """Render session and animal selectors against an already-loaded manifest."""
    sessions = sorted(manifest["session"].unique())
    session_id = st.selectbox("Session", sessions)
    animals = sorted(manifest.loc[manifest["session"] == session_id, "animal"].tolist())
    animal_id = st.selectbox("Animal", animals)
    return SessionKey(animal_id=animal_id, session_id=session_id, config_path=config_path)


def analysis_param_widgets(
    *,
    events: Optional[BehavioralEventsData],
    session_cache_key: str,
    animal_id: str,
) -> AnalysisParams:
    """Sidebar controls for shared decoding/geometry parameters.

    The "Min events / class" slider's max value is computed from the loaded
    events; before a session is loaded it falls back to a disabled-feeling
    range so the user knows the value will only become meaningful after load.
    """
    behavior_types = BehavioralEventsData.BEHAVIOR_TYPES
    abbrevs = list(behavior_types.keys())
    labels = list(behavior_types.values())

    beh_idx = st.selectbox(
        "Behavior type",
        range(len(abbrevs)),
        format_func=lambda i: f"{abbrevs[i]} — {labels[i]}",
    )
    behavior_type = abbrevs[beh_idx]

    cols = st.columns(2)
    with cols[0]:
        t_start = st.number_input("Window start (s)", value=-1.0, step=0.25)
    with cols[1]:
        t_end = st.number_input("Window end (s)", value=2.0, step=0.25)

    bin_size = st.number_input("Bin size (s)", value=0.5, step=0.05, min_value=0.01)
    cv_folds = st.slider("CV folds", 2, 10, 5)

    if events is not None:
        counts = count_events_per_opponent(
            events, animal_id, behavior_type, session_cache_key,
        )
        max_events = max(counts.values()) if counts else 2
    else:
        max_events = 2

    min_evts = st.slider(
        "Min events / class",
        min_value=2,
        max_value=max(2, max_events),
        value=min(5, max(2, max_events)),
        help="Opponents with fewer events than this are excluded.",
    )
    if events is not None and max_events <= 2:
        st.caption(
            f"⚠ Loaded events for behavior **{behavior_type}** "
            f"don't allow more than {max_events} events / class."
        )

    return AnalysisParams(
        behavior_type=behavior_type,
        time_window=(float(t_start), float(t_end)),
        time_bin_size=float(bin_size),
        cv_folds=int(cv_folds),
        min_events_per_class=int(min_evts),
    )


def population_param_widgets(base: AnalysisParams) -> PopulationParams:
    """Sidebar controls specific to population geometry."""
    method = st.radio("Method", ["pca", "umap"], horizontal=True)
    n_comp = st.slider("Components", 2, 10, 3)
    norm_method = st.selectbox("Normalization", ["none", "zscore", "baseline"])
    alignment = st.radio("Alignment", ["start", "end", "center"], horizontal=True)
    return PopulationParams(
        base=base,
        method=method,
        n_components=int(n_comp),
        normalize_method=norm_method,
        alignment=alignment,
    )


def plot_picker(label: str, options: Sequence[str], key: Optional[str] = None) -> str:
    """Pill-style picker (segmented control on Streamlit ≥1.34, radio elsewhere)."""
    seg = getattr(st, "segmented_control", None)
    if seg is not None:
        choice = seg(label, list(options), default=options[0], key=key)
        return choice if choice is not None else options[0]
    return st.radio(label, list(options), horizontal=True, key=key)


def session_info_header(summary: dict, key: SessionKey) -> None:
    """One-line summary banner shown above the tabs after Load Session."""
    st.markdown(f"### Session `{key.session_id}` · animal `{key.animal_id}`")
    cols = st.columns(4)
    with cols[0]:
        st.metric(
            "Quality cells",
            summary["n_cells"] if summary["n_cells"] is not None else "—",
        )
    with cols[1]:
        st.metric(
            "Duration",
            f"{summary['duration_min']:.1f} min" if summary["duration_min"] else "—",
        )
    with cols[2]:
        st.metric(
            "Events",
            summary["n_events"] if summary["n_events"] is not None else "—",
        )
    with cols[3]:
        st.metric("Opponents", len(summary["opponents"]) if summary["opponents"] else 0)

    flags = []
    flags.append("✓ tracking" if summary["has_tracking"] else "✗ tracking")
    flags.append("✓ sync" if summary["has_sync"] else "✗ sync")
    st.caption(" · ".join(flags))


def cache_controls() -> None:
    """Sidebar block showing disk cache stats and a Clear-all button."""
    from gui.runners import cache_summary, clear_all_cache  # local import: avoid cycle

    info = cache_summary()
    st.caption(f".gui_cache: {info['n_files']} entries · {info['total_kb']:,.0f} KB")
    if st.button("Clear all cached results", use_container_width=True):
        n = clear_all_cache()
        st.toast(f"Cleared {n} cache entries.", icon="🧹")
        st.rerun()


def status_chips(summary: Optional[dict]) -> None:
    """Compact green/red chips for the sidebar after a session is loaded."""
    if summary is None:
        return
    chips: Iterable[tuple[str, bool]] = [
        ("behavior", summary["n_events"] is not None),
        ("sync", summary["has_sync"]),
        ("tracking", summary["has_tracking"]),
        ("cells", summary["n_cells"] is not None),
    ]
    parts = [f"{'✓' if ok else '✗'} {name}" for name, ok in chips]
    st.caption(" · ".join(parts))
