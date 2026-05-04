"""Typed state for the Streamlit GUI.

Replaces ad-hoc tuples / dicts with named dataclasses so loaders, runners, and
tabs can take a single typed argument instead of unpacking by string keys.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Optional, Tuple

import streamlit as st


SESSION_STATE_KEY = "loaded_session"


@dataclass(frozen=True)
class SessionKey:
    animal_id: str
    session_id: str
    config_path: Optional[str]  # str (not Path) so the dataclass is hashable

    def as_loader_args(self) -> Tuple[str, str, Optional[str]]:
        return self.animal_id, self.session_id, self.config_path

    def as_cache_key(self) -> str:
        return f"{self.animal_id}|{self.session_id}|{self.config_path}"


@dataclass(frozen=True)
class AnalysisParams:
    behavior_type: str
    time_window: Tuple[float, float]
    time_bin_size: float
    cv_folds: int
    min_events_per_class: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PopulationParams:
    base: AnalysisParams
    method: str
    n_components: int
    normalize_method: str
    alignment: str

    def as_dict(self) -> dict:
        return {**self.base.as_dict(), **{
            "method": self.method,
            "n_components": self.n_components,
            "normalize_method": self.normalize_method,
            "alignment": self.alignment,
        }}


def set_loaded_session(key: SessionKey) -> None:
    st.session_state[SESSION_STATE_KEY] = key


def loaded_session_key() -> Optional[SessionKey]:
    return st.session_state.get(SESSION_STATE_KEY)


def is_session_loaded(key: SessionKey) -> bool:
    return loaded_session_key() == key
