"""Typed state for the Streamlit GUI.

Replaces ad-hoc tuples / dicts with named dataclasses so loaders, runners, and
tabs can take a single typed argument instead of unpacking by string keys.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal, Optional, Tuple

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


@dataclass(frozen=True)
class InterBrainParams:
    """Parameters for the Inter-Brain tab.

    The *focal* animal is supplied via the surrounding :class:`SessionKey`;
    ``partner_animal_ids`` lists the other animals recorded simultaneously
    in the same session. Keeping the focal on ``SessionKey`` and partners
    in params means the four existing tabs are unaffected by this addition
    and the disk cache key (``SessionKey`` + ``params``) invalidates
    correctly when the partner set or any analysis parameter changes.
    """

    partner_animal_ids: Tuple[str, ...]
    bin_size: float
    smoothing_sigma_sec: float
    n_components: int
    n_shuffles: int
    t_window: Optional[Tuple[float, float]]
    method: str
    reg: float
    cv_folds: int
    max_lag_bins: int
    alpha: float
    event_window: float
    behavior_type: Optional[str]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SocialSpatialParams:
    """Parameters for the Social Place Fields tab.

    ``focal`` is the animal whose spikes generate the rate maps; ``targets``
    are the animals (self + partners) whose positions the maps are built over.
    Both live in params (not only on :class:`SessionKey`) so the disk cache key
    invalidates when either changes.
    """

    focal: str
    targets: Tuple[str, ...]
    bin_size_cm: float
    smoothing_sigma_cm: float
    speed_threshold_cms: float
    speed_filter_subject: Literal["focal", "target", "none"]
    n_shuffles: int
    use_quality_cells: bool

    def as_dict(self) -> dict:
        return asdict(self)


def set_loaded_session(key: SessionKey) -> None:
    st.session_state[SESSION_STATE_KEY] = key


def loaded_session_key() -> Optional[SessionKey]:
    return st.session_state.get(SESSION_STATE_KEY)


def is_session_loaded(key: SessionKey) -> bool:
    return loaded_session_key() == key
