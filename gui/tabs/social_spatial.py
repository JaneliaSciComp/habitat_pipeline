"""Streamlit Social Place Fields tab.

Occupancy-normalized rate maps of a focal animal's cells over each (self and
partner) animal's allocentric position. Calls ``ephys.social_spatial_fields``
for the math and ``ephys.social_spatial_plots`` for the figures; the heavy run
is wrapped by :func:`gui.runners.cached_step` so the disk cache invalidates on
any parameter, focal, or target-set change.
"""
from __future__ import annotations

import logging

import numpy as np
import streamlit as st

from ephys.social_spatial_fields import compute_social_place_fields
from ephys.social_spatial_plots import (
    plot_cell_classification_summary,
    plot_field_stability,
    plot_rate_maps_grid,
)
from gui.plotting import show_fig
from gui.runners import cached_step
from gui.state import SessionKey, SocialSpatialParams
from gui.tabs.inter_brain import _session_animals
from ingestion.focal_session import load_focal_session_inputs

log = logging.getLogger(__name__)


def render(key: SessionKey, params: SocialSpatialParams | None = None) -> None:
    animals = _session_animals(key.config_path, key.session_id)
    if not animals:
        st.warning(f"No animals listed for session **{key.session_id}**.")
        return

    # ----- focal + target selection ----------------------------------------
    default_focal = key.animal_id if key.animal_id in animals else animals[0]
    focal = st.radio(
        "Focal animal (whose spikes)", animals,
        index=animals.index(default_focal), horizontal=True, key="spf_focal",
        help="The animal whose cells' rate maps are computed.",
    )
    targets = st.multiselect(
        "Target animals (whose position)", animals, default=animals,
        key="spf_targets",
        help="Build rate maps over each of these animals' (x, y). Include the "
             "focal animal for its self place field.",
    )
    if not targets:
        st.info("Select at least one target animal.")
        return

    # ----- parameter widgets -----------------------------------------------
    cols = st.columns(3)
    with cols[0]:
        bin_size = st.slider("Bin size (cm)", 1.0, 20.0, 5.0, step=1.0,
                             key="spf_bin", help="Spatial bin width.")
    with cols[1]:
        smoothing = st.slider("Smoothing σ (cm)", 0.0, 20.0, 5.0, step=1.0,
                              key="spf_smooth", help="Gaussian smoothing (0 = off).")
    with cols[2]:
        speed_thr = st.slider("Speed gate (cm/s)", 0.0, 30.0, 5.0, step=1.0,
                              key="spf_speed", help="Remove samples below this speed.")

    cols2 = st.columns(2)
    with cols2[0]:
        subject = st.radio("Speed-gate subject", ["target", "focal", "none"],
                           horizontal=True, key="spf_subject",
                           help="Whose speed gates the samples. Default = target.")
    with cols2[1]:
        n_shuffles = st.slider("Shuffles", 50, 1000, 200, step=50,
                               key="spf_shuffles",
                               help="Circular-shift shuffles for significance.")
    use_quality = st.checkbox("Quality cells only", value=True, key="spf_quality")

    spf_params = SocialSpatialParams(
        focal=focal,
        targets=tuple(targets),
        bin_size_cm=float(bin_size),
        smoothing_sigma_cm=float(smoothing),
        speed_threshold_cms=float(speed_thr),
        speed_filter_subject=subject,
        n_shuffles=int(n_shuffles),
        use_quality_cells=bool(use_quality),
    )

    # ----- run / cache ------------------------------------------------------
    result = cached_step(
        prefix="social_spatial",
        key=key,
        params=spf_params.as_dict(),
        run_fn=lambda: _run(key, spf_params),
        button_label="Run Social Place Fields",
        spinner_label="Building rate maps + significance across targets...",
    )
    if result is None:
        return

    df = result.cell_classification
    if df.empty:
        st.warning("No cells available (check quality filter / spike counts).")
        return

    # ----- cluster selector (ranked by max Skaggs across targets) ----------
    bits_cols = [f"bits_per_spike_{t}" for t in spf_params.targets
                 if f"bits_per_spike_{t}" in df.columns]
    df = df.assign(_maxbits=df[bits_cols].max(axis=1) if bits_cols else 0.0)
    df = df.sort_values("_maxbits", ascending=False)
    labels = {
        int(r.cluster_id): f"{int(r.cluster_id)} — {r.category} "
                           f"({r._maxbits:.2f} b/s)"
        for r in df.itertuples()
    }
    cluster_id = st.selectbox(
        "Cluster (ranked by max bits/spike across targets)",
        list(labels.keys()), format_func=lambda c: labels[c], key="spf_cluster",
    )

    show_fig(plot_rate_maps_grid(result, cluster_id=int(cluster_id)))

    st.markdown("#### Population summary")
    show_fig(plot_cell_classification_summary(result))
    show_fig(plot_field_stability(result))


def _run(key: SessionKey, params: SocialSpatialParams):
    """Load focal ephys + session tracking and compute social place fields."""
    inputs = load_focal_session_inputs(
        key.session_id, params.focal, config_path=key.config_path,
    )
    return compute_social_place_fields(
        inputs.ks_focal, inputs.tracking, inputs.sync,
        focal_animal=params.focal,
        target_animals=list(params.targets),
        pixels_per_cm=inputs.pixels_per_cm,
        bin_size_cm=params.bin_size_cm,
        smoothing_sigma_cm=params.smoothing_sigma_cm,
        speed_threshold_cms=params.speed_threshold_cms,
        speed_filter_subject=params.speed_filter_subject,
        n_shuffles=params.n_shuffles,
        use_quality_cells=params.use_quality_cells,
    )
