"""Streamlit Partner Distance tab.

Decode the continuous distance between the loaded *focal* animal and a chosen
*partner* from the focal's neural activity (regression). Calls into
``ephys.decode_partner_distance`` for the math and
``ephys.decode_partner_distance_plots`` for the plots; the heavy run is wrapped
by :func:`gui.runners.cached_step` so the disk cache invalidates on any
parameter change.
"""
from __future__ import annotations

import logging

import streamlit as st

from ephys.decode_partner_distance import (
    _analyze,
    build_distance_binned_data,
    load_partner_distance_inputs,
)
from ephys.decode_partner_distance_plots import (
    plot_distance_tuning_curves,
    plot_partner_distance_summary,
    plot_per_cell_r2_distribution,
    plot_predicted_vs_actual_scatter,
    plot_predicted_vs_actual_timeseries,
)
from gui.plotting import show_fig
from gui.runners import cached_step
from gui.state import PartnerDistanceParams, SessionKey
from gui.tabs.inter_brain import _session_animals
from gui.widgets import plot_picker

log = logging.getLogger(__name__)


VIEWS = [
    "Summary",
    "Distance Tuning",
    "Per-cell R²",
    "Predicted vs Actual",
    "Time Series",
]


def render(key: SessionKey, params: PartnerDistanceParams | None = None) -> None:
    candidates = [a for a in _session_animals(key.config_path, key.session_id)
                  if a != key.animal_id]
    if not candidates:
        st.warning(
            f"No other animals recorded in session **{key.session_id}** — "
            "partner-distance decoding needs a partner."
        )
        return

    partner = st.selectbox(
        "Partner animal",
        candidates,
        key="pd_partner",
        help="The specific other animal whose distance from the focal "
             "(implanted) animal will be decoded.",
    )

    cols = st.columns(3)
    with cols[0]:
        bin_size = st.number_input(
            "Bin size (s)", value=0.5, step=0.1, min_value=0.05,
            key="pd_bin_size",
            help="Time-bin width for the shared rate/distance grid.",
        )
    with cols[1]:
        smoothing = st.number_input(
            "Smoothing σ (s)", value=0.25, step=0.05, min_value=0.0,
            key="pd_smoothing",
            help="Gaussian smoothing across time per cell (0 = off).",
        )
    with cols[2]:
        alpha = st.number_input(
            "Ridge α", value=1.0, step=0.5, min_value=0.0, key="pd_alpha",
            help="Ridge regularization for the regression.",
        )

    cols2 = st.columns(3)
    with cols2[0]:
        n_distance_bins = st.slider(
            "Distance bins", 5, 40, 15, key="pd_dbins",
            help="Bins for the 1-D distance tuning curves.",
        )
    with cols2[1]:
        cv_folds = st.slider(
            "CV folds", 2, 10, 5, key="pd_cv",
            help="Contiguous-block cross-validation folds (no shuffling).",
        )
    with cols2[2]:
        n_shuffles = st.slider(
            "Null shuffles", 0, 300, 50, step=10, key="pd_shuffles",
            help="Circular-shift shuffles for the null (0 = skip null).",
        )

    pd_params = PartnerDistanceParams(
        partner=partner,
        bin_size=float(bin_size),
        smoothing_sigma_sec=float(smoothing),
        n_distance_bins=int(n_distance_bins),
        tuning_smoothing_sigma=1.0,
        alpha=float(alpha),
        cv_folds=int(cv_folds),
        null=("shuffle" if n_shuffles > 0 else None),
        n_shuffles=int(n_shuffles),
    )

    result = cached_step(
        prefix="partner_distance",
        key=key,
        params=pd_params.as_dict(),
        run_fn=lambda: _run(key, pd_params),
        button_label="Run Partner-Distance Decoding",
        spinner_label="Binning rates + distance, regressing, running null...",
    )
    if result is None:
        return

    if result.get("status") != "success":
        st.warning(f"Decode did not succeed: {result.get('status')}")
        return

    cv = result["cv_r2"]
    partial = result.get("cv_r2_partial")
    null_r2 = result.get("null_r2")
    mcols = st.columns(4)
    mcols[0].metric("Population CV R²", f"{cv:.3f}")
    mcols[1].metric("RMSE", f"{result['rmse']:.2f} {result['units']}")
    if partial is not None:
        mcols[2].metric("Partial R² (beyond self-motion)", f"{partial:.3f}")
    if null_r2 is not None:
        mcols[3].metric("Null R²", f"{null_r2:.3f}")

    view = plot_picker("View", VIEWS, key="pd_view")
    if view == "Summary":
        show_fig(plot_partner_distance_summary(result))
    elif view == "Distance Tuning":
        show_fig(plot_distance_tuning_curves(result))
    elif view == "Per-cell R²":
        show_fig(plot_per_cell_r2_distribution(result))
    elif view == "Predicted vs Actual":
        show_fig(plot_predicted_vs_actual_scatter(result))
    elif view == "Time Series":
        show_fig(plot_predicted_vs_actual_timeseries(result))


def _run(key: SessionKey, p: PartnerDistanceParams) -> dict:
    """Load focal ephys + session tracking, bin rates + distance, run the decode."""
    inputs = load_partner_distance_inputs(
        key.session_id, key.animal_id, config_path=key.config_path,
    )
    data = build_distance_binned_data(
        inputs.ks_focal, inputs.tracking, inputs.sync, key.animal_id, p.partner,
        pixels_per_cm=inputs.pixels_per_cm,
        bin_size=p.bin_size,
        smoothing_sigma_sec=(p.smoothing_sigma_sec if p.smoothing_sigma_sec > 0 else None),
    )
    return _analyze(
        data["firing_rates"], data["distance"], data["nuisance"],
        data["bin_centers"], data["units"], key.animal_id, p.partner,
        alpha=p.alpha, cv_folds=p.cv_folds,
        n_distance_bins=p.n_distance_bins,
        tuning_smoothing_sigma=p.tuning_smoothing_sigma,
        null=p.null, n_shuffles=p.n_shuffles,
        nuisance_names=data["nuisance_names"],
    )
