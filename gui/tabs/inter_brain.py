"""Streamlit Inter-Brain tab.

Multi-animal shared-subspace analysis between the loaded *focal* animal
and one (or more) simultaneously-recorded *partners*. Calls into
``ephys.inter_brain_dynamics`` for the math and
``ephys.inter_brain_plots`` for the plots; the heavy run is wrapped by
:func:`gui.runners.cached_step` so the disk cache invalidates on any
parameter change.
"""
from __future__ import annotations

import logging

import numpy as np
import streamlit as st

from ephys.inter_brain_plots import (
    plot_canonical_correlations,
    plot_cross_animal_correlation,
    plot_inter_brain_summary,
    plot_shared_dimensions,
    plot_shared_vs_behavior,
    plot_time_lagged_cca,
    plot_variance_partition,
)
from ephys.run_inter_brain import _analyze
from gui.plotting import show_fig
from gui.runners import cached_step
from gui.state import InterBrainParams, SessionKey
from gui.widgets import plot_picker
from ingestion.data_paths import get_animals_and_sessions
from ingestion.multi_animal_session import MultiAnimalSession
from video.behavior_features import build_behavior_feature_matrix
from video.tracking_import import load_tracking_data

log = logging.getLogger(__name__)


VIEWS = [
    "Summary",
    "Canonical Correlations",
    "Variance Partition",
    "Shared Dimensions",
    "Cross-Animal Correlation",
    "Time-Lagged CCA",
    "Behavior Regression",
]


@st.cache_data(show_spinner=False)
def _session_animals(config_path, session_id: str) -> list:
    """Cached list of animal IDs recorded in *session_id*."""
    try:
        manifest = get_animals_and_sessions(config_path=config_path)
    except Exception as e:
        log.warning("Could not load session manifest: %s", e)
        return []
    return sorted(manifest.loc[manifest["session"] == session_id, "animal"].tolist())


def render(key: SessionKey, params: InterBrainParams | None = None) -> None:
    candidates = [a for a in _session_animals(key.config_path, key.session_id)
                  if a != key.animal_id]
    if not candidates:
        st.warning(
            f"No other animals recorded in session **{key.session_id}** — "
            "inter-brain analysis needs at least one partner."
        )
        return

    # ----- partner + analysis param widgets ---------------------------------
    partner = st.selectbox(
        "Partner animal",
        candidates,
        key="ib_partner",
        help="The other simultaneously-recorded animal to fit shared "
             "subspace against.",
    )

    cols = st.columns(3)
    with cols[0]:
        bin_size = st.number_input(
            "Bin size (s)", value=0.5, step=0.1, min_value=0.05,
            key="ib_bin_size",
            help="Spike-bin width for the shared-grid rate matrices.",
        )
    with cols[1]:
        smoothing = st.number_input(
            "Smoothing σ (s)", value=0.25, step=0.05, min_value=0.0,
            key="ib_smoothing",
            help="Gaussian smoothing across time per cell (0 = off).",
        )
    with cols[2]:
        n_components = st.slider(
            "K (shared dims)", 1, 20, 5, key="ib_K",
            help="Dimensionality of the shared subspace.",
        )

    cols2 = st.columns(3)
    with cols2[0]:
        n_shuffles = st.slider(
            "Shuffles", 10, 500, 50, step=10, key="ib_n_shuffles",
            help="Circular-shift shuffles for the null distribution.",
        )
    with cols2[1]:
        max_lag_bins = st.slider(
            "Max lag (bins)", 1, 30, 10, key="ib_max_lag",
            help="± lag range for the time-lagged CCA sweep.",
        )
    with cols2[2]:
        alpha = st.number_input(
            "Ridge α", value=1.0, step=0.5, min_value=0.0, key="ib_alpha",
            help="Ridge regularization for the behavior regression.",
        )

    ib_params = InterBrainParams(
        partner_animal_ids=(partner,),
        bin_size=float(bin_size),
        smoothing_sigma_sec=float(smoothing),
        n_components=int(n_components),
        n_shuffles=int(n_shuffles),
        t_window=None,
        method="regularized",
        reg=1e-3,
        cv_folds=5,
        max_lag_bins=int(max_lag_bins),
        alpha=float(alpha),
        event_window=1.0,
        behavior_type=None,
    )

    # ----- run / cache ------------------------------------------------------
    result = cached_step(
        prefix="inter_brain",
        key=key,
        params=ib_params.as_dict(),
        run_fn=lambda: _run(key, partner, ib_params),
        button_label="Run Inter-Brain Analysis",
        spinner_label="Fitting shared subspace + null + regression...",
    )
    if result is None:
        return

    # ----- view picker ------------------------------------------------------
    view = plot_picker("View", VIEWS, key="ib_view")
    fit = result["fit"]
    t_bins = result["bin_centers"][fit.valid_mask]

    if view == "Summary":
        show_fig(plot_inter_brain_summary(
            fit,
            shuffle_null=result["shuffle_null"],
            t_bins=t_bins,
            cross_corr=result["cross_corr"],
            time_lagged=result["time_lagged"],
            regression_results=result["regression_results"],
            bin_size_sec=ib_params.bin_size,
        ))
    elif view == "Canonical Correlations":
        show_fig(plot_canonical_correlations(
            fit, shuffle_null=result["shuffle_null"],
        ))
    elif view == "Variance Partition":
        show_fig(plot_variance_partition(fit))
    elif view == "Shared Dimensions":
        max_k = fit.n_components
        n_show = min(3, max_k)
        k_dims = tuple(range(n_show))
        show_fig(plot_shared_dimensions(fit, t_bins=t_bins, k_dims=k_dims))
    elif view == "Cross-Animal Correlation":
        show_fig(plot_cross_animal_correlation(result["cross_corr"]))
    elif view == "Time-Lagged CCA":
        lags, ccs = result["time_lagged"]
        show_fig(plot_time_lagged_cca(lags, ccs, bin_size_sec=ib_params.bin_size))
    elif view == "Behavior Regression":
        if result["regression_results"] is None:
            st.info(
                "Behavior regression was skipped (tracking or sync unavailable)."
            )
        else:
            show_fig(plot_shared_vs_behavior(fit, result["regression_results"]))


def _run(key: SessionKey, partner: str, ib_params: InterBrainParams) -> dict:
    """Build the MultiAnimalSession, bin rates + features, run the analysis."""
    session = MultiAnimalSession(
        session_id=key.session_id,
        animal_ids=[key.animal_id, partner],
        config_path=key.config_path,
    )
    bin_centers, rates_by_animal = session.get_common_binned_rates(
        bin_size_sec=ib_params.bin_size,
        smoothing_sigma_sec=(
            ib_params.smoothing_sigma_sec if ib_params.smoothing_sigma_sec > 0 else None
        ),
        use_cache=True,
    )
    X_A = rates_by_animal[key.animal_id].T
    X_B = rates_by_animal[partner].T

    behavior_by_animal = None
    try:
        tracking = load_tracking_data(session.dsm_by_animal[session.sync_from_animal])
        try:
            tracking.synchronize_with_ephys(session.sync)
        except Exception as e:
            log.warning("Tracking sync failed: %s", e)
        event_types = (
            [ib_params.behavior_type] if ib_params.behavior_type else None
        )
        beh_A = build_behavior_feature_matrix(
            tracking, session.events, session.sync, bin_centers,
            focal=key.animal_id, partner=partner,
            event_window_sec=ib_params.event_window,
            event_types=event_types,
        )
        beh_B = build_behavior_feature_matrix(
            tracking, session.events, session.sync, bin_centers,
            focal=partner, partner=key.animal_id,
            event_window_sec=ib_params.event_window,
            event_types=event_types,
        )
        behavior_by_animal = {key.animal_id: beh_A, partner: beh_B}
    except Exception as e:
        log.warning("Skipping behavior regression: %s", e)

    return _analyze(
        np.asarray(X_A), np.asarray(X_B), np.asarray(bin_centers),
        animal_ids=(key.animal_id, partner),
        bin_size=ib_params.bin_size,
        smoothing=(
            ib_params.smoothing_sigma_sec if ib_params.smoothing_sigma_sec > 0 else None
        ),
        t_window=ib_params.t_window,
        n_components=ib_params.n_components,
        max_K=ib_params.n_components,
        method=ib_params.method,
        reg=ib_params.reg,
        cv_folds=ib_params.cv_folds,
        n_shuffles=ib_params.n_shuffles,
        max_lag_bins=ib_params.max_lag_bins,
        alpha=ib_params.alpha,
        behavior_by_animal=behavior_by_animal,
    )
