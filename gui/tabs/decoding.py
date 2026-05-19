import streamlit as st

from ephys.decode_event_outcome import (
    decode_event_outcome_population,
    decode_event_outcome_time_resolved,
)
from ephys.decode_opponent_identity import (
    decode_opponent_identity_population,
    decode_opponent_identity_time_resolved,
    plot_best_cells_decoding,
    plot_decoding_accuracy_distribution,
    plot_decoding_summary,
    plot_time_resolved_decoding,
    plot_top_cells_firing_rates,
)
from gui.loaders import get_ks_data, get_synced_behavior
from gui.plotting import show_fig
from gui.runners import cached_step
from gui.state import AnalysisParams, SessionKey
from gui.widgets import plot_picker


VIEWS = [
#    "Accuracy Distribution",
#    "Best Cells",
    "Summary",
    "Top Cells Firing Rates",
    "Time-Resolved Accuracy",
]


def render(key: SessionKey, params: AnalysisParams) -> None:
    events = get_synced_behavior(*key.as_loader_args())
    if events is None:
        st.warning("No behavioral event data — cannot run decoding.")
        return

    ks_data = get_ks_data(*key.as_loader_args())

    label_mode = st.radio(
        "Label mode",
        ["opponent", "group", "outcome"],
        horizontal=True,
        key="decoding_label_mode",
        help="opponent: decode each individual opponent rat. "
             "group: decode two ID-half groups (self vs others, relative to the focal animal). "
             "outcome: decode event outcome (winner vs loser).",
    )

    view = plot_picker("View", VIEWS, key="decoding_view")

    if view == "Time-Resolved Accuracy":
        cols = st.columns(2)
        with cols[0]:
            bin_step = st.number_input(
                "Bin step (s)",
                value=float(params.time_bin_size) / 2,
                min_value=0.01,
                step=0.05,
                help="Step between successive bin starts. Smaller = more "
                     "overlap (sliding window). Equal to bin size = no overlap.",
            )
        with cols[1]:
            n_shuffles = st.slider(
                "Shuffle nulls",
                0, 20, 5, step=1,
                help="Label-permutation shuffles for a chance band. 0 = skip "
                     "(faster); each adds a full per-bin LDA pass.",
            )
        tr_params = {
            **params.as_dict(),
            "time_bin_step": float(bin_step),
            "n_shuffles": n_shuffles,
            "label_mode": label_mode,
        }

        def _run_tr():
            if label_mode == "outcome":
                return decode_event_outcome_time_resolved(
                    ks_data=ks_data,
                    behavior_data=events,
                    animal_of_interest=key.animal_id,
                    use_quality_cells=True,
                    alignment="end",
                    time_window=params.time_window,
                    time_bin_size=params.time_bin_size,
                    time_bin_step=float(bin_step),
                    cv_folds=params.cv_folds,
                    # min_events_per_class=params.min_events_per_class,
                    n_shuffles=n_shuffles,
                )
            return decode_opponent_identity_time_resolved(
                ks_data=ks_data,
                behavior_data=events,
                animal_of_interest=key.animal_id,
                behavior_type=params.behavior_type,
                use_quality_cells=True,
                alignment="start",
                time_window=params.time_window,
                time_bin_size=params.time_bin_size,
                time_bin_step=float(bin_step),
                cv_folds=params.cv_folds,
                min_events_per_class=params.min_events_per_class,
                n_shuffles=n_shuffles,
                label_mode=label_mode,
            )

        results_tr = cached_step(
            prefix="decoding_time_resolved",
            key=key,
            params=tr_params,
            run_fn=_run_tr,
            button_label="Run Time-Resolved Decoding",
            spinner_label="Running per-bin LDAs across all cells...",
        )
        if results_tr is None:
            return
        if results_tr.get("status") == "failed":
            st.error(f"Decoding failed: {results_tr.get('error', 'unknown error')}")
            return
        show_fig(plot_time_resolved_decoding(results_tr))
        return

    def _run():
        if label_mode == "outcome":
            return decode_event_outcome_population(
                ks_data=ks_data,
                behavior_data=events,
                animal_of_interest=key.animal_id,
                use_quality_cells=True,
                alignment="end",
                time_window=params.time_window,
                time_bin_size=params.time_bin_size,
                cv_folds=params.cv_folds,
                # min_events_per_class=params.min_events_per_class,
            )
        return decode_opponent_identity_population(
            ks_data=ks_data,
            behavior_data=events,
            animal_of_interest=key.animal_id,
            behavior_type=params.behavior_type,
            use_quality_cells=True,
            alignment="start",
            time_window=params.time_window,
            time_bin_size=params.time_bin_size,
            cv_folds=params.cv_folds,
            min_events_per_class=params.min_events_per_class,
            label_mode=label_mode,
        )

    results = cached_step(
        prefix="decoding",
        key=key,
        params={**params.as_dict(), "label_mode": label_mode},
        run_fn=_run,
        button_label="Run Decoding",
        spinner_label="Running LDA decoding across all cells...",
    )
    if results is None:
        return

    if results.get("status") == "failed":
        st.error(f"Decoding failed: {results.get('error', 'unknown error')}")
        return

    if view == "Accuracy Distribution":
        show_fig(plot_decoding_accuracy_distribution(results))
    elif view == "Best Cells":
        n = st.slider("Top N cells", 5, 20, 10)
        show_fig(plot_best_cells_decoding(results, n_top_cells=n))
    elif view == "Summary":
        show_fig(plot_decoding_summary(results))
    elif view == "Top Cells Firing Rates":
        show_fig(
            plot_top_cells_firing_rates(
                ks_data, events, results,
                time_window=params.time_window,
                # time_bin_size=params.time_bin_size,
            )
        )
