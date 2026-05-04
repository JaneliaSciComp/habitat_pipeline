import streamlit as st

from ephys.decode_opponent_identity import (
    decode_opponent_identity_population,
    plot_best_cells_decoding,
    plot_decoding_accuracy_distribution,
    plot_decoding_summary,
    plot_top_cells_firing_rates,
)
from gui.loaders import get_ks_data, get_synced_behavior
from gui.plotting import show_fig
from gui.runners import cached_step
from gui.state import AnalysisParams, SessionKey
from gui.widgets import plot_picker


VIEWS = ["Accuracy Distribution", "Best Cells", "Summary", "Top Cells Firing Rates"]


def render(key: SessionKey, params: AnalysisParams) -> None:
    events = get_synced_behavior(*key.as_loader_args())
    if events is None:
        st.warning("No behavioral event data — cannot run decoding.")
        return

    ks_data = get_ks_data(*key.as_loader_args())

    def _run():
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
        )

    results = cached_step(
        prefix="decoding",
        key=key,
        params=params.as_dict(),
        run_fn=_run,
        button_label="Run Decoding",
        spinner_label="Running LDA decoding across all cells...",
    )
    if results is None:
        return

    if results.get("status") == "failed":
        st.error(f"Decoding failed: {results.get('error', 'unknown error')}")
        return

    view = plot_picker("View", VIEWS, key="decoding_view")

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
                time_bin_size=params.time_bin_size,
            )
        )
