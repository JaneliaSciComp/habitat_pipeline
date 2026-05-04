import streamlit as st

from ephys.population_geometry import PopulationGeometryAnalyzer
from gui.loaders import get_ks_data, get_synced_behavior
from gui.plotting import show_fig
from gui.runners import cached_step
from gui.state import PopulationParams, SessionKey
from gui.widgets import plot_picker


VIEWS = ["Population Dynamics", "PCA Summary", "Normalized Population Matrix"]


def render(key: SessionKey, params: PopulationParams) -> None:
    events = get_synced_behavior(*key.as_loader_args())
    if events is None:
        st.warning("No behavioral event data — cannot run population geometry.")
        return

    ks_data = get_ks_data(*key.as_loader_args())
    analyzer = PopulationGeometryAnalyzer(ks_data, events)
    base = params.base

    def _run():
        starts, ends, labels = events.extract_opponent_labels(
            animal_of_interest=key.animal_id,
            behavior_type=base.behavior_type,
            min_events_per_class=base.min_events_per_class,
        )
        pop_matrix = analyzer.construct_population_matrix(
            event_starts=starts,
            event_ends=ends,
            event_labels=labels,
            time_window=base.time_window,
            time_bin_size=base.time_bin_size,
            alignment=params.alignment,
            normalize_method=params.normalize_method,
            use_quality_cells=True,
        )
        try:
            reduced = analyzer.apply_dimensionality_reduction(
                pop_matrix, method=params.method, n_components=params.n_components,
            )
        except ImportError:
            st.error("Install umap-learn to use UMAP: pip install umap-learn")
            raise
        return pop_matrix, reduced

    cached = cached_step(
        prefix="population",
        key=key,
        params=params.as_dict(),
        run_fn=_run,
        button_label="Run Geometry",
        spinner_label="Building population matrix and reducing dimensions...",
    )
    if cached is None:
        return
    pop_matrix, reduced = cached

    view = plot_picker("View", VIEWS, key="population_view")

    if view == "Population Dynamics":
        show_individual = st.checkbox("Show individual trials")
        show_fig(analyzer.plot_population_dynamics(reduced, show_individual=show_individual))
    elif view == "PCA Summary":
        show_fig(analyzer.plot_pca_summary(reduced, pop_matrix))
    elif view == "Normalized Population Matrix":
        show_fig(analyzer.plot_normalized_population_matrix(pop_matrix["population_data"]))
