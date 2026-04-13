import streamlit as st

from ephys.population_geometry import PopulationGeometryAnalyzer
from gui.cache import cache_path, load_cache, save_cache
from gui.loaders import get_behavior_data, get_ks_data


def render(animal_id: str, session_id: str, config_path, decode_params: dict):
    events = get_behavior_data(animal_id, session_id, config_path)
    if events is None:
        st.warning("No behavioral event data — cannot run population geometry.")
        return

    ks_data = get_ks_data(animal_id, session_id, config_path)

    st.subheader("Dimensionality Reduction Parameters")
    method = st.radio("Method", ["pca", "umap"], horizontal=True)
    n_comp = st.slider("Components", 2, 10, 3)
    norm_method = st.selectbox("Normalization", ["none", "zscore", "baseline"])
    alignment = st.radio("Alignment", ["start", "end", "center"], horizontal=True)

    pop_params = {
        **decode_params,
        "method": method,
        "n_components": n_comp,
        "normalize_method": norm_method,
        "alignment": alignment,
    }

    pkl = cache_path("population", animal_id, session_id, config_path, pop_params)
    cached = load_cache(pkl)

    if cached is not None:
        pop_matrix, reduced_data = cached
        st.success("Loaded from disk cache.")
    else:
        if st.button("Run Population Geometry Analysis"):
            with st.spinner("Building population matrix and reducing dimensions..."):
                event_starts, event_ends, event_labels = events.extract_opponent_labels(
                    animal_of_interest=animal_id,
                    behavior_type=decode_params["behavior_type"],
                    min_events_per_class=decode_params["min_events_per_class"],
                )
                analyzer = PopulationGeometryAnalyzer(ks_data, events)
                pop_matrix = analyzer.construct_population_matrix(
                    event_starts=event_starts,
                    event_ends=event_ends,
                    event_labels=event_labels,
                    time_window=decode_params["time_window"],
                    time_bin_size=decode_params["time_bin_size"],
                    alignment=alignment,
                    normalize_method=norm_method,
                    use_quality_cells=True,
                )
                try:
                    reduced_data = analyzer.apply_dimensionality_reduction(
                        pop_matrix, method=method, n_components=n_comp
                    )
                except ImportError:
                    st.error("Install umap-learn to use UMAP: pip install umap-learn")
                    return
                save_cache(pkl, (pop_matrix, reduced_data))
            st.success("Done — result cached to disk.")
        else:
            st.info("Configure parameters above, then press Run.")
            return

    # Re-init analyzer for plotting (lightweight, no heavy computation)
    analyzer = PopulationGeometryAnalyzer(ks_data, events)

    view = st.radio(
        "View",
        ["Population Dynamics", "PCA Summary", "Normalized Population Matrix"],
        horizontal=True,
    )

    if view == "Population Dynamics":
        show_ind = st.checkbox("Show individual trials")
        st.pyplot(analyzer.plot_population_dynamics(reduced_data, show_individual=show_ind))

    elif view == "PCA Summary":
        st.pyplot(analyzer.plot_pca_summary(reduced_data, pop_matrix))

    elif view == "Normalized Population Matrix":
        st.pyplot(
            analyzer.plot_normalized_population_matrix(pop_matrix["population_data"])
        )
