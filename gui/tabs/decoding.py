import streamlit as st

from ephys.decode_opponent_identity import (
    decode_opponent_identity_population,
    plot_best_cells_decoding,
    plot_decoding_accuracy_distribution,
    plot_decoding_summary,
    plot_top_cells_firing_rates,
)
from gui.cache import cache_path, load_cache, save_cache
from gui.loaders import get_behavior_data, get_ks_data


def render(animal_id: str, session_id: str, config_path, decode_params: dict):
    events = get_behavior_data(animal_id, session_id, config_path)
    if events is None:
        st.warning("No behavioral event data — cannot run decoding.")
        return

    ks_data = get_ks_data(animal_id, session_id, config_path)

    pkl = cache_path("decoding", animal_id, session_id, config_path, decode_params)
    results = load_cache(pkl)

    if results is not None:
        st.success("Loaded from disk cache.")
    else:
        if st.button("Run Decoding (may take several minutes)"):
            with st.spinner("Running LDA decoding across all cells..."):
                results = decode_opponent_identity_population(
                    ks_data=ks_data,
                    behavior_data=events,
                    animal_of_interest=animal_id,
                    behavior_type=decode_params["behavior_type"],
                    use_quality_cells=True,
                    alignment="start",
                    time_window=decode_params["time_window"],
                    time_bin_size=decode_params["time_bin_size"],
                    cv_folds=decode_params["cv_folds"],
                    min_events_per_class=decode_params["min_events_per_class"],
                )
                save_cache(pkl, results)
            st.success("Done — result cached to disk.")
        else:
            st.info("Configure parameters in the sidebar, then press Run.")
            return

    if results.get("status") == "failed":
        st.error(f"Decoding failed: {results.get('error', 'unknown error')}")
        return

    view = st.radio(
        "View",
        ["Accuracy Distribution", "Best Cells", "Summary", "Top Cells Firing Rates"],
        horizontal=True,
    )

    if view == "Accuracy Distribution":
        st.pyplot(plot_decoding_accuracy_distribution(results))

    elif view == "Best Cells":
        n = st.slider("Top N cells", 5, 20, 10)
        st.pyplot(plot_best_cells_decoding(results, n_top_cells=n))

    elif view == "Summary":
        st.pyplot(plot_decoding_summary(results))

    elif view == "Top Cells Firing Rates":
        st.pyplot(
            plot_top_cells_firing_rates(
                ks_data,
                events,
                results,
                time_window=decode_params["time_window"],
                time_bin_size=decode_params["time_bin_size"],
            )
        )
