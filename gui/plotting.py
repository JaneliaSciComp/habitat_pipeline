"""Display helpers for matplotlib figures inside Streamlit."""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st


def show_fig(fig) -> None:
    """Render a matplotlib figure and close it to avoid leaking memory.

    Streamlit re-renders top-to-bottom on every interaction, so figures
    created inside ``render`` accumulate unless explicitly closed.
    """
    if fig is None:
        st.info("No figure produced for this view.")
        return
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
