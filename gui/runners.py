"""Cached compute step used by the Decoding and Population Geometry tabs.

Both tabs follow the same pattern:

    pkl = cache_path(prefix, key, params)
    cached = load_cache(pkl)
    if cached is not None:
        st.success("Loaded from disk cache")
    else:
        if st.button("Run ..."):
            with st.spinner(...):
                cached = run_fn()
                save_cache(pkl, cached)
        else:
            st.info("Press Run.")
            return

This module exposes :func:`cached_step` which captures that pattern in one
place and adds a "Clear cache" affordance.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st

from gui.cache import CACHE_DIR, cache_path, load_cache, save_cache
from gui.state import SessionKey


def _cache_chip(pkl: Path) -> None:
    if pkl.exists():
        size_kb = pkl.stat().st_size / 1024
        mtime = datetime.fromtimestamp(pkl.stat().st_mtime).strftime("%H:%M:%S")
        st.caption(f"✓ cached · {size_kb:,.0f} KB · last run {mtime}")
    else:
        st.caption("✗ not cached")


def cached_step(
    *,
    prefix: str,
    key: SessionKey,
    params: dict,
    run_fn: Callable[[], Any],
    button_label: str,
    spinner_label: str = "Running...",
    not_run_message: str = "Configure parameters in the sidebar, then press Run.",
) -> Optional[Any]:
    """Render the run/cache controls and return the computed result.

    Returns ``None`` if the user has not yet pressed Run on a cold cache.
    """
    pkl = cache_path(
        prefix,
        key.animal_id,
        key.session_id,
        key.config_path,
        params,
    )

    cols = st.columns([3, 1, 1])
    with cols[0]:
        _cache_chip(pkl)
    with cols[1]:
        run_clicked = st.button(
            button_label,
            type="primary",
            use_container_width=True,
            key=f"{prefix}_run_{pkl.stem}",
        )
    with cols[2]:
        clear_clicked = st.button(
            "Clear",
            use_container_width=True,
            help="Delete this cache entry.",
            key=f"{prefix}_clear_{pkl.stem}",
        )

    if clear_clicked and pkl.exists():
        pkl.unlink()
        st.rerun()

    cached = load_cache(pkl)
    if cached is not None and not run_clicked:
        return cached

    if not run_clicked:
        st.info(not_run_message)
        return None

    with st.spinner(spinner_label):
        result = run_fn()
        save_cache(pkl, result)
    st.toast("Done — result cached to disk.", icon="✅")
    return result


def cache_summary() -> dict:
    """Inventory of disk cache, used by the sidebar's Cache expander."""
    if not CACHE_DIR.exists():
        return {"n_files": 0, "total_kb": 0.0}
    files = list(CACHE_DIR.glob("*.pkl"))
    return {
        "n_files": len(files),
        "total_kb": sum(f.stat().st_size for f in files) / 1024,
    }


def clear_all_cache() -> int:
    """Remove every pickle in the disk cache. Returns the number deleted."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()
        n += 1
    return n
