"""
Shared helpers for video trajectory plotting.

Private module — do not import from outside ``video/``.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

COORD_COLS = ['center_x', 'center_y']


def clean_coords(df: pd.DataFrame, name: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays from df['center_x'/'center_y']; raise on empty/missing.

    Use in single-animal entry points where bad input is a programming error.
    """
    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"DataFrame is empty{f' for {name}' if name else ''}")

    cleaned = df.dropna(subset=COORD_COLS)
    if cleaned.empty:
        raise ValueError(f"No valid coordinate data found{f' for {name}' if name else ''}")

    return cleaned['center_x'].to_numpy(), cleaned['center_y'].to_numpy()


def coords_or_skip(df: pd.DataFrame, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Like clean_coords but returns None and logs a warning on bad input.

    Use in multi-animal aggregations where one bad animal shouldn't kill the plot.
    """
    if any(c not in df.columns for c in COORD_COLS):
        logger.warning("Skipping %s — missing coordinate columns", name)
        return None

    cleaned = df.dropna(subset=COORD_COLS)
    if cleaned.empty:
        logger.warning("Skipping %s — no valid coordinate data", name)
        return None

    return cleaned['center_x'].to_numpy(), cleaned['center_y'].to_numpy()


def resolve_palette(colors: Optional[List], n: int, default: str = "husl") -> List:
    """Return a list of at least *n* colors, falling back to a seaborn palette."""
    if colors is None:
        return list(sns.color_palette(default, n))
    if len(colors) < n:
        return (list(colors) * ((n // len(colors)) + 1))[:n]
    return list(colors)


def combined_bounds(
    tracking_dict: Dict[str, pd.DataFrame],
) -> Tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) across every animal's valid coords."""
    all_x: List[float] = []
    all_y: List[float] = []
    for name, df in tracking_dict.items():
        coords = coords_or_skip(df, name)
        if coords is None:
            continue
        x, y = coords
        all_x.extend(x)
        all_y.extend(y)

    if not all_x:
        raise ValueError("No valid coordinate data found for any animal")

    return float(np.min(all_x)), float(np.max(all_x)), float(np.min(all_y)), float(np.max(all_y))


def occupancy_maps(
    tracking_dict: Dict[str, pd.DataFrame],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-animal 2D histograms over the given bin edges. Skips bad animals."""
    maps: Dict[str, np.ndarray] = {}
    for name, df in tracking_dict.items():
        coords = coords_or_skip(df, name)
        if coords is None:
            continue
        x, y = coords
        counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        maps[name] = counts
    return maps


def setup_spatial_axes(ax, *, grid: bool = True) -> None:
    """Apply standard spatial-axis formatting: labels, equal aspect, inverted y."""
    ax.set_xlabel('X Coordinate (pixels)')
    ax.set_ylabel('Y Coordinate (pixels)')
    if grid:
        ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
