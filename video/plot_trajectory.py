"""
Path Visualization Module

Utilities for visualizing tracked animal paths using ``center_x`` / ``center_y``
coordinates. Supports individual paths, multi-animal comparisons, occupancy
heatmaps, territorial maps (binned and Voronoi), and proximity networks.
"""

import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.spatial import Voronoi

from video._plot_helpers import (
    clean_coords,
    combined_bounds,
    coords_or_skip,
    occupancy_maps,
    resolve_palette,
    setup_spatial_axes,
)

logger = logging.getLogger(__name__)

DEFAULT_BINS = 50
DEFAULT_MIN_OCCUPANCY = 20
DEFAULT_PROXIMITY_PX = 100.0
DEFAULT_MIN_INTERACTION_FRAMES = 30


# ---------------------------------------------------------------------------
# Single-animal plots
# ---------------------------------------------------------------------------

def plot_animal_path(df: pd.DataFrame, animal_name: str,
                    figsize: Tuple[int, int] = (10, 8),
                    cmap: str = 'viridis',
                    show_start_end: bool = True,
                    show_stats: bool = True,
                    title: Optional[str] = None) -> plt.Figure:
    """Plot the movement path for a single animal with a time-gradient line.

    Args:
        df: DataFrame with 'center_x' and 'center_y' columns.
        animal_name: Name of the animal (used in title and labels).
        figsize: Figure size as (width, height).
        cmap: Colormap name for the time gradient.
        show_start_end: If True, mark the start (green) and end (red) points.
        show_stats: If True, overlay total distance and point count.
        title: Custom title; defaults to ``f"Movement Path: {animal_name}"``.

    Raises:
        ValueError: if 'center_x' / 'center_y' are missing or all-NaN.
    """
    x_coords, y_coords = clean_coords(df, animal_name)

    fig, ax = plt.subplots(figsize=figsize)

    if len(x_coords) > 1:
        points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments, cmap=cmap, linewidths=1.5, alpha=0.8,
        )
        lc.set_array(np.linspace(0, 1, len(segments)))
        line = ax.add_collection(lc)

        cbar = plt.colorbar(line, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Time Progression (Start → End)')
    else:
        ax.plot(x_coords[0], y_coords[0], 'o', color='blue',
                markersize=8, label=animal_name)

    if show_start_end and len(x_coords) > 1:
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
        ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End')

    setup_spatial_axes(ax)
    ax.legend()

    ax.set_title(title or f'Movement Path: {animal_name}')

    total_distance = calculate_path_distance(x_coords, y_coords)
    if show_stats:
        ax.text(
            0.02, 0.98,
            f'Total Distance: {total_distance:.1f} pixels\nData Points: {len(x_coords)}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

    plt.tight_layout()
    logger.info("Plotted path for %s: %d points, %.1f px total distance",
                animal_name, len(x_coords), total_distance)
    return fig


def plot_path_heatmap(df: pd.DataFrame, animal_name: str,
                     figsize: Tuple[int, int] = (10, 8),
                     bins: int = DEFAULT_BINS,
                     cmap: str = 'hot',
                     title: Optional[str] = None) -> plt.Figure:
    """2D occupancy histogram (log-scaled) for a single animal."""
    x_coords, y_coords = clean_coords(df, animal_name)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('black')

    _, _, _, im = ax.hist2d(x_coords, y_coords, bins=bins, cmap=cmap, norm=LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Time Spent (frames) - Log Scale')

    ax.set_xlabel('X Coordinate (pixels)')
    ax.set_ylabel('Y Coordinate (pixels)')
    ax.invert_yaxis()
    ax.set_title(title or f'Position Heatmap: {animal_name}')

    plt.tight_layout()
    logger.info("Created heatmap for %s: %d data points", animal_name, len(x_coords))
    return fig


# ---------------------------------------------------------------------------
# Multi-animal plots
# ---------------------------------------------------------------------------

def plot_multiple_paths(tracking_dict: Dict[str, pd.DataFrame],
                       figsize: Tuple[int, int] = (12, 10),
                       colors: Optional[List[str]] = None,
                       show_start_end: bool = True,
                       title: Optional[str] = None) -> plt.Figure:
    """Plot movement paths for multiple animals on the same axes."""
    if not tracking_dict:
        raise ValueError("No tracking data provided")

    palette = resolve_palette(colors, len(tracking_dict))

    fig, ax = plt.subplots(figsize=figsize)

    for i, (animal_name, df) in enumerate(tracking_dict.items()):
        coords = coords_or_skip(df, animal_name)
        if coords is None:
            continue
        x_coords, y_coords = coords
        color = palette[i]

        ax.plot(x_coords, y_coords, '-', color=color, linewidth=2,
                alpha=0.7, label=animal_name)

        if show_start_end and len(x_coords) > 1:
            ax.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=8,
                    markeredgecolor='white', markeredgewidth=2)
            ax.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=8,
                    markeredgecolor='white', markeredgewidth=2)

    setup_spatial_axes(ax)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(title or f'Movement Paths Comparison ({len(tracking_dict)} animals)')

    plt.tight_layout()
    logger.info("Plotted paths for %d animals", len(tracking_dict))
    return fig


def plot_territorial_occupancy(tracking_dict: Dict[str, pd.DataFrame],
                             figsize: Tuple[int, int] = (12, 10),
                             bins: int = DEFAULT_BINS,
                             colors: Optional[List[str]] = None,
                             title: Optional[str] = None,
                             min_occupancy: int = 5) -> plt.Figure:
    """Territorial map: each spatial bin colored by the animal that occupied it longest."""
    if not tracking_dict:
        raise ValueError("No tracking data provided")

    animal_names = list(tracking_dict.keys())
    palette = resolve_palette(colors, len(animal_names), default="Set2")

    x_min, x_max, y_min, y_max = combined_bounds(tracking_dict)
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)

    maps = occupancy_maps(tracking_dict, x_edges, y_edges)
    if not maps:
        raise ValueError("No valid occupancy data could be generated")

    # Stack per-animal occupancy → pick winning animal per bin (with min threshold).
    ordered_names = [n for n in animal_names if n in maps]
    stack = np.stack([maps[n] for n in ordered_names], axis=0)  # (n_animals, bins, bins)
    max_per_bin = stack.max(axis=0)
    winner = stack.argmax(axis=0)
    territorial_map = np.where(max_per_bin >= min_occupancy, winner, -1)

    fig, ax = plt.subplots(figsize=figsize)

    territorial_colors = ['lightgray'] + [palette[animal_names.index(n)] for n in ordered_names]
    cmap = ListedColormap(territorial_colors)
    extent = [x_min, x_max, y_min, y_max]
    ax.imshow(territorial_map.T, extent=extent, origin='lower',
              cmap=cmap, vmin=-1, vmax=len(ordered_names) - 1, alpha=0.8)

    X, Y = np.meshgrid(x_edges[:-1] + np.diff(x_edges) / 2,
                       y_edges[:-1] + np.diff(y_edges) / 2)
    ax.contour(X, Y, max_per_bin.T, levels=5, colors='black',
               alpha=0.3, linewidths=0.5)

    for animal_name in ordered_names:
        coords = coords_or_skip(tracking_dict[animal_name], animal_name)
        if coords is None:
            continue
        x, y = coords
        ax.plot(x, y, '-', color=palette[animal_names.index(animal_name)],
                linewidth=0.5, alpha=0.4)

    ax.set_xlabel('X Coordinate (pixels)')
    ax.set_ylabel('Y Coordinate (pixels)')
    ax.invert_yaxis()

    legend_elements = [Patch(facecolor='lightgray', label='Unoccupied')]
    for name in ordered_names:
        legend_elements.append(Patch(facecolor=palette[animal_names.index(name)], label=name))
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title(title or f'Territorial Occupancy Map ({len(animal_names)} animals)')

    total_bins = bins * bins
    occupied_bins = int(np.sum(territorial_map >= 0))
    ax.text(
        0.02, 0.02,
        f'Grid: {bins}×{bins}\nOccupied: {occupied_bins}/{total_bins} bins\n'
        f'Min occupancy: {min_occupancy} frames',
        transform=ax.transAxes, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    plt.tight_layout()
    logger.info("Created territorial occupancy map for %d animals", len(animal_names))
    for idx, name in enumerate(ordered_names):
        territory_size = int(np.sum(territorial_map == idx))
        logger.info("  %s: %d territorial bins (%.1f%%)",
                    name, territory_size, territory_size / total_bins * 100)

    return fig


def plot_voronoi_territories(tracking_dict: Dict[str, pd.DataFrame],
                           figsize: Tuple[int, int] = (12, 10),
                           colors: Optional[List[str]] = None,
                           title: Optional[str] = None,
                           bins: int = DEFAULT_BINS,
                           min_occupancy: int = DEFAULT_MIN_OCCUPANCY,
                           show_paths: bool = False,
                           show_seeds: bool = False,
                           alpha: float = 0.6) -> plt.Figure:
    """Voronoi territorial diagram seeded by high-occupancy bin centers."""
    if not tracking_dict:
        raise ValueError("No tracking data provided")

    animal_names = list(tracking_dict.keys())
    palette = resolve_palette(colors, len(animal_names), default="Set2")

    x_min, x_max, y_min, y_max = combined_bounds(tracking_dict)
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)
    x_centers = x_edges[:-1] + np.diff(x_edges) / 2
    y_centers = y_edges[:-1] + np.diff(y_edges) / 2

    maps = occupancy_maps(tracking_dict, x_edges, y_edges)
    if not maps:
        raise ValueError("No valid occupancy data could be generated")

    # Collect Voronoi seed points from high-occupancy bins of each animal.
    all_points: List[List[float]] = []
    point_labels: List[int] = []

    for name, occupancy in maps.items():
        animal_idx = animal_names.index(name)
        rows, cols = np.where(occupancy >= min_occupancy)
        if rows.size == 0:
            logger.warning("No high-occupancy areas found for %s", name)
            continue
        for i, j in zip(rows, cols):
            all_points.append([x_centers[i], y_centers[j]])
            point_labels.append(animal_idx)

    if not all_points:
        raise ValueError("No valid seed points found for Voronoi diagram")

    seeds = np.asarray(all_points)
    seed_labels = np.asarray(point_labels)

    try:
        vor = Voronoi(seeds)
    except Exception as e:
        raise ValueError(f"Failed to create Voronoi diagram: {e}") from e

    fig, ax = plt.subplots(figsize=figsize)

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        vertices = vor.vertices[region]
        if (vertices[:, 0].min() > x_max or vertices[:, 0].max() < x_min or
                vertices[:, 1].min() > y_max or vertices[:, 1].max() < y_min):
            continue

        color = palette[seed_labels[point_idx]]
        ax.add_patch(Polygon(
            vertices, closed=True, facecolor=color,
            edgecolor=color, alpha=alpha, linewidth=0,
        ))

    if show_seeds:
        for animal_idx, name in enumerate(animal_names):
            mask = seed_labels == animal_idx
            if not mask.any():
                continue
            ax.scatter(seeds[mask, 0], seeds[mask, 1], c=[palette[animal_idx]],
                       s=20, alpha=0.9, edgecolors='white', linewidths=0.5,
                       marker='s', zorder=10)

    if show_paths:
        for name, df in tracking_dict.items():
            if name not in maps:
                continue
            coords = coords_or_skip(df, name)
            if coords is None:
                continue
            x, y = coords
            ax.plot(x, y, '-', color=palette[animal_names.index(name)],
                    linewidth=1, alpha=0.5, zorder=5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    setup_spatial_axes(ax, grid=False)

    legend_elements = [
        Patch(facecolor=palette[animal_names.index(name)], alpha=alpha,
              label=f'{name} territory')
        for name in maps
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title(title or 'Voronoi Territorial Diagram (Occupancy-Based)')

    n_points = len(seeds)
    n_regions = sum(1 for r in vor.regions if r and -1 not in r)
    ax.text(
        0.02, 0.02,
        f'Seed points: {n_points}\nVoronoi regions: {n_regions}\n'
        f'Grid: {bins}×{bins}\nMin occupancy: {min_occupancy}',
        transform=ax.transAxes, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    plt.tight_layout()
    logger.info("Created Voronoi territorial diagram with %d seed points from %d animals",
                n_points, len(animal_names))
    return fig


# ---------------------------------------------------------------------------
# Proximity / social network
# ---------------------------------------------------------------------------

def compute_proximity_interactions(
    tracking_dict: Dict[str, pd.DataFrame],
    proximity_threshold: float = DEFAULT_PROXIMITY_PX,
    min_interaction_time: int = DEFAULT_MIN_INTERACTION_FRAMES,
) -> Tuple[List[str], Dict[Tuple[str, str], Dict[str, float]]]:
    """Compute pairwise proximity interaction stats from per-animal tracking data.

    Aligns animals by ``frame`` (created if missing), computes vectorised pairwise
    distances on frames where both animals have valid coordinates, and returns
    the per-pair interaction summary for pairs that meet ``min_interaction_time``.

    Returns:
        (animal_names, interactions) where ``animal_names`` is the order used for
        analysis and ``interactions`` maps ``(name_i, name_j)`` (i < j by index)
        to a dict with ``proximity_frames``, ``total_frames``, ``interaction_strength``,
        ``avg_distance``, ``min_distance``, ``interaction_events``.
    """
    cleaned: Dict[str, pd.DataFrame] = {}
    max_frame = 0

    for name, df in tracking_dict.items():
        if 'center_x' not in df.columns or 'center_y' not in df.columns:
            logger.warning("Skipping %s — missing coordinate columns", name)
            continue
        df_clean = df.dropna(subset=['center_x', 'center_y']).copy()
        if df_clean.empty:
            logger.warning("Skipping %s — no valid coordinate data", name)
            continue
        if 'frame' not in df_clean.columns:
            df_clean['frame'] = range(len(df_clean))
        cleaned[name] = df_clean
        max_frame = max(max_frame, int(df_clean['frame'].max()))

    if len(cleaned) < 2:
        raise ValueError("Need at least 2 animals with valid data for proximity analysis")

    animal_names = list(cleaned.keys())
    n_animals = len(animal_names)
    n_frames = max_frame + 1

    x_coords = np.full((n_animals, n_frames), np.nan)
    y_coords = np.full((n_animals, n_frames), np.nan)
    for idx, name in enumerate(animal_names):
        df = cleaned[name]
        frame_idx = df['frame'].astype(int).to_numpy()
        x_coords[idx, frame_idx] = df['center_x'].to_numpy()
        y_coords[idx, frame_idx] = df['center_y'].to_numpy()

    interactions: Dict[Tuple[str, str], Dict[str, float]] = {}
    for i, j in combinations(range(n_animals), 2):
        valid = ~(np.isnan(x_coords[i]) | np.isnan(y_coords[i]) |
                  np.isnan(x_coords[j]) | np.isnan(y_coords[j]))
        if int(np.sum(valid)) < min_interaction_time:
            continue

        dx = x_coords[i, valid] - x_coords[j, valid]
        dy = y_coords[i, valid] - y_coords[j, valid]
        distances = np.sqrt(dx * dx + dy * dy)

        proximity_mask = distances <= proximity_threshold
        proximity_frames = int(np.sum(proximity_mask))
        total_frames = int(distances.size)
        if proximity_frames < min_interaction_time:
            continue

        valid_indices = np.where(valid)[0]
        interactions[(animal_names[i], animal_names[j])] = {
            'proximity_frames': proximity_frames,
            'total_frames': total_frames,
            'interaction_strength': proximity_frames / total_frames,
            'avg_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'interaction_events': valid_indices[proximity_mask].tolist(),
        }

    logger.info("Found %d significant interactions between animals", len(interactions))
    return animal_names, interactions


def plot_proximity_network(tracking_dict: Dict[str, pd.DataFrame],
                         proximity_threshold: float = DEFAULT_PROXIMITY_PX,
                         min_interaction_time: int = DEFAULT_MIN_INTERACTION_FRAMES,
                         figsize: Tuple[int, int] = (10, 8),
                         node_size_factor: float = 1000,
                         title: Optional[str] = None,
                         colors: Optional[List[str]] = None,
                         layout_type: str = 'spring',
                         animals: Optional[List[str]] = None) -> plt.Figure:
    """Network graph of social proximity between animals."""
    if animals is not None:
        missing = [a for a in animals if a not in tracking_dict]
        if missing:
            raise ValueError(f"Animals not found in tracking data: {missing}")
        tracking_dict = {a: tracking_dict[a] for a in animals}

    if len(tracking_dict) < 2:
        raise ValueError("Need at least 2 animals for proximity analysis")

    logger.info("Analyzing proximity interactions with threshold %s pixels...",
                proximity_threshold)
    animal_names, interactions = compute_proximity_interactions(
        tracking_dict, proximity_threshold, min_interaction_time,
    )
    palette = resolve_palette(colors, len(animal_names), default="Set2")

    G = nx.Graph()
    for idx, name in enumerate(animal_names):
        df = tracking_dict[name]
        activity = calculate_path_statistics(df).get('total_distance', 0.0)
        G.add_node(name, activity=activity, color=palette[idx], frames=len(df))

    edge_weights: List[float] = []
    edge_colors: List[float] = []
    for (a, b), details in interactions.items():
        weight = details['interaction_strength']
        G.add_edge(a, b, weight=weight,
                   proximity_frames=details['proximity_frames'],
                   avg_distance=details['avg_distance'],
                   min_distance=details['min_distance'])
        edge_weights.append(weight * 10)
        edge_colors.append(weight)

    if not G.edges():
        logger.warning("No significant interactions found between animals")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            'No significant interactions detected\n'
            f'(threshold: {proximity_threshold} pixels, min time: {min_interaction_time} frames)',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
        )
        ax.set_title(title or 'Proximity Network (No Interactions)')
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    layouts = {
        'spring': lambda g: nx.spring_layout(g, k=2, iterations=50),
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
    }
    pos = layouts.get(layout_type, layouts['spring'])(G)

    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors,
                           edge_cmap=plt.cm.Reds, alpha=0.6, ax=ax)

    activities = [G.nodes[n]['activity'] for n in G.nodes()]
    max_activity = max(activities) or 1.0
    node_sizes = [a / max_activity * node_size_factor + 200 for a in activities]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.8, edgecolors='black', linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold',
                            font_color='white', ax=ax)

    legend_elements: List = []
    for idx, name in enumerate(animal_names):
        node = G.nodes[name]
        legend_elements.append(Patch(
            facecolor=palette[idx],
            label=f"{name} ({node['frames']} frames, {node['activity']:.0f}px moved)",
        ))
    legend_elements.extend([
        Line2D([0], [0], color='red', lw=1, alpha=0.6, label='Weak interaction'),
        Line2D([0], [0], color='red', lw=3, alpha=0.6, label='Strong interaction'),
    ])
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title or f'Animal Proximity Network\n'
                          f'({len(G.nodes())} animals, {len(G.edges())} interactions)',
                 pad=20, fontsize=14, fontweight='bold')

    stats_lines = [
        f'Proximity threshold: {proximity_threshold} pixels',
        f'Min interaction time: {min_interaction_time} frames',
        f'Network density: {nx.density(G):.3f}',
    ]
    if G.edges():
        stats_lines.append(f'Average clustering: {nx.average_clustering(G):.3f}')
    ax.text(0.02, 0.02, '\n'.join(stats_lines), transform=ax.transAxes,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    logger.info("Proximity network: %d animals, %d edges, density %.3f",
                len(G.nodes()), len(G.edges()), nx.density(G))
    return fig


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calculate_path_distance(x_coords: np.ndarray, y_coords: np.ndarray) -> float:
    """Total Euclidean distance traveled along a path."""
    if len(x_coords) != len(y_coords):
        raise ValueError("Coordinate arrays must have the same length")
    if len(x_coords) < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2)))


def calculate_path_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """Summary statistics about a single-animal movement path."""
    coords = coords_or_skip(df, name="<stats>")
    if coords is None:
        return {}

    x, y = coords
    total_distance = calculate_path_distance(x, y)
    return {
        'total_distance': total_distance,
        'data_points': len(x),
        'x_range': float(np.max(x) - np.min(x)),
        'y_range': float(np.max(y) - np.min(y)),
        'x_center': float(np.mean(x)),
        'y_center': float(np.mean(y)),
        'x_std': float(np.std(x)),
        'y_std': float(np.std(y)),
        'avg_speed': total_distance / (len(x) - 1) if len(x) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_visualization(fig: plt.Figure, output_path: Union[str, Path],
                      dpi: int = 300, format: str = 'png') -> None:
    """Save a matplotlib figure to file, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info("Saved visualization to: %s", output_path)
