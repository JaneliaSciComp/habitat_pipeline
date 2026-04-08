"""
Path Visualization Module

This module provides utilities for visualizing tracked animal paths using
center_x and center_y coordinates from tracking data. Supports individual
animal paths, multi-animal comparisons, and animated trajectories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi, voronoi_plot_2d
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import seaborn as sns
import networkx as nx
from itertools import combinations


def plot_animal_path(df: pd.DataFrame, animal_name: str, 
                    figsize: Tuple[int, int] = (10, 8),
                    color: Optional[str] = None,
                    show_start_end: bool = True,
                    title: Optional[str] = None) -> plt.Figure:
    """
    Plot the movement path for a single animal.
    
    Args:
        df: DataFrame with tracking data for one animal containing 'center_x', 'center_y'
        animal_name: Name of the animal for labeling
        figsize: Figure size as (width, height)
        color: Color for the path line. If None, uses default color cycle
        show_start_end: If True, marks start and end points
        title: Custom title for the plot. If None, generates default title
        
    Returns:
        matplotlib.Figure: The created figure
        
    Raises:
        ValueError: If required columns are missing
    """
    # Check required columns
    required_cols = ['center_x', 'center_y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Remove rows with NaN coordinates
    df_clean = df.dropna(subset=['center_x', 'center_y'])
    
    if df_clean.empty:
        raise ValueError("No valid coordinate data found")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get coordinates
    x_coords = df_clean['center_x'].values
    y_coords = df_clean['center_y'].values
    
    # Create path with gradient colors representing time
    if len(x_coords) > 1:
        # Create line segments
        points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create colors representing time progression
        colors = np.linspace(0, 1, len(segments))
        
        # Create LineCollection with gradient colors
        lc = LineCollection(segments, cmap='viridis' if color is None else color, 
                          linewidths=1.5, alpha=0.8)
        lc.set_array(colors)
        
        # Add to plot
        line = ax.add_collection(lc)
        
        # Add colorbar to show time progression
        cbar = plt.colorbar(line, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Time Progression (Start → End)')
    else:
        # Single point - just plot as a dot
        ax.plot(x_coords[0], y_coords[0], 'o', color=color if color else 'blue', 
               markersize=8, label=animal_name)
    
    # Mark start and end points
    if show_start_end and len(x_coords) > 1:
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
        ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End')
    
    # Formatting
    ax.set_xlabel('X Coordinate (pixels)')
    ax.set_ylabel('Y Coordinate (pixels)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    # Invert y-axis to match image coordinates (origin at top-left)
    ax.invert_yaxis()
    
    # Set title
    if title is None:
        title = f'Movement Path: {animal_name}'
    ax.set_title(title)
    
    # Add statistics
    total_distance = calculate_path_distance(x_coords, y_coords)
    ax.text(0.02, 0.98, f'Total Distance: {total_distance:.1f} pixels\nData Points: {len(x_coords)}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    print(f"Plotted path for {animal_name}: {len(x_coords)} points, {total_distance:.1f} pixels total distance")
    
    return fig


def plot_multiple_paths(tracking_dict: Dict[str, pd.DataFrame],
                       figsize: Tuple[int, int] = (12, 10),
                       colors: Optional[List[str]] = None,
                       show_start_end: bool = True,
                       title: Optional[str] = None) -> plt.Figure:
    """
    Plot movement paths for multiple animals on the same figure.
    
    Args:
        tracking_dict: Dictionary where keys are animal names and values are DataFrames
        figsize: Figure size as (width, height)
        colors: List of colors for each animal. If None, uses default color palette
        show_start_end: If True, marks start and end points for each animal
        title: Custom title for the plot
        
    Returns:
        matplotlib.Figure: The created figure
    """
    if not tracking_dict:
        raise ValueError("No tracking data provided")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up colors
    if colors is None:
        colors = sns.color_palette("husl", len(tracking_dict))
    elif len(colors) < len(tracking_dict):
        # Extend colors if not enough provided
        colors = (colors * ((len(tracking_dict) // len(colors)) + 1))[:len(tracking_dict)]
    
    # Plot each animal's path
    all_x_coords = []
    all_y_coords = []
    
    for i, (animal_name, df) in enumerate(tracking_dict.items()):
        # Check required columns
        if 'center_x' not in df.columns or 'center_y' not in df.columns:
            print(f"Warning: Skipping {animal_name} - missing coordinate columns")
            continue
        
        # Remove rows with NaN coordinates
        df_clean = df.dropna(subset=['center_x', 'center_y'])
        
        if df_clean.empty:
            print(f"Warning: Skipping {animal_name} - no valid coordinate data")
            continue
        
        x_coords = df_clean['center_x'].values
        y_coords = df_clean['center_y'].values
        
        # Store coordinates for axis scaling
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)
        
        color = colors[i]
        
        # Plot path
        ax.plot(x_coords, y_coords, '-', color=color, linewidth=2, alpha=0.7, label=animal_name)
        
        # Mark start and end points
        if show_start_end and len(x_coords) > 1:
            ax.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=8, 
                   markeredgecolor='white', markeredgewidth=2)
            ax.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=8,
                   markeredgecolor='white', markeredgewidth=2)
    
    # Formatting
    ax.set_xlabel('X Coordinate (pixels)')
    ax.set_ylabel('Y Coordinate (pixels)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    
    # Invert y-axis to match image coordinates
    ax.invert_yaxis()
    
    # Set title
    if title is None:
        title = f'Movement Paths Comparison ({len(tracking_dict)} animals)'
    ax.set_title(title)
    
    plt.tight_layout()
    print(f"Plotted paths for {len(tracking_dict)} animals")
    
    return fig


def plot_path_heatmap(df: pd.DataFrame, animal_name: str,
                     figsize: Tuple[int, int] = (10, 8),
                     bins: int = 50,
                     cmap: str = 'hot',
                     title: Optional[str] = None) -> plt.Figure:
    """
    Create a 2D histogram heatmap showing where the animal spends most time.
    
    Args:
        df: DataFrame with tracking data containing 'center_x', 'center_y'
        animal_name: Name of the animal for labeling
        figsize: Figure size as (width, height)
        bins: Number of bins for the 2D histogram
        cmap: Colormap name for the heatmap
        title: Custom title for the plot
        
    Returns:
        matplotlib.Figure: The created figure
    """
    # Check required columns
    required_cols = ['center_x', 'center_y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with NaN coordinates
    df_clean = df.dropna(subset=['center_x', 'center_y'])
    
    if df_clean.empty:
        raise ValueError("No valid coordinate data found")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set black background
    ax.set_facecolor('black')
    # fig.patch.set_facecolor('black')
    
    x_coords = df_clean['center_x'].values
    y_coords = df_clean['center_y'].values
    
    # Create 2D histogram with log scale
    counts, xedges, yedges, im = ax.hist2d(x_coords, y_coords, bins=bins, cmap=cmap, norm=LogNorm())
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Time Spent (frames) - Log Scale')
    
    # Formatting
    ax.set_xlabel('X Coordinate (pixels)')
    ax.set_ylabel('Y Coordinate (pixels)')
    ax.invert_yaxis()
    
    # Set title
    if title is None:
        title = f'Position Heatmap: {animal_name}'
    ax.set_title(title)
    
    plt.tight_layout()
    print(f"Created heatmap for {animal_name}: {len(x_coords)} data points")
    
    return fig


def calculate_path_distance(x_coords: np.ndarray, y_coords: np.ndarray) -> float:
    """
    Calculate the total distance traveled along a path.
    
    Args:
        x_coords: Array of x coordinates
        y_coords: Array of y coordinates
        
    Returns:
        float: Total distance in pixels
    """
    if len(x_coords) != len(y_coords):
        raise ValueError("Coordinate arrays must have the same length")
    
    if len(x_coords) < 2:
        return 0.0
    
    # Calculate distances between consecutive points
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    distances = np.sqrt(dx**2 + dy**2)
    
    return np.sum(distances)


def calculate_path_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate various statistics about an animal's movement path.
    
    Args:
        df: DataFrame with tracking data containing 'center_x', 'center_y'
        
    Returns:
        Dictionary with path statistics
    """
    # Remove rows with NaN coordinates
    df_clean = df.dropna(subset=['center_x', 'center_y'])
    
    if df_clean.empty:
        return {}
    
    x_coords = df_clean['center_x'].values
    y_coords = df_clean['center_y'].values
    
    stats = {
        'total_distance': calculate_path_distance(x_coords, y_coords),
        'data_points': len(x_coords),
        'x_range': np.max(x_coords) - np.min(x_coords),
        'y_range': np.max(y_coords) - np.min(y_coords),
        'x_center': np.mean(x_coords),
        'y_center': np.mean(y_coords),
        'x_std': np.std(x_coords),
        'y_std': np.std(y_coords)
    }
    
    if len(x_coords) > 1:
        # Calculate average speed (distance per frame)
        stats['avg_speed'] = stats['total_distance'] / (len(x_coords) - 1)
    else:
        stats['avg_speed'] = 0.0
    
    return stats


def plot_territorial_occupancy(tracking_dict: Dict[str, pd.DataFrame],
                             figsize: Tuple[int, int] = (12, 10),
                             bins: int = 50,
                             colors: Optional[List[str]] = None,
                             title: Optional[str] = None,
                             min_occupancy: int = 5) -> plt.Figure:
    """
    Create a territorial map showing which animal occupied each location for the most time.
    
    Args:
        tracking_dict: Dictionary where keys are animal names and values are DataFrames
        figsize: Figure size as (width, height)
        bins: Number of bins for spatial discretization
        colors: List of colors for each animal. If None, uses default color palette
        title: Custom title for the plot
        min_occupancy: Minimum number of frames required to claim a territory
        
    Returns:
        matplotlib.Figure: The created figure showing territorial occupancy
    """
    if not tracking_dict:
        raise ValueError("No tracking data provided")
    
    # Set up colors for animals
    animal_names = list(tracking_dict.keys())
    if colors is None:
        colors = sns.color_palette("Set2", len(animal_names))
    elif len(colors) < len(animal_names):
        colors = (colors * ((len(animal_names) // len(colors)) + 1))[:len(animal_names)]
    
    # Find overall coordinate bounds
    all_x_coords = []
    all_y_coords = []
    
    for animal_name, df in tracking_dict.items():
        if 'center_x' in df.columns and 'center_y' in df.columns:
            df_clean = df.dropna(subset=['center_x', 'center_y'])
            if not df_clean.empty:
                all_x_coords.extend(df_clean['center_x'].values)
                all_y_coords.extend(df_clean['center_y'].values)
    
    if not all_x_coords:
        raise ValueError("No valid coordinate data found for any animal")
    
    x_min, x_max = np.min(all_x_coords), np.max(all_x_coords)
    y_min, y_max = np.min(all_y_coords), np.max(all_y_coords)
    
    # Create spatial bins
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)
    
    # Initialize occupancy matrix for each animal
    occupancy_maps = {}
    
    for i, (animal_name, df) in enumerate(tracking_dict.items()):
        if 'center_x' not in df.columns or 'center_y' not in df.columns:
            print(f"Warning: Skipping {animal_name} - missing coordinate columns")
            continue
            
        df_clean = df.dropna(subset=['center_x', 'center_y'])
        if df_clean.empty:
            print(f"Warning: Skipping {animal_name} - no valid coordinate data")
            continue
        
        # Create 2D histogram for this animal
        x_coords = df_clean['center_x'].values
        y_coords = df_clean['center_y'].values
        
        counts, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])
        occupancy_maps[animal_name] = counts
    
    if not occupancy_maps:
        raise ValueError("No valid occupancy data could be generated")
    
    # Create territorial ownership map
    territorial_map = np.zeros((bins, bins), dtype=int)
    max_occupancy_map = np.zeros((bins, bins))
    
    # For each spatial bin, find which animal spent most time there
    for i in range(bins):
        for j in range(bins):
            max_count = 0
            dominant_animal_idx = -1
            
            for animal_idx, (animal_name, occupancy) in enumerate(occupancy_maps.items()):
                if occupancy[i, j] > max_count and occupancy[i, j] >= min_occupancy:
                    max_count = occupancy[i, j]
                    dominant_animal_idx = animal_idx
            
            territorial_map[i, j] = dominant_animal_idx
            max_occupancy_map[i, j] = max_count
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap for territorial display
    from matplotlib.colors import ListedColormap
    
    # Add neutral color for unoccupied areas
    territorial_colors = ['lightgray'] + [colors[i] for i in range(len(animal_names))]
    cmap = ListedColormap(territorial_colors)
    
    # Plot territorial map
    extent = [x_min, x_max, y_min, y_max]
    im = ax.imshow(territorial_map.T, extent=extent, origin='lower', 
                   cmap=cmap, vmin=-1, vmax=len(animal_names)-1, alpha=0.8)
    
    # Overlay occupancy intensity as contours
    X, Y = np.meshgrid(x_edges[:-1] + np.diff(x_edges)/2, 
                       y_edges[:-1] + np.diff(y_edges)/2)
    
    # Plot contour lines showing occupancy intensity
    contours = ax.contour(X, Y, max_occupancy_map.T, levels=5, colors='black', 
                         alpha=0.3, linewidths=0.5)
    
    # Add animal paths as thin lines for reference
    for i, (animal_name, df) in enumerate(tracking_dict.items()):
        if animal_name not in occupancy_maps:
            continue
            
        df_clean = df.dropna(subset=['center_x', 'center_y'])
        if df_clean.empty:
            continue
            
        x_coords = df_clean['center_x'].values
        y_coords = df_clean['center_y'].values
        
        ax.plot(x_coords, y_coords, '-', color=colors[i], 
               linewidth=0.5, alpha=0.4)
    
    # Formatting
    ax.set_xlabel('X Coordinate (pixels)')
    ax.set_ylabel('Y Coordinate (pixels)')
    ax.invert_yaxis()
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightgray', label='Unoccupied')]
    for i, animal_name in enumerate(animal_names):
        if animal_name in occupancy_maps:
            legend_elements.append(Patch(facecolor=colors[i], label=animal_name))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set title
    if title is None:
        title = f'Territorial Occupancy Map ({len(animal_names)} animals)'
    ax.set_title(title)
    
    # Add text box with statistics
    total_bins = bins * bins
    occupied_bins = np.sum(territorial_map >= 0)
    stats_text = f'Grid: {bins}×{bins}\nOccupied: {occupied_bins}/{total_bins} bins\nMin occupancy: {min_occupancy} frames'
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    print(f"Created territorial occupancy map for {len(animal_names)} animals")
    
    # Print territorial statistics
    for i, animal_name in enumerate(animal_names):
        if animal_name in occupancy_maps:
            territory_size = np.sum(territorial_map == i)
            print(f"  {animal_name}: {territory_size} territorial bins ({territory_size/total_bins*100:.1f}%)")
    
    return fig


def plot_voronoi_territories(tracking_dict: Dict[str, pd.DataFrame],
                           figsize: Tuple[int, int] = (12, 10),
                           colors: Optional[List[str]] = None,
                           title: Optional[str] = None,
                           bins: int = 50,
                           min_occupancy: int = 20,
                           show_paths: bool = False,
                           show_seeds: bool = False,
                           alpha: float = 0.6) -> plt.Figure:
    """
    Create a Voronoi diagram-based territorial map using binned occupancy data.
    
    Uses spatial binning to identify significant occupancy areas and creates Voronoi
    territories based on these high-occupancy locations, excluding areas where animals
    just passed through briefly.
    
    Args:
        tracking_dict: Dictionary where keys are animal names and values are DataFrames
        figsize: Figure size as (width, height)
        colors: List of colors for each animal. If None, uses default color palette
        title: Custom title for the plot
        bins: Number of spatial bins for occupancy calculation
        min_occupancy: Minimum frames required for a bin to be used as Voronoi seed
        show_paths: If True, overlay animal paths
        show_seeds: If True, display the seed points used for Voronoi calculation
        alpha: Transparency of Voronoi regions
        
    Returns:
        matplotlib.Figure: The created figure showing Voronoi territories
    """
    if not tracking_dict:
        raise ValueError("No tracking data provided")
    
    # Set up colors for animals
    animal_names = list(tracking_dict.keys())
    if colors is None:
        colors = sns.color_palette("Set2", len(animal_names))
    elif len(colors) < len(animal_names):
        colors = (colors * ((len(animal_names) // len(colors)) + 1))[:len(animal_names)]
    
    # Find overall coordinate bounds
    all_x_coords = []
    all_y_coords = []
    
    for animal_name, df in tracking_dict.items():
        if 'center_x' in df.columns and 'center_y' in df.columns:
            df_clean = df.dropna(subset=['center_x', 'center_y'])
            if not df_clean.empty:
                all_x_coords.extend(df_clean['center_x'].values)
                all_y_coords.extend(df_clean['center_y'].values)
    
    if not all_x_coords:
        raise ValueError("No valid coordinate data found for any animal")
    
    x_min, x_max = np.min(all_x_coords), np.max(all_x_coords)
    y_min, y_max = np.min(all_y_coords), np.max(all_y_coords)
    
    # Create spatial bins
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)
    
    # Calculate bin centers for Voronoi seeds
    x_centers = x_edges[:-1] + np.diff(x_edges) / 2
    y_centers = y_edges[:-1] + np.diff(y_edges) / 2
    
    # Initialize occupancy data
    occupancy_maps = {}
    animal_data = {}
    
    # Calculate occupancy maps for each animal
    for animal_idx, (animal_name, df) in enumerate(tracking_dict.items()):
        if 'center_x' not in df.columns or 'center_y' not in df.columns:
            print(f"Warning: Skipping {animal_name} - missing coordinate columns")
            continue
            
        df_clean = df.dropna(subset=['center_x', 'center_y'])
        if df_clean.empty:
            print(f"Warning: Skipping {animal_name} - no valid coordinate data")
            continue
        
        # Create 2D histogram for this animal
        x_coords = df_clean['center_x'].values
        y_coords = df_clean['center_y'].values
        
        counts, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])
        occupancy_maps[animal_name] = counts
        
        animal_data[animal_name] = {
            'all_coords': (x_coords, y_coords),
            'color': colors[animal_idx]
        }
    
    if not occupancy_maps:
        raise ValueError("No valid occupancy data could be generated")
    
    # Collect Voronoi seed points from high-occupancy bins
    all_points = []
    point_labels = []
    
    for animal_idx, (animal_name, occupancy) in enumerate(occupancy_maps.items()):
        # Find bins with sufficient occupancy
        high_occupancy_indices = np.where(occupancy >= min_occupancy)
        
        if len(high_occupancy_indices[0]) == 0:
            print(f"Warning: No high-occupancy areas found for {animal_name}")
            continue
        
        # Create seed points from bin centers of high-occupancy areas
        for i, j in zip(high_occupancy_indices[0], high_occupancy_indices[1]):
            seed_point = [x_centers[i], y_centers[j]]
            all_points.append(seed_point)
            point_labels.append(animal_idx)
    
    if not all_points:
        raise ValueError("No valid seed points found for Voronoi diagram")
    
    # Convert to numpy arrays
    all_points = np.array(all_points)
    point_labels = np.array(point_labels)
    
    # Create Voronoi diagram
    try:
        vor = Voronoi(all_points)
    except Exception as e:
        raise ValueError(f"Failed to create Voronoi diagram: {e}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add some padding for plot bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding_x = x_range * 0.02
    padding_y = y_range * 0.02
    
    plot_x_min, plot_x_max = x_min - padding_x, x_max + padding_x
    plot_y_min, plot_y_max = y_min - padding_y, y_max + padding_y
    
    # Plot Voronoi regions
    from matplotlib.patches import Polygon
    
    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        
        if not region or -1 in region:
            # Skip infinite regions
            continue
        
        # Get vertices of the region
        vertices = vor.vertices[region]
        
        # Check if region is within plot bounds (approximately)
        if (vertices[:, 0].min() > plot_x_max or vertices[:, 0].max() < plot_x_min or
            vertices[:, 1].min() > plot_y_max or vertices[:, 1].max() < plot_y_min):
            continue
        
        # Color region based on which animal the point belongs to
        animal_idx = point_labels[point_idx]
        color = colors[animal_idx]
        
        # Create and add polygon
        polygon = Polygon(vertices, closed=True, facecolor=color, 
                         edgecolor=color, alpha=alpha, linewidth=0)
        ax.add_patch(polygon)
    
    # Show seed points as small dots if requested
    if show_seeds:
        for animal_idx, (animal_name, data) in enumerate(animal_data.items()):
            if animal_name in occupancy_maps:
                # Get seed points for this animal
                animal_seeds = all_points[point_labels == animal_idx]
                if len(animal_seeds) > 0:
                    ax.scatter(animal_seeds[:, 0], animal_seeds[:, 1], c=data['color'], 
                              s=20, alpha=0.9, edgecolors='white', linewidths=0.5,
                              marker='s', zorder=10)
    
    # Show animal paths if requested
    if show_paths:
        for animal_name, data in animal_data.items():
            x_coords, y_coords = data['all_coords']
            ax.plot(x_coords, y_coords, '-', color=data['color'], 
                   linewidth=1, alpha=0.5, zorder=5)
    
    # Set plot limits
    # ax.set_xlim(plot_x_min, plot_x_max)
    # ax.set_ylim(plot_y_min, plot_y_max)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Formatting
    ax.set_xlabel('X Coordinate (pixels)')
    ax.set_ylabel('Y Coordinate (pixels)')
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = []
    
    for animal_idx, animal_name in enumerate(animal_names):
        if animal_name in animal_data:
            legend_elements.append(Patch(facecolor=colors[animal_idx], 
                                       alpha=alpha, label=f'{animal_name} territory'))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set title
    if title is None:
        title = f'Voronoi Territorial Diagram (Occupancy-Based)'
    ax.set_title(title)
    
    # Add statistics text
    n_points = len(all_points)
    n_regions = len([r for r in vor.regions if r and -1 not in r])
    
    stats_text = f'Seed points: {n_points}\nVoronoi regions: {n_regions}\nGrid: {bins}×{bins}\nMin occupancy: {min_occupancy}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    print(f"Created occupancy-based Voronoi territorial diagram with {n_points} seed points from {len(animal_names)} animals")
    
    # Print statistics for each animal
    for animal_idx, (animal_name, data) in enumerate(animal_data.items()):
        if animal_name in occupancy_maps:
            n_animal_seeds = np.sum(point_labels == animal_idx)
            total_occupancy = np.sum(occupancy_maps[animal_name])
            high_occ_bins = np.sum(occupancy_maps[animal_name] >= min_occupancy)
            # print(f"  {animal_name}: {n_animal_seeds} seed points, {high_occ_bins} high-occupancy bins, {total_occupancy:.0f} total frames")
    
    return fig


def plot_proximity_network(tracking_dict: Dict[str, pd.DataFrame],
                         proximity_threshold: float = 100,
                         min_interaction_time: int = 30,
                         figsize: Tuple[int, int] = (10, 8),
                         node_size_factor: float = 1000,
                         title: Optional[str] = None,
                         colors: Optional[List[str]] = None,
                         layout_type: str = 'spring',
                         animals: Optional[List[str]] = None) -> plt.Figure:
    """
    Create a network graph showing social connections between animals based on proximity.
    
    Analyzes tracking data to identify when animals are in close proximity and creates
    a network visualization where nodes represent animals and edges represent the strength
    of their social interactions (time spent in proximity).
    
    Args:
        tracking_dict: Dictionary where keys are animal names and values are DataFrames
        proximity_threshold: Distance threshold (pixels) for considering animals "close"
        min_interaction_time: Minimum frames of proximity to create an edge
        figsize: Figure size as (width, height)
        node_size_factor: Scaling factor for node sizes
        title: Custom title for the plot
        colors: List of colors for each animal. If None, uses default palette
        layout_type: Network layout algorithm ('spring', 'circular', 'kamada_kawai')
        animals: Optional list of animal names to include. If None, uses all animals
        
    Returns:
        matplotlib.Figure: The created network visualization
    """
    # Filter tracking_dict to include only specified animals
    if animals is not None:
        # Validate that all specified animals exist in tracking_dict
        missing_animals = [animal for animal in animals if animal not in tracking_dict]
        if missing_animals:
            raise ValueError(f"Animals not found in tracking data: {missing_animals}")
        
        # Filter the tracking dictionary
        tracking_dict = {animal: tracking_dict[animal] for animal in animals}
    
    if len(tracking_dict) < 2:
        raise ValueError("Need at least 2 animals for proximity analysis")
    
    animal_names = list(tracking_dict.keys())
    n_animals = len(animal_names)
    
    # Set up colors
    if colors is None:
        colors = sns.color_palette("Set2", n_animals)
    elif len(colors) < n_animals:
        colors = (colors * ((n_animals // len(colors)) + 1))[:n_animals]
    
    # Clean tracking data and ensure temporal alignment
    cleaned_data = {}
    max_frames = 0
    
    for animal_name, df in tracking_dict.items():
        if 'center_x' not in df.columns or 'center_y' not in df.columns:
            print(f"Warning: Skipping {animal_name} - missing coordinate columns")
            continue
            
        df_clean = df.dropna(subset=['center_x', 'center_y']).copy()
        if df_clean.empty:
            print(f"Warning: Skipping {animal_name} - no valid coordinate data")
            continue
        
        # Ensure we have frame numbers for temporal alignment
        if 'frame' not in df_clean.columns:
            df_clean['frame'] = range(len(df_clean))
        
        cleaned_data[animal_name] = df_clean
        max_frames = max(max_frames, df_clean['frame'].max())
    
    if len(cleaned_data) < 2:
        raise ValueError("Need at least 2 animals with valid data for proximity analysis")
    
    # Create numpy arrays for vectorized distance calculations
    animal_names = list(cleaned_data.keys())
    n_animals = len(animal_names)
    max_frame = int(max_frames) + 1
    
    # Initialize coordinate arrays: [animal_index, frame] = coordinate (NaN if missing)
    x_coords = np.full((n_animals, max_frame), np.nan)
    y_coords = np.full((n_animals, max_frame), np.nan)
    
    # Fill coordinate arrays
    for animal_idx, animal_name in enumerate(animal_names):
        df = cleaned_data[animal_name]
        frame_indices = df['frame'].astype(int).values
        x_coords[animal_idx, frame_indices] = df['center_x'].values
        y_coords[animal_idx, frame_indices] = df['center_y'].values
    
    # Calculate pairwise proximity interactions using vectorized operations
    proximity_matrix = np.zeros((n_animals, n_animals))
    interaction_details = {}
    
    print(f"Analyzing proximity interactions with threshold {proximity_threshold} pixels...")
    
    for i, (animal1, animal2) in enumerate(combinations(animal_names, 2)):
        animal1_idx = animal_names.index(animal1)
        animal2_idx = animal_names.index(animal2)
        
        # Get coordinate arrays for both animals
        x1 = x_coords[animal1_idx, :]
        y1 = y_coords[animal1_idx, :]
        x2 = x_coords[animal2_idx, :]
        y2 = y_coords[animal2_idx, :]
        
        # Find frames where both animals have valid coordinates
        valid_frames_mask = ~(np.isnan(x1) | np.isnan(y1) | np.isnan(x2) | np.isnan(y2))
        
        if np.sum(valid_frames_mask) < min_interaction_time:
            continue
        
        # Calculate distances for all valid frames at once (vectorized)
        dx = x1[valid_frames_mask] - x2[valid_frames_mask]
        dy = y1[valid_frames_mask] - y2[valid_frames_mask]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Calculate interaction metrics
        proximity_mask = distances <= proximity_threshold
        proximity_frames = np.sum(proximity_mask)
        total_frames = len(distances)
        
        if proximity_frames < min_interaction_time:
            continue
        
        # Get frame numbers for interaction events
        valid_frame_indices = np.where(valid_frames_mask)[0]
        interaction_events = valid_frame_indices[proximity_mask].tolist()
        
        # Calculate statistics
        total_distance = np.sum(distances)
        min_distance = np.min(distances)
        avg_distance = np.mean(distances)
        interaction_strength = proximity_frames / total_frames
            
        # Store interaction metrics
        idx1 = animal1_idx
        idx2 = animal2_idx
        
        proximity_matrix[idx1, idx2] = interaction_strength
        proximity_matrix[idx2, idx1] = interaction_strength
        
        interaction_details[(animal1, animal2)] = {
            'proximity_frames': proximity_frames,
            'total_frames': total_frames,
            'interaction_strength': interaction_strength,
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'interaction_events': interaction_events
        }
    
    print(f"Found {len(interaction_details)} significant interactions between animals")

    # Create network graph
    G = nx.Graph()
    
    # Add nodes (animals)
    for i, animal_name in enumerate(animal_names):
        # Node size based on total movement/activity
        df = cleaned_data[animal_name]
        total_movement = calculate_path_statistics(df)['total_distance']
        
        G.add_node(animal_name, 
                   activity=total_movement,
                   color=colors[i],
                   frames=len(df))
    
    # Add edges (interactions)
    edge_weights = []
    edge_colors = []
    
    for (animal1, animal2), details in interaction_details.items():
        weight = details['interaction_strength']
        G.add_edge(animal1, animal2, 
                   weight=weight,
                   proximity_frames=details['proximity_frames'],
                   avg_distance=details['avg_distance'],
                   min_distance=details['min_distance'])
        
        edge_weights.append(weight * 10)  # Scale for visualization
        # Edge color intensity based on interaction strength
        edge_colors.append(weight)
    
    if len(G.edges()) == 0:
        print("Warning: No significant interactions found between animals")
        # Create figure with just nodes
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No significant interactions detected\n' +
               f'(threshold: {proximity_threshold} pixels, min time: {min_interaction_time} frames)',
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title(title or 'Proximity Network (No Interactions)')
        return fig
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout algorithm
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Draw edges with varying thickness and color
    if edge_weights:
        edges = nx.draw_networkx_edges(G, pos, 
                                     width=edge_weights,
                                     edge_color=edge_colors,
                                     edge_cmap=plt.cm.Reds,
                                     alpha=0.6,
                                     ax=ax)
    
    # Draw nodes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_sizes = [G.nodes[node]['activity'] / max([G.nodes[n]['activity'] for n in G.nodes()]) * 
                  node_size_factor + 200 for node in G.nodes()]  # Base size + scaled activity
    
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=2,
                          ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos,
                           font_size=12,
                           font_weight='bold',
                           font_color='white',
                           ax=ax)
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # Animal legend
    for i, animal_name in enumerate(animal_names):
        activity = G.nodes[animal_name]['activity']
        frames = G.nodes[animal_name]['frames']
        legend_elements.append(Patch(facecolor=colors[i], 
                                   label=f'{animal_name} ({frames} frames, {activity:.0f}px moved)'))
    
    # Interaction strength legend
    if edge_weights:
        legend_elements.extend([
            Line2D([0], [0], color='red', lw=1, alpha=0.6, label='Weak interaction'),
            Line2D([0], [0], color='red', lw=3, alpha=0.6, label='Strong interaction')
        ])
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Formatting
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Set title
    if title is None:
        title = f'Animal Proximity Network\n({len(G.nodes())} animals, {len(G.edges())} interactions)'
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    
    # Add network statistics
    stats_text = f'Proximity threshold: {proximity_threshold} pixels\n'
    stats_text += f'Min interaction time: {min_interaction_time} frames\n'
    stats_text += f'Network density: {nx.density(G):.3f}'
    
    if len(G.edges()) > 0:
        avg_clustering = nx.average_clustering(G)
        stats_text += f'\nAverage clustering: {avg_clustering:.3f}'
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Print detailed interaction statistics
    print(f"\nProximity Network Analysis Results:")
    print(f"  Animals analyzed: {len(animal_names)}")
    print(f"  Significant interactions: {len(interaction_details)}")
    print(f"  Network density: {nx.density(G):.3f}")
    
    # if interaction_details:
    #     print("\nDetailed Interactions:")
    #     for (animal1, animal2), details in interaction_details.items():
            # print(f"  {animal1} ↔ {animal2}:")
            # print(f"    Proximity time: {details['proximity_frames']} / {details['total_frames']} frames ({details['interaction_strength']*100:.1f}%)")
            # print(f"    Avg distance: {details['avg_distance']:.1f} pixels")
            # print(f"    Min distance: {details['min_distance']:.1f} pixels")
    
    return fig


def save_visualization(fig: plt.Figure, output_path: Union[str, Path],
                      dpi: int = 300, format: str = 'png') -> None:
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: The figure to save
        output_path: Path where to save the figure
        dpi: Resolution for the saved figure
        format: File format ('png', 'pdf', 'svg', etc.)
    """
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Path Visualization Example:")
    print("-" * 40)
    
    # Create example tracking data
    np.random.seed(42)
    n_frames = 1000
    
    # Mouse path (small movements)
    mouse_x = 100 + np.cumsum(np.random.randn(n_frames) * 2)
    mouse_y = 100 + np.cumsum(np.random.randn(n_frames) * 2)
    
    # Rat path (larger movements)  
    rat_x = 200 + np.cumsum(np.random.randn(n_frames) * 4)
    rat_y = 150 + np.cumsum(np.random.randn(n_frames) * 3)
    
    # Create DataFrames
    mouse_df = pd.DataFrame({
        'frame': range(n_frames),
        'center_x': mouse_x,
        'center_y': mouse_y
    })
    
    rat_df = pd.DataFrame({
        'frame': range(n_frames),
        'center_x': rat_x,
        'center_y': rat_y
    })
    
    tracking_data = {
        'mouse1': mouse_df,
        'rat1': rat_df
    }
    
    # Plot individual paths
    print("Creating individual path plots...")
    mouse_fig = plot_animal_path(mouse_df, 'mouse1')
    rat_fig = plot_animal_path(rat_df, 'rat1')
    
    # Plot multiple paths
    print("Creating multi-animal comparison...")
    multi_fig = plot_multiple_paths(tracking_data)
    
    # Create heatmaps
    print("Creating position heatmaps...")
    mouse_heatmap = plot_path_heatmap(mouse_df, 'mouse1')
    rat_heatmap = plot_path_heatmap(rat_df, 'rat1')
    
    # Calculate statistics
    print("\nPath Statistics:")
    for animal_name, df in tracking_data.items():
        stats = calculate_path_statistics(df)
        print(f"\n{animal_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
    
    plt.show()