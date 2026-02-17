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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import seaborn as sns


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
    
    # Plot path
    x_coords = df_clean['center_x'].values
    y_coords = df_clean['center_y'].values
    
    ax.plot(x_coords, y_coords, '-', color=color, linewidth=1.5, alpha=0.7, label=animal_name)
    
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
    
    x_coords = df_clean['center_x'].values
    y_coords = df_clean['center_y'].values
    
    # Create 2D histogram
    counts, xedges, yedges, im = ax.hist2d(x_coords, y_coords, bins=bins, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Time Spent (frames)')
    
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