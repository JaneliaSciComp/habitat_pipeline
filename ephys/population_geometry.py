"""
Population-Level Neural Geometry Analysis Module

This module provides tools for analyzing neural population dynamics and geometry:
- Population firing rate matrix construction
- Dimensionality reduction (PCA, UMAP)
- Neural state space trajectory analysis
- Cross-condition comparisons (event types, opponents)
- Advanced visualization of population dynamics

Author: Mikhail Proskurin
Created: March 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Scientific computing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# Try to import UMAP (optional dependency)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


class PopulationGeometryAnalyzer:
    """
    Analyze neural population geometry and dynamics across behavioral contexts.
    
    This class provides methods to:
    1. Construct population firing rate matrices
    2. Apply dimensionality reduction techniques
    3. Analyze neural trajectories through state space
    4. Compare population dynamics across conditions
    5. Visualize population-level neural geometry
    """
    
    def __init__(self, ks_data, behavior_data=None):
        """
        Initialize the population geometry analyzer.
        
        Parameters:
        -----------
        ks_data : KilosortData
            Processed electrophysiology data
        behavior_data : BehavioralEventsData, optional
            Behavioral event data for condition-based analysis
        """
        self.ks_data = ks_data
        self.behavior_data = behavior_data
        
        # Analysis results storage
        self.population_matrices = {}
        self.reduced_data = {}
        self.analysis_metadata = {}
    
    def construct_population_matrix(self, 
                                  event_starts: np.ndarray,
                                  event_ends: np.ndarray,
                                  event_labels: np.ndarray,
                                  time_window: Tuple[float, float] = (-2.0, 2.0),
                                  time_bin_size: float = 0.1,
                                  alignment: str = 'start',
                                  normalize_method: str = 'none',
                                  min_spikes_per_bin: int = 0,
                                  use_quality_cells: bool = True) -> Dict:
        """
        Construct population firing rate matrices aligned to behavioral events.
        
        Parameters:
        -----------
        event_starts : np.ndarray
            Event start times (seconds)
        event_ends : np.ndarray  
            Event end times (seconds)
        event_labels : np.ndarray
            Event condition labels
        time_window : tuple, default=(-2.0, 2.0)
            Time window around alignment point (seconds)
        time_bin_size : float, default=0.1
            Size of time bins (seconds)
        alignment : str, default='start'
            Alignment point ('start', 'end', 'center')
        normalize_method : str, default='none'
            Normalization method ('zscore', 'baseline', 'none')
        min_spikes_per_bin : int, default=0
            Minimum spikes per bin for inclusion
        use_quality_cells : bool, default=True
            Whether to filter cells by quality metrics before analysis
            
        Returns:
        --------
        dict : Population matrix data with metadata
        """
        
        # Determine alignment times
        if alignment == 'start':
            align_times = event_starts
        elif alignment == 'end':
            align_times = event_ends
        elif alignment == 'center':
            align_times = (event_starts + event_ends) / 2
        else:
            raise ValueError(f"Unknown alignment: {alignment}")
        
        # Handle quality cell filtering
        if use_quality_cells:
            # Check if filter results already exist
            if hasattr(self.ks_data, 'filter_results') and self.ks_data.filter_results is not None:
                filter_results = self.ks_data.filter_results
            else:
                filter_results = self.ks_data.filter_cells_by_firing_patterns()
            
            # Get indices of quality cells
            selected_cell_indices = [i for i, cluster_id in enumerate(self.ks_data.ks_ids) 
                                   if cluster_id in filter_results['passed_clusters']]
            selected_cluster_ids = filter_results['passed_clusters']
        else:
            # Use all cells
            selected_cell_indices = list(range(len(self.ks_data.ks_ids)))
            selected_cluster_ids = self.ks_data.ks_ids
        
        if len(selected_cluster_ids) == 0:
            raise ValueError("No cells selected for analysis (all cells failed quality criteria)")
        
        # Create time bins
        n_bins = int((time_window[1] - time_window[0]) / time_bin_size)
        time_bins = np.linspace(time_window[0], time_window[1], n_bins + 1)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        # Get unique conditions
        unique_labels = np.unique(event_labels)
        n_conditions = len(unique_labels)
        n_cells = len(selected_cluster_ids)
        
        # Initialize population matrices
        # Shape: [conditions, events_per_condition, cells, time_bins]
        population_data = {}
        event_counts = {}
        
        for label in unique_labels:
            label_mask = event_labels == label
            label_events = np.sum(label_mask)
            event_counts[label] = label_events
            
            # Initialize matrix for this condition
            population_data[label] = np.zeros((label_events, n_cells, n_bins))
            
            label_align_times = align_times[label_mask]
            
            # Fill matrix for each event and cell
            for event_idx, align_time in enumerate(label_align_times):
                for matrix_cell_idx, (selected_idx, cell_id) in enumerate(zip(selected_cell_indices, selected_cluster_ids)):
                    spike_times = self.ks_data.spike_times_by_cell[selected_idx]
                    
                    # Get spikes in window
                    window_start = align_time + time_window[0]
                    window_end = align_time + time_window[1]
                    
                    window_spikes = spike_times[
                        (spike_times >= window_start) & 
                        (spike_times < window_end)
                    ]
                    
                    if len(window_spikes) > 0:
                        # Convert to relative times
                        relative_times = window_spikes - align_time
                        
                        # Bin spikes
                        spike_counts, _ = np.histogram(relative_times, bins=time_bins)
                        
                        # Convert to firing rate (spikes/second)
                        firing_rates = spike_counts / time_bin_size
                        
                        population_data[label][event_idx, matrix_cell_idx, :] = firing_rates
        
        # Apply normalization
        if normalize_method != 'none':
            population_data = self._normalize_population_data(
                population_data, normalize_method, time_window, bin_centers
            )
        
        # Create result structure
        result = {
            'population_data': population_data,
            'time_bins': bin_centers,
            'event_counts': event_counts,
            'unique_labels': unique_labels,
            'metadata': {
                'time_window': time_window,
                'time_bin_size': time_bin_size,
                'alignment': alignment,
                'normalize_method': normalize_method,
                'n_cells': n_cells,
                'n_bins': n_bins,
                'n_conditions': n_conditions,
                'total_events': len(event_starts),
                'selected_cell_ids': selected_cluster_ids
            }
        }
        
        # Store for later use
        matrix_key = f"{alignment}_{normalize_method}"
        self.population_matrices[matrix_key] = result
        
        return result
    
    def _normalize_population_data(self, pop_data: Dict, method: str, 
                                 time_window: Tuple[float, float], 
                                 bin_centers: np.ndarray) -> Dict:
        """Apply normalization to population data."""
        
        normalized_data = {}
        
        if method == 'zscore':
            # Z-score normalize across time for each cell
            for label, data in pop_data.items():
                # Shape: [events, cells, time_bins]
                normalized = np.zeros_like(data)
                
                for cell_idx in range(data.shape[1]):
                    cell_data = data[:, cell_idx, :].flatten()
                    if np.std(cell_data) > 0:
                        cell_zscore = zscore(cell_data)
                        normalized[:, cell_idx, :] = cell_zscore.reshape(data.shape[0], -1)
                    else:
                        normalized[:, cell_idx, :] = cell_data.reshape(data.shape[0], -1)
                
                normalized_data[label] = normalized
        
        elif method == 'baseline':
            # Baseline normalize using pre-event period
            baseline_mask = bin_centers < 0
            
            for label, data in pop_data.items():
                normalized = np.zeros_like(data)
                
                for cell_idx in range(data.shape[1]):
                    cell_data = data[:, cell_idx, :]
                    
                    # Calculate baseline for each event
                    baseline_rates = np.mean(cell_data[:, baseline_mask], axis=1, keepdims=True)
                    baseline_std = np.std(cell_data[:, baseline_mask], axis=1, keepdims=True)
                    
                    # Avoid division by zero
                    baseline_std[baseline_std == 0] = 1.0
                    
                    # Normalize: (firing_rate - baseline_mean) / baseline_std
                    normalized[:, cell_idx, :] = (cell_data - baseline_rates) / baseline_std
                
                normalized_data[label] = normalized
        
        else:
            normalized_data = pop_data
        
        return normalized_data
    
    def apply_dimensionality_reduction(self, 
                                     population_matrix: Dict,
                                     method: str = 'pca',
                                     n_components: int = 3,
                                     **method_kwargs) -> Dict:
        """
        Apply dimensionality reduction to population data.
        
        Parameters:
        -----------
        population_matrix : dict
            Result from construct_population_matrix()
        method : str, default='pca'
            Reduction method ('pca', 'umap')
        n_components : int, default=3
            Number of dimensions to reduce to
        **method_kwargs : 
            Additional arguments for the reduction method
            
        Returns:
        --------
        dict : Reduced data with trajectories and metadata
        """
        
        pop_data = population_matrix['population_data']
        time_bins = population_matrix['time_bins']
        unique_labels = population_matrix['unique_labels']
        
        # Prepare data for reduction
        # Flatten across events and time, keep cells as features
        all_data = []
        all_labels = []
        all_times = []
        all_events = []
        
        for label in unique_labels:
            data = pop_data[label]  # [events, cells, time_bins]
            n_events, n_cells, n_time = data.shape
            
            # Reshape to [events * time_bins, cells]
            reshaped = data.reshape(-1, n_cells)
            all_data.append(reshaped)
            
            # Track labels, times, and events
            label_array = np.repeat(label, n_events * n_time)
            all_labels.extend(label_array)
            
            time_array = np.tile(time_bins, n_events)
            all_times.extend(time_array)
            
            event_array = np.repeat(np.arange(n_events), n_time)
            all_events.extend(event_array)
        
        # Combine all data
        X = np.vstack(all_data)  # [total_timepoints, cells]
        labels = np.array(all_labels)
        times = np.array(all_times)
        events = np.array(all_events)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, **method_kwargs)
            X_reduced = reducer.fit_transform(X)
            
            # Calculate explained variance
            explained_var = reducer.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            reduction_info = {
                'explained_variance_ratio': explained_var,
                'cumulative_variance': cumulative_var,
                'components': reducer.components_,
                'reducer': reducer
            }
        
        elif method.lower() == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            
            # Default UMAP parameters
            umap_params = {
                'n_neighbors': 15,
                'min_dist': 0.1,
                'metric': 'euclidean',
                'random_state': 42
            }
            umap_params.update(method_kwargs)
            
            reducer = umap.UMAP(n_components=n_components, **umap_params)
            X_reduced = reducer.fit_transform(X)
            
            reduction_info = {
                'reducer': reducer,
                'umap_params': umap_params
            }
            
        
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Organize reduced data by condition and reshape back to trajectories
        reduced_trajectories = {}
        
        start_idx = 0
        for label in unique_labels:
            n_events = population_matrix['event_counts'][label]
            n_time = len(time_bins)
            
            end_idx = start_idx + (n_events * n_time)
            label_reduced = X_reduced[start_idx:end_idx]
            
            # Reshape back to [events, time_bins, components]
            trajectory = label_reduced.reshape(n_events, n_time, n_components)
            reduced_trajectories[label] = trajectory
            
            start_idx = end_idx
        
        # Create result structure
        result = {
            'reduced_trajectories': reduced_trajectories,
            'reduced_data_flat': X_reduced,
            'labels_flat': labels,
            'times_flat': times,
            'events_flat': events,
            'time_bins': time_bins,
            'unique_labels': unique_labels,
            'method': method,
            'n_components': n_components,
            'reduction_info': reduction_info,
            'original_shape': X.shape,
        }
        
        # Store for later use
        reduction_key = f"{method}_{n_components}d"
        self.reduced_data[reduction_key] = result
        
        return result
        
    def plot_population_dynamics(self, 
                               reduced_data: Dict,
                               show_individual: bool = False,
                               time_range: Optional[Tuple[float, float]] = None,
                               figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """
        Plot 3D neural population trajectories from reduced dimensional data.
        
        Parameters:
        -----------
        reduced_data : dict
            Result from apply_dimensionality_reduction()
        show_individual : bool, default=False
            Whether to show all individual trajectories (True) or just mean trajectories (False)
        time_range : tuple, optional
            Time range to plot (start, end) in seconds
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : Generated figure
        """
        
        trajectories = reduced_data['reduced_trajectories']
        time_bins = reduced_data['time_bins']
        unique_labels = reduced_data['unique_labels']
        
        # Filter time range if specified
        if time_range is not None:
            time_mask = (time_bins >= time_range[0]) & (time_bins <= time_range[1])
        else:
            time_mask = slice(None)
        
        # Color scheme - one color per condition
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        for i, label in enumerate(unique_labels):
            traj = trajectories[label]  # [events, time_bins, components]
            n_events = traj.shape[0]
            
            if show_individual:
                # Plot all individual trajectories with same color per condition
                for event_idx in range(n_events):
                    trajectory = traj[event_idx][time_mask]  # [time_bins, components]
                    
                    # Use lighter alpha for individual trajectories
                    alpha = 0.6 if n_events <= 5 else 0.3
                    linewidth = 1.5 if n_events <= 5 else 1.0
                    
                    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                           color=colors[i], alpha=alpha, linewidth=linewidth,
                           label=f'{label}' if event_idx == 0 else "")
                    
                    # Mark start and end points for each trajectory
                    ax.scatter(*trajectory[0], color=colors[i], s=30, marker='o', alpha=0.7)
                    ax.scatter(*trajectory[-1], color=colors[i], s=30, marker='s', alpha=0.7)
            else:
                # Plot only mean trajectory per condition
                mean_traj = np.mean(traj, axis=0)[time_mask]  # [time_bins, components]
                
                ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2],
                       color=colors[i], linewidth=3, label=f'{label} (n={n_events})')
                
                # Mark start and end points
                ax.scatter(*mean_traj[0], color=colors[i], s=100, marker='o', alpha=0.8)
                ax.scatter(*mean_traj[-1], color=colors[i], s=100, marker='s', alpha=0.8)
        
        # Set labels and title
        method_name = reduced_data['method'].upper()
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_zlabel(f'{method_name} Component 3')
        
        if show_individual:
            ax.set_title('Neural Population Trajectories - All Individual Trials')
        else:
            ax.set_title('Neural Population Trajectories - Mean per Condition')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"✅ 3D trajectory visualization created!")
        return fig
    

def run_population_analysis_pipeline(ks_data, 
                                   behavior_data,
                                   animal_of_interest: str,
                                   behavior_type: str = 'EC',
                                   time_window: Tuple[float, float] = (-2.0, 2.0),
                                   time_bin_size: float = 0.5,
                                   reduction_method: str = 'pca',
                                   n_components: int = 3,
                                   min_events_per_class: int = 10,
                                   use_quality_cells: bool = True,
                                   create_plots: bool = True,
                                   save_results: bool = False,
                                   save_path: Optional[str] = None) -> Dict:
    """
    Complete pipeline for population-level neural geometry analysis.
    
    Parameters:
    -----------
    ks_data : KilosortData
        Processed electrophysiology data
    behavior_data : BehavioralEventsData
        Behavioral event data
    animal_of_interest : str
        Animal ID to analyze
    behavior_type : str, default='EC'
        Type of behavioral events to analyze
    time_window : tuple, default=(-2.0, 2.0)
        Time window around events (seconds)
    time_bin_size : float, default=0.5
        Size of time bins (seconds)  
    reduction_method : str, default='pca'
        Dimensionality reduction method ('pca', 'umap')
    n_components : int, default=3
        Number of dimensions for reduction
    min_events_per_class : int, default=10
        Minimum events per condition
    use_quality_cells : bool, default=True
        Whether to filter cells by quality metrics before analysis
    create_plots : bool, default=True
        Whether to create visualization plots
    save_results : bool, default=False
        Whether to save results to file
    save_path : str, optional
        Path to save results
        
    Returns:
    --------
    dict : Complete analysis results
    """
    
    # Extract behavioral events with opponent labels
    event_starts, event_ends, opponent_labels = behavior_data.extract_opponent_labels(
        animal_of_interest=animal_of_interest,
        behavior_type=behavior_type,
        min_events_per_class=min_events_per_class
    )
    
    if len(event_starts) == 0:
        print("No behavioral events found!")
        return {'status': 'failed', 'error': 'No events found'}
    
    # Initialize analyzer
    analyzer = PopulationGeometryAnalyzer(ks_data, behavior_data)
    
    # Step 1: Construct population matrices
    pop_matrix = analyzer.construct_population_matrix(
        event_starts=event_starts,
        event_ends=event_ends,
        event_labels=opponent_labels,
        time_window=time_window,
        time_bin_size=time_bin_size,
        alignment='start',
        normalize_method='none',
        use_quality_cells=use_quality_cells
    )
    
    # Step 2: Apply dimensionality reduction
    reduced_data = analyzer.apply_dimensionality_reduction(
        pop_matrix,
        method=reduction_method,
        n_components=n_components
    )
    
    # Step 3: Create visualizations
    figures = {}
    if create_plots:
        # Create both mean and individual trajectory plots
        try:
            # Plot mean trajectories
            fig_mean = analyzer.plot_population_dynamics(
                reduced_data,
                show_individual=False
            )
            figures['mean_trajectories'] = fig_mean
            plt.show()
            
            # Plot individual trajectories
            fig_individual = analyzer.plot_population_dynamics(
                reduced_data,
                show_individual=True
            )
            figures['individual_trajectories'] = fig_individual
            plt.show()
            
        except Exception as e:
            print(f"Error creating trajectory plots: {e}")
    
    # Step 4: Save results
    if save_results:
        if save_path is None:
            save_path = f"population_analysis_{animal_of_interest}_{behavior_type}_{reduction_method}.pkl"
        analyzer.save_analysis(save_path)
    
    # Compile results
    results = {
        'status': 'success',
        'analyzer': analyzer,
        'population_matrix': pop_matrix,
        'reduced_data': reduced_data,
        'figures': figures,
        'parameters': {
            'animal_of_interest': animal_of_interest,
            'behavior_type': behavior_type,
            'time_window': time_window,
            'time_bin_size': time_bin_size,
            'reduction_method': reduction_method,
            'n_components': n_components,
            'min_events_per_class': min_events_per_class
        }
    }
    
    return results


if __name__ == "__main__":
    """Command line interface for population analysis."""
    import argparse
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from ingestion.data_paths import DataStorageManager
        from ingestion.kilosort_data_import import KilosortData
        from video.behavioral_events import BehavioralEventsData
        from ingestion.ephys_sync import DataSyncManager
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Population-Level Neural Geometry Analysis")
    parser.add_argument("--animal_id", required=True, help="Animal ID")
    parser.add_argument("--session_id", required=True, help="Session ID")
    parser.add_argument("--behavior_type", default="F", help="Behavior type (default: F)")
    parser.add_argument("--reduction_method", default="pca", choices=["pca", "umap"], 
                       help="Dimensionality reduction method")
    parser.add_argument("--n_components", type=int, default=3, help="Number of components")
    parser.add_argument("--time_window", nargs=2, type=float, default=[-2.0, 2.0],
                       help="Time window around events")
    parser.add_argument("--time_bin_size", type=float, default=0.1, help="Time bin size")
    parser.add_argument("--use_quality_cells", action="store_true", default=True, help="Filter cells by quality metrics")
    parser.add_argument("--save_results", action="store_true", help="Save results to file")
    parser.add_argument("--save_path", help="Path to save results")
    
    args = parser.parse_args()
    
    print(f"Loading data for animal {args.animal_id}, session {args.session_id}...")
    
    # Load data
    data_storage = DataStorageManager(args.animal_id, args.session_id, auto_load=True)
    ks_data = KilosortData(data_storage)
    behavior_data = BehavioralEventsData(data_storage)
    
    # Synchronize behavioral data with ephys
    sync_manager = DataSyncManager(data_storage, dio_channel=1)
    behavior_data.synchronize_with_ephys(sync_manager, create_new_columns=True)
    
    # Run analysis pipeline
    results = run_population_analysis_pipeline(
        ks_data=ks_data,
        behavior_data=behavior_data,
        animal_of_interest=f"rat{args.animal_id}",
        behavior_type=args.behavior_type,
        time_window=tuple(args.time_window),
        time_bin_size=args.time_bin_size,
        reduction_method=args.reduction_method,
        n_components=args.n_components,
        use_quality_cells=args.use_quality_cells,
        create_plots=True,
        save_results=args.save_results,
        save_path=args.save_path
    )
    
    if results['status'] == 'success':
        print("Analysis completed successfully!")
    else:
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)