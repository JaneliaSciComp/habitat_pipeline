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
        self.trajectories = {}
        self.analysis_metadata = {}
    
    def construct_population_matrix(self, 
                                  event_starts: np.ndarray,
                                  event_ends: np.ndarray,
                                  event_labels: np.ndarray,
                                  time_window: Tuple[float, float] = (-2.0, 2.0),
                                  time_bin_size: float = 0.1,
                                  alignment: str = 'start',
                                  normalize_method: str = 'zscore',
                                  min_spikes_per_bin: int = 0) -> Dict:
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
        normalize_method : str, default='zscore'
            Normalization method ('zscore', 'baseline', 'none')
        min_spikes_per_bin : int, default=0
            Minimum spikes per bin for inclusion
            
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
        
        # Create time bins
        n_bins = int((time_window[1] - time_window[0]) / time_bin_size)
        time_bins = np.linspace(time_window[0], time_window[1], n_bins + 1)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        # Get unique conditions
        unique_labels = np.unique(event_labels)
        n_conditions = len(unique_labels)
        n_cells = len(self.ks_data.ks_ids)
        
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
                for cell_idx, cell_id in enumerate(self.ks_data.ks_ids):
                    spike_times = self.ks_data.spike_times_by_cell[cell_idx]
                    
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
                        
                        population_data[label][event_idx, cell_idx, :] = firing_rates
        
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
                'total_events': len(event_starts)
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
    

