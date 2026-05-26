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
from scipy import stats
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
            selected_cluster_ids, spike_times_list = self.ks_data.get_filtered_cells_spike_times()
        else:
            spike_times_list = self.ks_data.spike_times_by_cell
            selected_cluster_ids = list(self.ks_data.ks_ids)

        if len(spike_times_list) == 0:
            raise ValueError("No cells selected for analysis (all cells failed quality criteria)")
        
        # Create time bins
        n_bins = int((time_window[1] - time_window[0]) / time_bin_size)
        time_bins = np.linspace(time_window[0], time_window[1], n_bins + 1)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        # Get unique conditions
        unique_labels = np.unique(event_labels)
        n_conditions = len(unique_labels)
        n_cells = len(spike_times_list)
        
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
                for matrix_cell_idx, spike_times in enumerate(spike_times_list):
                    
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
            
            # Transpose to [events, time_bins, cells] then reshape to [events * time_bins, cells]
            # This preserves the (event, time) pairing for each observation
            transposed = data.transpose(0, 2, 1)  # [events, time_bins, cells]
            reshaped = transposed.reshape(-1, n_cells)  # [events * time_bins, cells]
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
                    ax.scatter(*trajectory[0,:3], color=colors[i], s=30, marker='o', alpha=0.7)
                    ax.scatter(*trajectory[-1,:3], color=colors[i], s=30, marker='s', alpha=0.7)
            else:
                # Plot only mean trajectory per condition
                mean_traj = np.mean(traj, axis=0)[time_mask]  # [time_bins, components]
                
                ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2],
                       color=colors[i], linewidth=3, label=f'{label} (n={n_events})')
                
                # Mark start and end points
                ax.scatter(*mean_traj[0,:3], color=colors[i], s=100, marker='o', alpha=0.8)
                ax.scatter(*mean_traj[-1,:3], color=colors[i], s=100, marker='s', alpha=0.8)
        
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
        
        return fig
    
    def plot_population_dynamics_interactive(self,
                                           reduced_data: Dict,
                                           show_individual: bool = False,
                                           time_range: Optional[Tuple[float, float]] = None):
        """
        Interactive 3D plot of neural population trajectories using plotly.
        
        Parameters:
        -----------
        reduced_data : dict
            Result from apply_dimensionality_reduction()
        show_individual : bool, default=False
            Whether to show all individual trajectories (True) or just mean trajectories (False)
        time_range : tuple, optional
            Time range to plot (start, end) in seconds
            
        Returns:
        --------
        plotly.graph_objects.Figure : Interactive 3D figure
        """
        import plotly.graph_objects as go
        
        trajectories = reduced_data['reduced_trajectories']
        time_bins = reduced_data['time_bins']
        unique_labels = reduced_data['unique_labels']
        method_name = reduced_data['method'].upper()
        
        # Filter time range if specified
        if time_range is not None:
            time_mask = (time_bins >= time_range[0]) & (time_bins <= time_range[1])
        else:
            time_mask = slice(None)
        
        # Color palette
        import plotly.express as px
        palette = px.colors.qualitative.Set1
        
        fig = go.Figure()
        
        for i, label in enumerate(unique_labels):
            color = palette[i % len(palette)]
            traj = trajectories[label]  # [events, time_bins, components]
            n_events = traj.shape[0]
            
            if show_individual:
                for event_idx in range(n_events):
                    t = traj[event_idx][time_mask]
                    fig.add_trace(go.Scatter3d(
                        x=t[:, 0], y=t[:, 1], z=t[:, 2],
                        mode='lines',
                        line=dict(color=color, width=3),
                        opacity=0.5,
                        name=str(label),
                        legendgroup=str(label),
                        showlegend=(event_idx == 0),
                    ))
                    # Start marker
                    fig.add_trace(go.Scatter3d(
                        x=[t[0, 0]], y=[t[0, 1]], z=[t[0, 2]],
                        mode='markers',
                        marker=dict(color=color, size=4, symbol='circle'),
                        showlegend=False, legendgroup=str(label),
                    ))
            else:
                mean_traj = np.mean(traj, axis=0)[time_mask]
                fig.add_trace(go.Scatter3d(
                    x=mean_traj[:, 0], y=mean_traj[:, 1], z=mean_traj[:, 2],
                    mode='lines',
                    line=dict(color=color, width=5),
                    name=f'{label} (n={n_events})',
                ))
                # Start / end markers
                fig.add_trace(go.Scatter3d(
                    x=[mean_traj[0, 0]], y=[mean_traj[0, 1]], z=[mean_traj[0, 2]],
                    mode='markers',
                    marker=dict(color=color, size=6, symbol='circle'),
                    name=f'{label} start', showlegend=False,
                ))
                fig.add_trace(go.Scatter3d(
                    x=[mean_traj[-1, 0]], y=[mean_traj[-1, 1]], z=[mean_traj[-1, 2]],
                    mode='markers',
                    marker=dict(color=color, size=6, symbol='square'),
                    name=f'{label} end', showlegend=False,
                ))
        
        title = 'Neural Population Trajectories'
        if show_individual:
            title += ' - All Individual Trials'
        else:
            title += ' - Mean per Condition'
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f'{method_name} Component 1',
                yaxis_title=f'{method_name} Component 2',
                zaxis_title=f'{method_name} Component 3',
            ),
            width=900,
            height=700,
        )
        
        return fig

    def plot_pca_summary(self, 
                        reduced_data: Dict,
                        pop_data: Dict,
                        figsize: Tuple[float, float] = (16, 12)) -> plt.Figure:
        """
        Create comprehensive PCA summary plots including covariance matrix, 
        component weights, and variance explained.
        
        Parameters:
        -----------
        reduced_data : dict
            Result from apply_dimensionality_reduction() with method='pca'
        pop_data : dict
            Original population data from construct_population_matrix()['population_data']
        figsize : tuple, default=(16, 12)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure : Generated figure with PCA summary
        """
        
        if reduced_data['method'].lower() != 'pca':
            raise ValueError("This method only works with PCA reduced data")
        
        reduction_info = reduced_data['reduction_info']
        
        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Covariance matrix heatmap (top-left)
        # Use original population data directly for accurate covariance calculation
        all_data = []
        for label in pop_data.keys():
            data = pop_data[label]  # [events, cells, time_bins]
            # Transpose to [events, time_bins, cells] then reshape to [events*time_bins, cells]
            # This preserves the (event, time) pairing for covariance calculation
            transposed = data.transpose(0, 2, 1)  # [events, time_bins, cells]
            reshaped = transposed.reshape(-1, data.shape[1])  # [events*time_bins, cells]
            all_data.append(reshaped)
        
        # Combine all data and compute correlation matrix
        combined_data = np.vstack(all_data)  # [total_timepoints, cells]
        corr_matrix = np.corrcoef(combined_data.T)  # [cells, cells]
        
        # Plot correlation matrix
        im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax1.set_title('Cross Correlation Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Neuron Index')
        ax1.set_ylabel('Neuron Index')
        
        # Add colorbar for correlation matrix
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Correlation Coefficient', rotation=270, labelpad=15)
        
        # 2. PCA component weights heatmap (top-right)
        components = reduction_info['components']
        n_components_to_show = min(10, components.shape[0])  # Show up to 10 components
        
        im2 = ax2.imshow(components[:n_components_to_show], 
                        cmap='RdBu_r', aspect='auto', vmin=-np.abs(components).max(), 
                        vmax=np.abs(components).max(), interpolation='none')
        ax2.set_title(f'PCA Component Weights (First {n_components_to_show} PCs)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Neuron Index')
        ax2.set_ylabel('Principal Component')
        ax2.set_yticks(range(n_components_to_show))
        ax2.set_yticklabels([f'PC{i+1}' for i in range(n_components_to_show)])
        
        # Add colorbar for component weights
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Component Loading', rotation=270, labelpad=15)
        
        # 3. Individual variance explained (bottom-left)
        explained_var = reduction_info['explained_variance_ratio']
        n_vars_to_show = min(15, len(explained_var))  # Show up to 15 components
        
        bars = ax3.bar(range(1, n_vars_to_show + 1), explained_var[:n_vars_to_show] * 100,
                      color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax3.set_title('Individual Variance Explained', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Variance Explained (%)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 1:  # Only label bars > 1%
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Cumulative variance explained (bottom-right)
        cumulative_var = reduction_info['cumulative_variance']
        n_cum_to_show = min(20, len(cumulative_var))  # Show up to 20 components
        
        line = ax4.plot(range(1, n_cum_to_show + 1), cumulative_var[:n_cum_to_show] * 100,
                       'o-', color='darkred', linewidth=2, markersize=4, markerfacecolor='red')
        
        ax4.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Principal Components')
        ax4.set_ylabel('Cumulative Variance Explained (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add horizontal reference lines
        for threshold in [80, 90, 95]:
            ax4.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
            ax4.text(1, threshold + 1, f'{threshold}%', fontsize=9, color='gray')
        
        # Find and annotate key milestones
        if len(cumulative_var) > 0:
            # Find components for 80%, 90%, 95% variance
            for threshold in [0.8, 0.9, 0.95]:
                idx = np.where(cumulative_var >= threshold)[0]
                if len(idx) > 0:
                    first_idx = idx[0] + 1  # +1 because we use 1-based indexing
                    if first_idx <= n_cum_to_show:
                        ax4.scatter(first_idx, threshold * 100, color='orange', s=60, zorder=10)
                        ax4.annotate(f'PC{first_idx}', 
                                   (first_idx, threshold * 100),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=9, fontweight='bold')
        
        # Add overall title and summary statistics
        n_neurons = components.shape[1]
        total_var_3pc = cumulative_var[2] * 100 if len(cumulative_var) >= 3 else 0
        
        fig.suptitle(f'PCA Analysis Summary\n'
                    f'{n_neurons} neurons, {len(explained_var)} components, '
                    f'First 3 PCs explain {total_var_3pc:.1f}% variance', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Make room for suptitle
        
        # Find components needed for different variance thresholds
        for threshold_pct, threshold_val in [(80, 0.8), (90, 0.9), (95, 0.95)]:
            idx = np.where(cumulative_var >= threshold_val)[0]
            if len(idx) > 0:
                print(f"{threshold_pct}% variance in {idx[0] + 1} components")
        
        return fig
    
    def plot_normalized_population_matrix(self, pop_data: Dict[str, np.ndarray], 
                                        figsize: Tuple[float, float] = (14, 10)) -> plt.Figure:
        """
        Plot flattened population matrix with each cell normalized as z-score.
        
        This method creates a heatmap visualization of the neural population activity
        where each cell's firing rate is z-score normalized across all events and time bins.
        
        Parameters:
        -----------
        pop_data : Dict[str, np.ndarray]
            Population data dictionary with keys as behavior labels and values as
            3D arrays (n_events, n_cells, n_time_bins)
        figsize : Tuple[float, float], optional
            Figure size (width, height). Default is (14, 10)
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        
        # Collect all data and flatten for each cell
        all_data = []
        cell_labels = []
        event_labels = []
        time_labels = []
        
        for label in pop_data.keys():
            data = pop_data[label]  # (n_events, n_cells, n_time_bins)
            all_data.append(data)
            
        # Concatenate all data across events and conditions
        full_data = np.concatenate(all_data, axis=0)  # (total_events, n_cells, n_time_bins)
        n_total_events, n_cells, n_time_bins = full_data.shape
                
        # Flatten data for each cell and apply z-score normalization
        normalized_matrix = np.zeros((n_cells, n_total_events * n_time_bins))
        
        for cell_idx in range(n_cells):
            # Get all activity for this cell across events and time
            cell_data = full_data[:, cell_idx, :].flatten()  # (n_total_events * n_time_bins,)
            
            # Apply z-score normalization
            if np.std(cell_data) > 0:
                normalized_matrix[cell_idx, :] = stats.zscore(cell_data)
            else:
                normalized_matrix[cell_idx, :] = cell_data  # Keep as is if no variance
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(normalized_matrix, 
                      aspect='auto', 
                      cmap='RdBu_r', 
                      vmin=-3, vmax=3,  # Reasonable z-score range
                      interpolation='nearest')
        
        # Customize appearance
        ax.set_xlabel('Time Points (Events × Time Bins)', fontsize=12)
        ax.set_ylabel('Cells', fontsize=12)
        ax.set_title('Population Matrix: Z-Score Normalized Neural Activity\n'
                    f'({n_cells} cells × {n_total_events} events × {n_time_bins} time bins)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Z-Score', rotation=270, labelpad=20, fontsize=12)
        
        # Add subtle grid lines to separate events
        for event_idx in range(1, n_total_events):
            ax.axvline(x=event_idx * n_time_bins - 0.5, 
                      color='gray', alpha=0.3, linewidth=0.5)
        
        # Add event markers on x-axis
        if n_total_events <= 20:  # Only show ticks if not too many events
            event_centers = [(i * n_time_bins + n_time_bins/2 - 0.5) for i in range(n_total_events)]
            ax.set_xticks(event_centers[::max(1, len(event_centers)//10)])
            ax.set_xticklabels([f'E{i+1}' for i in range(0, n_total_events, 
                               max(1, n_total_events//10))], fontsize=10)
        else:
            # For many events, just show sparse labels
            n_ticks = min(10, n_total_events)
            tick_positions = np.linspace(0, n_total_events * n_time_bins - 1, n_ticks)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f'E{int(pos/n_time_bins)+1}' 
                               for pos in tick_positions], fontsize=10)
        
        # Show every nth cell label or sparse labels for many cells
        if n_cells <= 50:
            ax.set_yticks(range(0, n_cells, max(1, n_cells//20)))
            ax.set_yticklabels([f'C{i+1}' for i in range(0, n_cells, max(1, n_cells//20))], 
                              fontsize=10)
        else:
            # For many cells, show sparse labels
            n_y_ticks = min(20, n_cells)
            y_positions = np.linspace(0, n_cells-1, n_y_ticks).astype(int)
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f'C{pos+1}' for pos in y_positions], fontsize=10)
        
        # Add statistics text
        mean_activity = np.mean(normalized_matrix)
        std_activity = np.std(normalized_matrix)
        max_activity = np.max(normalized_matrix)
        min_activity = np.min(normalized_matrix)
        
        stats_text = (f'Activity Statistics:\n'
                     f'Mean: {mean_activity:.3f}\n'
                     f'Std: {std_activity:.3f}\n'
                     f'Range: [{min_activity:.2f}, {max_activity:.2f}]')
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        return fig

    def compute_opponent_similarity(self,
                                    animal_of_interest: str,
                                    behavior_type: str = 'EC',
                                    windows: Union[Tuple[float, float], List[Tuple[float, float]]] = (-1.0, 0.0),
                                    alignment: str = 'start',
                                    use_quality_cells: bool = True,
                                    min_events_per_opponent: int = 5,
                                    top_n_cells: Optional[int] = None,
                                    time_range: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Compute pairwise Pearson similarity between mean population activity
        vectors for each opponent the focal animal encountered.

        For each window in ``windows``, the mean firing rate per cell per
        opponent (averaged across that opponent's events) is computed. Per-cell
        means are concatenated across windows into a single feature vector per
        opponent of length ``n_cells * n_windows``. Pairwise Pearson
        correlation across opponents yields the similarity matrix.

        Parameters
        ----------
        animal_of_interest : str
            Focal animal id (``'631'`` or ``'rat631'`` — auto-prefixed).
        behavior_type : str, default='EC'
            Behavior abbreviation (passed through to
            ``BehavioralEventsData.extract_opponent_labels``).
        windows : (start, end) tuple or list of such tuples, default=(-1.0, 0.0)
            One or more analysis windows (seconds, relative to the alignment
            point). Multi-window features are concatenated.
        alignment : {'start', 'end', 'center'}, default='start'
            Event alignment point.
        use_quality_cells : bool, default=True
            Restrict to quality-filtered cells.
        min_events_per_opponent : int, default=5
            Drop opponents with fewer than this many events.
        top_n_cells : int, optional
            If set, keep only the ``top_n_cells`` cells with the highest mean
            firing rate across all selected events and windows. Applied after
            quality filtering. ``None`` (default) keeps all selected cells.
        time_range : (min_time, max_time) tuple, optional
            If set, restrict to events whose start time (in ephys seconds)
            falls within ``[min_time, max_time]`` inclusive. Either bound may
            be ``None`` to leave that side unbounded. ``min_events_per_opponent``
            is re-applied after this filter, so opponents below the threshold
            within the range are dropped.

        Returns
        -------
        dict
            Result with keys ``similarity_matrix``, ``opponents``,
            ``mean_activity``, ``event_counts``, ``parameters``,
            ``behavioral_summary``, ``status``. On failure (no events after
            filtering) returns ``{'status': 'failed', 'error': ..., 'parameters': ...}``.

        Notes
        -----
        Internally calls ``construct_population_matrix`` once per window with
        ``time_bin_size`` equal to the window width (so each event yields a
        single mean-rate bin per cell). Successive calls overwrite
        ``self.population_matrices[f"{alignment}_none"]``; the returned dict
        holds everything needed downstream.
        """
        if self.behavior_data is None:
            raise ValueError(
                "compute_opponent_similarity requires behavior_data; "
                "pass it to PopulationGeometryAnalyzer(ks_data, behavior_data)."
            )

        # Normalize windows -> list of (start, end) tuples
        def _is_single_window(w) -> bool:
            try:
                return (
                    len(w) == 2
                    and isinstance(w[0], (int, float))
                    and isinstance(w[1], (int, float))
                )
            except TypeError:
                return False

        if _is_single_window(windows):
            window_list: List[Tuple[float, float]] = [(float(windows[0]), float(windows[1]))]
        else:
            window_list = [(float(w[0]), float(w[1])) for w in windows]

        for w_start, w_end in window_list:
            if w_end <= w_start:
                raise ValueError(f"Window {(w_start, w_end)} has non-positive width")

        if top_n_cells is not None and top_n_cells <= 0:
            raise ValueError(f"top_n_cells must be positive or None, got {top_n_cells}")

        time_range_norm: Optional[Tuple[Optional[float], Optional[float]]] = None
        if time_range is not None:
            if len(time_range) != 2:
                raise ValueError(f"time_range must be (min_time, max_time), got {time_range}")
            t_min = None if time_range[0] is None else float(time_range[0])
            t_max = None if time_range[1] is None else float(time_range[1])
            if t_min is not None and t_max is not None and t_max <= t_min:
                raise ValueError(f"time_range max ({t_max}) must exceed min ({t_min})")
            time_range_norm = (t_min, t_max)

        base_params = {
            'animal_of_interest': animal_of_interest,
            'behavior_type': behavior_type,
            'windows': window_list,
            'alignment': alignment,
            'use_quality_cells': use_quality_cells,
            'min_events_per_opponent': min_events_per_opponent,
            'top_n_cells': top_n_cells,
            'time_range': time_range_norm,
            'n_windows': len(window_list),
            'class_label': 'Opponent',
            'analysis_title': 'Opponent Representation Similarity',
        }

        # Extract opponent labels (ephys-synced timestamps).
        # If time_range is set, defer the min-events filter until after we
        # restrict to the requested range, so the threshold reflects events
        # actually used downstream.
        extract_min_events = 1 if time_range_norm is not None else min_events_per_opponent
        event_starts, event_ends, opponent_labels = self.behavior_data.extract_opponent_labels(
            animal_of_interest=animal_of_interest,
            behavior_type=behavior_type,
            min_events_per_class=extract_min_events,
        )

        if len(event_starts) == 0:
            return {
                'status': 'failed',
                'error': (
                    f"No '{behavior_type}' events for {animal_of_interest} "
                    f"after min_events_per_opponent={min_events_per_opponent} filter."
                ),
                'parameters': base_params,
            }

        if time_range_norm is not None:
            t_min, t_max = time_range_norm
            in_range = np.ones(len(event_starts), dtype=bool)
            if t_min is not None:
                in_range &= event_starts >= t_min
            if t_max is not None:
                in_range &= event_starts <= t_max
            event_starts = event_starts[in_range]
            event_ends = event_ends[in_range]
            opponent_labels = opponent_labels[in_range]

            if len(event_starts) == 0:
                return {
                    'status': 'failed',
                    'error': (
                        f"No '{behavior_type}' events for {animal_of_interest} "
                        f"within time_range={time_range_norm}."
                    ),
                    'parameters': base_params,
                }

            # Re-apply per-opponent threshold within the restricted range
            uniq, counts = np.unique(opponent_labels, return_counts=True)
            keep_opps = set(uniq[counts >= min_events_per_opponent].tolist())
            if not keep_opps:
                return {
                    'status': 'failed',
                    'error': (
                        f"No opponent reaches min_events_per_opponent="
                        f"{min_events_per_opponent} within time_range={time_range_norm}."
                    ),
                    'parameters': base_params,
                }
            mask = np.array([lbl in keep_opps for lbl in opponent_labels])
            event_starts = event_starts[mask]
            event_ends = event_ends[mask]
            opponent_labels = opponent_labels[mask]

        # Per-window: opp -> mean firing rate vector across cells
        per_window_means: List[Dict[str, np.ndarray]] = []
        n_cells_final: Optional[int] = None

        for w_start, w_end in window_list:
            width = w_end - w_start
            pop_matrix = self.construct_population_matrix(
                event_starts=event_starts,
                event_ends=event_ends,
                event_labels=opponent_labels,
                time_window=(w_start, w_end),
                time_bin_size=width,
                alignment=alignment,
                normalize_method='none',
                use_quality_cells=use_quality_cells,
            )
            n_cells_final = pop_matrix['metadata']['n_cells']

            window_means: Dict[str, np.ndarray] = {}
            for opp, data in pop_matrix['population_data'].items():
                # data shape: (n_events_opp, n_cells, 1)
                window_means[str(opp)] = data[:, :, 0].mean(axis=0)
            per_window_means.append(window_means)

        # Deterministic opponent ordering
        opponents_sorted = sorted(per_window_means[0].keys())

        # Optional top-N cell selection by overall mean firing rate across
        # all opponents and windows.
        selected_cell_indices: Optional[np.ndarray] = None
        if top_n_cells is not None and n_cells_final is not None:
            # Sum each cell's mean rate across opponents and windows
            cell_rate_sum = np.zeros(n_cells_final)
            for window_means in per_window_means:
                for opp in opponents_sorted:
                    cell_rate_sum += window_means[opp]
            n_keep = min(top_n_cells, n_cells_final)
            # Indices of the top-n cells (descending by rate)
            selected_cell_indices = np.argsort(cell_rate_sum)[::-1][:n_keep]
            selected_cell_indices.sort()  # stable order for interpretability
            for window_means in per_window_means:
                for opp in window_means:
                    window_means[opp] = window_means[opp][selected_cell_indices]
            n_cells_final = int(n_keep)

        # Concatenate per-window vectors -> feature matrix [n_opponents, n_cells * n_windows]
        feature_rows = [
            np.concatenate([per_window_means[i][opp] for i in range(len(window_list))])
            for opp in opponents_sorted
        ]
        mean_activity = np.vstack(feature_rows)

        # Pairwise Pearson correlation
        similarity_matrix = np.corrcoef(mean_activity)

        # Event counts per opponent
        unique_labels, counts = np.unique(opponent_labels, return_counts=True)
        event_counts = {str(lbl): int(c) for lbl, c in zip(unique_labels, counts)}

        return {
            'status': 'success',
            'similarity_matrix': similarity_matrix,
            'opponents': np.array(opponents_sorted),
            'mean_activity': mean_activity,
            'event_counts': event_counts,
            'selected_cell_indices': selected_cell_indices,
            'parameters': {**base_params, 'n_cells': n_cells_final},
            'behavioral_summary': {
                'n_events': int(len(opponent_labels)),
                'unique_classes': unique_labels,
                'class_counts': event_counts,
            },
        }

    def plot_opponent_similarity_matrix(self,
                                        result: Dict,
                                        figsize: Tuple[float, float] = (13, 7),
                                        cmap: str = 'RdBu_r',
                                        annotate: bool = True,
                                        vmin: Optional[float] = None,
                                        vmax: Optional[float] = None) -> plt.Figure:
        """
        Plot the opponent similarity matrix from ``compute_opponent_similarity``
        with a side bar-plot summarising within- vs between-group similarity.

        Opponents are split into ``'low'`` / ``'high'`` halves by their
        trailing numeric ID (same rule as
        ``BehavioralEventsData._assign_id_groups``). The half whose median
        ID is closer to the focal animal's ID is labelled ``own``; the
        other is labelled ``other``. The bar plot shows the mean (± SEM)
        Pearson *r* of:

        - within-own pairs (upper triangle of the own×own block)
        - within-other pairs (upper triangle of the other×other block)
        - between-group pairs (all own×other entries)

        Individual pair values are overlaid as jittered points.

        Parameters
        ----------
        result : dict
            Successful result dict from ``compute_opponent_similarity``.
        figsize : tuple, default=(13, 7)
        cmap : str, default='RdBu_r'
        annotate : bool, default=True
            Write the correlation value in each heatmap cell.
        vmin, vmax : float, optional
            Heatmap colour-bar range. ``None`` (default) autoscales to the
            matrix min/max.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if result.get('status') != 'success':
            raise ValueError(
                f"Cannot plot — result status is '{result.get('status')}'. "
                f"Error: {result.get('error', 'unknown')}"
            )

        sim = result['similarity_matrix']
        opponents = result['opponents']
        params = result['parameters']

        if vmin is None:
            vmin = float(np.nanmin(sim))
        if vmax is None:
            vmax = float(np.nanmax(sim))

        # Split opponents into low/high halves by trailing numeric id, then
        # relabel as own/other relative to the focal animal's numeric id.
        import re
        from video.behavioral_events import BehavioralEventsData
        opp_strs = [str(o) for o in opponents]
        try:
            group_map = BehavioralEventsData._assign_id_groups(opp_strs)
            groups = np.array([group_map[s] for s in opp_strs])
            low_idx = np.where(groups == 'low')[0]
            high_idx = np.where(groups == 'high')[0]
            grouping_error: Optional[str] = None
        except ValueError as e:
            low_idx = np.array([], dtype=int)
            high_idx = np.array([], dtype=int)
            grouping_error = str(e)

        def _trailing_int(s: str) -> Optional[int]:
            m = re.search(r'(\d+)$', str(s))
            return int(m.group(1)) if m else None

        # Decide which half is 'own' for the focal animal
        own_half: Optional[str] = None
        focal_num = _trailing_int(params['animal_of_interest'])
        if grouping_error is None and focal_num is not None:
            low_nums = [n for n in (_trailing_int(opponents[i]) for i in low_idx) if n is not None]
            high_nums = [n for n in (_trailing_int(opponents[i]) for i in high_idx) if n is not None]
            if low_nums and high_nums:
                low_med = float(np.median(low_nums))
                high_med = float(np.median(high_nums))
                own_half = 'low' if abs(focal_num - low_med) <= abs(focal_num - high_med) else 'high'
            elif low_nums:
                own_half = 'low'
            elif high_nums:
                own_half = 'high'

        if own_half == 'high':
            own_idx, other_idx = high_idx, low_idx
        else:
            # Default to low as own when focal cannot be placed
            own_idx, other_idx = low_idx, high_idx

        def _offdiag(block: np.ndarray) -> np.ndarray:
            n = block.shape[0]
            if n < 2:
                return np.array([])
            iu = np.triu_indices(n, k=1)
            return block[iu]

        within_own = _offdiag(sim[np.ix_(own_idx, own_idx)])
        within_other = _offdiag(sim[np.ix_(other_idx, other_idx)])
        between = (
            sim[np.ix_(own_idx, other_idx)].ravel()
            if len(own_idx) and len(other_idx) else np.array([])
        )

        fig, (ax_hm, ax_bar) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={'width_ratios': [3, 1]},
        )

        sns.heatmap(
            sim,
            ax=ax_hm,
            xticklabels=list(opponents),
            yticklabels=list(opponents),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=annotate,
            fmt='.2f',
            square=True,
            cbar_kws={'label': 'Pearson r'},
        )

        windows_str = ', '.join(f'({s:g}, {e:g})s' for s, e in params['windows'])
        title_main = params.get('analysis_title', 'Opponent Representation Similarity')
        title_sub = (
            f"{params['animal_of_interest']} | {params['behavior_type']} events | "
            f"{params['n_cells']} cells | windows: {windows_str}"
        )
        ax_hm.set_title(f"{title_main}\n{title_sub}", fontsize=12)
        ax_hm.set_xlabel('Opponent')
        ax_hm.set_ylabel('Opponent')
        plt.setp(ax_hm.get_xticklabels(), rotation=45, ha='right')

        # Bar plot: within- vs between-group similarity
        categories = ['Within own', 'Within other', 'Between']
        pools = [within_own, within_other, between]
        means = [float(np.mean(p)) if len(p) else np.nan for p in pools]
        sems = [
            float(np.std(p, ddof=1) / np.sqrt(len(p))) if len(p) > 1 else 0.0
            for p in pools
        ]
        n_pairs = [len(p) for p in pools]
        bar_colors = ['#4C72B0', '#C44E52', '#8172B2']

        x = np.arange(len(categories))
        ax_bar.bar(x, means, yerr=sems, capsize=4,
                   color=bar_colors, edgecolor='black', linewidth=0.6)

        rng = np.random.default_rng(0)
        for xi, p in zip(x, pools):
            if len(p) == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(p))
            ax_bar.scatter(np.full(len(p), xi, dtype=float) + jitter, p,
                           color='black', s=12, alpha=0.6, zorder=3)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(
            [f'{c}\n(n={n})' for c, n in zip(categories, n_pairs)],
            fontsize=10,
        )
        ax_bar.set_ylabel('Pearson r')
        ax_bar.axhline(0, color='gray', linewidth=0.6)
        ax_bar.set_title('Within- vs between-group', fontsize=12)
        ax_bar.grid(True, axis='y', alpha=0.3)

        if grouping_error is not None:
            ax_bar.text(0.5, 0.5, f'Grouping unavailable:\n{grouping_error}',
                        transform=ax_bar.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
        else:
            own_lbls = ', '.join(str(o) for o in opponents[own_idx])
            other_lbls = ', '.join(str(o) for o in opponents[other_idx])
            own_half_str = own_half if own_half is not None else 'low (default)'
            # fig.text(
            #     0.01, 0.01,
            #     f"focal {params['animal_of_interest']} → {own_half_str} half   "
            #     f"|   own: {own_lbls}   |   other: {other_lbls}",
            #     fontsize=9, color='gray',
            # )

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        return fig

    def compute_event_similarity_to_reference(self,
                                              animal_of_interest: str,
                                              reference_animal: str,
                                              behavior_type: str = 'EC',
                                              windows: Union[Tuple[float, float], List[Tuple[float, float]]] = (-1.0, 0.0),
                                              alignment: str = 'start',
                                              use_quality_cells: bool = True,
                                              min_events_per_opponent: int = 5,
                                              top_n_cells: Optional[int] = None,
                                              time_range: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Track how each non-reference opponent's per-event population activity
        correlates with a reference opponent's mean activity, as a function
        of the opponent's serial event number.

        For each opponent ``opp != reference_animal`` and each of ``opp``'s
        events with ``animal_of_interest`` (chronological order), the Pearson
        correlation is computed between that single event's per-cell mean
        firing-rate vector and the mean per-cell vector across all events of
        ``reference_animal``. Per-cell means are concatenated across all
        entries in ``windows`` (so the feature vector has length
        ``n_cells * n_windows``), matching ``compute_opponent_similarity``.
        Cell selection (quality filter, optional ``top_n_cells``) also
        matches that function.

        Parameters
        ----------
        animal_of_interest : str
            Focal animal id.
        reference_animal : str
            Opponent whose mean activity is the reference template. Accepts
            either ``'632'`` or ``'rat632'`` — matched by trailing digits
            against the opponent labels returned by
            ``extract_opponent_labels``.
        behavior_type, windows, alignment, use_quality_cells,
        min_events_per_opponent, top_n_cells, time_range :
            See ``compute_opponent_similarity``. ``time_range`` filters by
            event start time (ephys seconds) before per-opponent thresholding
            and reference resolution.

        Returns
        -------
        dict
            On success::

                {
                    'status': 'success',
                    'reference_animal': <resolved label>,
                    'similarity_traces': {opp: 1-D array of Pearson r},
                    'event_indices':     {opp: 1-based serial event numbers},
                    'event_times':       {opp: event-start times (ephys s)},
                    'reference_mean_activity': 1-D array (n_features,),
                    'event_counts': {opp: n_events},
                    'selected_cell_indices': np.ndarray or None,
                    'parameters': {...},
                }

            On failure (no events, unknown reference) ``{'status': 'failed',
            'error': ..., 'parameters': ...}``.
        """
        if self.behavior_data is None:
            raise ValueError(
                "compute_event_similarity_to_reference requires behavior_data."
            )

        def _is_single_window(w) -> bool:
            try:
                return (
                    len(w) == 2
                    and isinstance(w[0], (int, float))
                    and isinstance(w[1], (int, float))
                )
            except TypeError:
                return False

        if _is_single_window(windows):
            window_list: List[Tuple[float, float]] = [(float(windows[0]), float(windows[1]))]
        else:
            window_list = [(float(w[0]), float(w[1])) for w in windows]

        for w_start, w_end in window_list:
            if w_end <= w_start:
                raise ValueError(f"Window {(w_start, w_end)} has non-positive width")

        if top_n_cells is not None and top_n_cells <= 0:
            raise ValueError(f"top_n_cells must be positive or None, got {top_n_cells}")

        time_range_norm: Optional[Tuple[Optional[float], Optional[float]]] = None
        if time_range is not None:
            if len(time_range) != 2:
                raise ValueError(f"time_range must be (min_time, max_time), got {time_range}")
            t_min = None if time_range[0] is None else float(time_range[0])
            t_max = None if time_range[1] is None else float(time_range[1])
            if t_min is not None and t_max is not None and t_max <= t_min:
                raise ValueError(f"time_range max ({t_max}) must exceed min ({t_min})")
            time_range_norm = (t_min, t_max)

        base_params = {
            'animal_of_interest': animal_of_interest,
            'reference_animal': reference_animal,
            'behavior_type': behavior_type,
            'windows': window_list,
            'alignment': alignment,
            'use_quality_cells': use_quality_cells,
            'min_events_per_opponent': min_events_per_opponent,
            'top_n_cells': top_n_cells,
            'time_range': time_range_norm,
            'n_windows': len(window_list),
            'class_label': 'Opponent',
            'analysis_title': 'Event-wise similarity to reference',
        }

        # If time_range is set, defer the min-events filter until after the
        # range restriction so the threshold reflects events actually used.
        extract_min_events = 1 if time_range_norm is not None else min_events_per_opponent
        event_starts, event_ends, opponent_labels = self.behavior_data.extract_opponent_labels(
            animal_of_interest=animal_of_interest,
            behavior_type=behavior_type,
            min_events_per_class=extract_min_events,
        )
        event_starts = np.asarray(event_starts)
        event_ends = np.asarray(event_ends)
        opponent_labels = np.asarray(opponent_labels)

        if len(event_starts) == 0:
            return {
                'status': 'failed',
                'error': (
                    f"No '{behavior_type}' events for {animal_of_interest} "
                    f"after min_events_per_opponent={min_events_per_opponent} filter."
                ),
                'parameters': base_params,
            }

        if time_range_norm is not None:
            t_min, t_max = time_range_norm
            in_range = np.ones(len(event_starts), dtype=bool)
            if t_min is not None:
                in_range &= event_starts >= t_min
            if t_max is not None:
                in_range &= event_starts <= t_max
            event_starts = event_starts[in_range]
            event_ends = event_ends[in_range]
            opponent_labels = opponent_labels[in_range]

            if len(event_starts) == 0:
                return {
                    'status': 'failed',
                    'error': (
                        f"No '{behavior_type}' events for {animal_of_interest} "
                        f"within time_range={time_range_norm}."
                    ),
                    'parameters': base_params,
                }

            uniq, counts = np.unique(opponent_labels, return_counts=True)
            keep_opps = set(uniq[counts >= min_events_per_opponent].tolist())
            if not keep_opps:
                return {
                    'status': 'failed',
                    'error': (
                        f"No opponent reaches min_events_per_opponent="
                        f"{min_events_per_opponent} within time_range={time_range_norm}."
                    ),
                    'parameters': base_params,
                }
            mask = np.array([lbl in keep_opps for lbl in opponent_labels])
            event_starts = event_starts[mask]
            event_ends = event_ends[mask]
            opponent_labels = opponent_labels[mask]

        # Sort events chronologically so per-opponent indexing reflects serial order
        order = np.argsort(event_starts, kind='stable')
        event_starts = event_starts[order]
        event_ends = event_ends[order]
        opponent_labels = opponent_labels[order]

        # Resolve reference label by trailing digits (accepts '632' or 'rat632')
        import re
        unique_label_strs = sorted({str(l) for l in opponent_labels})
        ref_str = str(reference_animal)
        if ref_str not in unique_label_strs:
            m = re.search(r'(\d+)$', ref_str)
            ref_num = m.group(1) if m else None
            if ref_num is not None:
                matches = []
                for l in unique_label_strs:
                    mm = re.search(r'(\d+)$', l)
                    if mm is not None and mm.group(1) == ref_num:
                        matches.append(l)
                if matches:
                    ref_str = matches[0]
            if ref_str not in unique_label_strs:
                return {
                    'status': 'failed',
                    'error': (
                        f"Reference animal '{reference_animal}' not among opponents "
                        f"{unique_label_strs} for {animal_of_interest} "
                        f"(after min_events_per_opponent={min_events_per_opponent})."
                    ),
                    'parameters': base_params,
                }

        # Per-window: opp -> [n_events_opp, n_cells]
        per_window_per_event: List[Dict[str, np.ndarray]] = []
        n_cells_total: Optional[int] = None

        for w_start, w_end in window_list:
            width = w_end - w_start
            pop_matrix = self.construct_population_matrix(
                event_starts=event_starts,
                event_ends=event_ends,
                event_labels=opponent_labels,
                time_window=(w_start, w_end),
                time_bin_size=width,
                alignment=alignment,
                normalize_method='none',
                use_quality_cells=use_quality_cells,
            )
            n_cells_total = pop_matrix['metadata']['n_cells']
            window_per_event: Dict[str, np.ndarray] = {}
            for opp, data in pop_matrix['population_data'].items():
                # data shape: (n_events_opp, n_cells, 1)
                window_per_event[str(opp)] = data[:, :, 0]
            per_window_per_event.append(window_per_event)

        # Concatenate windows -> per-opp feature matrix [n_events_opp, n_cells * n_windows]
        opp_event_features: Dict[str, np.ndarray] = {}
        for opp in per_window_per_event[0]:
            opp_event_features[opp] = np.concatenate(
                [per_window_per_event[i][opp] for i in range(len(window_list))],
                axis=1,
            )

        # Optional top-N cell selection (same rule as compute_opponent_similarity)
        selected_cell_indices: Optional[np.ndarray] = None
        if top_n_cells is not None and n_cells_total is not None:
            cell_rate_sum = np.zeros(n_cells_total)
            for window_per_event in per_window_per_event:
                for opp, mat in window_per_event.items():
                    cell_rate_sum += mat.mean(axis=0)
            n_keep = min(top_n_cells, n_cells_total)
            selected_cell_indices = np.argsort(cell_rate_sum)[::-1][:n_keep]
            selected_cell_indices.sort()
            feature_index = np.concatenate([
                selected_cell_indices + i * n_cells_total
                for i in range(len(window_list))
            ])
            for opp in opp_event_features:
                opp_event_features[opp] = opp_event_features[opp][:, feature_index]
            n_cells_total = int(n_keep)

        # Reference mean across reference's events
        ref_features = opp_event_features[ref_str]
        ref_mean = ref_features.mean(axis=0)
        ref_var = float(np.std(ref_mean))

        # Per non-reference opponent: correlate each event with ref_mean
        other_opps = [o for o in sorted(opp_event_features) if o != ref_str]
        similarity_traces: Dict[str, np.ndarray] = {}
        event_indices: Dict[str, np.ndarray] = {}
        event_times: Dict[str, np.ndarray] = {}

        for opp in other_opps:
            feats = opp_event_features[opp]
            n_ev = feats.shape[0]
            sims = np.full(n_ev, np.nan)
            if ref_var > 0:
                for j in range(n_ev):
                    v = feats[j]
                    if np.std(v) > 0:
                        sims[j] = np.corrcoef(v, ref_mean)[0, 1]
            similarity_traces[opp] = sims
            event_indices[opp] = np.arange(1, n_ev + 1)
            event_times[opp] = event_starts[opponent_labels == opp]

        unique_labels, counts = np.unique(opponent_labels, return_counts=True)
        event_counts = {str(lbl): int(c) for lbl, c in zip(unique_labels, counts)}

        return {
            'status': 'success',
            'reference_animal': ref_str,
            'similarity_traces': similarity_traces,
            'event_indices': event_indices,
            'event_times': event_times,
            'reference_mean_activity': ref_mean,
            'event_counts': event_counts,
            'selected_cell_indices': selected_cell_indices,
            'parameters': {**base_params, 'n_cells': n_cells_total},
        }

    def plot_event_similarity_to_reference(self,
                                           result: Dict,
                                           figsize: Tuple[float, float] = (10, 6),
                                           show_markers: bool = True,
                                           smooth_window: Optional[int] = None) -> plt.Figure:
        """
        Plot the result of ``compute_event_similarity_to_reference``.

        One line per non-reference opponent: x = serial event number,
        y = Pearson r between that event's population activity vector
        and the reference animal's mean activity vector.

        Parameters
        ----------
        result : dict
            Successful result from ``compute_event_similarity_to_reference``.
        figsize : tuple, default=(10, 6)
        show_markers : bool, default=True
            Draw a marker at each event.
        smooth_window : int, optional
            If set, overlay a centred rolling mean of this width. Raw points
            stay as faded markers (if ``show_markers``).
        """
        if result.get('status') != 'success':
            raise ValueError(
                f"Cannot plot — result status is '{result.get('status')}'. "
                f"Error: {result.get('error', 'unknown')}"
            )

        traces = result['similarity_traces']
        indices = result['event_indices']
        ref = result['reference_animal']
        params = result['parameters']

        opps = sorted(traces)
        palette = sns.color_palette('tab10', n_colors=max(len(opps), 1))

        fig, ax = plt.subplots(figsize=figsize)

        for i, opp in enumerate(opps):
            x = indices[opp]
            y = traces[opp]
            color = palette[i % len(palette)]
            label = f'{opp} (n={len(y)})'

            if smooth_window is not None and smooth_window > 1 and len(y) >= 2:
                w = min(smooth_window, len(y))
                kernel = np.ones(w) / w
                fill = np.nanmean(y) if np.any(~np.isnan(y)) else 0.0
                y_filled = np.where(np.isnan(y), fill, y)
                y_smooth = np.convolve(y_filled, kernel, mode='same')
                ax.plot(x, y_smooth, color=color, linewidth=2, label=label)
                if show_markers:
                    ax.plot(x, y, 'o', color=color, alpha=0.35, markersize=4)
            else:
                ax.plot(
                    x, y,
                    marker='o' if show_markers else None,
                    color=color, linewidth=1.5, markersize=5,
                    label=label,
                )

        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_xlabel('Serial event number')
        ax.set_ylabel(f'Pearson r to mean activity for {ref}')
        windows_str = ', '.join(f'({s:g}, {e:g})s' for s, e in params['windows'])
        ax.set_title(
            f"Event-wise similarity to reference {ref}\n"
            f"{params['animal_of_interest']} | {params['behavior_type']} events | "
            f"{params['n_cells']} cells | windows: {windows_str}"
        )
        ax.legend(title='Opponent', loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Continuous PCA trajectory with event overlay
# ---------------------------------------------------------------------------

def plot_pca_trajectory_with_events(
    ks_data,
    behavior_data,
    animal_of_interest: str,
    behavior_types: Optional[List[str]] = None,
    bin_size: float = 0.5,
    n_components: int = 3,
    filtered_only: bool = True,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    min_events_per_class: int = 1,
    marker_size: int = 6,
    line_width: float = 1.0,
    line_alpha: float = 0.3,
):
    """
    Bin the full recording into a firing-rate matrix (filtered cells),
    run PCA, and plot the 3-D trajectory with event markers colour-coded
    by opponent animal.

    Parameters
    ----------
    ks_data : KilosortData
        Must have ``spike_times_by_cell`` populated.
    behavior_data : BehavioralEventsData
        Must already be synchronised with ephys (``ts_start_ephys`` present).
    animal_of_interest : str
        The implanted animal (e.g. ``"rat631"`` or ``"631"``).
    behavior_types : list of str, optional
        Behaviour-type abbreviations to include (e.g. ``["F", "C"]``).
        ``None`` → all available types.
    bin_size : float
        Temporal bin width in seconds.
    n_components : int
        Number of PCA components (≥3 for 3-D plot).
    filtered_only : bool
        Use only quality-filtered cells.
    start_time, end_time : float, optional
        Restrict the time range.
    min_events_per_class : int
        Minimum events per opponent to include that opponent.
    marker_size : int
        Size of event markers.
    line_width : float
        Width of the background trajectory line.
    line_alpha : float
        Opacity of the background trajectory line.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3-D figure.
    dict
        Result dictionary with keys ``pca_model``, ``scores``,
        ``bin_centers``, ``explained_variance``, ``event_info``.
    """
    import plotly.graph_objects as go
    import plotly.express as px

    # --- 1. Build firing-rate matrix (n_cells × n_bins) ---
    spks, bin_centers = ks_data.bin_spike_times(
        bin_size_sec=bin_size,
        t_start=start_time,
        t_end=end_time,
        filtered_only=filtered_only,
    )
    n_cells, n_bins = spks.shape
    print(f"Firing-rate matrix: {n_cells} cells × {n_bins} bins "
          f"({bin_centers[0]:.1f}–{bin_centers[-1]:.1f} s)")

    # --- 2. Z-score and PCA ---
    X = zscore(spks, axis=1)          # z-score each cell across time
    X = np.nan_to_num(X, 0.0)        # zero-variance cells → 0
    pca = PCA(n_components=min(n_components, n_cells))
    scores = pca.fit_transform(X.T)   # (n_bins, n_components)

    ev = pca.explained_variance_ratio_
    print(f"PCA explained variance: "
          + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(ev)))

    # --- 3. Gather events ---
    if not animal_of_interest.startswith('rat'):
        animal_of_interest = f"rat{animal_of_interest}"

    if behavior_types is None:
        behavior_types = behavior_data.get_available_event_types()

    all_event_starts = []
    all_event_ends = []
    all_opponents = []
    all_btypes = []

    for bt in behavior_types:
        try:
            starts, ends, opponents = behavior_data.extract_opponent_labels(
                animal_of_interest, behavior_type=bt,
                min_events_per_class=min_events_per_class,
            )
        except Exception:
            continue
        if len(starts) == 0:
            continue
        all_event_starts.append(starts)
        all_event_ends.append(ends)
        all_opponents.append(opponents)
        all_btypes.append(np.full(len(starts), bt))

    if all_event_starts:
        ev_starts = np.concatenate(all_event_starts)
        ev_ends = np.concatenate(all_event_ends)
        ev_opponents = np.concatenate(all_opponents)
        ev_btypes = np.concatenate(all_btypes)
    else:
        ev_starts = np.array([])
        ev_ends = np.array([])
        ev_opponents = np.array([])
        ev_btypes = np.array([])

    # Map each event to the nearest time-bin index
    event_bin_idx = np.searchsorted(bin_centers, ev_starts).clip(0, n_bins - 1)

    # Decode behavior abbreviation for hover text
    btype_map = behavior_data.get_behavior_type_mapping()

    # --- 4. Build interactive plotly figure ---
    palette = px.colors.qualitative.Set1
    unique_opponents = np.unique(ev_opponents) if len(ev_opponents) > 0 else []
    opp_color = {opp: palette[i % len(palette)] for i, opp in enumerate(unique_opponents)}

    fig = go.Figure()

    # Background trajectory – colour-coded by time
    fig.add_trace(go.Scatter3d(
        x=scores[:, 0], y=scores[:, 1], z=scores[:, 2],
        mode='lines',
        line=dict(color=bin_centers, colorscale='Viridis',
                  width=line_width, showscale=True,
                  colorbar=dict(title='Time (s)', x=1.05, len=0.6)),
        opacity=line_alpha,
        name='Trajectory',
        hovertemplate='t=%{customdata:.1f}s<extra></extra>',
        customdata=bin_centers,
    ))

    # Event markers — one trace per opponent for legend grouping
    for opp in unique_opponents:
        mask = ev_opponents == opp
        idx = event_bin_idx[mask]
        btypes_here = ev_btypes[mask]
        hover = [
            f"t={bin_centers[bi]:.1f}s<br>"
            f"{btype_map.get(b, b)} vs {opp}"
            for bi, b in zip(idx, btypes_here)
        ]
        fig.add_trace(go.Scatter3d(
            x=scores[idx, 0],
            y=scores[idx, 1],
            z=scores[idx, 2],
            mode='markers',
            marker=dict(size=marker_size, color=opp_color[opp],
                        line=dict(width=0.5, color='black')),
            name=opp,
            hovertext=hover,
            hoverinfo='text',
        ))

    fig.update_layout(
        title=(f'PCA trajectory – {animal_of_interest}  '
               f'({n_cells} cells, bin={bin_size}s)'),
        scene=dict(
            xaxis_title=f'PC1 ({ev[0]:.1%})',
            yaxis_title=f'PC2 ({ev[1]:.1%})',
            zaxis_title=f'PC3 ({ev[2]:.1%})' if len(ev) >= 3 else 'PC3',
        ),
        width=950, height=750,
        legend=dict(title='Opponent', x=0.97, xanchor='right', y=1, yanchor='top'),
    )

    result = {
        'pca_model': pca,
        'scores': scores,
        'bin_centers': bin_centers,
        'explained_variance': ev,
        'n_cells': n_cells,
        'event_info': {
            'starts': ev_starts,
            'ends': ev_ends,
            'opponents': ev_opponents,
            'behavior_types': ev_btypes,
            'bin_indices': event_bin_idx,
        },
    }

    return fig, result


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
        from ingestion.kilosort_data_import import load_kilosort_data
        from video.behavioral_events import load_behavioral_events
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
    ks_data = load_kilosort_data(data_storage)
    behavior_data = load_behavioral_events(
        data_storage.get_behavioral_event_files(),
        session_id=data_storage.session_id,
    )
    
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