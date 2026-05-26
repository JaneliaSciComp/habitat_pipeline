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

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA

from ephys._rate_tensor import event_aligned_rates
from video.behavioral_events import BehavioralEventsData

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")


class PopulationGeometryAnalyzer:
    """
    Analyze neural population geometry and dynamics across behavioral contexts.

    Methods:
    1. Build event-aligned population firing-rate tensors
    2. Apply dimensionality reduction (PCA / UMAP)
    3. Visualise trajectories through state space
    4. Compare population activity across opponents (similarity matrix,
       event-wise similarity to a reference)
    """

    def __init__(self, ks_data, behavior_data=None):
        """
        Parameters
        ----------
        ks_data : KilosortData
            Processed electrophysiology data.
        behavior_data : BehavioralEventsData, optional
            Behavioral event data for condition-based analyses.
        """
        self.ks_data = ks_data
        self.behavior_data = behavior_data

    # ------------------------------------------------------------------ #
    # Population matrix construction
    # ------------------------------------------------------------------ #

    def construct_population_matrix(self,
                                    event_starts: np.ndarray,
                                    event_ends: np.ndarray,
                                    event_labels: np.ndarray,
                                    time_window: Tuple[float, float] = (-2.0, 2.0),
                                    time_bin_size: float = 0.1,
                                    alignment: str = 'start',
                                    normalize_method: str = 'none',
                                    use_quality_cells: bool = True) -> Dict:
        """
        Build event-aligned population firing-rate matrices.

        Parameters
        ----------
        event_starts, event_ends : np.ndarray
            Event start / end times (s, ephys clock).
        event_labels : np.ndarray
            Per-event condition label.
        time_window : tuple, default=(-2.0, 2.0)
            ``(start, end)`` of the analysis window relative to the alignment
            point.
        time_bin_size : float, default=0.1
            Width of each time bin (s).
        alignment : {'start', 'end', 'center'}, default='start'
        normalize_method : {'none', 'zscore', 'baseline'}, default='none'
        use_quality_cells : bool, default=True
            Filter cells by quality metrics before analysis.

        Returns
        -------
        dict
            Keys: ``population_data`` (``{label: (n_events_lbl, n_cells, n_bins)}``),
            ``time_bins`` (bin centers), ``event_counts``, ``unique_labels``,
            ``metadata``.
        """
        event_starts = np.asarray(event_starts)
        event_ends = np.asarray(event_ends)
        event_labels = np.asarray(event_labels)

        if alignment == 'start':
            align_times = event_starts
        elif alignment == 'end':
            align_times = event_ends
        elif alignment == 'center':
            align_times = (event_starts + event_ends) / 2
        else:
            raise ValueError(f"Unknown alignment: {alignment}")

        if use_quality_cells:
            selected_cluster_ids, spike_times_list = self.ks_data.get_filtered_cells_spike_times()
        else:
            spike_times_list = list(self.ks_data.spike_times_by_cell)
            selected_cluster_ids = list(self.ks_data.ks_ids)

        if len(spike_times_list) == 0:
            raise ValueError("No cells selected for analysis (all cells failed quality criteria)")

        n_bins = int((time_window[1] - time_window[0]) / time_bin_size)
        bin_starts = time_window[0] + np.arange(n_bins) * time_bin_size
        bin_ends = bin_starts + time_bin_size
        bin_centers = bin_starts + time_bin_size / 2

        # Single tensor build over all events; split by label afterwards.
        rates = event_aligned_rates(
            spike_times_list, align_times, time_window,
            bin_starts, bin_ends, time_bin_size,
        )  # (n_events, n_cells, n_bins)

        unique_labels = np.unique(event_labels)
        population_data: Dict = {}
        event_counts: Dict = {}
        for label in unique_labels:
            mask = event_labels == label
            population_data[label] = rates[mask]
            event_counts[label] = int(mask.sum())

        if normalize_method != 'none':
            population_data = self._normalize_population_data(
                population_data, normalize_method, bin_centers,
            )

        return {
            'population_data': population_data,
            'time_bins': bin_centers,
            'event_counts': event_counts,
            'unique_labels': unique_labels,
            'metadata': {
                'time_window': time_window,
                'time_bin_size': time_bin_size,
                'alignment': alignment,
                'normalize_method': normalize_method,
                'n_cells': len(spike_times_list),
                'n_bins': n_bins,
                'n_conditions': len(unique_labels),
                'total_events': len(event_starts),
                'selected_cell_ids': selected_cluster_ids,
            },
        }

    @staticmethod
    def _normalize_population_data(pop_data: Dict, method: str,
                                   bin_centers: np.ndarray) -> Dict:
        """Vectorised z-score or baseline normalisation of (n_events, n_cells, n_bins) tensors."""
        if method == 'zscore':
            # Per-cell zscore across (events × time) within each condition
            out = {}
            for label, data in pop_data.items():
                mean = data.mean(axis=(0, 2), keepdims=True)
                std = data.std(axis=(0, 2), keepdims=True)
                std = np.where(std == 0, 1.0, std)
                out[label] = (data - mean) / std
            return out

        if method == 'baseline':
            baseline_mask = bin_centers < 0
            out = {}
            for label, data in pop_data.items():
                base = data[:, :, baseline_mask]
                base_mean = base.mean(axis=2, keepdims=True)
                base_std = base.std(axis=2, keepdims=True)
                base_std = np.where(base_std == 0, 1.0, base_std)
                out[label] = (data - base_mean) / base_std
            return out

        return pop_data

    # ------------------------------------------------------------------ #
    # Dimensionality reduction
    # ------------------------------------------------------------------ #

    def apply_dimensionality_reduction(self,
                                       population_matrix: Dict,
                                       method: str = 'pca',
                                       n_components: int = 3,
                                       **method_kwargs) -> Dict:
        """
        Apply PCA or UMAP to a population matrix.

        Returns a dict with ``reduced_trajectories`` (``{label: (n_events,
        n_bins, n_components)}``), the flat reduced data + indexing arrays,
        and a ``reduction_info`` block (explained variance for PCA).
        """
        pop_data = population_matrix['population_data']
        time_bins = population_matrix['time_bins']
        unique_labels = population_matrix['unique_labels']

        # Flatten (events × time) into rows; cells become features.
        all_data = []
        all_labels: List = []
        all_times: List = []
        all_events: List = []

        for label in unique_labels:
            data = pop_data[label]  # (n_events, n_cells, n_bins)
            n_events, n_cells, n_time = data.shape
            # Transpose to (events, time, cells) then reshape to (events*time, cells)
            reshaped = data.transpose(0, 2, 1).reshape(-1, n_cells)
            all_data.append(reshaped)
            all_labels.extend(np.repeat(label, n_events * n_time))
            all_times.extend(np.tile(time_bins, n_events))
            all_events.extend(np.repeat(np.arange(n_events), n_time))

        X = np.vstack(all_data)
        labels = np.array(all_labels)
        times = np.array(all_times)
        events = np.array(all_events)

        method_lower = method.lower()
        if method_lower == 'pca':
            reducer = PCA(n_components=n_components, **method_kwargs)
            X_reduced = reducer.fit_transform(X)
            reduction_info = {
                'explained_variance_ratio': reducer.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(reducer.explained_variance_ratio_),
                'components': reducer.components_,
                'reducer': reducer,
            }
        elif method_lower == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            umap_params = {
                'n_neighbors': 15, 'min_dist': 0.1,
                'metric': 'euclidean', 'random_state': 42,
            }
            umap_params.update(method_kwargs)
            reducer = umap.UMAP(n_components=n_components, **umap_params)
            X_reduced = reducer.fit_transform(X)
            reduction_info = {'reducer': reducer, 'umap_params': umap_params}
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

        # Re-organise reduced rows back into per-condition trajectories.
        reduced_trajectories: Dict = {}
        start_idx = 0
        n_time = len(time_bins)
        for label in unique_labels:
            n_events_lbl = population_matrix['event_counts'][label]
            end_idx = start_idx + n_events_lbl * n_time
            reduced_trajectories[label] = (
                X_reduced[start_idx:end_idx]
                .reshape(n_events_lbl, n_time, n_components)
            )
            start_idx = end_idx

        return {
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

    # ------------------------------------------------------------------ #
    # Trajectory plots
    # ------------------------------------------------------------------ #

    def plot_population_dynamics(self,
                                 reduced_data: Dict,
                                 show_individual: bool = False,
                                 time_range: Optional[Tuple[float, float]] = None,
                                 figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """3-D matplotlib plot of population trajectories (mean per condition by default)."""
        trajectories = reduced_data['reduced_trajectories']
        time_bins = reduced_data['time_bins']
        unique_labels = reduced_data['unique_labels']

        time_mask = (
            (time_bins >= time_range[0]) & (time_bins <= time_range[1])
            if time_range is not None else slice(None)
        )
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for i, label in enumerate(unique_labels):
            traj = trajectories[label]
            n_events = traj.shape[0]

            if show_individual:
                alpha = 0.6 if n_events <= 5 else 0.3
                linewidth = 1.5 if n_events <= 5 else 1.0
                for event_idx in range(n_events):
                    t = traj[event_idx][time_mask]
                    ax.plot(t[:, 0], t[:, 1], t[:, 2],
                            color=colors[i], alpha=alpha, linewidth=linewidth,
                            label=f'{label}' if event_idx == 0 else "")
                    ax.scatter(*t[0, :3], color=colors[i], s=30, marker='o', alpha=0.7)
                    ax.scatter(*t[-1, :3], color=colors[i], s=30, marker='s', alpha=0.7)
            else:
                mean_traj = np.mean(traj, axis=0)[time_mask]
                ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2],
                        color=colors[i], linewidth=3, label=f'{label} (n={n_events})')
                ax.scatter(*mean_traj[0, :3], color=colors[i], s=100, marker='o', alpha=0.8)
                ax.scatter(*mean_traj[-1, :3], color=colors[i], s=100, marker='s', alpha=0.8)

        method_name = reduced_data['method'].upper()
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_zlabel(f'{method_name} Component 3')
        ax.set_title(
            'Neural Population Trajectories - All Individual Trials' if show_individual
            else 'Neural Population Trajectories - Mean per Condition'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    def plot_population_dynamics_interactive(self,
                                             reduced_data: Dict,
                                             show_individual: bool = False,
                                             time_range: Optional[Tuple[float, float]] = None):
        """Interactive plotly version of :meth:`plot_population_dynamics`."""
        import plotly.express as px
        import plotly.graph_objects as go

        trajectories = reduced_data['reduced_trajectories']
        time_bins = reduced_data['time_bins']
        unique_labels = reduced_data['unique_labels']
        method_name = reduced_data['method'].upper()

        time_mask = (
            (time_bins >= time_range[0]) & (time_bins <= time_range[1])
            if time_range is not None else slice(None)
        )
        palette = px.colors.qualitative.Set1
        fig = go.Figure()

        for i, label in enumerate(unique_labels):
            color = palette[i % len(palette)]
            traj = trajectories[label]
            n_events = traj.shape[0]

            if show_individual:
                for event_idx in range(n_events):
                    t = traj[event_idx][time_mask]
                    fig.add_trace(go.Scatter3d(
                        x=t[:, 0], y=t[:, 1], z=t[:, 2],
                        mode='lines', line=dict(color=color, width=3),
                        opacity=0.5, name=str(label),
                        legendgroup=str(label), showlegend=(event_idx == 0),
                    ))
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
                    mode='lines', line=dict(color=color, width=5),
                    name=f'{label} (n={n_events})',
                ))
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

        title_suffix = (' - All Individual Trials' if show_individual
                        else ' - Mean per Condition')
        fig.update_layout(
            title='Neural Population Trajectories' + title_suffix,
            scene=dict(
                xaxis_title=f'{method_name} Component 1',
                yaxis_title=f'{method_name} Component 2',
                zaxis_title=f'{method_name} Component 3',
            ),
            width=900, height=700,
        )
        return fig

    # ------------------------------------------------------------------ #
    # PCA summary plot
    # ------------------------------------------------------------------ #

    def plot_pca_summary(self,
                         reduced_data: Dict,
                         pop_data: Dict,
                         figsize: Tuple[float, float] = (16, 12)) -> plt.Figure:
        """2x2 PCA diagnostic: correlation matrix, component weights, variance explained."""
        if reduced_data['method'].lower() != 'pca':
            raise ValueError("This method only works with PCA reduced data")

        reduction_info = reduced_data['reduction_info']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1) Cell × cell correlation matrix
        combined = np.vstack([
            data.transpose(0, 2, 1).reshape(-1, data.shape[1])
            for data in pop_data.values()
        ])
        corr_matrix = np.corrcoef(combined.T)
        im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax1.set_title('Cross Correlation Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Neuron Index')
        ax1.set_ylabel('Neuron Index')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Correlation Coefficient', rotation=270, labelpad=15)

        # 2) PCA component weights
        components = reduction_info['components']
        n_show = min(10, components.shape[0])
        vmax = np.abs(components).max()
        im2 = ax2.imshow(components[:n_show], cmap='RdBu_r', aspect='auto',
                         vmin=-vmax, vmax=vmax, interpolation='none')
        ax2.set_title(f'PCA Component Weights (First {n_show} PCs)',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Neuron Index')
        ax2.set_ylabel('Principal Component')
        ax2.set_yticks(range(n_show))
        ax2.set_yticklabels([f'PC{i+1}' for i in range(n_show)])
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Component Loading', rotation=270, labelpad=15)

        # 3) Individual variance explained
        explained_var = reduction_info['explained_variance_ratio']
        n_var = min(15, len(explained_var))
        bars = ax3.bar(range(1, n_var + 1), explained_var[:n_var] * 100,
                       color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_title('Individual Variance Explained', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Variance Explained (%)')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax3.text(bar.get_x() + bar.get_width() / 2., h + 0.1,
                         f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

        # 4) Cumulative variance + milestone markers
        cumulative_var = reduction_info['cumulative_variance']
        n_cum = min(20, len(cumulative_var))
        ax4.plot(range(1, n_cum + 1), cumulative_var[:n_cum] * 100,
                 'o-', color='darkred', linewidth=2, markersize=4, markerfacecolor='red')
        ax4.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Principal Components')
        ax4.set_ylabel('Cumulative Variance Explained (%)')
        ax4.grid(True, alpha=0.3)
        for threshold in (80, 90, 95):
            ax4.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
            ax4.text(1, threshold + 1, f'{threshold}%', fontsize=9, color='gray')
        for threshold in (0.8, 0.9, 0.95):
            idx = np.where(cumulative_var >= threshold)[0]
            if len(idx) and idx[0] + 1 <= n_cum:
                first_idx = idx[0] + 1
                ax4.scatter(first_idx, threshold * 100, color='orange', s=60, zorder=10)
                ax4.annotate(f'PC{first_idx}',
                             (first_idx, threshold * 100),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=9, fontweight='bold')

        n_neurons = components.shape[1]
        total_var_3pc = cumulative_var[2] * 100 if len(cumulative_var) >= 3 else 0
        fig.suptitle(
            f'PCA Analysis Summary\n'
            f'{n_neurons} neurons, {len(explained_var)} components, '
            f'First 3 PCs explain {total_var_3pc:.1f}% variance',
            fontsize=16, fontweight='bold', y=0.95,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        return fig

    # ------------------------------------------------------------------ #
    # Normalised population matrix heatmap
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_normalized_population_matrix(pop_data: Dict[str, np.ndarray],
                                          figsize: Tuple[float, float] = (14, 10)) -> plt.Figure:
        """Heatmap of z-scored per-cell activity across all events × time bins."""
        full_data = np.concatenate(list(pop_data.values()), axis=0)
        n_total_events, n_cells, n_time_bins = full_data.shape

        # Flatten per cell across (events × time), then z-score.
        cell_rows = full_data.transpose(1, 0, 2).reshape(n_cells, -1)
        mean = cell_rows.mean(axis=1, keepdims=True)
        std = cell_rows.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        normalized = (cell_rows - mean) / std

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(normalized, aspect='auto', cmap='RdBu_r',
                       vmin=-3, vmax=3, interpolation='nearest')
        ax.set_xlabel('Time Points (Events × Time Bins)', fontsize=12)
        ax.set_ylabel('Cells', fontsize=12)
        ax.set_title(
            'Population Matrix: Z-Score Normalized Neural Activity\n'
            f'({n_cells} cells × {n_total_events} events × {n_time_bins} time bins)',
            fontsize=14, fontweight='bold', pad=20,
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Z-Score', rotation=270, labelpad=20, fontsize=12)

        # Event separators
        for event_idx in range(1, n_total_events):
            ax.axvline(x=event_idx * n_time_bins - 0.5,
                       color='gray', alpha=0.3, linewidth=0.5)

        # Sparse x-axis labels
        n_x_ticks = min(10, n_total_events)
        x_step = max(1, n_total_events // n_x_ticks)
        x_positions = [i * n_time_bins + n_time_bins / 2 - 0.5
                       for i in range(0, n_total_events, x_step)]
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'E{i+1}' for i in range(0, n_total_events, x_step)],
                           fontsize=10)

        # Sparse y-axis labels
        n_y_ticks = min(20, n_cells)
        y_positions = np.linspace(0, n_cells - 1, n_y_ticks).astype(int)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f'C{pos+1}' for pos in y_positions], fontsize=10)

        stats_text = (
            f'Activity Statistics:\n'
            f'Mean: {np.mean(normalized):.3f}\n'
            f'Std: {np.std(normalized):.3f}\n'
            f'Range: [{np.min(normalized):.2f}, {np.max(normalized):.2f}]'
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Similarity-analysis helpers (shared by both compute_* methods)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_windows(
        windows: Union[Tuple[float, float], List[Tuple[float, float]]],
    ) -> List[Tuple[float, float]]:
        """Coerce ``windows`` to a list of ``(start, end)`` tuples; validate widths."""
        def _is_single(w) -> bool:
            try:
                return (
                    len(w) == 2
                    and isinstance(w[0], (int, float))
                    and isinstance(w[1], (int, float))
                )
            except TypeError:
                return False

        if _is_single(windows):
            window_list = [(float(windows[0]), float(windows[1]))]
        else:
            window_list = [(float(w[0]), float(w[1])) for w in windows]

        for w_start, w_end in window_list:
            if w_end <= w_start:
                raise ValueError(f"Window {(w_start, w_end)} has non-positive width")
        return window_list

    @staticmethod
    def _normalize_time_range(
        time_range: Optional[Tuple[Optional[float], Optional[float]]],
    ) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """Validate / canonicalise an optional ``(min, max)`` time-range filter."""
        if time_range is None:
            return None
        if len(time_range) != 2:
            raise ValueError(f"time_range must be (min_time, max_time), got {time_range}")
        t_min = None if time_range[0] is None else float(time_range[0])
        t_max = None if time_range[1] is None else float(time_range[1])
        if t_min is not None and t_max is not None and t_max <= t_min:
            raise ValueError(f"time_range max ({t_max}) must exceed min ({t_min})")
        return (t_min, t_max)

    @staticmethod
    def _filter_events_by_time_range(
        event_starts: np.ndarray,
        event_ends: np.ndarray,
        labels: np.ndarray,
        time_range_norm: Optional[Tuple[Optional[float], Optional[float]]],
        min_events_per_class: int,
        behavior_type: str,
        animal_of_interest: str,
        base_params: Dict,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict]:
        """Apply ``time_range`` filter then re-threshold per class.

        Returns the filtered ``(starts, ends, labels)`` tuple, or a failure
        dict ``{'status': 'failed', 'error': ..., 'parameters': base_params}``.
        """
        if len(event_starts) == 0:
            return {
                'status': 'failed',
                'error': (
                    f"No '{behavior_type}' events for {animal_of_interest} "
                    f"after min_events_per_opponent={min_events_per_class} filter."
                ),
                'parameters': base_params,
            }

        if time_range_norm is None:
            return event_starts, event_ends, labels

        t_min, t_max = time_range_norm
        in_range = np.ones(len(event_starts), dtype=bool)
        if t_min is not None:
            in_range &= event_starts >= t_min
        if t_max is not None:
            in_range &= event_starts <= t_max
        event_starts = event_starts[in_range]
        event_ends = event_ends[in_range]
        labels = labels[in_range]

        if len(event_starts) == 0:
            return {
                'status': 'failed',
                'error': (
                    f"No '{behavior_type}' events for {animal_of_interest} "
                    f"within time_range={time_range_norm}."
                ),
                'parameters': base_params,
            }

        uniq, counts = np.unique(labels, return_counts=True)
        keep = set(uniq[counts >= min_events_per_class].tolist())
        if not keep:
            return {
                'status': 'failed',
                'error': (
                    f"No opponent reaches min_events_per_opponent="
                    f"{min_events_per_class} within time_range={time_range_norm}."
                ),
                'parameters': base_params,
            }
        mask = np.array([lbl in keep for lbl in labels])
        return event_starts[mask], event_ends[mask], labels[mask]

    def _build_per_opponent_features(
        self,
        event_starts: np.ndarray,
        event_ends: np.ndarray,
        opponent_labels: np.ndarray,
        window_list: List[Tuple[float, float]],
        alignment: str,
        use_quality_cells: bool,
        top_n_cells: Optional[int],
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Optional[int]]:
        """Build per-opponent ``(n_events_opp, n_cells × n_windows)`` feature matrices.

        For each window in ``window_list``, build a single-bin population matrix
        (``time_bin_size`` = window width), then concatenate per-cell rates
        across windows. Optionally keep only the top-``N`` highest-firing cells
        (ranked by mean rate summed across opponents and windows).
        """
        per_window: List[Dict[str, np.ndarray]] = []
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
            per_window.append({
                str(opp): data[:, :, 0]
                for opp, data in pop_matrix['population_data'].items()
            })

        opp_features: Dict[str, np.ndarray] = {
            opp: np.concatenate([per_window[i][opp] for i in range(len(window_list))],
                                axis=1)
            for opp in per_window[0]
        }

        selected_cell_indices: Optional[np.ndarray] = None
        if top_n_cells is not None and n_cells_total is not None:
            cell_rate_sum = np.zeros(n_cells_total)
            for window_means in per_window:
                for mat in window_means.values():
                    cell_rate_sum += mat.mean(axis=0)
            n_keep = min(top_n_cells, n_cells_total)
            selected_cell_indices = np.argsort(cell_rate_sum)[::-1][:n_keep]
            selected_cell_indices.sort()
            feature_index = np.concatenate([
                selected_cell_indices + i * n_cells_total
                for i in range(len(window_list))
            ])
            for opp in opp_features:
                opp_features[opp] = opp_features[opp][:, feature_index]
            n_cells_total = int(n_keep)

        return opp_features, selected_cell_indices, n_cells_total

    # ------------------------------------------------------------------ #
    # Opponent-representation similarity matrix
    # ------------------------------------------------------------------ #

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
        Pairwise Pearson similarity between per-opponent mean population vectors.

        For each window in ``windows``, the mean firing rate per cell per
        opponent (averaged across that opponent's events) is computed.
        Per-cell means are concatenated across windows into a single feature
        vector per opponent (length ``n_cells × n_windows``). The output
        ``similarity_matrix`` is the pairwise Pearson r across opponents.

        Parameters
        ----------
        animal_of_interest : str
        behavior_type : str, default='EC'
        windows : (start, end) tuple or list of such tuples, default=(-1.0, 0.0)
        alignment : {'start', 'end', 'center'}, default='start'
        use_quality_cells : bool, default=True
        min_events_per_opponent : int, default=5
        top_n_cells : int, optional
            Keep only this many highest-firing cells (across all events and
            windows). ``None`` keeps all selected cells.
        time_range : (min_time, max_time), optional
            Restrict to events whose start time (ephys seconds) is within
            ``[min, max]``; either bound may be ``None``.

        Returns
        -------
        dict
            ``status``, ``similarity_matrix``, ``opponents``, ``mean_activity``,
            ``event_counts``, ``selected_cell_indices``, ``parameters``,
            ``behavioral_summary``. On failure: ``{'status': 'failed',
            'error': ..., 'parameters': ...}``.
        """
        if self.behavior_data is None:
            raise ValueError(
                "compute_opponent_similarity requires behavior_data; "
                "pass it to PopulationGeometryAnalyzer(ks_data, behavior_data)."
            )

        window_list = self._normalize_windows(windows)
        time_range_norm = self._normalize_time_range(time_range)
        if top_n_cells is not None and top_n_cells <= 0:
            raise ValueError(f"top_n_cells must be positive or None, got {top_n_cells}")

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

        # If a time_range is set, defer the min-events filter until after we
        # restrict to the range so the threshold reflects events used downstream.
        extract_min = 1 if time_range_norm is not None else min_events_per_opponent
        event_starts, event_ends, opponent_labels = self.behavior_data.extract_opponent_labels(
            animal_of_interest=animal_of_interest,
            behavior_type=behavior_type,
            min_events_per_class=extract_min,
        )
        event_starts = np.asarray(event_starts)
        event_ends = np.asarray(event_ends)
        opponent_labels = np.asarray(opponent_labels)

        filtered = self._filter_events_by_time_range(
            event_starts, event_ends, opponent_labels,
            time_range_norm, min_events_per_opponent,
            behavior_type, animal_of_interest, base_params,
        )
        if isinstance(filtered, dict):
            return filtered
        event_starts, event_ends, opponent_labels = filtered

        opp_features, selected_cell_indices, n_cells_final = self._build_per_opponent_features(
            event_starts, event_ends, opponent_labels,
            window_list, alignment, use_quality_cells, top_n_cells,
        )

        opponents_sorted = sorted(opp_features)
        mean_activity = np.vstack([opp_features[opp].mean(axis=0)
                                   for opp in opponents_sorted])
        similarity_matrix = np.corrcoef(mean_activity)

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
        """Heatmap of the similarity matrix + within/between-group summary bars."""
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
            own_idx, other_idx = low_idx, high_idx

        def _offdiag(block: np.ndarray) -> np.ndarray:
            n = block.shape[0]
            if n < 2:
                return np.array([])
            iu = np.triu_indices(n, k=1)
            return block[iu]

        within_own = _offdiag(sim[np.ix_(own_idx, own_idx)])
        within_other = _offdiag(sim[np.ix_(other_idx, other_idx)])
        between = (sim[np.ix_(own_idx, other_idx)].ravel()
                   if len(own_idx) and len(other_idx) else np.array([]))

        fig, (ax_hm, ax_bar) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={'width_ratios': [3, 1]},
        )

        sns.heatmap(
            sim, ax=ax_hm,
            xticklabels=list(opponents), yticklabels=list(opponents),
            cmap=cmap, vmin=vmin, vmax=vmax,
            annot=annotate, fmt='.2f', square=True,
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

        categories = ['Within own', 'Within other', 'Between']
        pools = [within_own, within_other, between]
        means = [float(np.mean(p)) if len(p) else np.nan for p in pools]
        sems = [float(np.std(p, ddof=1) / np.sqrt(len(p))) if len(p) > 1 else 0.0
                for p in pools]
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

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        return fig

    # ------------------------------------------------------------------ #
    # Event-wise similarity to a reference opponent
    # ------------------------------------------------------------------ #

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
        Per-event Pearson similarity to a reference opponent's mean activity.

        For each non-reference opponent ``opp``, return a 1-D array of Pearson
        correlations between each of ``opp``'s events (chronological order) and
        the reference's mean per-cell vector. Feature construction (windows,
        top-N cells, quality filter) matches :meth:`compute_opponent_similarity`.

        Returns
        -------
        dict
            On success: ``status``, ``reference_animal``, ``similarity_traces``,
            ``event_indices``, ``event_times``, ``reference_mean_activity``,
            ``event_counts``, ``selected_cell_indices``, ``parameters``. On
            failure: ``{'status': 'failed', 'error': ..., 'parameters': ...}``.
        """
        if self.behavior_data is None:
            raise ValueError("compute_event_similarity_to_reference requires behavior_data.")

        window_list = self._normalize_windows(windows)
        time_range_norm = self._normalize_time_range(time_range)
        if top_n_cells is not None and top_n_cells <= 0:
            raise ValueError(f"top_n_cells must be positive or None, got {top_n_cells}")

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

        extract_min = 1 if time_range_norm is not None else min_events_per_opponent
        event_starts, event_ends, opponent_labels = self.behavior_data.extract_opponent_labels(
            animal_of_interest=animal_of_interest,
            behavior_type=behavior_type,
            min_events_per_class=extract_min,
        )
        event_starts = np.asarray(event_starts)
        event_ends = np.asarray(event_ends)
        opponent_labels = np.asarray(opponent_labels)

        filtered = self._filter_events_by_time_range(
            event_starts, event_ends, opponent_labels,
            time_range_norm, min_events_per_opponent,
            behavior_type, animal_of_interest, base_params,
        )
        if isinstance(filtered, dict):
            return filtered
        event_starts, event_ends, opponent_labels = filtered

        # Sort chronologically so per-opp indexing reflects serial event order.
        order = np.argsort(event_starts, kind='stable')
        event_starts = event_starts[order]
        event_ends = event_ends[order]
        opponent_labels = opponent_labels[order]

        # Resolve reference label by trailing digits (accepts '632' or 'rat632').
        unique_label_strs = sorted({str(l) for l in opponent_labels})
        ref_str = str(reference_animal)
        if ref_str not in unique_label_strs:
            m = re.search(r'(\d+)$', ref_str)
            ref_num = m.group(1) if m else None
            if ref_num is not None:
                for l in unique_label_strs:
                    mm = re.search(r'(\d+)$', l)
                    if mm is not None and mm.group(1) == ref_num:
                        ref_str = l
                        break
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

        opp_features, selected_cell_indices, n_cells_total = self._build_per_opponent_features(
            event_starts, event_ends, opponent_labels,
            window_list, alignment, use_quality_cells, top_n_cells,
        )

        ref_features = opp_features[ref_str]
        ref_mean = ref_features.mean(axis=0)
        ref_var = float(np.std(ref_mean))

        other_opps = [o for o in sorted(opp_features) if o != ref_str]
        similarity_traces: Dict[str, np.ndarray] = {}
        event_indices: Dict[str, np.ndarray] = {}
        event_times: Dict[str, np.ndarray] = {}

        for opp in other_opps:
            feats = opp_features[opp]
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
        Plot per-opponent similarity traces from
        :meth:`compute_event_similarity_to_reference`.

        x = serial event number, y = Pearson r between the event's population
        activity vector and the reference's mean activity vector.
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

    Returns ``(plotly Figure, result_dict)``.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    spks, bin_centers = ks_data.bin_spike_times(
        bin_size_sec=bin_size,
        t_start=start_time,
        t_end=end_time,
        filtered_only=filtered_only,
    )
    n_cells, n_bins = spks.shape
    print(f"Firing-rate matrix: {n_cells} cells × {n_bins} bins "
          f"({bin_centers[0]:.1f}–{bin_centers[-1]:.1f} s)")

    X = zscore(spks, axis=1)
    X = np.nan_to_num(X, 0.0)
    pca = PCA(n_components=min(n_components, n_cells))
    scores = pca.fit_transform(X.T)

    ev = pca.explained_variance_ratio_
    print(f"PCA explained variance: "
          + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(ev)))

    if not animal_of_interest.startswith('rat'):
        animal_of_interest = f"rat{animal_of_interest}"

    if behavior_types is None:
        behavior_types = behavior_data.get_available_event_types()

    all_event_starts, all_event_ends, all_opponents, all_btypes = [], [], [], []
    for bt in behavior_types:
        try:
            starts, ends, opponents = behavior_data.extract_opponent_labels(
                animal_of_interest, behavior_type=bt,
                min_events_per_class=min_events_per_class,
            )
        except (KeyError, ValueError):
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

    event_bin_idx = np.searchsorted(bin_centers, ev_starts).clip(0, n_bins - 1)
    btype_map = behavior_data.get_behavior_type_mapping()

    palette = px.colors.qualitative.Set1
    unique_opponents = np.unique(ev_opponents) if len(ev_opponents) > 0 else []
    opp_color = {opp: palette[i % len(palette)] for i, opp in enumerate(unique_opponents)}

    fig = go.Figure()

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

    for opp in unique_opponents:
        mask = ev_opponents == opp
        idx = event_bin_idx[mask]
        btypes_here = ev_btypes[mask]
        hover = [
            f"t={bin_centers[bi]:.1f}s<br>{btype_map.get(b, b)} vs {opp}"
            for bi, b in zip(idx, btypes_here)
        ]
        fig.add_trace(go.Scatter3d(
            x=scores[idx, 0], y=scores[idx, 1], z=scores[idx, 2],
            mode='markers',
            marker=dict(size=marker_size, color=opp_color[opp],
                        line=dict(width=0.5, color='black')),
            name=opp, hovertext=hover, hoverinfo='text',
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


__all__ = [
    'PopulationGeometryAnalyzer',
    'plot_pca_trajectory_with_events',
]
