#!/usr/bin/env python3
"""
Opponent Identity Decoding from Single Cell Ephys Activity

This module provides functions for decoding opponent animal identity during
behavioral events using single-cell ephys activity and cross-validated Linear
Discriminant Analysis (LDA).

The shared LDA/CV/feature/cell-select machinery lives in
``ephys/_lda_decoding.py``; this module is the opponent-flavored wrapper that
extracts opponent / group labels and shapes results for plotting.

Usage:
    from ephys.decode_opponent_identity import decode_opponent_identity_population
    from ingestion.kilosort_data_import import load_kilosort_data
    from video.behavioral_events import load_behavioral_events
    from ingestion.data_paths import DataStorageManager

    data_manager = DataStorageManager("631", "20251216", auto_load=True)
    ks_data = load_kilosort_data(data_manager)
    behavior_data = load_behavioral_events(
        data_manager.get_behavioral_event_files(),
        session_id=data_manager.session_id,
    )

    results = decode_opponent_identity_population(
        ks_data=ks_data,
        behavior_data=behavior_data,
        animal_of_interest="631",
        behavior_type='F',
        use_quality_cells=True,
        alignment='start',
        time_window=(-0.5, 1.0),
        time_bin_size=0.1,
        cv_folds=5,
    )
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ephys._lda_decoding import (
    align_spikes_to_events,
    extract_firing_rate_features,
    run_population_per_cell_decode,
    run_time_resolved_population_decode,
    select_quality_cells,
    single_cell_lda_decode,
)


def decode_opponent_identity_single_cell(spike_times: np.ndarray,
                                         event_times: np.ndarray,
                                         opponent_labels: np.ndarray,
                                         alignment: str = 'start',
                                         time_window: Tuple[float, float] = (-1.0, 2.0),
                                         time_bin_size: float = 0.5,
                                         cv_folds: int = 5,
                                         min_events_per_class: int = 5,
                                         selected_opponents: Optional[List[str]] = None) -> Dict:
    """Decode opponent identity from single-cell activity using cross-validated LDA.

    Thin wrapper around ``single_cell_lda_decode`` that applies an optional
    ``selected_opponents`` mask first. Result dict keys are unchanged.
    """
    if selected_opponents is not None:
        mask = np.isin(opponent_labels, selected_opponents)
        event_times = event_times[mask]
        opponent_labels = opponent_labels[mask]

    return single_cell_lda_decode(
        spike_times=spike_times,
        event_times=event_times,
        labels=opponent_labels,
        time_window=time_window,
        time_bin_size=time_bin_size,
        cv_folds=cv_folds,
        min_events_per_class=min_events_per_class,
    )


def _extract_labels(behavior_data,
                    animal_of_interest: str,
                    behavior_type: Optional[str],
                    min_events_per_class: int,
                    label_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if label_mode == 'group':
        return behavior_data.extract_group_labels(
            animal_of_interest, behavior_type, min_events_per_class
        )
    return behavior_data.extract_opponent_labels(
        animal_of_interest, behavior_type, min_events_per_class
    )


def decode_opponent_identity_population(ks_data,
                                        behavior_data,
                                        animal_of_interest: str,
                                        behavior_type: str = None,
                                        use_quality_cells: bool = True,
                                        quality_thresholds: Dict = None,
                                        alignment: str = 'start',
                                        time_window: Tuple[float, float] = (-1.0, 2.0),
                                        time_bin_size: float = 0.5,
                                        cv_folds: int = 5,
                                        min_events_per_class: int = 5,
                                        selected_opponents: Optional[List[str]] = None,
                                        label_mode: str = 'opponent') -> Dict:
    """Decode opponent identity across the population (per-cell LDA)."""
    if label_mode not in ('opponent', 'group'):
        raise ValueError(f"label_mode must be 'opponent' or 'group', got {label_mode!r}")

    try:
        event_start_times, event_end_times, opponent_labels = _extract_labels(
            behavior_data, animal_of_interest, behavior_type,
            min_events_per_class, label_mode,
        )
        if len(event_start_times) == 0:
            raise ValueError(f"No events found for behavior type '{behavior_type}'")
        print(f"Found {len(event_start_times)} behavioral events")
    except Exception as e:
        print(f"Error extracting behavioral events: {e}")
        return {'error': str(e), 'status': 'failed'}

    if alignment == 'start':
        event_times = event_start_times
    elif alignment == 'end':
        event_times = event_end_times
    else:
        raise ValueError("alignment must be 'start' or 'end'")

    if selected_opponents is not None:
        mask = np.isin(opponent_labels, selected_opponents)
        event_times = event_times[mask]
        opponent_labels = opponent_labels[mask]

    selected_cell_indices, selected_cluster_ids, quality_thresholds = select_quality_cells(
        ks_data, use_quality_cells=use_quality_cells, quality_thresholds=quality_thresholds,
    )
    if use_quality_cells:
        print(f"Using {len(selected_cluster_ids)} quality-filtered cells")
    else:
        print(f"Using all {len(selected_cluster_ids)} cells")

    if len(selected_cluster_ids) == 0:
        print("No cells selected for analysis")
        return {'error': 'No cells selected', 'status': 'failed'}

    cell_results, successful_cluster_ids, accuracies = run_population_per_cell_decode(
        ks_data=ks_data,
        event_times=event_times,
        labels=opponent_labels,
        selected_cell_indices=selected_cell_indices,
        selected_cluster_ids=selected_cluster_ids,
        time_window=time_window,
        time_bin_size=time_bin_size,
        cv_folds=cv_folds,
        min_events_per_class=min_events_per_class,
    )

    n_total = len(selected_cluster_ids)
    return {
        'cell_results': cell_results,
        'successful_cells': successful_cluster_ids,
        'n_successful_cells': len(successful_cluster_ids),
        'n_total_cells': n_total,
        'success_rate': len(successful_cluster_ids) / n_total if n_total > 0 else 0,
        'population_accuracy_mean': float(np.mean(accuracies)) if accuracies else np.nan,
        'population_accuracy_std': float(np.std(accuracies)) if accuracies else np.nan,
        'population_accuracy_median': float(np.median(accuracies)) if accuracies else np.nan,
        'best_cell_accuracy': float(np.max(accuracies)) if accuracies else np.nan,
        'best_cell_id': successful_cluster_ids[int(np.argmax(accuracies))] if accuracies else None,
        'parameters': {
            'animal_of_interest': animal_of_interest,
            'behavior_type': behavior_type,
            'use_quality_cells': use_quality_cells,
            'quality_thresholds': quality_thresholds,
            'alignment': alignment,
            'time_window': time_window,
            'time_bin_size': time_bin_size,
            'cv_folds': cv_folds,
            'min_events_per_class': min_events_per_class,
            'label_mode': label_mode,
        },
        'behavioral_summary': {
            'n_events': len(event_times),
            'unique_opponents': np.unique(opponent_labels),
            'opponent_counts': dict(zip(*np.unique(opponent_labels, return_counts=True))),
        },
        'status': 'success',
    }


def decode_opponent_identity_time_resolved(ks_data,
                                           behavior_data,
                                           animal_of_interest: str,
                                           behavior_type: str = None,
                                           use_quality_cells: bool = True,
                                           quality_thresholds: Dict = None,
                                           alignment: str = 'start',
                                           time_window: Tuple[float, float] = (-1.0, 2.0),
                                           time_bin_size: float = 0.5,
                                           time_bin_step: Optional[float] = None,
                                           cv_folds: int = 5,
                                           min_events_per_class: int = 5,
                                           n_shuffles: int = 0,
                                           selected_opponents: Optional[List[str]] = None,
                                           label_mode: str = 'opponent') -> Dict:
    """Population (multi-cell) LDA decoding of opponent identity per time bin."""
    if label_mode not in ('opponent', 'group'):
        return {'error': f"label_mode must be 'opponent' or 'group', got {label_mode!r}",
                'status': 'failed'}
    try:
        event_start_times, event_end_times, opponent_labels = _extract_labels(
            behavior_data, animal_of_interest, behavior_type,
            min_events_per_class, label_mode,
        )
        if len(event_start_times) == 0:
            raise ValueError(f"No events found for behavior type '{behavior_type}'")
    except Exception as e:
        return {'error': str(e), 'status': 'failed'}

    if alignment == 'start':
        event_times = event_start_times
    elif alignment == 'end':
        event_times = event_end_times
    else:
        raise ValueError("alignment must be 'start' or 'end'")

    if selected_opponents is not None:
        mask = np.isin(opponent_labels, selected_opponents)
        event_times = event_times[mask]
        opponent_labels = opponent_labels[mask]

    selected_cell_indices, selected_cluster_ids, _ = select_quality_cells(
        ks_data, use_quality_cells=use_quality_cells, quality_thresholds=quality_thresholds,
    )
    if len(selected_cluster_ids) == 0:
        return {'error': 'No cells selected', 'status': 'failed'}

    core = run_time_resolved_population_decode(
        ks_data=ks_data,
        event_times=event_times,
        labels=opponent_labels,
        selected_cell_indices=selected_cell_indices,
        selected_cluster_ids=selected_cluster_ids,
        time_window=time_window,
        time_bin_size=time_bin_size,
        time_bin_step=time_bin_step,
        cv_folds=cv_folds,
        n_shuffles=n_shuffles,
    )
    if core.get('status') != 'success':
        return core

    core['unique_opponents'] = core.pop('unique_classes')
    core['parameters'] = {
        'animal_of_interest': animal_of_interest,
        'behavior_type': behavior_type,
        'use_quality_cells': use_quality_cells,
        'alignment': alignment,
        'time_window': time_window,
        'time_bin_size': time_bin_size,
        'time_bin_step': core.pop('time_bin_step'),
        'cv_folds': cv_folds,
        'min_events_per_class': min_events_per_class,
        'n_shuffles': n_shuffles,
        'label_mode': label_mode,
    }
    return core


# Visualization functions

def plot_time_resolved_decoding(results: Dict,
                                figsize: Tuple[int, int] = (13, 5)) -> plt.Figure:
    """Plot the population-LDA accuracy curve as a function of time around the event,
    plus the confusion matrix for the best-accuracy time bin on the right.

    Expects the dict returned by ``decode_opponent_identity_time_resolved``.
    Shows accuracy ± CV-fold SEM, the chance level, and (when present) the
    label-shuffle 95% band.
    """
    if results.get('status') != 'success':
        return None

    acc = results['accuracy_by_bin']
    sem = results['accuracy_sem_by_bin']
    t = results['bin_centers']
    chance = results['chance_level']
    best_idx = results.get('best_bin_index')

    fig, (ax, ax_cm) = plt.subplots(1, 2, figsize=figsize,
                                    gridspec_kw={'width_ratios': [2, 1]})

    ax.fill_between(t, acc - sem, acc + sem, color='steelblue', alpha=0.25)
    ax.plot(t, acc, color='steelblue', linewidth=2.0,
            label=f'Population LDA ({results["n_cells"]} cells)')

    if results.get('shuffle_null') is not None:
        null = results['shuffle_null']
        lo, hi = np.nanpercentile(null, [2.5, 97.5], axis=0)
        ax.fill_between(t, lo, hi, color='gray', alpha=0.25,
                        label='Shuffle 95% band')

    ax.axhline(chance, color='red', linestyle='--', alpha=0.7,
               label=f'Chance ({chance:.1%})')
    ax.axvline(0, color='black', linestyle='--', alpha=0.4, linewidth=0.8)

    if best_idx is not None:
        ax.axvline(t[best_idx], color='goldenrod', linestyle=':', linewidth=1.5,
                   alpha=0.9, label=f'Best bin ({t[best_idx]:.2f}s)')

    params = results.get('parameters', {})
    btype = params.get('behavior_type', '?')
    n_opp = len(results.get('unique_opponents', []))
    ax.set_xlabel('Time from event (s)')
    ax.set_ylabel('Decoding accuracy')
    ax.set_title(f'Time-resolved population decoding of opponent identity\n'
                 f'behavior={btype} · {n_opp} opponents · {results["n_events"]} events')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    cm = results.get('best_bin_confusion_matrix')
    if cm is not None and best_idx is not None:
        unique_opponents = results.get('unique_opponents', [])
        im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(unique_opponents))
        ax_cm.set_xticks(tick_marks)
        ax_cm.set_yticks(tick_marks)
        ax_cm.set_xticklabels([f'{o}' for o in unique_opponents])
        ax_cm.set_yticklabels([f'{o}' for o in unique_opponents])
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        best_acc = results.get('best_bin_accuracy')
        title = f'Best bin: t={t[best_idx]:.2f}s'
        if best_acc is not None:
            title += f'\naccuracy={best_acc:.1%}'
        ax_cm.set_title(title)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, format(int(cm[i, j]), 'd'),
                           ha='center', va='center',
                           color='white' if cm[i, j] > thresh else 'black')
        plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    else:
        ax_cm.axis('off')
        ax_cm.text(0.5, 0.5, 'No confusion matrix available',
                   ha='center', va='center', transform=ax_cm.transAxes)

    fig.tight_layout()
    return fig


def plot_decoding_accuracy_distribution(results: Dict,
                                      figsize: Tuple[int, int] = (9, 5),
                                      save_path: str = None) -> plt.Figure:
    """
    Plot distribution of decoding accuracies across cells.

    Parameters:
    -----------
    results : Dict
        Results from decode_opponent_identity_population()
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure : The created figure
    """
    if results['status'] != 'success' or results['n_successful_cells'] == 0:
        print("No successful results to plot")
        return None

    # Extract accuracies
    accuracies = []
    for cluster_id in results['successful_cells']:
        acc = results['cell_results'][cluster_id]['accuracy']
        if not np.isnan(acc):
            accuracies.append(acc)

    if len(accuracies) == 0:
        print("No valid accuracies to plot")
        return None

    # Calculate chance level based on number of classes
    n_classes = len(results['behavioral_summary']['unique_opponents'])
    chance_level = 1.0 / n_classes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.axvline(chance_level, color='red', linestyle='--', alpha=0.7, label=f'Chance ({chance_level:.1%})')
    ax1.axvline(np.mean(accuracies), color='orange', linestyle='-', linewidth=2,
               label=f'Mean ({np.mean(accuracies):.1%})')
    ax1.set_xlabel('Decoding Accuracy')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Distribution of Opponent Identity Decoding Accuracies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot([accuracies], labels=['All Cells'])
    ax2.axhline(chance_level, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Decoding Accuracy')
    ax2.set_title('Accuracy Distribution Summary')
    ax2.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"""
    N cells: {len(accuracies)}
    Mean: {np.mean(accuracies):.1%}
    Median: {np.median(accuracies):.1%}
    Std: {np.std(accuracies):.1%}
    Best: {np.max(accuracies):.1%}
    > Chance: {np.sum(np.array(accuracies) > chance_level)}/{len(accuracies)}
    """

    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f"Opponent Identity Decoding Results\n" +
                f"Behavior: {results['parameters']['behavior_type']}, " +
                f"Alignment: {results['parameters']['alignment']}", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")

    return fig


def plot_best_cells_decoding(results: Dict,
                           n_top_cells: int = 10,
                           figsize: Tuple[int, int] = (12, 6),
                           save_path: str = None) -> plt.Figure:
    """
    Plot decoding performance of top performing cells.

    Parameters:
    -----------
    results : Dict
        Results from decode_opponent_identity_population()
    n_top_cells : int, default=10
        Number of top cells to show
    figsize : tuple, default=(12, 8)
        Figure size (width, height)
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure : The created figure
    """
    if results['status'] != 'success' or results['n_successful_cells'] == 0:
        print("No successful results to plot")
        return None

    # Get top performing cells
    cell_accs = [(cluster_id, results['cell_results'][cluster_id]['accuracy'])
                 for cluster_id in results['successful_cells']
                 if not np.isnan(results['cell_results'][cluster_id]['accuracy'])]

    if len(cell_accs) == 0:
        print("No valid accuracies to plot")
        return None

    # Calculate chance level based on number of classes
    n_classes = len(results['behavioral_summary']['unique_opponents'])
    chance_level = 1.0 / n_classes

    # Sort by accuracy and take top N
    cell_accs_sorted = sorted(cell_accs, key=lambda x: x[1], reverse=True)
    top_cells = cell_accs_sorted[:min(n_top_cells, len(cell_accs_sorted))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar plot of top cells
    cluster_ids = [str(cell[0]) for cell in top_cells]
    accuracies = [cell[1] for cell in top_cells]

    bars = ax1.barh(range(len(top_cells)), accuracies, color='lightcoral')
    ax1.set_yticks(range(len(top_cells)))
    ax1.set_yticklabels(cluster_ids)
    ax1.set_xlabel('Decoding Accuracy')
    ax1.set_ylabel('Cluster ID')
    ax1.set_title(f'Top {len(top_cells)} Cells by Decoding Accuracy')
    ax1.axvline(chance_level, color='red', linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3)

    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.1%}', va='center')

    # Confusion matrix for best cell
    best_cell_id = top_cells[0][0]
    best_result = results['cell_results'][best_cell_id]

    if best_result['confusion_matrix'] is not None:
        cm = best_result['confusion_matrix']
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title(f'Confusion Matrix - Best Cell (ID: {best_cell_id})')

        # Add labels
        unique_opponents = results['behavioral_summary']['unique_opponents']
        tick_marks = np.arange(len(unique_opponents))
        ax2.set_xticks(tick_marks)
        ax2.set_yticks(tick_marks)
        ax2.set_xticklabels([f'Rat {o}' for o in unique_opponents])
        ax2.set_yticklabels([f'Rat {o}' for o in unique_opponents])
        ax2.set_xlabel('Predicted Opponent')
        ax2.set_ylabel('True Opponent')

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.colorbar(im, ax=ax2)

    plt.suptitle(f"Top Performing Cells - Opponent Identity Decoding\n" +
                f"Behavior: {results['parameters']['behavior_type']}, " +
                f"Best accuracy: {top_cells[0][1]:.1%}", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")

    return fig


def plot_decoding_summary(results: Dict,
                         figsize: Tuple[int, int] = (15, 10),
                         save_path: str = None) -> plt.Figure:
    """
    Create comprehensive summary plot of decoding results.

    Parameters:
    -----------
    results : Dict
        Results from decode_opponent_identity_population()
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure : The created figure
    """
    if results['status'] != 'success':
        print("No successful results to plot")
        return None

    # Calculate chance level based on number of classes
    n_classes = len(results['behavioral_summary']['unique_opponents'])
    chance_level = 1.0 / n_classes

    fig = plt.figure(figsize=figsize)

    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Accuracy distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [results['cell_results'][cid]['accuracy']
                 for cid in results['successful_cells']
                 if not np.isnan(results['cell_results'][cid]['accuracy'])]

    if accuracies:
        ax1.hist(accuracies, bins=15, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.axvline(chance_level, color='red', linestyle='--', alpha=0.7, label='Chance')
        ax1.axvline(np.mean(accuracies), color='orange', linestyle='-',
                   linewidth=2, label=f'Mean ({np.mean(accuracies):.1%})')
        ax1.set_xlabel('Accuracy')
        ax1.set_ylabel('Count')
        ax1.set_title('Accuracy Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Confusion matrix for best cell
    ax2 = fig.add_subplot(gs[0, 1])
    cell_accs = [
        (cid, results['cell_results'][cid]['accuracy'])
        for cid in results['successful_cells']
        if not np.isnan(results['cell_results'][cid]['accuracy'])
    ]
    best_cell_id = max(cell_accs, key=lambda x: x[1])[0] if cell_accs else None
    best_result = results['cell_results'].get(best_cell_id) if best_cell_id is not None else None

    if best_result is not None and best_result.get('confusion_matrix') is not None:
        cm = best_result['confusion_matrix']
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title(f'Confusion Matrix — Best Cell (ID: {best_cell_id})')

        unique_opponents = results['behavioral_summary']['unique_opponents']
        tick_marks = np.arange(len(unique_opponents))
        ax2.set_xticks(tick_marks)
        ax2.set_yticks(tick_marks)
        ax2.set_xticklabels([f'{o}' for o in unique_opponents])
        ax2.set_yticklabels([f'{o}' for o in unique_opponents])
        ax2.set_xlabel('Predicted Opponent')
        ax2.set_ylabel('True Opponent')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.colorbar(im, ax=ax2)
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'No confusion matrix available',
                 ha='center', va='center', transform=ax2.transAxes)

    # 3. Behavioral event summary
    ax3 = fig.add_subplot(gs[0, 2])
    opponent_counts = results['behavioral_summary']['opponent_counts']
    opponents = list(opponent_counts.keys())
    counts = list(opponent_counts.values())

    bars = ax3.bar([f'{o}' for o in opponents], counts, color='lightblue')
    ax3.set_xlabel('Opponent Identity')
    ax3.set_ylabel('Number of Events')
    ax3.set_title(f"Behavioral Events\n(Total: {results['behavioral_summary']['n_events']})")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')

    # 4. Top cells performance
    ax4 = fig.add_subplot(gs[1, :2])
    if accuracies:
        # Sort cells by accuracy
        cell_accs = [(cid, results['cell_results'][cid]['accuracy'])
                     for cid in results['successful_cells']
                     if not np.isnan(results['cell_results'][cid]['accuracy'])]
        cell_accs_sorted = sorted(cell_accs, key=lambda x: x[1], reverse=True)

        # Show top 20 cells
        top_n = min(20, len(cell_accs_sorted))
        top_cells = cell_accs_sorted[:top_n]

        x_pos = range(top_n)
        accuracies_top = [cell[1] for cell in top_cells]

        bars = ax4.bar(x_pos, accuracies_top, color='lightcoral')
        ax4.axhline(chance_level, color='red', linestyle='--', alpha=0.7, label='Chance')
        ax4.set_xlabel('Cell Rank')
        ax4.set_ylabel('Accuracy')
        ax4.set_title(f'Top {top_n} Cells by Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Highlight best cell
        if len(bars) > 0:
            bars[0].set_color('red')
            bars[0].set_alpha(0.8)

    # 5. Parameter summary text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    params = results['parameters']
    param_text = f"""
    ANALYSIS PARAMETERS:

    Behavior Type: {params['behavior_type']}
    Alignment: {params['alignment']}
    Time Window: {params['time_window'][0]:.1f} to {params['time_window'][1]:.1f}s
    Time Bin Size: {params['time_bin_size']:.1f}s
    CV Folds: {params['cv_folds']}
    Quality Filtered: {params['use_quality_cells']}

    RESULTS SUMMARY:

    Population Accuracy: {results['population_accuracy_mean']:.1%} ± {results['population_accuracy_std']:.1%}
    Best Cell Accuracy: {results['best_cell_accuracy']:.1%}
    Best Cell ID: {results['best_cell_id']}
    Cells > Chance: {np.sum(np.array(accuracies) > chance_level) if accuracies else 0}/{len(accuracies) if accuracies else 0}
    """

    ax5.text(0.05, 0.95, param_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle(f"Opponent Identity Decoding - Complete Analysis Summary",
                fontsize=16, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot: {save_path}")

    return fig


def plot_top_cells_firing_rates(ks_data,
                               behavior_data,
                               test_results: Dict,
                               time_window: Tuple[float, float] = (-2.0, 2.0),
                               time_bin_size: float = 0.2,
                               n_top_cells: int = 12,
                               figsize: Tuple[int, int] = (15, 10),
                               save_path: str = None) -> plt.Figure:
    """
    Plot average firing rates around event times for top performing cells by opponent class.

    Parameters:
    -----------
    ks_data : KilosortData
        Kilosort electrophysiology data
    behavior_data : BehavioralEventsData
        Behavioral events data
    test_results : Dict
        Results from decode_opponent_identity_population()
    time_window : tuple, default=(-0.5, 1.0)
        Time window around events (start, end) in seconds
    time_bin_size : float, default=0.05
        Size of time bins in seconds for PETH
    n_top_cells : int, default=6
        Number of top cells to plot
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    plt.Figure : The created figure
    """
    if test_results['status'] != 'success' or test_results['n_successful_cells'] == 0:
        print("No successful results to plot")
        return None

    # Get top performing cells
    cell_accs = [(cluster_id, test_results['cell_results'][cluster_id]['accuracy'])
                 for cluster_id in test_results['successful_cells']
                 if not np.isnan(test_results['cell_results'][cluster_id]['accuracy'])]

    if len(cell_accs) == 0:
        print("No valid accuracies to plot")
        return None

    # Sort by accuracy and take top N
    cell_accs_sorted = sorted(cell_accs, key=lambda x: x[1], reverse=True)
    top_cells = cell_accs_sorted[:min(n_top_cells, len(cell_accs_sorted))]

    # Extract behavioral events and opponent labels
    params = test_results['parameters']
    try:
        event_start_times, event_end_times, opponent_labels = behavior_data.extract_opponent_labels(
            animal_of_interest=params.get('animal_of_interest', ''),
            behavior_type=params['behavior_type'],
            min_events_per_class=params['min_events_per_class']
        )

        if len(event_start_times) == 0:
            print("No behavioral events found")
            return None

    except Exception as e:
        print(f"Error extracting behavioral events: {e}")
        return None

    # Choose alignment times
    if params['alignment'] == 'start':
        event_times = event_start_times
    else:
        event_times = event_end_times

    # Get unique opponent classes and colors
    unique_opponents = np.unique(opponent_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_opponents)))

    # Create time bins
    bin_edges = np.arange(time_window[0], time_window[1] + time_bin_size, time_bin_size)
    bin_centers = bin_edges[:-1] + time_bin_size / 2

    # Set up subplots
    n_rows = int(np.ceil(n_top_cells / 3))
    n_cols = min(3, n_top_cells)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                            sharex=True, sharey=True)

    if n_top_cells == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each top cell
    for cell_idx, (cluster_id, accuracy) in enumerate(top_cells):
        if cell_idx >= n_top_cells:
            break

        # Get subplot
        row = cell_idx // n_cols
        col = cell_idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[0, col] if n_cols > 1 else axes[0]

        # Get spike times for this cell
        cell_position = None
        for i, ks_id in enumerate(ks_data.ks_ids):
            if ks_id == cluster_id:
                cell_position = i
                break

        if cell_position is None:
            print(f"Warning: Cell {cluster_id} not found in spike data")
            continue

        spike_times = ks_data.spike_times_by_cell[cell_position]

        # Process each opponent class
        for opp_idx, opponent in enumerate(unique_opponents):
            # Get events for this opponent class
            class_mask = opponent_labels == opponent
            class_event_times = event_times[class_mask]

            if len(class_event_times) == 0:
                continue

            # Align spikes to events for this class
            aligned_spikes = align_spikes_to_events(spike_times, class_event_times, time_window)

            # Calculate firing rates for each trial
            trial_firing_rates = []
            for trial_spikes in aligned_spikes:
                if len(trial_spikes) > 0:
                    counts, _ = np.histogram(trial_spikes, bins=bin_edges)
                    firing_rate = counts / time_bin_size  # Convert to Hz
                else:
                    firing_rate = np.zeros(len(bin_centers))
                trial_firing_rates.append(firing_rate)

            # Convert to array and calculate statistics
            trial_firing_rates = np.array(trial_firing_rates)
            mean_firing_rate = np.mean(trial_firing_rates, axis=0)
            sem_firing_rate = np.std(trial_firing_rates, axis=0) / np.sqrt(len(trial_firing_rates))

            # Plot mean with SEM shading
            color = colors[opp_idx]
            ax.plot(bin_centers, mean_firing_rate, color=color, linewidth=2,
                   label=f'Opponent {opponent} (n={len(class_event_times)})')
            ax.fill_between(bin_centers,
                          mean_firing_rate - sem_firing_rate,
                          mean_firing_rate + sem_firing_rate,
                          color=color, alpha=0.3)

        # Formatting
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'Cell {cluster_id}\nAccuracy: {accuracy:.1%}', fontsize=10)
        ax.grid(True, alpha=0.3)

        if cell_idx == 0:  # Add legend to first subplot
            ax.legend(fontsize=8, loc='upper right')

    # Remove empty subplots
    total_subplots = n_rows * n_cols
    for idx in range(n_top_cells, total_subplots):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1:
            fig.delaxes(axes[row, col])
        elif n_cols > 1:
            fig.delaxes(axes[0, col])

    # Set common labels
    fig.text(0.5, 0.02, 'Time from Event (s)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Firing Rate (Hz)', va='center', rotation=90, fontsize=12)

    plt.suptitle(f'Peri-Event Firing Rates - Top {len(top_cells)} Cells\n' +
                f'Behavior: {params["behavior_type"]}, ' +
                f'Alignment: {params["alignment"]}, ' +
                f'Window: {time_window[0]:.1f} to {time_window[1]:.1f}s',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved firing rate plot: {save_path}")

    return fig


def plot_top_cells_rasters(ks_data,
                           behavior_data,
                           test_results: Dict,
                           time_window: Tuple[float, float] = (-2.0, 2.0),
                           n_top_cells: int = 12,
                           figsize: Tuple[int, int] = (15, 10),
                           save_path: str = None) -> plt.Figure:
    """
    Plot spike rasters around event times for top performing cells.

    Each row in a panel is one event; rows are grouped/sorted by opponent
    identity, with horizontal separators and color-coded tick marks per class.

    Parameters mirror ``plot_top_cells_firing_rates``.

    Returns:
    --------
    plt.Figure : The created figure
    """
    if test_results['status'] != 'success' or test_results['n_successful_cells'] == 0:
        print("No successful results to plot")
        return None

    cell_accs = [(cluster_id, test_results['cell_results'][cluster_id]['accuracy'])
                 for cluster_id in test_results['successful_cells']
                 if not np.isnan(test_results['cell_results'][cluster_id]['accuracy'])]

    if len(cell_accs) == 0:
        print("No valid accuracies to plot")
        return None

    cell_accs_sorted = sorted(cell_accs, key=lambda x: x[1], reverse=True)
    top_cells = cell_accs_sorted[:min(n_top_cells, len(cell_accs_sorted))]

    params = test_results['parameters']
    try:
        event_start_times, event_end_times, opponent_labels = behavior_data.extract_opponent_labels(
            animal_of_interest=params.get('animal_of_interest', ''),
            behavior_type=params['behavior_type'],
            min_events_per_class=params['min_events_per_class']
        )
        if len(event_start_times) == 0:
            print("No behavioral events found")
            return None
    except Exception as e:
        print(f"Error extracting behavioral events: {e}")
        return None

    if params['alignment'] == 'start':
        event_times = event_start_times
    else:
        event_times = event_end_times

    # Sort events by opponent identity so each class forms a contiguous block
    unique_opponents = np.unique(opponent_labels)
    sort_idx = np.argsort(opponent_labels, kind='stable')
    event_times_sorted = event_times[sort_idx]
    opponent_labels_sorted = opponent_labels[sort_idx]

    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_opponents)))
    color_map = {opp: colors[i] for i, opp in enumerate(unique_opponents)}

    # Boundaries between opponent blocks (in row indices)
    block_boundaries = []
    for i in range(1, len(opponent_labels_sorted)):
        if opponent_labels_sorted[i] != opponent_labels_sorted[i - 1]:
            block_boundaries.append(i)

    n_top = len(top_cells)
    n_cols = min(3, n_top)
    n_rows = int(np.ceil(n_top / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)

    n_events = len(event_times_sorted)

    for cell_idx, (cluster_id, accuracy) in enumerate(top_cells):
        row = cell_idx // n_cols
        col = cell_idx % n_cols
        ax = axes[row, col]

        cell_position = None
        for i, ks_id in enumerate(ks_data.ks_ids):
            if ks_id == cluster_id:
                cell_position = i
                break
        if cell_position is None:
            print(f"Warning: Cell {cluster_id} not found in spike data")
            continue

        spike_times = ks_data.spike_times_by_cell[cell_position]
        aligned_spikes = align_spikes_to_events(spike_times, event_times_sorted, time_window)

        # Group spikes by opponent class so each class is plotted in one eventplot call
        for opponent in unique_opponents:
            class_rows = np.where(opponent_labels_sorted == opponent)[0]
            if len(class_rows) == 0:
                continue
            class_spike_lists = [aligned_spikes[r] for r in class_rows]
            ax.eventplot(class_spike_lists,
                         lineoffsets=class_rows,
                         linelengths=0.85,
                         linewidths=0.6,
                         colors=[color_map[opponent]])

        # Block separators between opponent classes
        for boundary in block_boundaries:
            ax.axhline(boundary - 0.5, color='black', linestyle='-',
                       alpha=0.3, linewidth=0.6)

        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlim(time_window)
        ax.set_ylim(-0.5, n_events - 0.5)
        ax.invert_yaxis()
        ax.set_title(f'Cell {cluster_id}\nAccuracy: {accuracy:.1%}', fontsize=10)

        if cell_idx == 0:
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], color=color_map[opp], lw=2,
                       label=f'Opponent {opp} '
                             f'(n={int(np.sum(opponent_labels_sorted == opp))})')
                for opp in unique_opponents
            ]
            ax.legend(handles=legend_handles, fontsize=8, loc='upper right')

    # Remove empty subplots
    for idx in range(n_top, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    fig.text(0.5, 0.02, 'Time from Event (s)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Event (sorted by opponent)', va='center',
             rotation=90, fontsize=12)

    plt.suptitle(f'Spike Rasters - Top {n_top} Cells\n' +
                 f'Behavior: {params["behavior_type"]}, ' +
                 f'Alignment: {params["alignment"]}, ' +
                 f'Window: {time_window[0]:.1f} to {time_window[1]:.1f}s',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved raster plot: {save_path}")

    return fig


# Command line interface

def main():
    """Command line interface for opponent identity decoding."""
    parser = argparse.ArgumentParser(description='Decode opponent identity from ephys activity')
    parser.add_argument('--animal_id', type=str, required=True, help='Animal identifier')
    parser.add_argument('--session_id', type=str, required=True, help='Session identifier')
    parser.add_argument('--behavior_type', type=str, default='F', help='Behavior type to analyze')
    parser.add_argument('--alignment', type=str, default='start', choices=['start', 'end'],
                      help='Event alignment point')
    parser.add_argument('--time_window', type=float, nargs=2, default=[-0.5, 1.0],
                      help='Time window around events (start end)')
    parser.add_argument('--time_bin_size', type=float, default=0.1, help='Time bin size in seconds')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--use_quality_cells', action='store_true', help='Filter by cell quality')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    parser.add_argument('--output_dir', type=str, help='Output directory for plots')

    args = parser.parse_args()

    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))

    try:
        from ingestion.kilosort_data_import import load_kilosort_data
        from ingestion.data_paths import get_kilosort_path
        from video.behavioral_events import load_behavioral_events
        from ingestion.data_paths import DataStorageManager
    except ImportError as e:
        print(f"Import error: {e}")
        return 1

    # Load data
    print(f"Loading data for animal {args.animal_id}, session {args.session_id}")

    try:
        # Load ephys data
        kilosort_path = get_kilosort_path(args.animal_id, args.session_id)[0]
        ks_data = load_kilosort_data(kilosort_path)

        print(f"Loaded {len(ks_data.ks_ids)} ephys clusters")

        # Load behavioral data
        data_manager = DataStorageManager(args.animal_id, args.session_id)
        behavior_data = load_behavioral_events(
            data_manager.get_behavioral_event_files(),
            session_id=data_manager.session_id,
        )
        print(f"Loaded {len(behavior_data.events_data)} behavioral events")

    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Run decoding analysis
    print("Running opponent identity decoding analysis...")

    results = decode_opponent_identity_population(
        ks_data=ks_data,
        behavior_data=behavior_data,
        animal_of_interest=args.animal_id,
        behavior_type=args.behavior_type,
        use_quality_cells=args.use_quality_cells,
        alignment=args.alignment,
        time_window=tuple(args.time_window),
        time_bin_size=args.time_bin_size,
        cv_folds=args.cv_folds
    )

    if results['status'] != 'success':
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return 1

    # Print summary
    print(f"\n=== OPPONENT IDENTITY DECODING RESULTS ===")
    print(f"Successful cells: {results['n_successful_cells']}/{results['n_total_cells']}")
    print(f"Population accuracy: {results['population_accuracy_mean']:.1%} ± {results['population_accuracy_std']:.1%}")
    print(f"Best cell accuracy: {results['best_cell_accuracy']:.1%} (Cell ID: {results['best_cell_id']})")

    # Generate plots
    if args.save_plots:
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        output_dir.mkdir(exist_ok=True)

        # Save summary plot
        summary_path = output_dir / f"opponent_decoding_summary_{args.animal_id}_{args.session_id}_{args.behavior_type}.png"
        plot_decoding_summary(results, save_path=summary_path)

        # Save detailed plots
        dist_path = output_dir / f"opponent_decoding_distribution_{args.animal_id}_{args.session_id}_{args.behavior_type}.png"
        plot_decoding_accuracy_distribution(results, save_path=dist_path)

        best_path = output_dir / f"opponent_decoding_best_cells_{args.animal_id}_{args.session_id}_{args.behavior_type}.png"
        plot_best_cells_decoding(results, save_path=best_path)

    else:
        # Show plots
        plot_decoding_summary(results)
        plt.show()

    print("Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
