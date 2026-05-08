#!/usr/bin/env python3
"""
Aggressive-Event Outcome Decoding from Single-Cell and Population Ephys Activity

Parallels ``ephys/decode_opponent_identity.py`` but the binary classification
target is whether ``animal_of_interest`` was the ``'winner'`` or the ``'loser'``
of each event. Labels are produced by
``BehavioralEventsData.extract_outcome_labels``.

The module exposes:

- ``decode_event_outcome_single_cell`` — per-cell cross-validated LDA.
- ``decode_event_outcome_population`` — per-cell decoding across the population,
  yielding a distribution of accuracies (one per cell).
- ``decode_event_outcome_time_resolved`` — population LDA with one classifier
  per time bin (multi-cell firing-rate vector → outcome).
- A set of plotting functions mirroring the opponent-identity module.

Spike-alignment and firing-rate utilities are imported from
``ephys.decode_opponent_identity``. The LDA/CV body is duplicated to keep the
two analyses textually independent; refactoring into a shared helper is left
for a future change.

Usage:
    from ephys.decode_event_outcome import (
        decode_event_outcome_population,
        decode_event_outcome_time_resolved,
        plot_decoding_summary,
        plot_time_resolved_decoding,
    )

    results = decode_event_outcome_population(
        ks_data=ks_data,
        behavior_data=behavior_data,
        animal_of_interest='631',
        time_window=(-1.0, 2.0),
        time_bin_size=0.5,
        cv_folds=5,
    )
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler

from ephys.decode_opponent_identity import (
    align_spikes_to_events,
    extract_firing_rate_features,
)


# ---------------------------------------------------------------------------
# Single-cell decoder
# ---------------------------------------------------------------------------

def decode_event_outcome_single_cell(spike_times: np.ndarray,
                                     event_times: np.ndarray,
                                     outcome_labels: np.ndarray,
                                     alignment: str = 'start',
                                     time_window: Tuple[float, float] = (-1.0, 2.0),
                                     time_bin_size: float = 0.5,
                                     cv_folds: int = 5,
                                     min_events_per_class: int = 5) -> Dict:
    """
    Decode event outcome (winner vs loser) from a single cell using CV LDA.

    Returns a dict matching the structure used by the opponent-identity module
    (``accuracy``, ``confusion_matrix``, ``cv_scores``, ``status`` ...).
    """
    unique_labels, counts = np.unique(outcome_labels, return_counts=True)
    if len(unique_labels) < 2 or np.min(counts) < min_events_per_class:
        return {
            'accuracy': np.nan,
            'accuracy_std': np.nan,
            'n_events': len(event_times),
            'n_classes': len(unique_labels),
            'class_counts': dict(zip(unique_labels, counts)),
            'confusion_matrix': None,
            'cv_scores': None,
            'status': 'insufficient_data',
        }

    try:
        aligned_spikes = align_spikes_to_events(spike_times, event_times, time_window)
        features = extract_firing_rate_features(aligned_spikes, time_window, time_bin_size)

        if np.all(features == 0):
            return {
                'accuracy': np.nan,
                'accuracy_std': np.nan,
                'n_events': len(event_times),
                'n_classes': len(unique_labels),
                'class_counts': dict(zip(unique_labels, counts)),
                'confusion_matrix': None,
                'cv_scores': None,
                'status': 'no_spikes',
            }

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if cv_folds == 'loo' or cv_folds == -1:
            cv = LeaveOneOut()
        else:
            cv = StratifiedKFold(n_splits=min(cv_folds, int(np.min(counts))),
                                 shuffle=True, random_state=42)

        lda = LinearDiscriminantAnalysis()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(lda, features_scaled, outcome_labels,
                                        cv=cv, scoring='accuracy')

        lda.fit(features_scaled, outcome_labels)
        predictions = lda.predict(features_scaled)
        conf_matrix = confusion_matrix(outcome_labels, predictions, labels=unique_labels)

        return {
            'accuracy': float(np.mean(cv_scores)),
            'accuracy_std': float(np.std(cv_scores)),
            'n_events': len(event_times),
            'n_classes': len(unique_labels),
            'class_counts': dict(zip(unique_labels, counts)),
            'confusion_matrix': conf_matrix,
            'cv_scores': cv_scores,
            'status': 'success',
        }

    except Exception as e:
        return {
            'accuracy': np.nan,
            'accuracy_std': np.nan,
            'n_events': len(event_times),
            'n_classes': len(unique_labels) if 'unique_labels' in locals() else 0,
            'class_counts': dict(zip(unique_labels, counts)) if 'unique_labels' in locals() else {},
            'confusion_matrix': None,
            'cv_scores': None,
            'status': f'error: {str(e)}',
        }


# ---------------------------------------------------------------------------
# Population decoder (per-cell loop)
# ---------------------------------------------------------------------------

def decode_event_outcome_population(ks_data,
                                    behavior_data,
                                    animal_of_interest: str,
                                    behavior_type: Optional[str] = None,
                                    use_quality_cells: bool = True,
                                    quality_thresholds: Optional[Dict] = None,
                                    alignment: str = 'start',
                                    time_window: Tuple[float, float] = (-1.0, 2.0),
                                    time_bin_size: float = 0.5,
                                    cv_folds: int = 5,
                                    min_events_per_class: int = 5) -> Dict:
    """
    Decode event outcome (winner vs loser) cell-by-cell across the population.

    Parameters mirror ``decode_opponent_identity_population``. ``behavior_type``
    defaults to ``None``, in which case any event with both ``winner`` and
    ``loser`` populated is included.
    """
    try:
        event_start_times, event_end_times, outcome_labels = behavior_data.extract_outcome_labels(
            animal_of_interest, behavior_type, min_events_per_class
        )

        if len(event_start_times) == 0:
            scope = behavior_type if behavior_type is not None else 'aggressive'
            raise ValueError(
                f"No {scope} events with outcome labels found for {animal_of_interest}"
            )

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

    if use_quality_cells:
        if quality_thresholds is None:
            quality_thresholds = {
                'min_firing_rate': 0.5,
                'min_presence_ratio': 0.8,
                'max_cv_isi': 5.0,
            }
        filter_results = ks_data.filter_cells_by_firing_patterns(**quality_thresholds)
        selected_cell_indices = [i for i, cluster_id in enumerate(ks_data.ks_ids)
                                 if cluster_id in filter_results['passed_clusters']]
        selected_cluster_ids = filter_results['passed_clusters']
        print(f"Using {len(selected_cluster_ids)} quality-filtered cells")
    else:
        selected_cell_indices = list(range(len(ks_data.ks_ids)))
        selected_cluster_ids = list(ks_data.ks_ids)
        print(f"Using all {len(selected_cluster_ids)} cells")

    if len(selected_cluster_ids) == 0:
        print("No cells selected for analysis")
        return {'error': 'No cells selected', 'status': 'failed'}

    cell_results: Dict = {}
    successful_cells = 0

    for i, cell_idx in enumerate(selected_cell_indices):
        cluster_id = ks_data.ks_ids[cell_idx]
        spike_times = ks_data.spike_times_by_cell[cell_idx]

        if i % 50 == 0:
            print(f"Processing cell {i+1}/{len(selected_cell_indices)} (cluster {cluster_id})")

        result = decode_event_outcome_single_cell(
            spike_times=spike_times,
            event_times=event_times,
            outcome_labels=outcome_labels,
            alignment=alignment,
            time_window=time_window,
            time_bin_size=time_bin_size,
            cv_folds=cv_folds,
            min_events_per_class=min_events_per_class,
        )

        cell_results[cluster_id] = result
        if result['status'] == 'success':
            successful_cells += 1

    print(f"Successfully decoded {successful_cells}/{len(selected_cluster_ids)} cells")

    accuracies = []
    successful_cluster_ids = []
    for cluster_id, result in cell_results.items():
        if result['status'] == 'success' and not np.isnan(result['accuracy']):
            accuracies.append(result['accuracy'])
            successful_cluster_ids.append(cluster_id)

    population_results = {
        'cell_results': cell_results,
        'successful_cells': successful_cluster_ids,
        'n_successful_cells': len(successful_cluster_ids),
        'n_total_cells': len(selected_cluster_ids),
        'success_rate': len(successful_cluster_ids) / len(selected_cluster_ids) if len(selected_cluster_ids) > 0 else 0,
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
        },
        'behavioral_summary': {
            'n_events': len(event_times),
            'unique_outcomes': np.unique(outcome_labels),
            'outcome_counts': dict(zip(*np.unique(outcome_labels, return_counts=True))),
        },
        'status': 'success',
    }

    return population_results


# ---------------------------------------------------------------------------
# Time-resolved population decoder
# ---------------------------------------------------------------------------

def decode_event_outcome_time_resolved(ks_data,
                                       behavior_data,
                                       animal_of_interest: str,
                                       behavior_type: Optional[str] = None,
                                       use_quality_cells: bool = True,
                                       quality_thresholds: Optional[Dict] = None,
                                       alignment: str = 'start',
                                       time_window: Tuple[float, float] = (-1.0, 2.0),
                                       time_bin_size: float = 0.5,
                                       time_bin_step: Optional[float] = None,
                                       cv_folds: int = 5,
                                       min_events_per_class: int = 5,
                                       n_shuffles: int = 0) -> Dict:
    """Population (multi-cell) LDA decoding of event outcome per time bin.

    Mirrors ``decode_opponent_identity_time_resolved``. Labels are
    ``'winner'`` / ``'loser'`` from ``extract_outcome_labels``.
    """
    try:
        event_start_times, event_end_times, outcome_labels = behavior_data.extract_outcome_labels(
            animal_of_interest, behavior_type, min_events_per_class
        )
        if len(event_start_times) == 0:
            scope = behavior_type if behavior_type is not None else 'aggressive'
            raise ValueError(
                f"No {scope} events with outcome labels found for {animal_of_interest}"
            )
    except Exception as e:
        return {'error': str(e), 'status': 'failed'}

    if alignment == 'start':
        event_times = event_start_times
    elif alignment == 'end':
        event_times = event_end_times
    else:
        raise ValueError("alignment must be 'start' or 'end'")

    if use_quality_cells:
        if quality_thresholds is None:
            quality_thresholds = {
                'min_firing_rate': 0.5,
                'min_presence_ratio': 0.8,
                'max_cv_isi': 5.0,
            }
        filter_results = ks_data.filter_cells_by_firing_patterns(**quality_thresholds)
        passed = set(filter_results['passed_clusters'])
        selected_cell_indices = [i for i, cid in enumerate(ks_data.ks_ids) if cid in passed]
        selected_cluster_ids = [cid for cid in ks_data.ks_ids if cid in passed]
    else:
        selected_cell_indices = list(range(len(ks_data.ks_ids)))
        selected_cluster_ids = list(ks_data.ks_ids)

    if len(selected_cluster_ids) == 0:
        return {'error': 'No cells selected', 'status': 'failed'}

    step = float(time_bin_step) if time_bin_step is not None else float(time_bin_size)
    if step <= 0 or time_bin_size <= 0:
        return {'error': 'time_bin_size and time_bin_step must be positive', 'status': 'failed'}
    bin_starts = np.arange(time_window[0], time_window[1] - time_bin_size + 1e-9, step)
    if len(bin_starts) == 0:
        return {'error': 'time_window is shorter than time_bin_size', 'status': 'failed'}
    bin_centers = bin_starts + time_bin_size / 2
    n_bins = len(bin_starts)

    n_cells = len(selected_cell_indices)
    n_events = len(event_times)
    rates = np.zeros((n_cells, n_events, n_bins), dtype=np.float32)
    bin_ends = bin_starts + time_bin_size
    for ci, cell_idx in enumerate(selected_cell_indices):
        spike_times = ks_data.spike_times_by_cell[cell_idx]
        for ei, et in enumerate(event_times):
            in_window = (spike_times >= et + time_window[0]) & (spike_times < et + time_window[1])
            rel = np.sort(spike_times[in_window] - et)
            if len(rel) > 0:
                lo = np.searchsorted(rel, bin_starts, side='left')
                hi = np.searchsorted(rel, bin_ends, side='left')
                rates[ci, ei, :] = (hi - lo) / time_bin_size

    unique_labels, class_counts = np.unique(outcome_labels, return_counts=True)
    n_splits = min(cv_folds, int(np.min(class_counts)))
    if n_splits < 2:
        return {'error': f'Not enough events per class for {cv_folds}-fold CV', 'status': 'failed'}

    def _population_bin_accuracy(labels_vec):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        mean = np.full(n_bins, np.nan, dtype=np.float32)
        sem = np.full(n_bins, np.nan, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for bi in range(n_bins):
                X = rates[:, :, bi].T  # (n_events, n_cells)
                X_s = np.nan_to_num(StandardScaler().fit_transform(X), nan=0.0)
                if not np.any(X_s.std(axis=0) > 0):
                    continue
                scores = cross_val_score(
                    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                    X_s, labels_vec, cv=cv, scoring='accuracy',
                )
                mean[bi] = float(np.mean(scores))
                sem[bi] = float(np.std(scores) / np.sqrt(len(scores)))
        return mean, sem

    accuracy_by_bin, accuracy_sem_by_bin = _population_bin_accuracy(outcome_labels)

    best_bin_index = None
    best_bin_confusion_matrix = None
    if np.any(np.isfinite(accuracy_by_bin)):
        best_bin_index = int(np.nanargmax(accuracy_by_bin))
        X_best = rates[:, :, best_bin_index].T
        X_best_s = np.nan_to_num(StandardScaler().fit_transform(X_best), nan=0.0)
        if np.any(X_best_s.std(axis=0) > 0):
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = cross_val_predict(
                    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                    X_best_s, outcome_labels, cv=cv,
                )
            best_bin_confusion_matrix = confusion_matrix(
                outcome_labels, preds, labels=unique_labels)

    shuffle_null = None
    if n_shuffles > 0:
        rng = np.random.default_rng(0)
        shuffle_null = np.full((n_shuffles, n_bins), np.nan, dtype=np.float32)
        for s in range(n_shuffles):
            perm_labels = outcome_labels[rng.permutation(n_events)]
            shuf_mean, _ = _population_bin_accuracy(perm_labels)
            shuffle_null[s, :] = shuf_mean

    return {
        'cluster_ids': selected_cluster_ids,
        'accuracy_by_bin': accuracy_by_bin,
        'accuracy_sem_by_bin': accuracy_sem_by_bin,
        'bin_centers': bin_centers,
        'shuffle_null': shuffle_null,
        'chance_level': 1.0 / len(unique_labels),
        'unique_outcomes': unique_labels,
        'best_bin_index': best_bin_index,
        'best_bin_center': float(bin_centers[best_bin_index]) if best_bin_index is not None else None,
        'best_bin_accuracy': float(accuracy_by_bin[best_bin_index]) if best_bin_index is not None else None,
        'best_bin_confusion_matrix': best_bin_confusion_matrix,
        'parameters': {
            'animal_of_interest': animal_of_interest,
            'behavior_type': behavior_type,
            'use_quality_cells': use_quality_cells,
            'alignment': alignment,
            'time_window': time_window,
            'time_bin_size': time_bin_size,
            'time_bin_step': step,
            'cv_folds': cv_folds,
            'min_events_per_class': min_events_per_class,
            'n_shuffles': n_shuffles,
        },
        'n_cells': n_cells,
        'n_events': n_events,
        'status': 'success',
    }


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def plot_time_resolved_decoding(results: Dict,
                                figsize: Tuple[int, int] = (13, 5)) -> Optional[plt.Figure]:
    """Plot population-LDA accuracy curve as a function of time around the event,
    plus the best-bin confusion matrix on the right.

    Expects the dict returned by ``decode_event_outcome_time_resolved``.
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
    btype = params.get('behavior_type', None)
    btype_str = btype if btype is not None else 'aggressive (any type)'
    ax.set_xlabel('Time from event (s)')
    ax.set_ylabel('Decoding accuracy')
    ax.set_title(f'Time-resolved population decoding of event outcome\n'
                 f'behavior={btype_str} · {results["n_events"]} events')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    cm = results.get('best_bin_confusion_matrix')
    if cm is not None and best_idx is not None:
        unique_outcomes = results.get('unique_outcomes', [])
        im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(unique_outcomes))
        ax_cm.set_xticks(tick_marks)
        ax_cm.set_yticks(tick_marks)
        ax_cm.set_xticklabels([f'{o}' for o in unique_outcomes])
        ax_cm.set_yticklabels([f'{o}' for o in unique_outcomes])
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
                                        save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot distribution of per-cell outcome-decoding accuracies."""
    if results['status'] != 'success' or results['n_successful_cells'] == 0:
        print("No successful results to plot")
        return None

    accuracies = []
    for cluster_id in results['successful_cells']:
        acc = results['cell_results'][cluster_id]['accuracy']
        if not np.isnan(acc):
            accuracies.append(acc)

    if len(accuracies) == 0:
        print("No valid accuracies to plot")
        return None

    n_classes = len(results['behavioral_summary']['unique_outcomes'])
    chance_level = 1.0 / n_classes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.axvline(chance_level, color='red', linestyle='--', alpha=0.7,
                label=f'Chance ({chance_level:.1%})')
    ax1.axvline(np.mean(accuracies), color='orange', linestyle='-', linewidth=2,
                label=f'Mean ({np.mean(accuracies):.1%})')
    ax1.set_xlabel('Decoding Accuracy')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Distribution of Event-Outcome Decoding Accuracies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.boxplot([accuracies], labels=['All Cells'])
    ax2.axhline(chance_level, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Decoding Accuracy')
    ax2.set_title('Accuracy Distribution Summary')
    ax2.grid(True, alpha=0.3)

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

    btype = results['parameters']['behavior_type']
    btype_str = btype if btype is not None else 'aggressive (any type)'
    plt.suptitle(f"Event Outcome Decoding Results\n"
                 f"Behavior: {btype_str}, "
                 f"Alignment: {results['parameters']['alignment']}",
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")

    return fig


def plot_best_cells_decoding(results: Dict,
                             n_top_cells: int = 10,
                             figsize: Tuple[int, int] = (12, 6),
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot decoding performance of top-performing cells, with the best cell's
    confusion matrix on the right."""
    if results['status'] != 'success' or results['n_successful_cells'] == 0:
        print("No successful results to plot")
        return None

    cell_accs = [(cluster_id, results['cell_results'][cluster_id]['accuracy'])
                 for cluster_id in results['successful_cells']
                 if not np.isnan(results['cell_results'][cluster_id]['accuracy'])]

    if len(cell_accs) == 0:
        print("No valid accuracies to plot")
        return None

    n_classes = len(results['behavioral_summary']['unique_outcomes'])
    chance_level = 1.0 / n_classes

    cell_accs_sorted = sorted(cell_accs, key=lambda x: x[1], reverse=True)
    top_cells = cell_accs_sorted[:min(n_top_cells, len(cell_accs_sorted))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

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

    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{acc:.1%}', va='center')

    best_cell_id = top_cells[0][0]
    best_result = results['cell_results'][best_cell_id]

    if best_result['confusion_matrix'] is not None:
        cm = best_result['confusion_matrix']
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title(f'Confusion Matrix - Best Cell (ID: {best_cell_id})')

        unique_outcomes = results['behavioral_summary']['unique_outcomes']
        tick_marks = np.arange(len(unique_outcomes))
        ax2.set_xticks(tick_marks)
        ax2.set_yticks(tick_marks)
        ax2.set_xticklabels([f'{o}' for o in unique_outcomes])
        ax2.set_yticklabels([f'{o}' for o in unique_outcomes])
        ax2.set_xlabel('Predicted Outcome')
        ax2.set_ylabel('True Outcome')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.colorbar(im, ax=ax2)

    btype = results['parameters']['behavior_type']
    btype_str = btype if btype is not None else 'aggressive (any type)'
    plt.suptitle(f"Top Performing Cells - Event Outcome Decoding\n"
                 f"Behavior: {btype_str}, "
                 f"Best accuracy: {top_cells[0][1]:.1%}", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")

    return fig


def plot_decoding_summary(results: Dict,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Comprehensive 6-panel summary plot of outcome-decoding results."""
    if results['status'] != 'success':
        print("No successful results to plot")
        return None

    n_classes = len(results['behavioral_summary']['unique_outcomes'])
    chance_level = 1.0 / n_classes

    fig = plt.figure(figsize=figsize)
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

        unique_outcomes = results['behavioral_summary']['unique_outcomes']
        tick_marks = np.arange(len(unique_outcomes))
        ax2.set_xticks(tick_marks)
        ax2.set_yticks(tick_marks)
        ax2.set_xticklabels([f'{o}' for o in unique_outcomes])
        ax2.set_yticklabels([f'{o}' for o in unique_outcomes])
        ax2.set_xlabel('Predicted Outcome')
        ax2.set_ylabel('True Outcome')

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

    # 3. Behavioral event summary (winner / loser counts)
    ax3 = fig.add_subplot(gs[0, 2])
    outcome_counts = results['behavioral_summary']['outcome_counts']
    outcomes = list(outcome_counts.keys())
    counts = list(outcome_counts.values())

    bars = ax3.bar([f'{o}' for o in outcomes], counts, color='lightblue')
    ax3.set_xlabel('Outcome')
    ax3.set_ylabel('Number of Events')
    ax3.set_title(f"Behavioral Events\n(Total: {results['behavioral_summary']['n_events']})")

    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom')

    # 4. Top cells performance
    ax4 = fig.add_subplot(gs[1, :2])
    if accuracies:
        cell_accs_sorted = sorted(cell_accs, key=lambda x: x[1], reverse=True)
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

        if len(bars) > 0:
            bars[0].set_color('red')
            bars[0].set_alpha(0.8)

    # 5. Parameter summary text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    params = results['parameters']
    btype = params['behavior_type']
    btype_str = btype if btype is not None else 'aggressive (any type)'
    param_text = f"""
    ANALYSIS PARAMETERS:

    Behavior Type: {btype_str}
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

    plt.suptitle("Event Outcome Decoding - Complete Analysis Summary",
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
                                save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot peri-event firing rates for top cells, split by outcome class."""
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
        event_start_times, event_end_times, outcome_labels = behavior_data.extract_outcome_labels(
            animal_of_interest=params.get('animal_of_interest', ''),
            behavior_type=params.get('behavior_type'),
            min_events_per_class=params.get('min_events_per_class', 5),
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

    unique_outcomes = np.unique(outcome_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_outcomes)))

    bin_edges = np.arange(time_window[0], time_window[1] + time_bin_size, time_bin_size)
    bin_centers = bin_edges[:-1] + time_bin_size / 2

    n_top = len(top_cells)
    n_cols = min(3, n_top)
    n_rows = int(np.ceil(n_top / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)

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

        for opp_idx, outcome in enumerate(unique_outcomes):
            class_mask = outcome_labels == outcome
            class_event_times = event_times[class_mask]
            if len(class_event_times) == 0:
                continue

            aligned_spikes = align_spikes_to_events(spike_times, class_event_times, time_window)

            trial_firing_rates = []
            for trial_spikes in aligned_spikes:
                if len(trial_spikes) > 0:
                    counts, _ = np.histogram(trial_spikes, bins=bin_edges)
                    firing_rate = counts / time_bin_size
                else:
                    firing_rate = np.zeros(len(bin_centers))
                trial_firing_rates.append(firing_rate)

            trial_firing_rates = np.array(trial_firing_rates)
            mean_firing_rate = np.mean(trial_firing_rates, axis=0)
            sem_firing_rate = np.std(trial_firing_rates, axis=0) / np.sqrt(len(trial_firing_rates))

            color = colors[opp_idx]
            ax.plot(bin_centers, mean_firing_rate, color=color, linewidth=2,
                    label=f'{outcome} (n={len(class_event_times)})')
            ax.fill_between(bin_centers,
                            mean_firing_rate - sem_firing_rate,
                            mean_firing_rate + sem_firing_rate,
                            color=color, alpha=0.3)

        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'Cell {cluster_id}\nAccuracy: {accuracy:.1%}', fontsize=10)
        ax.grid(True, alpha=0.3)

        if cell_idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    for idx in range(n_top, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    fig.text(0.5, 0.02, 'Time from Event (s)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Firing Rate (Hz)', va='center', rotation=90, fontsize=12)

    btype = params['behavior_type']
    btype_str = btype if btype is not None else 'aggressive (any type)'
    plt.suptitle(f'Peri-Event Firing Rates - Top {len(top_cells)} Cells\n'
                 f'Behavior: {btype_str}, '
                 f'Alignment: {params["alignment"]}, '
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
                           save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Plot spike rasters for top cells, sorted by outcome (winner/loser)."""
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
        event_start_times, event_end_times, outcome_labels = behavior_data.extract_outcome_labels(
            animal_of_interest=params.get('animal_of_interest', ''),
            behavior_type=params.get('behavior_type'),
            min_events_per_class=params.get('min_events_per_class', 5),
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

    unique_outcomes = np.unique(outcome_labels)
    sort_idx = np.argsort(outcome_labels, kind='stable')
    event_times_sorted = event_times[sort_idx]
    outcome_labels_sorted = outcome_labels[sort_idx]

    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_outcomes)))
    color_map = {opp: colors[i] for i, opp in enumerate(unique_outcomes)}

    block_boundaries = []
    for i in range(1, len(outcome_labels_sorted)):
        if outcome_labels_sorted[i] != outcome_labels_sorted[i - 1]:
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

        for outcome in unique_outcomes:
            class_rows = np.where(outcome_labels_sorted == outcome)[0]
            if len(class_rows) == 0:
                continue
            class_spike_lists = [aligned_spikes[r] for r in class_rows]
            ax.eventplot(class_spike_lists,
                         lineoffsets=class_rows,
                         linelengths=0.85,
                         linewidths=0.6,
                         colors=[color_map[outcome]])

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
                       label=f'{opp} '
                             f'(n={int(np.sum(outcome_labels_sorted == opp))})')
                for opp in unique_outcomes
            ]
            ax.legend(handles=legend_handles, fontsize=8, loc='upper right')

    for idx in range(n_top, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    fig.text(0.5, 0.02, 'Time from Event (s)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Event (sorted by outcome)', va='center',
             rotation=90, fontsize=12)

    btype = params['behavior_type']
    btype_str = btype if btype is not None else 'aggressive (any type)'
    plt.suptitle(f'Spike Rasters - Top {n_top} Cells\n'
                 f'Behavior: {btype_str}, '
                 f'Alignment: {params["alignment"]}, '
                 f'Window: {time_window[0]:.1f} to {time_window[1]:.1f}s',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved raster plot: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def main():
    """Command line interface for event outcome decoding."""
    parser = argparse.ArgumentParser(description='Decode event outcome (winner/loser) from ephys activity')
    parser.add_argument('--animal_id', type=str, required=True, help='Animal identifier')
    parser.add_argument('--session_id', type=str, required=True, help='Session identifier')
    parser.add_argument('--behavior_type', type=str, default=None,
                        help="Behavior type to analyze (default: any with winner/loser)")
    parser.add_argument('--alignment', type=str, default='start', choices=['start', 'end'],
                        help='Event alignment point')
    parser.add_argument('--time_window', type=float, nargs=2, default=[-1.0, 2.0],
                        help='Time window around events (start end)')
    parser.add_argument('--time_bin_size', type=float, default=0.5, help='Time bin size in seconds')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--min_events_per_class', type=int, default=5,
                        help='Minimum events per outcome class')
    parser.add_argument('--use_quality_cells', action='store_true', help='Filter by cell quality')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    parser.add_argument('--output_dir', type=str, help='Output directory for plots')

    args = parser.parse_args()

    sys.path.append(str(Path(__file__).parent.parent))

    try:
        from ingestion.kilosort_data_import import load_kilosort_data
        from ingestion.data_paths import DataStorageManager
        from video.behavioral_events import load_behavioral_events
    except ImportError as e:
        print(f"Import error: {e}")
        return 1

    print(f"Loading data for animal {args.animal_id}, session {args.session_id}")

    try:
        data_manager = DataStorageManager(args.animal_id, args.session_id, auto_load=True)
        ks_data = load_kilosort_data(data_manager)
        print(f"Loaded {len(ks_data.ks_ids)} ephys clusters")

        behavior_data = load_behavioral_events(
            data_manager.get_behavioral_event_files(),
            session_id=data_manager.session_id,
        )
        print(f"Loaded {len(behavior_data.events_data)} behavioral events")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    print("Running event outcome decoding analysis...")
    results = decode_event_outcome_population(
        ks_data=ks_data,
        behavior_data=behavior_data,
        animal_of_interest=args.animal_id,
        behavior_type=args.behavior_type,
        use_quality_cells=args.use_quality_cells,
        alignment=args.alignment,
        time_window=tuple(args.time_window),
        time_bin_size=args.time_bin_size,
        cv_folds=args.cv_folds,
        min_events_per_class=args.min_events_per_class,
    )

    if results['status'] != 'success':
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return 1

    print("\n=== EVENT OUTCOME DECODING RESULTS ===")
    print(f"Successful cells: {results['n_successful_cells']}/{results['n_total_cells']}")
    print(f"Population accuracy: {results['population_accuracy_mean']:.1%} "
          f"± {results['population_accuracy_std']:.1%}")
    print(f"Best cell accuracy: {results['best_cell_accuracy']:.1%} "
          f"(Cell ID: {results['best_cell_id']})")

    if args.save_plots:
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        btag = args.behavior_type if args.behavior_type is not None else 'any'

        summary_path = output_dir / f"outcome_decoding_summary_{args.animal_id}_{args.session_id}_{btag}.png"
        plot_decoding_summary(results, save_path=summary_path)

        dist_path = output_dir / f"outcome_decoding_distribution_{args.animal_id}_{args.session_id}_{btag}.png"
        plot_decoding_accuracy_distribution(results, save_path=dist_path)

        best_path = output_dir / f"outcome_decoding_best_cells_{args.animal_id}_{args.session_id}_{btag}.png"
        plot_best_cells_decoding(results, save_path=best_path)
    else:
        plot_decoding_summary(results)
        plt.show()

    print("Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
