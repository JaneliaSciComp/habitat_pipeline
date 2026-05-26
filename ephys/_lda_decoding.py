"""
Shared LDA decoding core for the per-cell and population analyses in
``ephys/decode_opponent_identity.py`` and ``ephys/decode_event_outcome.py``.

This module is label-agnostic: it does not know whether labels mean opponent
identity, win/loss, or anything else. Callers extract labels however they want
(via ``BehavioralEventsData`` methods, for example) and hand a label vector
plus event times to the helpers here.

Public surface:

- ``align_spikes_to_events(spike_times, event_times, time_window)``
- ``extract_firing_rate_features(aligned_spikes, time_window, time_bin_size)``
- ``single_cell_lda_decode(spike_times, event_times, labels, ...)``
- ``run_population_per_cell_decode(spike_times_list, cluster_ids, event_times, labels, ...)``
- ``run_time_resolved_population_decode(spike_times_list, cluster_ids, event_times, labels, ...)``

Callers select the cells they want to decode (e.g. via
``KilosortData.get_filtered_cells_spike_times``) and pass the resulting
parallel ``(cluster_ids, spike_times_list)`` arrays in. This module does
not know about ``KilosortData``.

The two outer wrappers in the existing modules pass labels through and
preserve their public result-dict schemas (with label-specific keys like
``unique_opponents`` / ``unique_outcomes``).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from ephys._rate_tensor import event_aligned_rates
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Spike alignment & firing-rate feature extraction
# ---------------------------------------------------------------------------

def align_spikes_to_events(spike_times: np.ndarray,
                           event_times: np.ndarray,
                           time_window: Tuple[float, float] = (-1.0, 2.0)) -> List[np.ndarray]:
    """Return per-event arrays of spike times relative to each event."""
    aligned_spikes = []
    for event_time in event_times:
        start_time = event_time + time_window[0]
        end_time = event_time + time_window[1]
        event_spikes = spike_times[(spike_times >= start_time) &
                                   (spike_times <= end_time)]
        aligned_spikes.append(event_spikes - event_time)
    return aligned_spikes


def extract_firing_rate_features(aligned_spikes: List[np.ndarray],
                                 time_window: Tuple[float, float] = (-1.0, 2.0),
                                 time_bin_size: float = 0.5) -> np.ndarray:
    """Return a (n_events, n_time_bins) firing-rate feature matrix."""
    bin_edges = np.arange(time_window[0], time_window[1] + time_bin_size, time_bin_size)
    n_bins = len(bin_edges) - 1
    firing_rates = np.zeros((len(aligned_spikes), n_bins))
    for i, spikes in enumerate(aligned_spikes):
        if len(spikes) > 0:
            counts, _ = np.histogram(spikes, bins=bin_edges)
            firing_rates[i] = counts / time_bin_size
    return firing_rates


# ---------------------------------------------------------------------------
# Single-cell decoder (label-agnostic)
# ---------------------------------------------------------------------------

def single_cell_lda_decode(spike_times: np.ndarray,
                           event_times: np.ndarray,
                           labels: np.ndarray,
                           time_window: Tuple[float, float] = (-1.0, 2.0),
                           time_bin_size: float = 0.5,
                           cv_folds=5,
                           min_events_per_class: int = 5) -> Dict:
    """Cross-validated LDA decode from one cell's binned firing rates.

    Returns the standard cell-result dict with keys: ``accuracy``,
    ``accuracy_std``, ``n_events``, ``n_classes``, ``class_counts``,
    ``confusion_matrix``, ``cv_scores``, ``status``.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
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

        features_scaled = StandardScaler().fit_transform(features)

        if cv_folds == 'loo' or cv_folds == -1:
            cv = LeaveOneOut()
        else:
            cv = StratifiedKFold(n_splits=min(int(cv_folds), int(np.min(counts))),
                                 shuffle=True, random_state=42)

        lda = LinearDiscriminantAnalysis()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(lda, features_scaled, labels,
                                        cv=cv, scoring='accuracy')

        lda.fit(features_scaled, labels)
        predictions = lda.predict(features_scaled)
        conf_matrix = confusion_matrix(labels, predictions, labels=unique_labels)

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
# Per-cell population decoder (label-agnostic)
# ---------------------------------------------------------------------------

def run_population_per_cell_decode(spike_times_list: Sequence[np.ndarray],
                                   cluster_ids: Sequence,
                                   event_times: np.ndarray,
                                   labels: np.ndarray,
                                   time_window: Tuple[float, float] = (-1.0, 2.0),
                                   time_bin_size: float = 0.5,
                                   cv_folds=5,
                                   min_events_per_class: int = 5,
                                   progress_every: int = 50
                                   ) -> Tuple[Dict, List, List[float]]:
    """Loop over selected cells running ``single_cell_lda_decode`` on each.

    ``spike_times_list`` and ``cluster_ids`` must be parallel arrays:
    ``spike_times_list[i]`` is the spike-time array for ``cluster_ids[i]``.

    Returns:
        cell_results           : ``{cluster_id: result_dict}`` for every selected cell.
        successful_cluster_ids : cluster IDs whose decode succeeded with a finite accuracy.
        accuracies             : accuracies for those successful cells (parallel to the ids).
    """
    cell_results: Dict = {}
    successful_count = 0
    n_total = len(spike_times_list)

    for i, (cluster_id, spike_times) in enumerate(zip(cluster_ids, spike_times_list)):
        if progress_every and i % progress_every == 0:
            print(f"Processing cell {i + 1}/{n_total} (cluster {cluster_id})")

        result = single_cell_lda_decode(
            spike_times=spike_times,
            event_times=event_times,
            labels=labels,
            time_window=time_window,
            time_bin_size=time_bin_size,
            cv_folds=cv_folds,
            min_events_per_class=min_events_per_class,
        )
        cell_results[cluster_id] = result
        if result['status'] == 'success':
            successful_count += 1

    print(f"Successfully decoded {successful_count}/{n_total} cells")

    accuracies: List[float] = []
    successful_cluster_ids: List = []
    for cluster_id, result in cell_results.items():
        if result['status'] == 'success' and not np.isnan(result['accuracy']):
            accuracies.append(result['accuracy'])
            successful_cluster_ids.append(cluster_id)

    return cell_results, successful_cluster_ids, accuracies


# ---------------------------------------------------------------------------
# Time-resolved population decoder (label-agnostic)
# ---------------------------------------------------------------------------

def _build_rates_tensor(spike_times_list: Sequence[np.ndarray],
                        event_times: np.ndarray,
                        time_window: Tuple[float, float],
                        bin_starts: np.ndarray,
                        bin_ends: np.ndarray,
                        time_bin_size: float) -> np.ndarray:
    """Return a (n_cells, n_events, n_bins) firing-rate tensor."""
    rates = event_aligned_rates(
        spike_times_list, event_times, time_window,
        bin_starts, bin_ends, time_bin_size,
    )
    return np.transpose(rates, (1, 0, 2))


def run_time_resolved_population_decode(spike_times_list: Sequence[np.ndarray],
                                        cluster_ids: Sequence,
                                        event_times: np.ndarray,
                                        labels: np.ndarray,
                                        time_window: Tuple[float, float] = (-1.0, 2.0),
                                        time_bin_size: float = 0.5,
                                        time_bin_step: Optional[float] = None,
                                        cv_folds: int = 5,
                                        n_shuffles: int = 0) -> Dict:
    """Population LDA per time bin: one classifier per bin across all cells.

    Returns a dict with keys: ``cluster_ids``, ``accuracy_by_bin``,
    ``accuracy_sem_by_bin``, ``bin_centers``, ``shuffle_null``, ``chance_level``,
    ``unique_classes``, ``best_bin_index``, ``best_bin_center``,
    ``best_bin_accuracy``, ``best_bin_confusion_matrix``, ``n_cells``,
    ``n_events``, ``status``, ``time_bin_step`` (the resolved step).

    The caller is responsible for adding ``parameters`` and any
    label-specific aliases (e.g. renaming ``unique_classes`` →
    ``unique_opponents`` / ``unique_outcomes``).
    """
    step = float(time_bin_step) if time_bin_step is not None else float(time_bin_size)
    if step <= 0 or time_bin_size <= 0:
        return {'error': 'time_bin_size and time_bin_step must be positive', 'status': 'failed'}
    bin_starts = np.arange(time_window[0], time_window[1] - time_bin_size + 1e-9, step)
    if len(bin_starts) == 0:
        return {'error': 'time_window is shorter than time_bin_size', 'status': 'failed'}
    bin_centers = bin_starts + time_bin_size / 2
    bin_ends = bin_starts + time_bin_size
    n_bins = len(bin_starts)

    rates = _build_rates_tensor(spike_times_list, event_times,
                                time_window, bin_starts, bin_ends, time_bin_size)
    n_cells, n_events, _ = rates.shape

    unique_labels, class_counts = np.unique(labels, return_counts=True)
    n_splits = min(int(cv_folds), int(np.min(class_counts)))
    if n_splits < 2:
        return {'error': f'Not enough events per class for {cv_folds}-fold CV', 'status': 'failed'}

    def _population_bin_accuracy(labels_vec):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        mean = np.full(n_bins, np.nan, dtype=np.float32)
        sem = np.full(n_bins, np.nan, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for bi in range(n_bins):
                X = rates[:, :, bi].T
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

    accuracy_by_bin, accuracy_sem_by_bin = _population_bin_accuracy(labels)

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
                    X_best_s, labels, cv=cv,
                )
            best_bin_confusion_matrix = confusion_matrix(
                labels, preds, labels=unique_labels)

    shuffle_null = None
    if n_shuffles > 0:
        rng = np.random.default_rng(0)
        shuffle_null = np.full((n_shuffles, n_bins), np.nan, dtype=np.float32)
        for s in range(n_shuffles):
            perm_labels = labels[rng.permutation(n_events)]
            shuf_mean, _ = _population_bin_accuracy(perm_labels)
            shuffle_null[s, :] = shuf_mean

    return {
        'cluster_ids': list(cluster_ids),
        'accuracy_by_bin': accuracy_by_bin,
        'accuracy_sem_by_bin': accuracy_sem_by_bin,
        'bin_centers': bin_centers,
        'shuffle_null': shuffle_null,
        'chance_level': 1.0 / len(unique_labels),
        'unique_classes': unique_labels,
        'best_bin_index': best_bin_index,
        'best_bin_center': float(bin_centers[best_bin_index]) if best_bin_index is not None else None,
        'best_bin_accuracy': float(accuracy_by_bin[best_bin_index]) if best_bin_index is not None else None,
        'best_bin_confusion_matrix': best_bin_confusion_matrix,
        'n_cells': n_cells,
        'n_events': n_events,
        'time_bin_step': step,
        'status': 'success',
    }
