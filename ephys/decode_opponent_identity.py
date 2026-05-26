#!/usr/bin/env python3
"""
Opponent Identity Decoding from Single Cell Ephys Activity

Decodes which opponent animal was involved in each behavioral event from
single-cell ephys activity using cross-validated LDA. The shared core
(spike alignment, feature extraction, LDA/CV, cell selection) lives in
``ephys/_lda_decoding.py``; the shared plots live in
``ephys/decoding_plots.py``. This module is the opponent-flavored wrapper.

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
    single_cell_lda_decode,
)
from ingestion.kilosort_data_import import _DEFAULT_QUALITY_THRESHOLDS
from ephys.decoding_plots import (
    plot_best_cells_decoding,
    plot_decoding_accuracy_distribution,
    plot_decoding_summary,
    plot_time_resolved_decoding,
    plot_top_cells_firing_rates,
    plot_top_cells_rasters,
)


_CLASS_LABEL = 'Opponent'
_ANALYSIS_TITLE = 'Opponent Identity Decoding'


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

    Thin wrapper around ``single_cell_lda_decode``; applies an optional
    ``selected_opponents`` filter first. Result dict keys are unchanged.
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
                                        behavior_type: Optional[str] = None,
                                        use_quality_cells: bool = True,
                                        quality_thresholds: Optional[Dict] = None,
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

    if use_quality_cells:
        quality_thresholds = dict(_DEFAULT_QUALITY_THRESHOLDS) if quality_thresholds is None else quality_thresholds
        selected_cluster_ids, spike_times_list = ks_data.get_filtered_cells_spike_times(**quality_thresholds)
        print(f"Using {len(selected_cluster_ids)} quality-filtered cells")
    else:
        selected_cluster_ids = list(ks_data.ks_ids)
        spike_times_list = list(ks_data.spike_times_by_cell)
        print(f"Using all {len(selected_cluster_ids)} cells")

    if len(selected_cluster_ids) == 0:
        print("No cells selected for analysis")
        return {'error': 'No cells selected', 'status': 'failed'}

    cell_results, successful_cluster_ids, accuracies = run_population_per_cell_decode(
        spike_times_list=spike_times_list,
        cluster_ids=selected_cluster_ids,
        event_times=event_times,
        labels=opponent_labels,
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
        'event_times': event_times,
        'labels': opponent_labels,
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
            'class_label': _CLASS_LABEL,
            'analysis_title': _ANALYSIS_TITLE,
        },
        'behavioral_summary': {
            'n_events': len(event_times),
            'unique_classes': np.unique(opponent_labels),
            'class_counts': dict(zip(*np.unique(opponent_labels, return_counts=True))),
        },
        'status': 'success',
    }


def decode_opponent_identity_time_resolved(ks_data,
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

    if use_quality_cells:
        thresholds = dict(_DEFAULT_QUALITY_THRESHOLDS) if quality_thresholds is None else quality_thresholds
        selected_cluster_ids, spike_times_list = ks_data.get_filtered_cells_spike_times(**thresholds)
    else:
        selected_cluster_ids = list(ks_data.ks_ids)
        spike_times_list = list(ks_data.spike_times_by_cell)
    if len(selected_cluster_ids) == 0:
        return {'error': 'No cells selected', 'status': 'failed'}

    core = run_time_resolved_population_decode(
        spike_times_list=spike_times_list,
        cluster_ids=selected_cluster_ids,
        event_times=event_times,
        labels=opponent_labels,
        time_window=time_window,
        time_bin_size=time_bin_size,
        time_bin_step=time_bin_step,
        cv_folds=cv_folds,
        n_shuffles=n_shuffles,
    )
    if core.get('status') != 'success':
        return core

    core['event_times'] = event_times
    core['labels'] = opponent_labels
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
        'class_label': _CLASS_LABEL,
        'analysis_title': _ANALYSIS_TITLE,
    }
    return core


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

    sys.path.append(str(Path(__file__).parent.parent))

    try:
        from ingestion.kilosort_data_import import load_kilosort_data
        from ingestion.data_paths import get_kilosort_path, DataStorageManager
        from video.behavioral_events import load_behavioral_events
    except ImportError as e:
        print(f"Import error: {e}")
        return 1

    print(f"Loading data for animal {args.animal_id}, session {args.session_id}")

    try:
        kilosort_path = get_kilosort_path(args.animal_id, args.session_id)[0]
        ks_data = load_kilosort_data(kilosort_path)
        print(f"Loaded {len(ks_data.ks_ids)} ephys clusters")

        data_manager = DataStorageManager(args.animal_id, args.session_id)
        behavior_data = load_behavioral_events(
            data_manager.get_behavioral_event_files(),
            session_id=data_manager.session_id,
        )
        print(f"Loaded {len(behavior_data.events_data)} behavioral events")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

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
        cv_folds=args.cv_folds,
    )

    if results['status'] != 'success':
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return 1

    print(f"\n=== OPPONENT IDENTITY DECODING RESULTS ===")
    print(f"Successful cells: {results['n_successful_cells']}/{results['n_total_cells']}")
    print(f"Population accuracy: {results['population_accuracy_mean']:.1%} "
          f"± {results['population_accuracy_std']:.1%}")
    print(f"Best cell accuracy: {results['best_cell_accuracy']:.1%} "
          f"(Cell ID: {results['best_cell_id']})")

    if args.save_plots:
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        output_dir.mkdir(exist_ok=True)
        summary_path = output_dir / f"opponent_decoding_summary_{args.animal_id}_{args.session_id}_{args.behavior_type}.png"
        plot_decoding_summary(results, save_path=summary_path)
        dist_path = output_dir / f"opponent_decoding_distribution_{args.animal_id}_{args.session_id}_{args.behavior_type}.png"
        plot_decoding_accuracy_distribution(results, save_path=dist_path)
        best_path = output_dir / f"opponent_decoding_best_cells_{args.animal_id}_{args.session_id}_{args.behavior_type}.png"
        plot_best_cells_decoding(results, save_path=best_path)
    else:
        plot_decoding_summary(results)
        plt.show()

    print("Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
