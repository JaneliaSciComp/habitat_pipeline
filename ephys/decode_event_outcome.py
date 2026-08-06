#!/usr/bin/env python3
"""
Aggressive-Event Outcome Decoding from Single-Cell and Population Ephys Activity

Parallels ``ephys/decode_opponent_identity.py`` but the binary classification
target is whether ``animal_of_interest`` was the ``'winner'`` or the ``'loser'``
of each event. Labels are produced by
``BehavioralEventsData.extract_outcome_labels``.

Shared LDA/CV/feature/cell-select machinery lives in
``ephys/_lda_decoding.py``; shared plots live in ``ephys/decoding_plots.py``.

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ephys._lda_decoding import (
    align_spikes_to_events,
    compute_population_significance,
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


_CLASS_LABEL = 'Outcome'
_ANALYSIS_TITLE = 'Event Outcome Decoding'


def decode_event_outcome_single_cell(spike_times: np.ndarray,
                                     event_times: np.ndarray,
                                     outcome_labels: np.ndarray,
                                     alignment: str = 'end',
                                     time_window: Tuple[float, float] = (-1.0, 2.0),
                                     time_bin_size: float = 0.5,
                                     cv_folds: int = 5,
                                     min_events_per_class: int = 5) -> Dict:
    """Decode event outcome (winner vs loser) from a single cell using CV LDA.

    Thin wrapper around ``single_cell_lda_decode``; result dict keys are
    unchanged.
    """
    return single_cell_lda_decode(
        spike_times=spike_times,
        event_times=event_times,
        labels=outcome_labels,
        time_window=time_window,
        time_bin_size=time_bin_size,
        cv_folds=cv_folds,
        min_events_per_class=min_events_per_class,
    )


def decode_event_outcome_population(ks_data,
                                    behavior_data,
                                    animal_of_interest: str,
                                    behavior_type: Optional[str] = None,
                                    use_quality_cells: bool = True,
                                    quality_thresholds: Optional[Dict] = None,
                                    alignment: str = 'end',
                                    time_window: Tuple[float, float] = (-1.0, 2.0),
                                    time_bin_size: float = 0.5,
                                    cv_folds: int = 5,
                                    min_events_per_class: int = 5,
                                    n_shuffles: int = 0,
                                    alpha: float = 0.05,
                                    seed: int = 0) -> Dict:
    """Decode event outcome (winner vs loser) cell-by-cell across the population.

    Parameters mirror ``decode_opponent_identity_population``. ``behavior_type``
    defaults to ``None``, in which case any event with both ``winner`` and
    ``loser`` populated is included.

    ``n_shuffles`` (default 0, i.e. off) opts into the rigor layer: a
    label-permutation significance test + Benjamini-Hochberg FDR correction
    across cells (see ``ephys._lda_decoding.compute_population_significance``).
    When enabled, ``results['significance']`` is a ``{cluster_id: {...}}``
    dict; when off, it is ``None`` and every other key is unchanged.
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
        labels=outcome_labels,
        time_window=time_window,
        time_bin_size=time_bin_size,
        cv_folds=cv_folds,
        min_events_per_class=min_events_per_class,
    )

    significance = None
    if n_shuffles > 0 and successful_cluster_ids:
        print(f"Computing label-permutation significance ({n_shuffles} shuffles)...")
        significance = compute_population_significance(
            spike_times_list=spike_times_list,
            cluster_ids=selected_cluster_ids,
            event_times=event_times,
            labels=outcome_labels,
            successful_cluster_ids=successful_cluster_ids,
            time_window=time_window,
            time_bin_size=time_bin_size,
            cv_folds=cv_folds,
            n_shuffles=n_shuffles,
            alpha=alpha,
            seed=seed,
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
        'labels': outcome_labels,
        'significance': significance,
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
            'n_shuffles': n_shuffles,
            'alpha': alpha,
            'class_label': _CLASS_LABEL,
            'analysis_title': _ANALYSIS_TITLE,
        },
        'behavioral_summary': {
            'n_events': len(event_times),
            'unique_classes': np.unique(outcome_labels),
            'class_counts': dict(zip(*np.unique(outcome_labels, return_counts=True))),
        },
        'status': 'success',
    }


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
        labels=outcome_labels,
        time_window=time_window,
        time_bin_size=time_bin_size,
        time_bin_step=time_bin_step,
        cv_folds=cv_folds,
        n_shuffles=n_shuffles,
    )
    if core.get('status') != 'success':
        return core

    core['event_times'] = event_times
    core['labels'] = outcome_labels
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
        'class_label': _CLASS_LABEL,
        'analysis_title': _ANALYSIS_TITLE,
    }
    return core


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
    parser.add_argument('--n_shuffles', type=int, default=0,
                        help='Label-permutation shuffles for the rigor-layer significance test '
                             '(0 = off, the default; e.g. 200 for a real significance pass)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='FDR threshold for the rigor-layer significance test')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    parser.add_argument('--output_dir', type=str, help='Output directory for plots')

    args = parser.parse_args()

    sys.path.append(str(Path(__file__).parent.parent))

    try:
        from ingestion.kilosort_data_import import load_kilosort_data
        from ingestion.data_paths import DataStorageManager
        from ingestion.ephys_sync import DataSyncManager
        from video.behavioral_events import load_behavioral_events
    except ImportError as e:
        print(f"Import error: {e}")
        return 1

    print(f"Loading data for animal {args.animal_id}, session {args.session_id}")

    try:
        data_manager = DataStorageManager(args.animal_id, args.session_id, auto_load=True)
        ks_data = load_kilosort_data(data_manager.get_kilosort_path())
        print(f"Loaded {len(ks_data.ks_ids)} ephys clusters")

        behavior_data = load_behavioral_events(
            data_manager.get_behavioral_event_files(),
            session_id=data_manager.session_id,
        )
        print(f"Loaded {len(behavior_data.events_data)} behavioral events")

        print("Synchronizing behavioral events to ephys clock...")
        sync = DataSyncManager(data_manager, dio_channel=1)
        behavior_data.synchronize_with_ephys(sync, create_new_columns=True)
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
        n_shuffles=args.n_shuffles,
        alpha=args.alpha,
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

    if results.get('significance'):
        n_sig = sum(1 for v in results['significance'].values() if v['significant'])
        print(f"Significant cells after FDR correction (q < {args.alpha}): "
              f"{n_sig}/{len(results['significance'])}")

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
