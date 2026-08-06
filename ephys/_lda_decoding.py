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
- ``compute_population_significance(spike_times_list, cluster_ids, event_times, labels, successful_cluster_ids, ...)``
  — opt-in label-permutation + FDR significance layer for the per-cell decoders (§5 rigor layer).

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
from ephys._stats_utils import (
    benjamini_hochberg,
    empirical_p_value,
    fdr_resolution,
    majority_class_baseline,
)
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

def _prepare_lda_features(spike_times: np.ndarray,
                          event_times: np.ndarray,
                          time_window: Tuple[float, float],
                          time_bin_size: float) -> Optional[np.ndarray]:
    """Align spikes to events and return standardized firing-rate features.

    Returns ``None`` if every bin is empty (no spikes anywhere in the
    window) — the caller should treat that as a ``'no_spikes'`` condition.
    Shared by ``single_cell_lda_decode`` and the permutation-null
    significance test below, since features don't depend on labels and can
    be computed once per cell and reused across label shuffles.
    """
    aligned_spikes = align_spikes_to_events(spike_times, event_times, time_window)
    features = extract_firing_rate_features(aligned_spikes, time_window, time_bin_size)
    if np.all(features == 0):
        return None
    return StandardScaler().fit_transform(features)


def _cv_lda_accuracy(features_scaled: np.ndarray,
                     labels: np.ndarray,
                     counts: np.ndarray,
                     cv_folds,
                     scoring: str = 'accuracy') -> np.ndarray:
    """Cross-validated LDA scores for one cell's features/labels.

    ``scoring`` defaults to plain ``'accuracy'`` (what every existing caller
    expects); pass ``'balanced_accuracy'`` for the prevalence-corrected
    variant, which is what to compare against 1/n_classes when classes are
    imbalanced.
    """
    if cv_folds == 'loo' or cv_folds == -1:
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=min(int(cv_folds), int(np.min(counts))),
                             shuffle=True, random_state=42)
    lda = LinearDiscriminantAnalysis()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_score(lda, features_scaled, labels, cv=cv, scoring=scoring)


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
    ``confusion_matrix``, ``cv_scores``, ``status``, plus two
    prevalence-aware additions:

    - ``baseline_accuracy`` — accuracy from always guessing the majority
      class. **This, not ``1/n_classes``, is what ``accuracy`` must beat**
      when classes are imbalanced: for a 12/7 split the majority baseline is
      63.2% while ``1/n_classes`` would say 50%.
    - ``balanced_accuracy`` — mean per-class recall, i.e. the
      prevalence-corrected score, whose chance level *is* ``1/n_classes``.
      ``nan`` where it can't be computed (e.g. leave-one-out folds).
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    baseline = majority_class_baseline(labels)
    if len(unique_labels) < 2 or np.min(counts) < min_events_per_class:
        return {
            'accuracy': np.nan,
            'accuracy_std': np.nan,
            'baseline_accuracy': baseline,
            'balanced_accuracy': np.nan,
            'n_events': len(event_times),
            'n_classes': len(unique_labels),
            'class_counts': dict(zip(unique_labels, counts)),
            'confusion_matrix': None,
            'cv_scores': None,
            'status': 'insufficient_data',
        }

    try:
        features_scaled = _prepare_lda_features(spike_times, event_times, time_window, time_bin_size)

        if features_scaled is None:
            return {
                'accuracy': np.nan,
                'accuracy_std': np.nan,
                'baseline_accuracy': baseline,
                'balanced_accuracy': np.nan,
                'n_events': len(event_times),
                'n_classes': len(unique_labels),
                'class_counts': dict(zip(unique_labels, counts)),
                'confusion_matrix': None,
                'cv_scores': None,
                'status': 'no_spikes',
            }

        cv_scores = _cv_lda_accuracy(features_scaled, labels, counts, cv_folds)

        try:
            balanced = float(np.mean(_cv_lda_accuracy(
                features_scaled, labels, counts, cv_folds, scoring='balanced_accuracy')))
        except Exception:
            balanced = np.nan

        lda = LinearDiscriminantAnalysis()
        lda.fit(features_scaled, labels)
        predictions = lda.predict(features_scaled)
        conf_matrix = confusion_matrix(labels, predictions, labels=unique_labels)

        return {
            'accuracy': float(np.mean(cv_scores)),
            'accuracy_std': float(np.std(cv_scores)),
            'baseline_accuracy': baseline,
            'balanced_accuracy': balanced,
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
            'baseline_accuracy': baseline,
            'balanced_accuracy': np.nan,
            'n_events': len(event_times),
            'n_classes': len(unique_labels) if 'unique_labels' in locals() else 0,
            'class_counts': dict(zip(unique_labels, counts)) if 'unique_labels' in locals() else {},
            'confusion_matrix': None,
            'cv_scores': None,
            'status': f'error: {str(e)}',
        }


# ---------------------------------------------------------------------------
# Rigor layer: label-permutation significance + FDR across cells
# ---------------------------------------------------------------------------

def compute_population_significance(spike_times_list: Sequence[np.ndarray],
                                     cluster_ids: Sequence,
                                     event_times: np.ndarray,
                                     labels: np.ndarray,
                                     successful_cluster_ids: Sequence,
                                     time_window: Tuple[float, float] = (-1.0, 2.0),
                                     time_bin_size: float = 0.5,
                                     cv_folds=5,
                                     n_shuffles: int = 200,
                                     alpha: float = 0.05,
                                     seed: int = 0,
                                     null_mode: str = 'per_cell') -> Dict:
    """Label-permutation significance for a per-cell decode, at two granularities.

    Re-decodes each of ``successful_cluster_ids`` under ``n_shuffles`` random
    label permutations. Features don't depend on labels, so they are computed
    once per cell (via ``_prepare_lda_features``) and reused across shuffles;
    only the label vector and the CV/LDA refit change per shuffle.

    Returns a structured dict with three parts:

    ``'per_cell'``
        ``{cluster_id: {'p_value', 'q_value', 'significant', 'n_shuffles'}}`` —
        each cell's accuracy ranked against a null, BH-FDR corrected across
        cells. This is the within-run multiple-comparison guardrail Phase 0
        flagged as missing (its synthetic demo found a naive
        accuracy-vs-chance screen flagged 9/24 cells vs. 6 truly tuned).
    ``'population'``
        ``{'p_value', 'observed_mean_accuracy', 'null_mean', 'null_std',
        'n_shuffles'}`` — one test of whether the *population mean* accuracy
        exceeds its permutation null. Free, since it reuses the same null
        matrix. Being a single test it needs no FDR correction and is well
        resolved at modest ``n_shuffles``, which makes it the appropriate
        headline statistic when the per-cell screen is under-resolved (see
        below) or when events are few.
    ``'resolution'``
        ``ephys._stats_utils.fdr_resolution`` output for this run, plus
        ``null_mode``. **Read this before interpreting a null per-cell
        result**: if ``resolvable`` is False, no single cell could have
        reached ``q < alpha`` regardless of effect size, so zero significant
        cells says nothing about the data. A warning is also raised in that
        case.

    ``null_mode``:

    - ``'per_cell'`` (default) — each cell is ranked against its own
      ``n_shuffles`` null draws. No cross-cell assumption, but the p-value
      floor is ``1/(n_shuffles+1)``, which BH multiplies by the number of
      cells; for a few hundred cells this needs thousands of shuffles to
      resolve.
    - ``'pooled'`` — every cell is ranked against the null accuracies of
      *all* cells pooled (``n_cells * n_shuffles`` draws), lowering the
      p-value floor by roughly the cell count at identical compute. Assumes
      cells share a common null accuracy distribution; that is reasonable
      for a screen but can be anti-conservative for atypical cells (e.g.
      very low firing rates, whose null is wider).

    Assumptions: labels are exchangeable under the null (nothing else, such
    as temporal autocorrelation, ties label identity to event order). One
    permutation is drawn per shuffle index and **shared across all cells**,
    so the pooled and population-level nulls stay valid permutation nulls of
    a coherent relabeling — drawing independent permutations per cell would
    shrink the population-mean null's variance and make that test
    anti-conservative. BH-FDR controls the expected false-discovery
    *proportion* across tested cells, not per-cell Type-I error.
    """
    if null_mode not in ('per_cell', 'pooled'):
        raise ValueError(f"null_mode must be 'per_cell' or 'pooled', got {null_mode!r}")

    _, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    n_events = len(labels)
    spikes_by_cluster = dict(zip(cluster_ids, spike_times_list))

    # One permutation per shuffle index, shared by every cell (see docstring).
    perms = [rng.permutation(n_events) for _ in range(n_shuffles)]

    tested_ids: List = []
    observed: List[float] = []
    null_rows: List[np.ndarray] = []
    for cluster_id in successful_cluster_ids:
        features_scaled = _prepare_lda_features(
            spikes_by_cluster[cluster_id], event_times, time_window, time_bin_size
        )
        if features_scaled is None:
            continue

        tested_ids.append(cluster_id)
        observed.append(
            float(np.mean(_cv_lda_accuracy(features_scaled, labels, counts, cv_folds)))
        )
        null_rows.append(np.array([
            float(np.mean(_cv_lda_accuracy(features_scaled, labels[perm], counts, cv_folds)))
            for perm in perms
        ], dtype=np.float64))

    if not tested_ids:
        return {'per_cell': {}, 'population': None,
                'resolution': fdr_resolution(1, max(n_shuffles, 1), alpha) | {'null_mode': null_mode}}

    observed_arr = np.array(observed, dtype=np.float64)
    null_matrix = np.vstack(null_rows)  # (n_cells, n_shuffles)
    n_cells = len(tested_ids)

    resolution = fdr_resolution(n_cells, n_shuffles, alpha)
    resolution['null_mode'] = null_mode
    if null_mode == 'pooled':
        # Pooling lowers the achievable p-floor by the cell count.
        resolution = fdr_resolution(n_cells, n_cells * n_shuffles, alpha)
        resolution['null_mode'] = null_mode
        resolution['n_shuffles_requested'] = n_shuffles
    if not resolution['resolvable']:
        warnings.warn(
            f"Under-resolved per-cell FDR: {n_cells} cells x {n_shuffles} shuffles gives a "
            f"p-value floor of {resolution['p_floor']:.2g}, so the best achievable q is "
            f"{resolution['best_achievable_q']:.2g} (>= alpha={alpha}). No single cell can reach "
            f"significance regardless of effect size; at least {resolution['min_tests_at_floor']} "
            f"cells must hit the floor simultaneously. Use n_shuffles>="
            f"{resolution['recommended_n_shuffles']}, or null_mode='pooled', or rely on the "
            f"population-level p-value instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    # `empirical_p_value` uses the add-one form and drops non-finite draws from
    # both numerator and denominator — see its docstring for why that matters.
    if null_mode == 'pooled':
        pooled = null_matrix.ravel()
        p_values = np.array([empirical_p_value(obs, pooled) for obs in observed_arr])
    else:
        p_values = np.array([
            empirical_p_value(observed_arr[i], null_matrix[i]) for i in range(n_cells)
        ])

    n_nan_draws = int(np.sum(~np.isfinite(null_matrix)))
    q_values = benjamini_hochberg(p_values)

    # Population-level test: mean accuracy across cells vs. its own null.
    # Free — the per-shuffle means of the retained null matrix.
    observed_mean = float(np.nanmean(observed_arr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column
        null_means = np.nanmean(null_matrix, axis=0)
    valid_means = null_means[np.isfinite(null_means)]
    population = {
        'p_value': empirical_p_value(observed_mean, null_means),
        'observed_mean_accuracy': observed_mean,
        'null_mean': float(np.mean(valid_means)) if valid_means.size else np.nan,
        'null_std': float(np.std(valid_means)) if valid_means.size else np.nan,
        'n_shuffles': n_shuffles,
        'n_valid_shuffles': int(valid_means.size),
        'n_cells': n_cells,
        'n_nan_draws': n_nan_draws,
    }

    return {
        'per_cell': {
            cluster_id: {
                'p_value': float(p),
                'q_value': float(q),
                'significant': bool(q < alpha),
                'n_shuffles': n_shuffles,
            }
            for cluster_id, p, q in zip(tested_ids, p_values, q_values)
        },
        'population': population,
        'resolution': resolution,
    }


# ---------------------------------------------------------------------------
# Shared CLI reporting for the rigor layer
# ---------------------------------------------------------------------------

def print_baseline_block(results: Dict) -> None:
    """Print the prevalence-aware baseline next to the reported accuracy.

    Exists because plain accuracy against imbalanced classes is easy to
    over-read: a 60.6% accuracy looks like signal until you see the
    majority-class baseline is 63.2%.
    """
    baseline = results.get('population_baseline_accuracy')
    if baseline is None or not np.isfinite(baseline):
        return
    accuracy = results.get('population_accuracy_mean', np.nan)
    verdict = ''
    if np.isfinite(accuracy):
        verdict = ('  <-- BELOW majority-class baseline'
                   if accuracy < baseline else '')
    print(f"Majority-class baseline: {baseline:.1%}{verdict}")
    balanced = results.get('population_balanced_accuracy_mean')
    if balanced is not None and np.isfinite(balanced):
        # `unique_classes` is a numpy array — no truthiness tests on it.
        classes = results.get('behavioral_summary', {}).get('unique_classes')
        n_classes = len(classes) if classes is not None else 0
        chance = f" (chance = {1.0 / n_classes:.1%})" if n_classes else ""
        print(f"Balanced accuracy (mean): {balanced:.1%}{chance}")


def print_significance_block(results: Dict, alpha: float) -> None:
    """Print the population-level p-value, per-cell FDR count, and — crucially —
    whether the per-cell screen had the resolution to detect anything."""
    population = results.get('significance_population')
    if population:
        print(f"Population-level permutation test: p = {population['p_value']:.4g} "
              f"(observed mean {population['observed_mean_accuracy']:.1%} vs null "
              f"{population['null_mean']:.1%} ± {population['null_std']:.1%}, "
              f"{population.get('n_valid_shuffles', population['n_shuffles'])} "
              f"usable shuffles)")
        if population.get('n_nan_draws'):
            print(f"  (note: {population['n_nan_draws']} non-finite null draws excluded — "
                  f"degenerate CV folds, expected with few events)")

    significance = results.get('significance')
    if significance:
        n_sig = sum(1 for v in significance.values() if v['significant'])
        print(f"Significant cells after FDR correction (q < {alpha}): "
              f"{n_sig}/{len(significance)}")

    resolution = results.get('significance_resolution')
    if resolution and not resolution.get('resolvable', True):
        print(f"  [!] Per-cell screen UNDER-RESOLVED: {resolution['n_tests']} cells x "
              f"{resolution['n_shuffles']} shuffles -> p-floor {resolution['p_floor']:.2g}, "
              f"best achievable q {resolution['best_achievable_q']:.2g} >= alpha {resolution['alpha']}.")
        print(f"      A null per-cell result here is UNINFORMATIVE, not evidence of absence. "
              f"Use --n_shuffles {resolution['recommended_n_shuffles']}, --null_mode pooled, "
              f"or read the population-level p-value above.")


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
