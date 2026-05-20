"""
Shared plotting functions for the ephys decoding modules.

Both ``ephys/decode_opponent_identity.py`` and
``ephys/decode_event_outcome.py`` produce result dicts in a unified schema
and re-export the plot functions defined here. Plot titles and axis labels
are driven by two cosmetic strings that the wrappers stash on the result:

    results['parameters']['class_label']     # e.g. 'Opponent', 'Outcome'
    results['parameters']['analysis_title']  # e.g. 'Opponent Identity Decoding'

Per-cell population results carry the unified keys
``behavioral_summary['unique_classes']`` and
``behavioral_summary['class_counts']``. Time-resolved population results carry
``unique_classes`` at the top level. Both result types include top-level
``event_times`` and ``labels`` arrays; the per-cell plots read those directly
instead of re-extracting from ``behavior_data``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ephys._lda_decoding import align_spikes_to_events


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _class_label(results: Dict, default: str = 'Class') -> str:
    return results.get('parameters', {}).get('class_label', default)


def _analysis_title(results: Dict, default: str = 'Decoding') -> str:
    return results.get('parameters', {}).get('analysis_title', default)


def _behavior_label(results: Dict) -> str:
    btype = results.get('parameters', {}).get('behavior_type')
    return btype if btype is not None else 'any'


def _draw_confusion_matrix(ax,
                           cm: np.ndarray,
                           ticklabels,
                           title: Optional[str] = None,
                           xlabel: str = 'Predicted',
                           ylabel: str = 'True'):
    """Draw a confusion matrix on ``ax``. Returns the imshow handle."""
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ticks = np.arange(len(ticklabels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{t}' for t in ticklabels])
    ax.set_yticklabels([f'{t}' for t in ticklabels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    return im


def _cluster_index_map(ks_data) -> Dict:
    """Map cluster_id -> position in ks_data.spike_times_by_cell."""
    return {cid: i for i, cid in enumerate(ks_data.ks_ids)}


def _accuracies_with_ids(results: Dict) -> List[Tuple]:
    """Return [(cluster_id, accuracy), ...] for cells that decoded successfully."""
    return [
        (cid, results['cell_results'][cid]['accuracy'])
        for cid in results['successful_cells']
        if not np.isnan(results['cell_results'][cid]['accuracy'])
    ]


# ---------------------------------------------------------------------------
# Time-resolved
# ---------------------------------------------------------------------------

def plot_time_resolved_decoding(results: Dict,
                                figsize: Tuple[int, int] = (13, 5)) -> Optional[plt.Figure]:
    """Population-LDA accuracy curve vs time, plus best-bin confusion matrix.

    Expects the dict returned by the time-resolved decoders. Shows the
    accuracy ± CV-fold SEM, the chance level, and (when present) the
    label-shuffle 95% band.
    """
    if results.get('status') != 'success':
        return None

    acc = results['accuracy_by_bin']
    sem = results['accuracy_sem_by_bin']
    t = results['bin_centers']
    chance = results['chance_level']
    best_idx = results.get('best_bin_index')
    unique_classes = results.get('unique_classes', [])
    n_classes = len(unique_classes)
    class_label = _class_label(results)
    analysis_title = _analysis_title(results)
    btype = _behavior_label(results)

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

    ax.set_xlabel('Time from event (s)')
    ax.set_ylabel('Decoding accuracy')
    ax.set_title(f'Time-resolved population decoding — {analysis_title}\n'
                 f'behavior={btype} · {n_classes} classes · {results["n_events"]} events')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    cm = results.get('best_bin_confusion_matrix')
    if cm is not None and best_idx is not None:
        best_acc = results.get('best_bin_accuracy')
        title = f'Best bin: t={t[best_idx]:.2f}s'
        if best_acc is not None:
            title += f'\naccuracy={best_acc:.1%}'
        im = _draw_confusion_matrix(
            ax_cm, cm, unique_classes, title=title,
            xlabel=f'Predicted {class_label}',
            ylabel=f'True {class_label}',
        )
        plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    else:
        ax_cm.axis('off')
        ax_cm.text(0.5, 0.5, 'No confusion matrix available',
                   ha='center', va='center', transform=ax_cm.transAxes)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-cell distribution
# ---------------------------------------------------------------------------

def plot_decoding_accuracy_distribution(results: Dict,
                                        figsize: Tuple[int, int] = (9, 5),
                                        save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Histogram + box plot of per-cell decoding accuracies."""
    if results['status'] != 'success' or results['n_successful_cells'] == 0:
        print("No successful results to plot")
        return None

    accuracies = [results['cell_results'][cid]['accuracy']
                  for cid in results['successful_cells']
                  if not np.isnan(results['cell_results'][cid]['accuracy'])]
    if len(accuracies) == 0:
        print("No valid accuracies to plot")
        return None

    n_classes = len(results['behavioral_summary']['unique_classes'])
    chance_level = 1.0 / n_classes
    analysis_title = _analysis_title(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.axvline(chance_level, color='red', linestyle='--', alpha=0.7,
                label=f'Chance ({chance_level:.1%})')
    ax1.axvline(np.mean(accuracies), color='orange', linestyle='-', linewidth=2,
                label=f'Mean ({np.mean(accuracies):.1%})')
    ax1.set_xlabel('Decoding Accuracy')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title(f'Distribution of {analysis_title} Accuracies')
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

    btype = _behavior_label(results)
    plt.suptitle(f"{analysis_title} Results\n"
                 f"Behavior: {btype}, "
                 f"Alignment: {results['parameters']['alignment']}",
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Best cells
# ---------------------------------------------------------------------------

def plot_best_cells_decoding(results: Dict,
                             n_top_cells: int = 10,
                             figsize: Tuple[int, int] = (12, 6),
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Top-cell bar chart + best-cell confusion matrix."""
    if results['status'] != 'success' or results['n_successful_cells'] == 0:
        print("No successful results to plot")
        return None

    cell_accs = _accuracies_with_ids(results)
    if not cell_accs:
        print("No valid accuracies to plot")
        return None

    n_classes = len(results['behavioral_summary']['unique_classes'])
    chance_level = 1.0 / n_classes
    class_label = _class_label(results)
    analysis_title = _analysis_title(results)

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
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{acc:.1%}', va='center')

    best_cell_id = top_cells[0][0]
    best_result = results['cell_results'][best_cell_id]
    if best_result['confusion_matrix'] is not None:
        im = _draw_confusion_matrix(
            ax2,
            best_result['confusion_matrix'],
            results['behavioral_summary']['unique_classes'],
            title=f'Confusion Matrix - Best Cell (ID: {best_cell_id})',
            xlabel=f'Predicted {class_label}',
            ylabel=f'True {class_label}',
        )
        plt.colorbar(im, ax=ax2)

    btype = _behavior_label(results)
    plt.suptitle(f"Top Performing Cells - {analysis_title}\n"
                 f"Behavior: {btype}, Best accuracy: {top_cells[0][1]:.1%}",
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Comprehensive summary
# ---------------------------------------------------------------------------

def plot_decoding_summary(results: Dict,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """6-panel summary plot of per-cell decoding results."""
    if results['status'] != 'success':
        print("No successful results to plot")
        return None

    n_classes = len(results['behavioral_summary']['unique_classes'])
    chance_level = 1.0 / n_classes
    class_label = _class_label(results)
    analysis_title = _analysis_title(results)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Accuracy distribution
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

    # 2. Best-cell confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cell_accs = _accuracies_with_ids(results)
    best_cell_id = max(cell_accs, key=lambda x: x[1])[0] if cell_accs else None
    best_result = results['cell_results'].get(best_cell_id) if best_cell_id is not None else None

    if best_result is not None and best_result.get('confusion_matrix') is not None:
        im = _draw_confusion_matrix(
            ax2,
            best_result['confusion_matrix'],
            results['behavioral_summary']['unique_classes'],
            title=f'Confusion Matrix — Best Cell (ID: {best_cell_id})',
            xlabel=f'Predicted {class_label}',
            ylabel=f'True {class_label}',
        )
        plt.colorbar(im, ax=ax2)
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'No confusion matrix available',
                 ha='center', va='center', transform=ax2.transAxes)

    # 3. Class counts
    ax3 = fig.add_subplot(gs[0, 2])
    class_counts = results['behavioral_summary']['class_counts']
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    bars = ax3.bar([f'{c}' for c in classes], counts, color='lightblue')
    ax3.set_xlabel(class_label)
    ax3.set_ylabel('Number of Events')
    ax3.set_title(f"Behavioral Events\n(Total: {results['behavioral_summary']['n_events']})")
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom')

    # 4. Top cells bar chart
    ax4 = fig.add_subplot(gs[1, :2])
    if accuracies:
        cell_accs_sorted = sorted(cell_accs, key=lambda x: x[1], reverse=True)
        top_n = min(20, len(cell_accs_sorted))
        top_cells = cell_accs_sorted[:top_n]
        accuracies_top = [cell[1] for cell in top_cells]

        bars = ax4.bar(range(top_n), accuracies_top, color='lightcoral')
        ax4.axhline(chance_level, color='red', linestyle='--', alpha=0.7, label='Chance')
        ax4.set_xticks(range(top_n))
        ax4.set_xticklabels([str(cell[0]) for cell in top_cells], rotation=45, ha='right')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Accuracy')
        ax4.set_title(f'Top {top_n} Cells by Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        if len(bars) > 0:
            bars[0].set_color('red')
            bars[0].set_alpha(0.8)

    # 5. Parameters / results summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    params = results['parameters']
    btype = _behavior_label(results)
    param_text = f"""
    ANALYSIS PARAMETERS:

    Behavior Type: {btype}
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

    plt.suptitle(f"{analysis_title} - Complete Analysis Summary",
                 fontsize=16, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Top-cell PETHs and rasters (need event_times + labels from results)
# ---------------------------------------------------------------------------

def _events_and_labels_from_results(test_results: Dict) -> Tuple[np.ndarray, np.ndarray]:
    if 'event_times' not in test_results or 'labels' not in test_results:
        raise KeyError(
            "result dict missing 'event_times' / 'labels'; "
            "regenerate via decode_*_population on the current code."
        )
    return test_results['event_times'], test_results['labels']


def plot_top_cells_firing_rates(ks_data,
                                behavior_data,
                                test_results: Dict,
                                time_window: Tuple[float, float] = (-2.0, 2.0),
                                time_bin_size: float = 0.2,
                                n_top_cells: int = 12,
                                figsize: Tuple[int, int] = (15, 10),
                                save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """Peri-event firing-rate curves for the top cells, split by class label.

    ``behavior_data`` is unused (kept for backward-compat call signatures);
    event times and labels are read directly from ``test_results``.
    """
    del behavior_data
    if test_results['status'] != 'success' or test_results['n_successful_cells'] == 0:
        print("No successful results to plot")
        return None

    cell_accs = _accuracies_with_ids(test_results)
    if not cell_accs:
        print("No valid accuracies to plot")
        return None
    cell_accs_sorted = sorted(cell_accs, key=lambda x: x[1], reverse=True)
    top_cells = cell_accs_sorted[:min(n_top_cells, len(cell_accs_sorted))]

    try:
        event_times, labels = _events_and_labels_from_results(test_results)
    except KeyError as e:
        print(f"Error: {e}")
        return None
    if len(event_times) == 0:
        print("No behavioral events found")
        return None

    unique_classes = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    bin_edges = np.arange(time_window[0], time_window[1] + time_bin_size, time_bin_size)
    bin_centers = bin_edges[:-1] + time_bin_size / 2

    n_top = len(top_cells)
    n_cols = min(3, n_top)
    n_rows = int(np.ceil(n_top / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=False, squeeze=False)

    cluster_idx = _cluster_index_map(ks_data)

    for cell_idx, (cluster_id, accuracy) in enumerate(top_cells):
        row = cell_idx // n_cols
        col = cell_idx % n_cols
        ax = axes[row, col]

        cell_position = cluster_idx.get(cluster_id)
        if cell_position is None:
            print(f"Warning: Cell {cluster_id} not found in spike data")
            continue

        spike_times = ks_data.spike_times_by_cell[cell_position]

        cell_max_fr = 0.0
        for class_idx, cls in enumerate(unique_classes):
            class_mask = labels == cls
            class_event_times = event_times[class_mask]
            if len(class_event_times) == 0:
                continue

            aligned_spikes = align_spikes_to_events(spike_times, class_event_times, time_window)
            trial_firing_rates = []
            for trial_spikes in aligned_spikes:
                if len(trial_spikes) > 0:
                    counts, _ = np.histogram(trial_spikes, bins=bin_edges)
                    trial_firing_rates.append(counts / time_bin_size)
                else:
                    trial_firing_rates.append(np.zeros(len(bin_centers)))
            trial_firing_rates = np.array(trial_firing_rates)
            mean_fr = np.mean(trial_firing_rates, axis=0)
            sem_fr = np.std(trial_firing_rates, axis=0) / np.sqrt(len(trial_firing_rates))

            if mean_fr.size:
                cell_max_fr = max(cell_max_fr, float(np.nanmax(mean_fr)))

            color = colors[class_idx]
            ax.plot(bin_centers, mean_fr, color=color, linewidth=2,
                    label=f'{cls} (n={len(class_event_times)})')
            ax.fill_between(bin_centers, mean_fr - sem_fr, mean_fr + sem_fr,
                            color=color, alpha=0.3)

        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'Cell {cluster_id}\nAccuracy: {accuracy:.1%}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.3*cell_max_fr if cell_max_fr > 0 else 1.0)
        if cell_idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    for idx in range(n_top, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    fig.text(0.5, 0.02, 'Time from Event (s)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Firing Rate (Hz)', va='center', rotation=90, fontsize=12)

    params = test_results['parameters']
    btype = _behavior_label(test_results)
    plt.suptitle(f'Peri-Event Firing Rates - Top {len(top_cells)} Cells\n'
                 f'Behavior: {btype}, Alignment: {params["alignment"]}, '
                 f'Window: {time_window[0]:.1f} to {time_window[1]:.1f}s',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0.03, 0.03, 1.0, 0.97])

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
    """Spike rasters for the top cells, with rows sorted/colored by class.

    ``behavior_data`` is unused (kept for backward-compat call signatures);
    event times and labels are read directly from ``test_results``.
    """
    del behavior_data
    if test_results['status'] != 'success' or test_results['n_successful_cells'] == 0:
        print("No successful results to plot")
        return None

    cell_accs = _accuracies_with_ids(test_results)
    if not cell_accs:
        print("No valid accuracies to plot")
        return None
    cell_accs_sorted = sorted(cell_accs, key=lambda x: x[1], reverse=True)
    top_cells = cell_accs_sorted[:min(n_top_cells, len(cell_accs_sorted))]

    try:
        event_times, labels = _events_and_labels_from_results(test_results)
    except KeyError as e:
        print(f"Error: {e}")
        return None
    if len(event_times) == 0:
        print("No behavioral events found")
        return None

    unique_classes = np.unique(labels)
    sort_idx = np.argsort(labels, kind='stable')
    event_times_sorted = event_times[sort_idx]
    labels_sorted = labels[sort_idx]

    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    block_boundaries = [
        i for i in range(1, len(labels_sorted))
        if labels_sorted[i] != labels_sorted[i - 1]
    ]

    n_top = len(top_cells)
    n_cols = min(3, n_top)
    n_rows = int(np.ceil(n_top / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)

    cluster_idx = _cluster_index_map(ks_data)
    n_events = len(event_times_sorted)

    for cell_idx, (cluster_id, accuracy) in enumerate(top_cells):
        row = cell_idx // n_cols
        col = cell_idx % n_cols
        ax = axes[row, col]

        cell_position = cluster_idx.get(cluster_id)
        if cell_position is None:
            print(f"Warning: Cell {cluster_id} not found in spike data")
            continue

        spike_times = ks_data.spike_times_by_cell[cell_position]
        aligned_spikes = align_spikes_to_events(spike_times, event_times_sorted, time_window)

        for cls in unique_classes:
            class_rows = np.where(labels_sorted == cls)[0]
            if len(class_rows) == 0:
                continue
            class_spike_lists = [aligned_spikes[r] for r in class_rows]
            ax.eventplot(class_spike_lists,
                         lineoffsets=class_rows,
                         linelengths=0.85,
                         linewidths=0.6,
                         colors=[color_map[cls]])

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
                Line2D([0], [0], color=color_map[cls], lw=2,
                       label=f'{cls} (n={int(np.sum(labels_sorted == cls))})')
                for cls in unique_classes
            ]
            ax.legend(handles=legend_handles, fontsize=8, loc='upper right')

    for idx in range(n_top, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    class_label = _class_label(test_results)
    fig.text(0.5, 0.02, 'Time from Event (s)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, f'Event (sorted by {class_label.lower()})',
             va='center', rotation=90, fontsize=12)

    params = test_results['parameters']
    btype = _behavior_label(test_results)
    plt.suptitle(f'Spike Rasters - Top {n_top} Cells\n'
                 f'Behavior: {btype}, Alignment: {params["alignment"]}, '
                 f'Window: {time_window[0]:.1f} to {time_window[1]:.1f}s',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved raster plot: {save_path}")

    return fig
