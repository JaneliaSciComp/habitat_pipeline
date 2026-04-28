import logging

import numpy as np
from scipy import stats

from ingestion.data_paths import DataStorageManager
from ingestion.trodes_to_python import readTrodesExtractedDataFile

logger = logging.getLogger(__name__)

SAMPLE_RATE = 30000.0  # Hz


def load_ephys_sync(data_manager, dio_channel=1):
    """
    Load and process ephys synchronization data using a DataStorageManager.

    Parameters:
    - data_manager: DataStorageManager instance containing session paths
    - dio_channel: int, DIO channel number to load (default is 1)

    Returns:
    - TSESync: np.ndarray, ephys sync timestamps (seconds)
    - TSBSync: np.ndarray, behavioral sync timestamps (seconds)
    - system_time_at_creation: float, system time when recording started (seconds)
    """
    dio_path = data_manager.get_dio_path(channel=dio_channel)
    if dio_path is None:
        raise FileNotFoundError(f"DIO channel {dio_channel} not available in DataStorageManager")

    fields = readTrodesExtractedDataFile(dio_path)
    dio_data = np.array(fields['data'])
    system_time_at_creation = int(fields['system_time_at_creation']) / 1e3

    time = dio_data['time']
    state = dio_data['state']
    TSESync = np.array(time[state == 1]) / SAMPLE_RATE

    pulse_log_path = data_manager.get_pulse_log_path()
    if pulse_log_path is None:
        raise FileNotFoundError("Pulse log path not available in DataStorageManager")

    with open(pulse_log_path, 'r') as f:
        lines = f.readlines()
    TSBSync = np.array([int(line) / 1e9 for line in lines[1:]])

    return TSESync, TSBSync, system_time_at_creation


def find_sync_mapping(TSESync, TSBSync, system_time_at_creation, search_window=300,
                      interval_tolerance=0.02, min_sequence_length=10):
    """
    Find the best linear mapping between ephys and behavior timestamps using inter-pulse intervals.

    Parameters:
    - TSESync: array of ephys sync timestamps (seconds)
    - TSBSync: array of behavior sync timestamps (seconds)
    - system_time_at_creation: float, system time when recording started (seconds)
    - search_window: float, time window around system_time_at_creation to search (seconds)
    - interval_tolerance: tolerance for matching intervals (seconds)
    - min_sequence_length: minimum length of matching sequences to consider

    Returns:
    - dict with 'slope', 'intercept', 'r_squared', 'n_matches', matched arrays, and
      'ephys_to_behavior' / 'behavior_to_ephys' conversion functions.
    """
    TSE = np.asarray(TSESync)
    TSB = np.asarray(TSBSync)
    intervals_E = np.diff(TSE)
    intervals_B = np.diff(TSB)

    start_time = system_time_at_creation - search_window / 2
    end_time = system_time_at_creation + search_window / 2
    valid_indices = np.where((TSB >= start_time) & (TSB <= end_time))[0]

    def find_matching_sequence(intervals_ref, intervals_target, start_ref):
        seq_ref = intervals_ref[start_ref:start_ref + min_sequence_length]
        for start_target in range(len(intervals_target) - min_sequence_length + 1):
            seq_target = intervals_target[start_target:start_target + min_sequence_length]
            if np.all(np.abs(seq_ref - seq_target) <= interval_tolerance):
                return start_target
        return -1

    search_start = int(valid_indices[0]) if len(valid_indices) > 0 else 0
    matches = []
    for start_b in range(search_start, len(intervals_B) - min_sequence_length + 1):
        match_e = find_matching_sequence(intervals_B, intervals_E, start_b)
        if match_e == -1:
            continue

        for i in range(min_sequence_length):
            matches.append((TSE[match_e + i + 1], TSB[start_b + i + 1]))
        logger.info("Found sync sequence match: B[%d] -> E[%d]", start_b, match_e)

        b_idx = start_b + min_sequence_length
        e_idx = match_e + min_sequence_length
        while b_idx < len(intervals_B) and e_idx < len(intervals_E):
            if abs(intervals_B[b_idx] - intervals_E[e_idx]) > interval_tolerance:
                break
            matches.append((TSE[e_idx + 1], TSB[b_idx + 1]))
            b_idx += 1
            e_idx += 1
        break

    if len(matches) < 2:
        raise ValueError("No sync match found within search window")

    matched_ephys = np.array([m[0] for m in matches])
    matched_behavior = np.array([m[1] for m in matches])

    slope, intercept, r_value, p_value, std_err = stats.linregress(matched_ephys, matched_behavior)

    mapping = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'n_matches': len(matched_ephys),
        'p_value': p_value,
        'std_err': std_err,
        'matched_ephys': matched_ephys,
        'matched_behavior': matched_behavior,
    }
    mapping['ephys_to_behavior'] = lambda t: slope * t + intercept
    mapping['behavior_to_ephys'] = lambda t: (t - intercept) / slope
    return mapping


def plot_sync_results(mapping_dict, TSESync=None, TSBSync=None, figsize=(15, 10)):
    """
    Plot synchronization mapping results and statistics.

    Parameters:
    - mapping_dict: dict returned by find_sync_mapping
    - TSESync: optional, full ephys sync array for context
    - TSBSync: optional, full behavior sync array for context
    - figsize: tuple, figure size for plots

    Returns:
    - matplotlib figure object, or None if matplotlib is unavailable
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; install with: pip install matplotlib")
        return None

    matched_ephys = mapping_dict['matched_ephys']
    matched_behavior = mapping_dict['matched_behavior']
    slope = mapping_dict['slope']
    intercept = mapping_dict['intercept']
    r_squared = mapping_dict['r_squared']
    n_matches = mapping_dict['n_matches']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Synchronization Mapping Results (n={n_matches} matches)', fontsize=16)

    # Scatter of matched timestamps with regression line
    ax1.scatter(matched_ephys, matched_behavior, alpha=0.6, color='blue', s=20)
    x_range = np.linspace(matched_ephys.min(), matched_ephys.max(), 100)
    ax1.plot(x_range, slope * x_range + intercept, 'r--', linewidth=2,
             label=f'y = {slope:.6f}x + {intercept:.3f}')
    ax1.set_xlabel('Ephys Timestamps (s)')
    ax1.set_ylabel('Behavior Timestamps (s)')
    ax1.set_title(f'Linear Mapping (R² = {r_squared:.6f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals
    residuals = matched_behavior - (slope * matched_ephys + intercept)
    ax2.scatter(matched_ephys, residuals, alpha=0.6, color='green', s=20)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Ephys Timestamps (s)')
    ax2.set_ylabel('Residuals (s)')
    ax2.set_title(f'Residuals (std = {np.std(residuals):.6f} s)')
    ax2.grid(True, alpha=0.3)

    # Interval comparison (if full arrays provided)
    if TSESync is not None and TSBSync is not None:
        ephys_converted = mapping_dict['ephys_to_behavior'](TSESync)
        start = max(TSBSync.min(), ephys_converted.min())
        end = min(TSBSync.max(), ephys_converted.max())
        tsb_overlap = TSBSync[(TSBSync >= start) & (TSBSync <= end)]
        tse_overlap = ephys_converted[(ephys_converted >= start) & (ephys_converted <= end)]

        if len(tsb_overlap) > 1 and len(tse_overlap) > 1:
            intervals_b = np.diff(tsb_overlap)
            intervals_e = np.diff(tse_overlap)
            min_len = min(len(intervals_b), len(intervals_e))
            intervals_b = intervals_b[:min_len]
            intervals_e = intervals_e[:min_len]

            ax3.scatter(intervals_e, intervals_b, alpha=0.6, color='purple', s=20)
            lim = max(intervals_e.max(), intervals_b.max())
            ax3.plot([0, lim], [0, lim], 'r--', linewidth=2)
            ax3.set_xlabel('Ephys Intervals (s)')
            ax3.set_ylabel('Behavior Intervals (s)')
            ax3.set_title('Interval Comparison')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient overlap\nfor interval comparison',
                     transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'Full arrays not provided\nfor interval analysis',
                 transform=ax3.transAxes, ha='center', va='center', fontsize=12)

    # Stats panel
    ax4.axis('off')
    stats_text = (
        f"Synchronization Statistics:\n\n"
        f"Linear Mapping:\n"
        f"  Slope: {slope:.8f}\n"
        f"  Intercept: {intercept:.6f} s\n"
        f"  R²: {r_squared:.6f}\n"
        f"  P-value: {mapping_dict.get('p_value', float('nan')):.2e}\n\n"
        f"Quality Metrics:\n"
        f"  Matched Points: {n_matches}\n"
        f"  Residual Std: {np.std(residuals):.6f} s\n"
        f"  Residual Max: {np.max(np.abs(residuals)):.6f} s\n"
        f"  Residual Mean: {np.mean(np.abs(residuals)):.6f} s\n\n"
        f"Time Range:\n"
        f"  Ephys: {matched_ephys.min():.3f} - {matched_ephys.max():.3f} s\n"
        f"  Behavior: {matched_behavior.min():.3f} - {matched_behavior.max():.3f} s\n"
        f"  Duration: {matched_ephys.max() - matched_ephys.min():.3f} s"
    )
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    return fig


class DataSyncManager:
    """
    Manage ephys ↔ behavior clock synchronization for a single session.

    Loads sync pulses from the configured DataStorageManager, fits a linear
    mapping between ephys and behavior timestamps, and exposes conversion
    functions in both directions.
    """

    def __init__(self, data_manager: DataStorageManager, dio_channel: int = 1):
        self.data_manager = data_manager
        self.dio_channel = dio_channel
        self.animal_id = data_manager.animal_id
        self.session_id = data_manager.session_id

        self.ephys_sync, self.behavior_sync, self.system_time = load_ephys_sync(
            data_manager, dio_channel
        )
        self.mapping = find_sync_mapping(
            self.ephys_sync, self.behavior_sync, self.system_time
        )
        logger.info("Sync mapping clock-rate ratio: %.8f", self.mapping['slope'])

    def convert_ephys_to_behavior(self, ephys_timestamps):
        """Convert ephys timestamps to behavioral timestamps."""
        return self.mapping['ephys_to_behavior'](ephys_timestamps)

    def convert_behavior_to_ephys(self, behavior_timestamps):
        """Convert behavioral timestamps to ephys timestamps."""
        return self.mapping['behavior_to_ephys'](behavior_timestamps)

    def plot_sync_results(self, figsize=(15, 10)):
        """Plot synchronization residuals and quality metrics."""
        return plot_sync_results(self.mapping, self.ephys_sync, self.behavior_sync, figsize)

    def __repr__(self) -> str:
        return (f"DataSyncManager({self.animal_id}/{self.session_id}, "
                f"DIO{self.dio_channel}, n_matches={self.mapping['n_matches']})")
