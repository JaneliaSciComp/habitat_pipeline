import numpy as np
from scipy import stats

from ingestion.data_paths import get_dio_path, get_pulse_log_path

SAMPLE_RATE = 30000.0  # Hz

def load_ephys_sync(animal_id, session_id, dio_channel=1, config_path=None):
    """
    Load and process ephys synchronization data for a given animal and session.

    Parameters:
    - animal_id: str, ID of the animal (e.g., "613")
    - session_id: str, ID of the session (e.g., "20251210")
    - dio_channel: int, DIO channel number to load (default is 1)
    - config_path: str or None, path to the configuration file (default is None)

    Returns:
    - TSESync: np.ndarray, timestamps of synchronization events
    - sync_params: dict, parameters of the synchronization fit
    """
    # Get DIO file path
    dio_path = get_dio_path(animal_id, session_id, dio_channel, config_path=config_path)
    
    # Load DIO data
    from ingestion.trodes_to_python import readTrodesExtractedDataFile
    fields = readTrodesExtractedDataFile(dio_path)
    dio_data = np.array(fields['data'])
    system_time_at_creation = int(fields['system_time_at_creation'])
    system_time_at_creation = system_time_at_creation / 1e3
    
    # Extract time and state
    time = dio_data['time']
    state = dio_data['state']
    
    # Get timestamps where state is 1 (sync pulses)
    TSESync = time[state == 1]   
    TSESync = np.array(TSESync) / SAMPLE_RATE  # Convert to seconds
    
    # Load pulse log data
    pulse_log_path = get_pulse_log_path(config_path=config_path)

    TSBSync = []
    with open(pulse_log_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the first row (header)
            time_stamp = int(line)/ 1e9
            TSBSync.append(time_stamp)
    TSBSync = np.array(TSBSync)

    
    return TSESync, TSBSync, system_time_at_creation


def find_sync_mapping_new(TSESync, TSBSync, system_time_at_creation, tolerance=0.1, search_window=300, initial_subset_size=10):
    """
    Find the best mapping between ephys sync timestamps (TSESync) and behavioral sync timestamps (TSBSync)
    by first matching a small subset and then extending the match as much as possible.
    
    Parameters:
    - TSESync: np.ndarray, ephys synchronization timestamps (smaller array)
    - TSBSync: np.ndarray, behavioral synchronization timestamps (larger array) 
    - system_time_at_creation: float, system time when recording started (seconds)
    - tolerance: float, tolerance for interval matching (default 0.1 seconds)
    - search_window: float, time window around system_time_at_creation to search (default 300 seconds)
    - initial_subset_size: int, size of initial subset to match (default 10)
    
    Returns:
    - best_offset: int, index offset in TSBSync that best matches TSESync
    - correlation: float, correlation coefficient of the best match
    - mapping_params: dict, additional parameters about the mapping
    """
    
    if len(TSESync) < 2:
        raise ValueError("TSESync must contain at least 2 timestamps for interval matching")
    
    # Use smaller subset size if TSESync is small
    subset_size = min(initial_subset_size, len(TSESync))
    
    # Use system_time_at_creation to narrow search in TSBSync
    start_time = system_time_at_creation - search_window/2
    end_time = system_time_at_creation + search_window/2
    
    valid_indices = np.where((TSBSync >= start_time) & (TSBSync <= end_time))[0]
    
    if len(valid_indices) < subset_size:
        # Expand search if not enough candidates found
        valid_indices = np.arange(len(TSBSync))
    
    best_correlation = -1
    best_offset = -1
    best_matched_length = 0
    best_scale_factor = 1.0
    
    # Step 1: Find matches for initial subset
    TSE_subset = TSESync[:subset_size]
    TSE_subset_intervals = np.diff(TSE_subset)
    
    candidate_offsets = []
    
    # Search for initial subset matches
    max_search_idx = len(valid_indices) - subset_size
    
    for i in range(max(0, max_search_idx)):
        start_idx = valid_indices[i] if i < len(valid_indices) else i
        
        if start_idx + subset_size > len(TSBSync):
            continue
            
        # Get corresponding subset from TSBSync
        TSB_subset = TSBSync[start_idx:start_idx + subset_size]
        TSB_subset_intervals = np.diff(TSB_subset)
        
        # Try different scale factors for this subset
        for scale in [0.99, 1.0, 1.01]:
            scaled_TSB_intervals = TSB_subset_intervals * scale
            
            # Calculate correlation and mean absolute difference
            if np.std(TSE_subset_intervals) > 0 and np.std(scaled_TSB_intervals) > 0:
                correlation = np.corrcoef(TSE_subset_intervals, scaled_TSB_intervals)[0, 1]
                abs_diffs = np.abs(TSE_subset_intervals - scaled_TSB_intervals)
                mean_abs_diff = np.mean(abs_diffs)
                
                # Check if this is a good initial match
                if not np.isnan(correlation) and correlation > 0.8 and mean_abs_diff < tolerance:
                    candidate_offsets.append((start_idx, scale, correlation, mean_abs_diff))
    
    if not candidate_offsets:
        raise ValueError("No suitable initial mapping found between TSESync subset and TSBSync")
    
    # Step 2: Extend the best candidates
    for start_offset, scale_factor, initial_corr, initial_diff in candidate_offsets:
        
        # Try to extend the match in both directions
        extended_length, final_correlation = _extend_match(
            TSESync, TSBSync, start_offset, scale_factor, tolerance
        )
        
        # Evaluate this extended match
        if extended_length > best_matched_length or (
            extended_length == best_matched_length and final_correlation > best_correlation
        ):
            best_matched_length = extended_length
            best_correlation = final_correlation
            best_offset = start_offset
            best_scale_factor = scale_factor
    
    if best_offset == -1:
        raise ValueError("No suitable extended mapping found between TSESync and TSBSync")
    
    # Calculate final mapping quality metrics
    TSB_matched = TSBSync[best_offset:best_offset + best_matched_length]
    TSE_matched = TSESync[:best_matched_length]
    
    TSB_intervals_matched = np.diff(TSB_matched) * best_scale_factor
    TSE_intervals_matched = np.diff(TSE_matched)
    
    mapping_params = {
        'scale_factor': best_scale_factor,
        'time_offset': TSBSync[best_offset] - TSESync[0],
        'mean_interval_diff': np.mean(np.abs(TSE_intervals_matched - TSB_intervals_matched)),
        'max_interval_diff': np.max(np.abs(TSE_intervals_matched - TSB_intervals_matched)),
        'matched_timestamps': best_matched_length,
        'total_timestamps': len(TSESync),
        'match_fraction': best_matched_length / len(TSESync),
        'search_window_used': search_window,
        'system_time_reference': system_time_at_creation,
        'initial_subset_size': subset_size
    }
    
    return best_offset, best_correlation, mapping_params


def _extend_match(TSESync, TSBSync, start_offset, scale_factor, tolerance):
    """
    Helper function to extend a match as far as possible in both directions.
    
    Parameters:
    - TSESync: full ephys sync array
    - TSBSync: full behavioral sync array  
    - start_offset: initial matching offset in TSBSync
    - scale_factor: scale factor for the match
    - tolerance: tolerance for interval matching
    
    Returns:
    - extended_length: number of successfully matched timestamps
    - correlation: correlation of the extended match
    """
    max_possible_length = min(len(TSESync), len(TSBSync) - start_offset)
    
    # Start with minimum length and extend forward
    best_length = 2  # minimum for correlation calculation
    
    for length in range(2, max_possible_length + 1):
        TSE_segment = TSESync[:length]
        TSB_segment = TSBSync[start_offset:start_offset + length]
        
        if len(TSB_segment) != length:
            break
            
        TSE_intervals = np.diff(TSE_segment)
        TSB_intervals = np.diff(TSB_segment) * scale_factor
        
        # Check if intervals still match within tolerance
        abs_diffs = np.abs(TSE_intervals - TSB_intervals)
        mean_abs_diff = np.mean(abs_diffs)
        
        if mean_abs_diff <= tolerance:
            # Calculate correlation for this length
            if np.std(TSE_intervals) > 0 and np.std(TSB_intervals) > 0:
                correlation = np.corrcoef(TSE_intervals, TSB_intervals)[0, 1]
                if not np.isnan(correlation) and correlation > 0.7:
                    best_length = length
                else:
                    break
            else:
                break
        else:
            break
    
    # Calculate final correlation for the best length
    if best_length >= 2:
        TSE_final = TSESync[:best_length]
        TSB_final = TSBSync[start_offset:start_offset + best_length]
        TSE_intervals_final = np.diff(TSE_final)
        TSB_intervals_final = np.diff(TSB_final) * scale_factor
        
        if np.std(TSE_intervals_final) > 0 and np.std(TSB_intervals_final) > 0:
            final_correlation = np.corrcoef(TSE_intervals_final, TSB_intervals_final)[0, 1]
        else:
            final_correlation = 0.0
    else:
        final_correlation = 0.0
    
    return best_length, final_correlation

def find_sync_mapping(TSBSync, TSESync, system_time_at_creation, search_window=300, interval_tolerance=0.02, min_sequence_length=10):
    """
    Find the best linear mapping between two timestamp arrays using inter-pulse intervals.
    
    Parameters:
    TSBSync: array of behavior sync timestamps
    TSESync: array of ephys sync timestamps  
    system_time_at_creation: float, system time when recording started (seconds)
    search_window: float, time window around system_time_at_creation to search (default 300 seconds)
    interval_tolerance: tolerance for matching intervals in seconds (default 20 ms)
    min_sequence_length: minimum length of matching sequences to consider
    min_matches: minimum number of matches required for reliable mapping
    
    Returns:
    dict with 'slope', 'intercept', 'r_squared', 'n_matches', and conversion functions
    """
    
    # Convert to numpy arrays 
    TSB = np.array(TSBSync)
    TSE = np.array(TSESync) 
    
    # Calculate inter-pulse intervals (differences)
    intervals_B = np.diff(TSB)
    intervals_E = np.diff(TSE)
    
    print(f"Behavior intervals: {len(intervals_B)}, Ephys intervals: {len(intervals_E)}")
    
    def find_matching_sequence(intervals_ref, intervals_target, start_ref, direction=1):
        """
        Find matching sequence starting from start_ref position.
        direction: 1 for forward search, -1 for backward search
        """
        if direction == 1:
            seq_ref = intervals_ref[start_ref:start_ref + min_sequence_length]
        else:
            seq_ref = intervals_ref[start_ref - min_sequence_length + 1:start_ref + 1]
        
        # Search for this sequence in the target array       
        search_range = len(intervals_target) - min_sequence_length + 1
        for start_target in range(search_range):
            seq_target = intervals_target[start_target:start_target + min_sequence_length]
            match = abs(seq_ref - seq_target) <= interval_tolerance
            if sum(match) == min_sequence_length:
                return start_target
        return -1    
    
    # Use system_time_at_creation to narrow search in TSBSync
    start_time = system_time_at_creation - search_window/2
    end_time = system_time_at_creation + search_window/2
    
    valid_indices = np.where((TSBSync >= start_time) & (TSBSync <= end_time))[0]
    

    # Find matches from beginning
    start_matches = []
    start_b = valid_indices[0] if len(valid_indices) > 0 else 0

    while start_b <= len(intervals_B) - min_sequence_length:
        # print(f"Searching for start match at B index {start_b}")
        match_start_e = find_matching_sequence(intervals_B, intervals_E, start_b, direction=1)
        if match_start_e != -1:
            # Record timestamp pairs for this sequence
            for i in range(min_sequence_length):
                start_matches.append((TSE[match_start_e + i + 1], TSB[start_b + i + 1]))
            print(f"Found start sequence match: B[{start_b}] -> E[{match_start_e}]")
            
            # Try to extend the matches one by one until end of arrays or tolerance exceeded
            current_b_idx = start_b + min_sequence_length
            current_e_idx = match_start_e + min_sequence_length
            
            while (current_b_idx < len(intervals_B) and 
                   current_e_idx < len(intervals_E)):
                
                # Check if the next intervals match within tolerance
                interval_b = intervals_B[current_b_idx]
                interval_e = intervals_E[current_e_idx]
                
                if abs(interval_b - interval_e) <= interval_tolerance:
                    # Add this match (remember intervals are between timestamps, so we add +1 to indices)
                    start_matches.append((TSE[current_e_idx + 1], TSB[current_b_idx + 1]))
                    # print(f"Extended match: B[{current_b_idx + 1}] -> E[{current_e_idx + 1}], interval diff: {abs(interval_b - interval_e):.4f}")
                    current_b_idx += 1
                    current_e_idx += 1
                else:
                    print(f"Extension stopped: interval tolerance exceeded. diff: {abs(interval_b - interval_e):.4f} > {interval_tolerance}")
                    break
            
            print(f"Final extended sequence length: {len(start_matches)}")
            break
        start_b += 1
        
    # Combine all matches
    all_matches = start_matches 
    
    # Extract matched timestamps
    matched_ephys = np.array([match[0] for match in all_matches])
    matched_behavior = np.array([match[1] for match in all_matches])
    
    # Remove duplicates if any
    # unique_pairs = list(set(zip(matched_ephys, matched_behavior)))
    # matched_ephys = np.array([pair[0] for pair in unique_pairs])
    # matched_behavior = np.array([pair[1] for pair in unique_pairs])

    # print(f"Found {len(matched_ephys)} unique matching timestamp pairs")

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(matched_ephys, matched_behavior)
    
    # Create mapping dictionary
    best_mapping = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'n_matches': len(matched_ephys),
        'p_value': p_value,
        'std_err': std_err,
        'matched_ephys': matched_ephys,
        'matched_behavior': matched_behavior
    }
    print("Interpolation slope", best_mapping["slope"])

    # Add conversion functions
    def ephys_to_behavior(t_ephys):
        """Convert ephys timestamps to behavior timestamps"""
        return best_mapping['slope'] * t_ephys  + best_mapping['intercept']
    
    def behavior_to_ephys(t_behavior):
        """Convert behavior timestamps to ephys timestamps"""
        return (t_behavior - best_mapping['intercept']) / best_mapping['slope'] 

    best_mapping['ephys_to_behavior'] = ephys_to_behavior
    best_mapping['behavior_to_ephys'] = behavior_to_ephys
    
    return best_mapping


def plot_sync_results(mapping_dict, TSESync=None, TSBSync=None, figsize=(15, 10)):
    """
    Plot synchronization mapping results and statistics.
    
    Parameters:
    - mapping_dict: dict returned by find_sync_mapping
    - TSESync: optional, full ephys sync array for context
    - TSBSync: optional, full behavior sync array for context
    - figsize: tuple, figure size for plots
    
    Returns:
    - fig: matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return None
    
    # Extract data from mapping dictionary
    matched_ephys = mapping_dict['matched_ephys']
    matched_behavior = mapping_dict['matched_behavior']
    slope = mapping_dict['slope']
    intercept = mapping_dict['intercept']
    r_squared = mapping_dict['r_squared']
    n_matches = mapping_dict['n_matches']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Synchronization Mapping Results (n={n_matches} matches)', fontsize=16)
    
    # Plot 1: Scatter plot of matched timestamps with regression line
    ax1.scatter(matched_ephys, matched_behavior, alpha=0.6, color='blue', s=20)
    
    # Add regression line
    x_range = np.linspace(matched_ephys.min(), matched_ephys.max(), 100)
    y_pred = slope * x_range + intercept
    ax1.plot(x_range, y_pred, 'r--', linewidth=2, label=f'y = {slope:.6f}x + {intercept:.3f}')
    
    ax1.set_xlabel('Ephys Timestamps (s)')
    ax1.set_ylabel('Behavior Timestamps (s)')
    ax1.set_title(f'Linear Mapping (R² = {r_squared:.6f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals plot
    y_pred_matched = slope * matched_ephys + intercept
    residuals = matched_behavior - y_pred_matched
    ax2.scatter(matched_ephys, residuals, alpha=0.6, color='green', s=20)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Ephys Timestamps (s)')
    ax2.set_ylabel('Residuals (s)')
    ax2.set_title(f'Residuals (std = {np.std(residuals):.6f} s)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Interval comparison (if we have full arrays)
    if TSESync is not None and TSBSync is not None:
        # Convert ephys timestamps to behavior time for comparison
        ephys_converted = mapping_dict['ephys_to_behavior'](TSESync)
        
        # Find overlapping time range
        start_time = max(TSBSync.min(), ephys_converted.min())
        end_time = min(TSBSync.max(), ephys_converted.max())
        
        # Plot intervals in overlapping region
        tsb_overlap = TSBSync[(TSBSync >= start_time) & (TSBSync <= end_time)]
        tse_overlap = ephys_converted[(ephys_converted >= start_time) & (ephys_converted <= end_time)]
        
        if len(tsb_overlap) > 1 and len(tse_overlap) > 1:
            intervals_b = np.diff(tsb_overlap)
            intervals_e = np.diff(tse_overlap)
            
            min_len = min(len(intervals_b), len(intervals_e))
            intervals_b = intervals_b[:min_len]
            intervals_e = intervals_e[:min_len]
            
            ax3.scatter(intervals_e, intervals_b, alpha=0.6, color='purple', s=20)
            ax3.plot([0, max(intervals_e.max(), intervals_b.max())], 
                    [0, max(intervals_e.max(), intervals_b.max())], 'r--', linewidth=2)
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
    
    # Plot 4: Statistics summary
    ax4.axis('off')
    
    # Prepare statistics text
    stats_text = f"""
Synchronization Statistics:

Linear Mapping:
  Slope: {slope:.8f}
  Intercept: {intercept:.6f} s
  R²: {r_squared:.6f}
  P-value: {mapping_dict.get('p_value', 'N/A'):.2e}

Quality Metrics:
  Matched Points: {n_matches}
  Residual Std: {np.std(residuals):.6f} s
  Residual Max: {np.max(np.abs(residuals)):.6f} s
  Residual Mean: {np.mean(np.abs(residuals)):.6f} s

Time Range:
  Ephys: {matched_ephys.min():.3f} - {matched_ephys.max():.3f} s
  Behavior: {matched_behavior.min():.3f} - {matched_behavior.max():.3f} s
  Duration: {matched_ephys.max() - matched_ephys.min():.3f} s
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Print summary to console
    print(f"\n=== Synchronization Mapping Summary ===")
    print(f"Matched {n_matches} timestamp pairs")
    print(f"Linear fit: y = {slope:.8f}x + {intercept:.6f}")
    print(f"R² = {r_squared:.6f}")
    print(f"Residual std = {np.std(residuals):.6f} s")
    print(f"Time range: {matched_ephys.max() - matched_ephys.min():.3f} s")
    
    return fig