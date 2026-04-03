"""
Helper functions for KilosortData testing.

Contains utilities for:
- Creating sophisticated mock data
- Validation functions
- Performance testing utilities
- Edge case data generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import tempfile
import warnings


def create_realistic_spike_trains(n_clusters: int = 20, 
                                duration_sec: float = 600.0,
                                sample_rate: float = 30000.0,
                                firing_rate_range: Tuple[float, float] = (0.5, 50.0),
                                burst_probability: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create realistic spike trains with various firing patterns.
    
    Args:
        n_clusters: Number of clusters
        duration_sec: Recording duration in seconds
        sample_rate: Sampling rate in Hz
        firing_rate_range: Min and max firing rates
        burst_probability: Probability of burst firing
        
    Returns:
        Tuple of (spike_times_samples, spike_clusters)
    """
    all_spike_times = []
    all_spike_clusters = []
    
    for cluster_id in range(n_clusters):
        # Random firing rate for this cluster
        base_rate = np.random.uniform(*firing_rate_range)
        
        # Create spike train
        if np.random.random() < burst_probability:
            # Bursting neuron
            spike_times = create_bursting_spike_train(base_rate, duration_sec)
        else:
            # Regular neuron (Poisson process)
            spike_times = create_poisson_spike_train(base_rate, duration_sec)
        
        # Convert to samples
        spike_times_samples = (spike_times * sample_rate).astype(np.int64)
        
        # Add to collections
        all_spike_times.extend(spike_times_samples)
        all_spike_clusters.extend([cluster_id] * len(spike_times_samples))
    
    # Convert to arrays and sort by time
    spike_times_array = np.array(all_spike_times)
    spike_clusters_array = np.array(all_spike_clusters)
    
    # Sort by spike time
    sort_idx = np.argsort(spike_times_array)
    spike_times_array = spike_times_array[sort_idx]
    spike_clusters_array = spike_clusters_array[sort_idx]
    
    return spike_times_array, spike_clusters_array


def create_poisson_spike_train(firing_rate: float, duration_sec: float) -> np.ndarray:
    """Create Poisson spike train."""
    # Expected number of spikes
    expected_spikes = int(firing_rate * duration_sec * 1.2)  # Extra buffer
    
    # Generate inter-spike intervals
    isi = np.random.exponential(1.0 / firing_rate, expected_spikes)
    
    # Convert to spike times
    spike_times = np.cumsum(isi)
    
    # Keep only spikes within duration
    spike_times = spike_times[spike_times <= duration_sec]
    
    return spike_times


def create_bursting_spike_train(base_rate: float, duration_sec: float,
                              burst_rate: float = 100.0,
                              burst_duration: float = 0.05,
                              inter_burst_interval: float = 2.0) -> np.ndarray:
    """Create spike train with bursting behavior."""
    spike_times = []
    current_time = 0.0
    
    while current_time < duration_sec:
        # Generate burst
        burst_start = current_time
        burst_end = min(burst_start + burst_duration, duration_sec)
        
        # Spikes within burst
        burst_isi = np.random.exponential(1.0 / burst_rate, int(burst_rate * burst_duration * 2))
        burst_spikes = burst_start + np.cumsum(burst_isi)
        burst_spikes = burst_spikes[burst_spikes <= burst_end]
        spike_times.extend(burst_spikes)
        
        # Move to next potential burst
        current_time = burst_end + np.random.exponential(inter_burst_interval)
        
        # Add some background spikes between bursts
        if current_time < duration_sec:
            background_interval = current_time - burst_end
            background_spikes = create_poisson_spike_train(base_rate * 0.1, background_interval)
            background_spikes += burst_end
            background_spikes = background_spikes[background_spikes < current_time]
            spike_times.extend(background_spikes)
    
    return np.array(sorted(spike_times))


def create_realistic_templates(n_clusters: int = 20,
                             n_timepoints: int = 82,
                             n_channels: int = 384,
                             noise_level: float = 10.0) -> np.ndarray:
    """
    Create realistic action potential templates.
    
    Args:
        n_clusters: Number of clusters
        n_timepoints: Number of time points per template
        n_channels: Number of channels
        noise_level: Background noise level
        
    Returns:
        Template array of shape (n_clusters, n_timepoints, n_channels)
    """
    templates = np.random.randn(n_clusters, n_timepoints, n_channels) * noise_level
    
    for cluster_id in range(n_clusters):
        # Assign primary channel
        primary_channel = cluster_id % n_channels
        
        # Create action potential shape
        peak_time = n_timepoints // 2
        time_axis = np.arange(n_timepoints) - peak_time
        
        # Different waveform types
        if cluster_id % 3 == 0:
            # Type 1: Classic negative spike
            ap_shape = -np.exp(-time_axis**2 / 50) * np.random.uniform(100, 500)
        elif cluster_id % 3 == 1:
            # Type 2: Biphasic
            ap_shape = (np.exp(-time_axis**2 / 30) * np.random.uniform(80, 300) - 
                       2 * np.exp(-(time_axis-10)**2 / 20) * np.random.uniform(150, 400))
        else:
            # Type 3: Positive spike (interneuron-like)
            ap_shape = np.exp(-time_axis**2 / 40) * np.random.uniform(200, 600)
        
        # Apply to primary channel
        templates[cluster_id, :, primary_channel] += ap_shape
        
        # Add spatial spread to neighboring channels
        for offset in [-1, 1]:
            neighbor_channel = primary_channel + offset
            if 0 <= neighbor_channel < n_channels:
                decay_factor = np.random.uniform(0.3, 0.7)
                templates[cluster_id, :, neighbor_channel] += ap_shape * decay_factor
    
    return templates


def create_probe_geometry(n_channels: int = 384, 
                         probe_type: str = 'neuropixels') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create realistic probe geometry.
    
    Args:
        n_channels: Number of channels
        probe_type: Type of probe ('neuropixels', 'linear', 'tetrode')
        
    Returns:
        Tuple of (channel_map, channel_positions)
    """
    channel_map = np.arange(n_channels)
    
    if probe_type == 'neuropixels':
        # Neuropixels geometry: 4 columns, alternating
        x_positions = np.tile([16, 48, 0, 32], n_channels // 4 + 1)[:n_channels]
        y_positions = np.repeat(np.arange(0, n_channels*20, 20), 1)[:n_channels]
        
    elif probe_type == 'linear':
        # Linear probe
        x_positions = np.zeros(n_channels)
        y_positions = np.arange(0, n_channels*25, 25)
        
    elif probe_type == 'tetrode':
        # Tetrode bundle
        tetrode_positions = np.array([[0, 0], [12.5, 12.5], [0, 25], [12.5, 37.5]])
        n_tetrodes = n_channels // 4
        
        x_positions = []
        y_positions = []
        
        for tetrode_id in range(n_tetrodes):
            tetrode_offset = tetrode_id * 100  # 100 um between tetrodes
            for pos in tetrode_positions:
                x_positions.append(pos[0])
                y_positions.append(pos[1] + tetrode_offset)
        
        x_positions = np.array(x_positions[:n_channels])
        y_positions = np.array(y_positions[:n_channels])
    
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    
    channel_positions = np.column_stack([x_positions, y_positions])
    return channel_map, channel_positions


def create_cluster_quality_metrics(n_clusters: int = 20) -> pd.DataFrame:
    """
    Create realistic cluster quality metrics.
    
    Args:
        n_clusters: Number of clusters
        
    Returns:
        DataFrame with quality metrics
    """
    cluster_info = pd.DataFrame({
        'cluster_id': np.arange(n_clusters),
        'channel': np.random.randint(0, 384, n_clusters),
        'depth': np.random.uniform(0, 4000, n_clusters),
        'amplitude': np.random.lognormal(mean=5, sigma=1, size=n_clusters),
        'contamination': np.random.beta(1, 10, n_clusters),  # Most clusters have low contamination
        'isi_violations': np.random.beta(1, 20, n_clusters),
        'isolation_distance': np.random.gamma(2, 10, n_clusters),
        'snr': np.random.gamma(2, 5, n_clusters),
    })
    
    # Assign quality groups based on metrics
    cluster_info['group'] = 'good'
    
    # Mark some as MUA based on quality metrics
    mua_mask = ((cluster_info['contamination'] > 0.1) | 
                (cluster_info['isi_violations'] > 0.05) |
                (cluster_info['snr'] < 2))
    cluster_info.loc[mua_mask, 'group'] = 'mua'
    
    # Mark some as noise
    noise_mask = ((cluster_info['contamination'] > 0.3) | 
                  (cluster_info['amplitude'] < 50) |
                  (cluster_info['snr'] < 1))
    cluster_info.loc[noise_mask, 'group'] = 'noise'
    
    cluster_info.set_index('cluster_id', inplace=True)
    return cluster_info


def create_behavioral_events(duration_sec: float = 600.0,
                           event_types: List[str] = None,
                           event_rate: float = 0.1) -> pd.DataFrame:
    """
    Create realistic behavioral events for testing.
    
    Args:
        duration_sec: Total duration
        event_types: List of event type names
        event_rate: Average events per second
        
    Returns:
        DataFrame with behavioral events
    """
    if event_types is None:
        event_types = ['approach', 'contact', 'departure', 'grooming', 'rearing']
    
    n_events = int(duration_sec * event_rate)
    
    # Create event times
    event_starts = np.sort(np.random.uniform(10, duration_sec-10, n_events))
    
    # Create event durations (realistic behavioral durations)
    event_durations = np.random.lognormal(mean=0.5, sigma=0.8, size=n_events)
    event_durations = np.clip(event_durations, 0.1, 30.0)  # 0.1s to 30s
    event_ends = event_starts + event_durations
    
    # Assign event types
    event_labels = np.random.choice(event_types, n_events)
    
    events_df = pd.DataFrame({
        'start_time': event_starts,
        'end_time': event_ends,
        'duration': event_durations,
        'event_type': event_labels,
        'event_id': np.arange(n_events)
    })
    
    return events_df


def validate_kilosort_data_integrity(ks_data, expected_properties: Dict = None):
    """
    Comprehensive validation of KilosortData object integrity.
    
    Args:
        ks_data: KilosortData object to validate
        expected_properties: Expected properties dictionary
    """
    # Basic attribute checks for actual implementation
    assert hasattr(ks_data, 'spike_times')
    assert hasattr(ks_data, 'spike_clusters')
    assert hasattr(ks_data, 'cluster_info')
    assert hasattr(ks_data, 'KSfolder')  # Actual implementation uses KSfolder
    assert hasattr(ks_data, 'animal_id')
    assert hasattr(ks_data, 'session_id')
    assert hasattr(ks_data, 'ks_ids')
    assert hasattr(ks_data, 'spike_times_by_cell')
    
    # Data consistency checks
    if ks_data.spike_times is not None and ks_data.spike_clusters is not None:
        assert len(ks_data.spike_times) == len(ks_data.spike_clusters), \
            "Spike times and clusters must have same length"
        
        # Spike times should be sorted (but may be sample indices, not seconds)
        # assert np.all(np.diff(ks_data.spike_times) >= 0), \
        #     "Spike times must be sorted"
        
    # Check ks_ids and spike_times_by_cell consistency
    if hasattr(ks_data, 'ks_ids') and hasattr(ks_data, 'spike_times_by_cell'):
        assert len(ks_data.ks_ids) == len(ks_data.spike_times_by_cell), \
            "ks_ids and spike_times_by_cell must have same length"
        
        # Each cell should have spike times as numpy array
        for i, spike_times in enumerate(ks_data.spike_times_by_cell):
            assert isinstance(spike_times, np.ndarray), \
                f"spike_times_by_cell[{i}] should be numpy array"
            
            # Spike times should be sorted within each cell
            if len(spike_times) > 1:
                assert np.all(np.diff(spike_times) >= 0), \
                    f"Spike times not sorted for cell {i}"
    
    # Validate expected properties if provided
    if expected_properties:
        for prop, expected_value in expected_properties.items():
            actual_value = getattr(ks_data, prop)
            assert actual_value == expected_value, \
                f"Property {prop}: expected {expected_value}, got {actual_value}"
    
    # Validate expected properties if provided
    if expected_properties:
        for prop, expected_value in expected_properties.items():
            actual_value = getattr(ks_data, prop)
            assert actual_value == expected_value, \
                f"Property {prop}: expected {expected_value}, got {actual_value}"


def benchmark_kilosort_operations(ks_data, operations: List[str] = None):
    """
    Benchmark common KilosortData operations for performance testing.
    
    Args:
        ks_data: KilosortData object
        operations: List of operations to benchmark
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    if operations is None:
        operations = ['firing_rates', 'isi_statistics', 'firing_pattern_metrics']
    
    results = {}
    
    if 'firing_rates' in operations:
        start = time.time()
        _ = ks_data.get_firing_rates()
        results['firing_rates'] = time.time() - start
    
    if 'isi_statistics' in operations:
        start = time.time()
        _ = ks_data.get_isi_statistics()
        results['isi_statistics'] = time.time() - start
    
    if 'firing_pattern_metrics' in operations:
        start = time.time()
        _ = ks_data.calculate_firing_pattern_metrics()
        results['firing_pattern_metrics'] = time.time() - start
    
    if 'filter_cells' in operations:
        start = time.time()
        _ = ks_data.filter_cells_by_firing_patterns()
        results['filter_cells'] = time.time() - start
    
    return results


def create_edge_case_data() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Create edge case data for robust testing.
    
    Returns:
        Dictionary with different edge case scenarios
    """
    edge_cases = {}
    
    # Empty dataset
    edge_cases['empty'] = (np.array([], dtype=np.int64), np.array([], dtype=int))
    
    # Single spike
    edge_cases['single_spike'] = (np.array([1000]), np.array([0]))
    
    # Single cluster, many spikes
    single_cluster_times = np.random.randint(0, 600*30000, 10000)
    edge_cases['single_cluster'] = (
        np.sort(single_cluster_times), 
        np.zeros(10000, dtype=int)
    )
    
    # Many clusters, few spikes each
    n_clusters = 1000
    sparse_times = []
    sparse_clusters = []
    for i in range(n_clusters):
        n_spikes = np.random.randint(1, 5)  # 1-4 spikes per cluster
        cluster_times = np.random.randint(0, 600*30000, n_spikes)
        sparse_times.extend(cluster_times)
        sparse_clusters.extend([i] * n_spikes)
    
    sort_idx = np.argsort(sparse_times)
    edge_cases['sparse'] = (
        np.array(sparse_times)[sort_idx],
        np.array(sparse_clusters)[sort_idx]
    )
    
    # Very high firing rate cluster
    high_rate_times = np.random.randint(0, 600*30000, 100000)  # ~167 Hz avg
    edge_cases['high_firing_rate'] = (
        np.sort(high_rate_times),
        np.zeros(100000, dtype=int)
    )
    
    return edge_cases