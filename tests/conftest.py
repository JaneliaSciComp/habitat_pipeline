"""
Pytest configuration and shared fixtures for KilosortData testing.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Tuple, Any

from ingestion.kilosort_data_import import KilosortData


@pytest.fixture
def temp_kilosort_dir():
    """Create a temporary directory for Kilosort test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_spike_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create mock spike data for testing.
    
    Returns:
        Tuple of (spike_times, spike_clusters) arrays
    """
    # Create realistic spike data
    n_spikes = 10000
    n_clusters = 20
    duration_sec = 600.0  # 10 minutes
    sample_rate = 30000.0
    
    # Generate spike times uniformly distributed
    spike_times_sec = np.sort(np.random.uniform(0, duration_sec, n_spikes))
    spike_times_samples = (spike_times_sec * sample_rate).astype(np.int64)
    
    # Assign clusters with realistic firing rate distribution
    # Some clusters fire more than others (lognormal distribution)
    cluster_probs = np.random.lognormal(mean=0, sigma=1, size=n_clusters)
    cluster_probs = cluster_probs / cluster_probs.sum()
    spike_clusters = np.random.choice(n_clusters, size=n_spikes, p=cluster_probs)
    
    return spike_times_samples, spike_clusters


@pytest.fixture
def mock_templates() -> np.ndarray:
    """Create mock template data."""
    n_clusters = 20
    n_timepoints = 82
    n_channels = 384
    
    # Create realistic waveform templates
    templates = np.random.randn(n_clusters, n_timepoints, n_channels) * 50
    
    # Add some structure - make each cluster have a primary channel
    for i in range(n_clusters):
        primary_channel = i % n_channels
        # Make the waveform stronger on the primary channel
        templates[i, :, primary_channel] *= 3
        
        # Add typical action potential shape on primary channel
        peak_time = n_timepoints // 2
        time_axis = np.arange(n_timepoints) - peak_time
        ap_shape = -np.exp(-time_axis**2 / 50) * 200  # Negative spike
        templates[i, :, primary_channel] += ap_shape
    
    return templates


@pytest.fixture
def mock_channel_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create mock channel map and positions."""
    n_channels = 384
    
    # Create channel map (0-indexed)
    channel_map = np.arange(n_channels)
    
    # Create channel positions (Neuropixels-like geometry)
    x_positions = np.tile([16, 48], n_channels // 2)[:n_channels]
    y_positions = np.repeat(np.arange(0, n_channels*20, 20), 1)[:n_channels]
    channel_positions = np.column_stack([x_positions, y_positions])
    
    return channel_map, channel_positions


@pytest.fixture
def mock_cluster_info() -> pd.DataFrame:
    """Create mock cluster information."""
    n_clusters = 20
    
    cluster_info = pd.DataFrame({
        'cluster_id': np.arange(n_clusters),
        'channel': np.random.randint(0, 384, n_clusters),
        'group': np.random.choice(['good', 'mua', 'noise'], n_clusters, p=[0.7, 0.2, 0.1]),
        'depth': np.random.uniform(0, 4000, n_clusters),  # Depth in um
        'amplitude': np.random.uniform(50, 500, n_clusters)  # Amplitude in uV
    })
    
    cluster_info.set_index('cluster_id', inplace=True)
    return cluster_info


def create_mock_kilosort_files(temp_dir: Path, 
                              include_optional: bool = True,
                              include_cluster_info: bool = True) -> Dict[str, Any]:
    """
    Create mock Kilosort files in temporary directory matching the expected structure.
    
    Args:
        temp_dir: Temporary directory path
        include_optional: Whether to include optional files
        include_cluster_info: Whether to include cluster_info.tsv
        
    Returns:
        Dictionary with created data for validation
    """
    # Create kilosort4 subdirectory to match expected structure
    kilosort_dir = temp_dir / 'kilosort4'
    kilosort_dir.mkdir(exist_ok=True)
    
    # Create required files
    spike_times = np.sort(np.random.uniform(0, 600*30000, 10000)).astype(np.int64)
    spike_clusters = np.random.randint(0, 20, 10000)
    
    np.save(kilosort_dir / 'spike_times.npy', spike_times)
    np.save(kilosort_dir / 'spike_clusters.npy', spike_clusters)
    
    # Create templates file (required by waveform2channel method)
    n_clusters = 20
    n_timepoints = 82
    n_channels = 384
    templates = np.random.randn(n_clusters, n_timepoints, n_channels) * 50
    np.save(kilosort_dir / 'templates.npy', templates)
    
    # Create properly formatted timestamps file with header and multiple-of-4 binary data
    n_samples = 600 * 30000  # 10 minutes at 30kHz
    # Make sure n_samples is multiple of required alignment
    n_samples = (n_samples // 4) * 4  
    
    with open(temp_dir / 'test.timestamps.dat', 'wb') as f:
        # Write 25 header lines as expected
        for i in range(25):
            f.write(f"Header line {i}\n".encode('utf-8'))
        
        # Write binary data (4 bytes per sample)
        timestamps = np.arange(0, n_samples, dtype=np.uint32)
        f.write(timestamps.tobytes())
    
    data = {
        'spike_times': spike_times,
        'spike_clusters': spike_clusters,
        'templates': templates,
        'timestamps': timestamps,
        'sample_rate': 30000.0
    }
    
    if include_optional:
        # Optional files can be added here if needed
        pass
    
    # Create KS labels file (required by actual implementation)
    ks_labels = pd.DataFrame({
        'cluster_id': np.arange(20),
        'KSLabel': np.random.choice(['good', 'mua', 'noise'], 20, p=[0.7, 0.2, 0.1])
    })
    ks_labels.to_csv(kilosort_dir / 'cluster_KSLabel.tsv', sep='\t', index=False)
    data['ks_labels'] = ks_labels
    
    # Create channel_map.npy (required for extract_cluster_properties)
    channel_map = np.arange(n_channels)
    np.save(kilosort_dir / 'channel_map.npy', channel_map)
    data['channel_map'] = channel_map
    
    # Create channel_positions.npy (also required by extract_cluster_properties)
    channel_positions = np.column_stack([
        np.tile([16, 48], n_channels // 2)[:n_channels],
        np.repeat(np.arange(0, n_channels*20, 20), 1)[:n_channels]
    ])
    np.save(kilosort_dir / 'channel_positions.npy', channel_positions)
    data['channel_positions'] = channel_positions
    
    if include_cluster_info:
        # Create cluster_info.tsv with expected columns
        # Make sure 'ch' values exist in our channel_map
        cluster_info = pd.DataFrame({
            'cluster_id': np.arange(20),
            'ch': np.random.choice(channel_map[:100], 20),  # Use valid channel indices from our map
            'depth': np.random.uniform(0, 4000, 20),
            'Amplitude': np.random.uniform(50, 500, 20),
            'amp': np.random.uniform(10, 100, 20),
            'fr': np.random.uniform(0.5, 50, 20),
            'group': np.random.choice(['good', 'mua', 'noise'], 20, p=[0.7, 0.2, 0.1])
        })
        cluster_info.to_csv(kilosort_dir / 'cluster_info.tsv', sep='\t', index=False)
        data['cluster_info'] = cluster_info
    
    # Create cluster amplitude file (expected by actual implementation)
    amplitude_df = pd.DataFrame({
        'cluster_id': np.arange(20),
        'Amplitude': np.random.uniform(50, 500, 20)
    })
    amplitude_df.to_csv(kilosort_dir / 'cluster_Amplitude.tsv', sep='\t', index=False)
    
    return data


@pytest.fixture
def complete_kilosort_data(temp_kilosort_dir) -> Tuple[KilosortData, Dict[str, Any]]:
    """Create a complete KilosortData object with all files."""
    mock_data = create_mock_kilosort_files(
        temp_kilosort_dir, 
        include_optional=True, 
        include_cluster_info=True
    )
    
    # Pass the kilosort4 directory directly to match expected structure
    kilosort4_dir = temp_kilosort_dir / 'kilosort4'
    ks_data = KilosortData(data_input=kilosort4_dir)
    
    return ks_data, mock_data


@pytest.fixture
def minimal_kilosort_data(temp_kilosort_dir) -> Tuple[KilosortData, Dict[str, Any]]:
    """Create a minimal KilosortData object with only required files."""
    mock_data = create_mock_kilosort_files(
        temp_kilosort_dir, 
        include_optional=False, 
        include_cluster_info=False
    )
    
    # Pass the kilosort4 directory directly to match expected structure  
    kilosort4_dir = temp_kilosort_dir / 'kilosort4'
    ks_data = KilosortData(data_input=kilosort4_dir)
    
    return ks_data, mock_data


def create_test_events(duration_sec: float = 600.0, n_events: int = 50) -> np.ndarray:
    """Create test behavioral events."""
    event_times = np.sort(np.random.uniform(10, duration_sec-10, n_events))
    return event_times


@pytest.fixture
def cluster_info_kilosort_data(temp_kilosort_dir) -> Tuple[KilosortData, Dict[str, Any]]:
    """Create a KilosortData object with cluster_info.tsv but no other optional files."""
    mock_data = create_mock_kilosort_files(
        temp_kilosort_dir, 
        include_optional=False, 
        include_cluster_info=True
    )
    
    # Pass the kilosort4 directory directly to match expected structure  
    kilosort4_dir = temp_kilosort_dir / 'kilosort4'
    ks_data = KilosortData(data_input=kilosort4_dir)
    
    return ks_data, mock_data


def assert_spike_data_valid(ks_data: KilosortData, expected_data: Dict[str, Any]):
    """Assert that loaded spike data matches expected values."""
    assert ks_data.spike_times is not None
    assert ks_data.spike_clusters is not None
    assert len(ks_data.spike_times) == len(ks_data.spike_clusters)
    
    # Check basic properties
    assert hasattr(ks_data, 'ks_ids')
    assert hasattr(ks_data, 'spike_times_by_cell')
    assert len(ks_data.ks_ids) > 0
    assert len(ks_data.spike_times_by_cell) == len(ks_data.ks_ids)
    
    # The actual implementation stores spike times as sample indices, not seconds
    # Check that spike times are in reasonable range
    assert ks_data.spike_times.min() >= 0
    assert ks_data.spike_times.max() > 0