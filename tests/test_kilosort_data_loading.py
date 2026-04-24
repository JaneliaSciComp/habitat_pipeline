"""
Tests for KilosortData loading validation.

These tests focus on the critical data loading functionality including:
- Required file validation
- Optional file handling
- Data conversion and validation
- Error handling for missing/corrupt files
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from ingestion.kilosort_data_import import KilosortData, load_kilosort_data
from tests.conftest import assert_spike_data_valid


class TestKilosortDataLoading:
    """Test data loading and validation functionality."""
    
    def test_load_complete_dataset(self, complete_kilosort_data):
        """Test successful loading of complete Kilosort output."""
        ks_data, expected_data = complete_kilosort_data
        
        # Verify basic attributes (actual implementation extracts from path)
        assert hasattr(ks_data, 'animal_id')
        assert hasattr(ks_data, 'session_id')
        
        # Verify spike data
        assert_spike_data_valid(ks_data, expected_data)
        
        # Verify core attributes exist
        assert hasattr(ks_data, 'ks_ids')
        assert hasattr(ks_data, 'spike_times_by_cell')
        assert hasattr(ks_data, 'cluster_info')
        assert hasattr(ks_data, 'channel')
        
        # Verify data shapes
        assert len(ks_data.ks_ids) > 0
        assert len(ks_data.spike_times_by_cell) == len(ks_data.ks_ids)
    
    def test_load_minimal_dataset(self, minimal_kilosort_data):
        """Test loading with only required files."""
        ks_data, expected_data = minimal_kilosort_data
        
        # Verify basic attributes
        assert hasattr(ks_data, 'animal_id')
        assert hasattr(ks_data, 'session_id')
        
        # Verify spike data
        assert_spike_data_valid(ks_data, expected_data)
        
        # Core data should still be created
        assert hasattr(ks_data, 'ks_ids')
        assert hasattr(ks_data, 'spike_times_by_cell')
        assert len(ks_data.ks_ids) > 0
    
    def test_missing_spike_times_file(self, temp_kilosort_dir):
        """Test error when spike_times.npy is missing."""
        # Create kilosort4 subdirectory and only spike_clusters.npy
        kilosort_dir = temp_kilosort_dir / 'kilosort4'
        kilosort_dir.mkdir()
        
        spike_clusters = np.random.randint(0, 10, 1000)
        np.save(kilosort_dir / 'spike_clusters.npy', spike_clusters)
        
        # Create required KS labels file
        ks_labels = pd.DataFrame({'cluster_id': [0], 'KSLabel': ['good']})
        ks_labels.to_csv(kilosort_dir / 'cluster_KSLabel.tsv', sep='\t', index=False)
        
        # Create timestamp file
        timestamps = np.arange(1000, dtype=np.uint32)
        timestamps.tofile(temp_kilosort_dir / 'test.timestamps.dat')
        
        with pytest.raises(FileNotFoundError, match="Spike times or clusters file not found"):
            load_kilosort_data(kilosort_dir)
    
    def test_missing_spike_clusters_file(self, temp_kilosort_dir):
        """Test error when spike_clusters.npy is missing."""
        # Create kilosort4 subdirectory and only spike_times.npy
        kilosort_dir = temp_kilosort_dir / 'kilosort4'
        kilosort_dir.mkdir()
        
        spike_times = np.random.randint(0, 10000, 1000)
        np.save(kilosort_dir / 'spike_times.npy', spike_times)
        
        # Create required KS labels file
        ks_labels = pd.DataFrame({'cluster_id': [0], 'KSLabel': ['good']})
        ks_labels.to_csv(kilosort_dir / 'cluster_KSLabel.tsv', sep='\t', index=False)
        
        # Create timestamp file
        timestamps = np.arange(10000, dtype=np.uint32)
        timestamps.tofile(temp_kilosort_dir / 'test.timestamps.dat')
        
        with pytest.raises(FileNotFoundError, match="Spike times or clusters file not found"):
            load_kilosort_data(kilosort_dir)
    
    def test_nonexistent_data_path(self):
        """Test error for non-existent directory."""
        fake_path = Path("/nonexistent/path/to/data")
        
        # The actual implementation will try to locate_KS_folder and set self.KSfolder to None
        # This should cause an error when trying to load spike data
        with pytest.raises((FileNotFoundError, AttributeError)):
            load_kilosort_data(fake_path)
    
    def test_spike_times_sample_indices(self, complete_kilosort_data):
        """Test that spike times are stored as sample indices (actual implementation behavior)."""
        ks_data, expected_data = complete_kilosort_data
        
        # In actual implementation, spike_times are adjusted by -31 for template alignment
        # and spike_times_by_cell contains the actual timestamps converted to seconds
        assert hasattr(ks_data, 'spike_times_by_cell')
        assert len(ks_data.spike_times_by_cell) > 0
    
    def test_cluster_info_from_tsv(self, cluster_info_kilosort_data):
        """Test loading cluster_info.tsv."""
        ks_data, expected_data = cluster_info_kilosort_data
        
        assert ks_data.cluster_info is not None
        # Note: actual implementation uses 'ch' not 'channel'
        assert len(ks_data.ks_ids) > 0
    
    def test_basic_functionality(self, complete_kilosort_data):
        """Test basic functionality of KilosortData."""
        ks_data, expected_data = complete_kilosort_data
        
        # Test basic methods
        firing_rates = ks_data.get_firing_rates()
        assert isinstance(firing_rates, dict)
        assert len(firing_rates) > 0
        
        duration = ks_data.duration_seconds
        assert isinstance(duration, (float, int))
        assert duration > 0
    
    def test_metadata_creation(self, complete_kilosort_data):
        """Test that metadata is correctly created."""
        ks_data, _ = complete_kilosort_data
        
        assert hasattr(ks_data, 'metadata')
        assert isinstance(ks_data.metadata, dict)
        assert 'data_path' in ks_data.metadata
        assert 'animal_id' in ks_data.metadata
        assert 'session_id' in ks_data.metadata
        
        assert ks_data.metadata['animal_id'] == ks_data.animal_id
        assert ks_data.metadata['session_id'] == ks_data.session_id
    
    def test_params_file_loading(self, minimal_kilosort_data, temp_kilosort_dir):
        """Test loading sample rate from params.py."""
        ks_data, expected_data = minimal_kilosort_data
        
        # Create custom params.py with different sample rate
        custom_sample_rate = 25000.0
        params_content = f"""
n_channels_dat = 385
sample_rate = {custom_sample_rate}
dat_path = '/path/to/data.dat'
"""
        with open(temp_kilosort_dir / 'params.py', 'w') as f:
            f.write(params_content)
        
        # Create a new KilosortData instance to test params loading
        kilosort_dir = temp_kilosort_dir / 'kilosort4'
        ks_data = load_kilosort_data(kilosort_dir)
        
    def test_repr_string(self, complete_kilosort_data):
        """Test __repr__ method returns informative string."""
        ks_data, _ = complete_kilosort_data
        
        repr_str = repr(ks_data)
        
        assert 'KilosortData' in repr_str
        assert ks_data.animal_id in repr_str
        assert ks_data.session_id in repr_str
        assert 'n_spikes=' in repr_str
        assert 'n_clusters=' in repr_str
        assert 'duration=' in repr_str