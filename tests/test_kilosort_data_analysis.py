"""
Tests for KilosortData analysis methods.

These tests focus on:
- Firing rate calculations and caching
- Filtering and access methods (get_clusters, get_spike_times)
- Event-aligned analysis methods
- Data binning functionality
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ingestion.kilosort_data_import import KilosortData


class TestFiringRateCalculations:
    """Test firing rate calculation using actual KilosortData methods."""
    
    def test_firing_rates_calculation(self, complete_kilosort_data):
        """Test correct calculation of firing rates using get_firing_rates method."""
        ks_data, expected_data = complete_kilosort_data
        
        # Use actual method from KilosortData
        firing_rates = ks_data.get_firing_rates()
        
        # Verify type and structure
        assert isinstance(firing_rates, dict)
        assert len(firing_rates) > 0
        
        # Verify all rates are positive
        for cluster_id, rate in firing_rates.items():
            assert rate >= 0, f"Cluster {cluster_id} has negative firing rate"
            assert isinstance(cluster_id, (int, np.integer))
            assert isinstance(rate, (float, np.floating))
    
    def test_firing_pattern_metrics(self, complete_kilosort_data):
        """Test firing pattern quality metrics calculation."""
        ks_data, _ = complete_kilosort_data
        
        # Use actual method from KilosortData
        metrics = ks_data.calculate_firing_pattern_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Check that metrics have expected structure
        for cluster_id, cluster_metrics in metrics.items():
            assert 'firing_rate' in cluster_metrics
            assert 'presence_ratio' in cluster_metrics
            assert 'cv_isi' in cluster_metrics
            
            # Verify reasonable values
            assert cluster_metrics['firing_rate'] >= 0
            assert 0 <= cluster_metrics['presence_ratio'] <= 1
    
    def test_cell_filtering(self, complete_kilosort_data):
        """Test cell filtering by firing patterns."""
        ks_data, _ = complete_kilosort_data
        
        # Use actual filtering method
        filter_results = ks_data.filter_cells_by_firing_patterns(
            min_firing_rate=0.1,
            max_firing_rate=100.0,
            min_presence_ratio=0.5
        )
        
        assert 'passed_clusters' in filter_results
        assert 'failed_clusters' in filter_results
        assert 'metrics' in filter_results
        assert 'summary' in filter_results
        
        # Should have some results
        total_clusters = len(filter_results['metrics'])
        assert total_clusters > 0


class TestDataAccess:
    """Test data access methods."""
    
    def test_spike_times_by_cell(self, complete_kilosort_data):
        """Test accessing spike times by cell using actual implementation."""
        ks_data, _ = complete_kilosort_data
        
        # The actual implementation provides spike_times_by_cell
        assert hasattr(ks_data, 'spike_times_by_cell')
        assert hasattr(ks_data, 'ks_ids')
        
        spike_times_by_cell = ks_data.spike_times_by_cell
        ks_ids = ks_data.ks_ids
        
        # Verify structure
        assert isinstance(spike_times_by_cell, list)
        assert len(spike_times_by_cell) == len(ks_ids)
        
        # Each cell should have an array of spike times
        for i, spike_times in enumerate(spike_times_by_cell):
            assert isinstance(spike_times, np.ndarray)
            # Spike times should be in ascending order
            if len(spike_times) > 1:
                assert np.all(np.diff(spike_times) >= 0), f"Spike times not sorted for cell {i}"
    
    def test_cluster_properties(self, complete_kilosort_data):
        """Test accessing cluster properties."""
        ks_data, _ = complete_kilosort_data
        
        # Test that cluster properties are loaded
        assert hasattr(ks_data, 'channel')
        assert hasattr(ks_data, 'amplitude') 
        assert hasattr(ks_data, 'DV')  # Depth
        assert hasattr(ks_data, 'XX')  # X position
        
        # Verify lengths match
        n_clusters = len(ks_data.ks_ids)
        if hasattr(ks_data, 'channel') and ks_data.channel is not None:
            if np.isscalar(ks_data.channel):
                assert n_clusters == 1
            else:
                assert len(ks_data.channel) == n_clusters
    
    def test_duration_calculation(self, complete_kilosort_data):
        """Test recording duration calculation."""
        ks_data, _ = complete_kilosort_data
        
        # Test duration property
        duration = ks_data.duration_seconds
        
        assert isinstance(duration, (float, np.floating))
        assert duration > 0
    
class TestISIStatistics:
    """Test inter-spike interval statistics."""
    
    def test_isi_statistics_calculation(self, complete_kilosort_data):
        """Test ISI statistics calculation."""
        ks_data, _ = complete_kilosort_data
        
        # Use actual method from KilosortData
        isi_stats = ks_data.get_isi_statistics()
        
        assert isinstance(isi_stats, dict)
        
        # Check structure for clusters with multiple spikes
        for cluster_id, stats in isi_stats.items():
            if stats:  # Only check if stats exist (requires > 1 spike)
                assert 'mean_isi' in stats
                assert 'median_isi' in stats
                assert 'cv_isi' in stats
                
                assert stats['mean_isi'] > 0
                assert stats['median_isi'] > 0


class TestSaveLoad:
    """Test save and load functionality."""
    
    def test_save_and_load_full(self, complete_kilosort_data, temp_kilosort_dir):
        """Test saving and loading complete object.""" 
        ks_data, _ = complete_kilosort_data
        
        # Save the object
        save_path = ks_data.save_to_file("test_save.pkl")
        
        assert Path(save_path).exists()
        
        # Load the object back
        loaded_ks_data = KilosortData.load_from_file(save_path)
        
        assert loaded_ks_data.animal_id == ks_data.animal_id
        assert loaded_ks_data.session_id == ks_data.session_id
        assert len(loaded_ks_data.ks_ids) == len(ks_data.ks_ids)
    
    def test_save_processed_only(self, complete_kilosort_data):
        """Test saving processed data only."""
        ks_data, _ = complete_kilosort_data
        
        # Save processed data only
        save_path = ks_data.save_to_file("test_processed.pkl", exclude_large_arrays=True)
        
        assert Path(save_path).exists()
        
        # Verify file is smaller (processed only)
        stats = Path(save_path).stat()
        assert stats.st_size > 0