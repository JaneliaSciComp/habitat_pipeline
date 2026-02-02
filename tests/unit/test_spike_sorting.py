"""Unit tests for spike sorting module."""

import pytest
import numpy as np

from habitat_pipeline.spike_sorting import (
    SpikeDetector,
    SpikeSorter,
    FeatureExtractor,
    WaveformAnalyzer
)


class TestSpikeDetector:
    """Test spike detection functionality."""
    
    def test_init(self):
        """Test spike detector initialization."""
        detector = SpikeDetector(sampling_rate=30000.0, threshold_factor=4.0)
        assert detector.sampling_rate == 30000.0
        assert detector.threshold_factor == 4.0
    
    def test_estimate_threshold(self):
        """Test threshold estimation."""
        detector = SpikeDetector(sampling_rate=30000.0, threshold_method='mad')
        
        data = np.random.randn(1000)
        threshold = detector.estimate_threshold(data)
        
        assert threshold > 0
        assert np.isfinite(threshold)
    
    def test_detect_spikes(self):
        """Test spike detection."""
        detector = SpikeDetector(sampling_rate=30000.0, threshold_factor=4.0)
        
        # Generate data with synthetic spikes
        data = np.random.randn(4, 10000) * 10
        
        # Add spikes
        for ch in range(4):
            for spike_pos in [1000, 3000, 5000]:
                spike = -100 * np.exp(-0.5 * ((np.arange(60) - 30) / 5)**2)
                data[ch, spike_pos:spike_pos + 60] += spike
        
        spike_times = detector.detect_spikes(data)
        
        assert len(spike_times) == 4
        for ch in range(4):
            assert len(spike_times[ch]) > 0  # Should detect some spikes


class TestSpikeSorter:
    """Test spike sorting functionality."""
    
    def test_init(self):
        """Test spike sorter initialization."""
        sorter = SpikeSorter(method='kmeans', n_clusters=3)
        assert sorter.method == 'kmeans'
        assert sorter.n_clusters == 3
    
    def test_extract_features(self):
        """Test feature extraction."""
        sorter = SpikeSorter(n_features=3)
        
        # Generate synthetic waveforms
        n_spikes = 100
        n_samples = 60
        waveforms = np.random.randn(n_spikes, n_samples)
        
        features = sorter.extract_features(waveforms)
        
        assert features.shape == (n_spikes, 3)
    
    def test_cluster(self):
        """Test clustering."""
        sorter = SpikeSorter(method='kmeans', n_clusters=3)
        
        # Generate features with clear clusters
        features = np.vstack([
            np.random.randn(30, 3) + [0, 0, 0],
            np.random.randn(30, 3) + [5, 0, 0],
            np.random.randn(30, 3) + [0, 5, 0]
        ])
        
        labels = sorter.cluster(features)
        
        assert len(labels) == 90
        assert len(np.unique(labels)) <= 3


class TestFeatureExtractor:
    """Test feature extraction."""
    
    def test_pca_features(self):
        """Test PCA feature extraction."""
        extractor = FeatureExtractor(method='pca', n_components=3)
        
        waveforms = np.random.randn(100, 60)
        features = extractor.extract(waveforms)
        
        assert features.shape == (100, 3)
    
    def test_peak_features(self):
        """Test peak-based feature extraction."""
        extractor = FeatureExtractor(method='peak')
        
        waveforms = np.random.randn(100, 60)
        features = extractor.extract(waveforms)
        
        assert features.shape == (100, 3)


class TestWaveformAnalyzer:
    """Test waveform analysis."""
    
    def test_mean_waveform(self):
        """Test mean waveform computation."""
        analyzer = WaveformAnalyzer(sampling_rate=30000.0)
        
        waveforms = np.random.randn(100, 60)
        mean_wf = analyzer.compute_mean_waveform(waveforms)
        
        assert len(mean_wf) == 60
    
    def test_snr_computation(self):
        """Test SNR computation."""
        analyzer = WaveformAnalyzer(sampling_rate=30000.0)
        
        # Create waveforms with consistent shape
        base_waveform = -100 * np.exp(-0.5 * ((np.arange(60) - 30) / 5)**2)
        waveforms = base_waveform[np.newaxis, :] + np.random.randn(100, 60) * 5
        
        snr = analyzer.compute_snr(waveforms)
        
        assert snr > 0  # Should have positive SNR


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
