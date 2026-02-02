"""Unit tests for preprocessing module."""

import pytest
import numpy as np

from habitat_pipeline.preprocessing import (
    SignalProcessor,
    BandpassFilter,
    NotchFilter,
    ArtifactDetector,
    ArtifactRemover,
    CommonAverageReference
)


class TestSignalProcessor:
    """Test signal processing functionality."""
    
    def test_init(self):
        """Test signal processor initialization."""
        processor = SignalProcessor(sampling_rate=30000.0)
        assert processor.sampling_rate == 30000.0
    
    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        sampling_rate = 30000.0
        processor = SignalProcessor(sampling_rate)
        
        # Generate test signal
        duration = 1.0
        t = np.arange(0, duration, 1/sampling_rate)
        
        # Signal with multiple frequencies
        signal = (
            np.sin(2 * np.pi * 100 * t) +  # Low frequency
            np.sin(2 * np.pi * 1000 * t) +  # Mid frequency
            np.sin(2 * np.pi * 10000 * t)   # High frequency
        )
        
        # Add channel dimension
        data = signal[np.newaxis, :]
        
        # Apply bandpass filter
        filtered = processor.bandpass_filter(data, lowcut=300, highcut=6000)
        
        assert filtered.shape == data.shape
        assert not np.array_equal(filtered, data)
    
    def test_notch_filter(self):
        """Test notch filtering."""
        sampling_rate = 30000.0
        processor = SignalProcessor(sampling_rate)
        
        # Generate test signal with line noise
        duration = 1.0
        t = np.arange(0, duration, 1/sampling_rate)
        signal = np.sin(2 * np.pi * 60 * t)  # 60 Hz line noise
        data = signal[np.newaxis, :]
        
        # Apply notch filter
        filtered = processor.notch_filter(data, freq=60)
        
        assert filtered.shape == data.shape
        # After notching, 60 Hz component should be reduced
        assert np.std(filtered) < np.std(data)
    
    def test_downsample(self):
        """Test downsampling."""
        sampling_rate = 30000.0
        processor = SignalProcessor(sampling_rate)
        
        # Generate test signal
        n_samples = 30000
        data = np.random.randn(4, n_samples)
        
        # Downsample by 10x
        downsampled, new_rate = processor.downsample(data, target_rate=3000.0)
        
        assert downsampled.shape[0] == data.shape[0]
        assert downsampled.shape[1] < data.shape[1]
        assert new_rate == 3000.0


class TestArtifactDetector:
    """Test artifact detection."""
    
    def test_init(self):
        """Test artifact detector initialization."""
        detector = ArtifactDetector(threshold_std=5.0)
        assert detector.threshold_std == 5.0
    
    def test_detect_amplitude_artifacts(self):
        """Test amplitude-based artifact detection."""
        detector = ArtifactDetector(threshold_std=5.0)
        
        # Generate data with artifacts
        data = np.random.randn(4, 1000) * 10
        
        # Add large amplitude artifact
        data[:, 500:510] = 500
        
        # Detect artifacts
        artifacts = detector.detect_amplitude_artifacts(data, sampling_rate=30000.0)
        
        assert len(artifacts) == data.shape[1]
        assert np.any(artifacts)  # Should detect some artifacts
    
    def test_detect_noisy_channels(self):
        """Test noisy channel detection."""
        detector = ArtifactDetector()
        
        # Generate data with one noisy channel
        data = np.random.randn(4, 1000) * 10
        data[2, :] *= 10  # Make channel 2 much noisier
        
        noisy_channels = detector.detect_noisy_channels(data)
        
        assert 2 in noisy_channels


class TestCommonAverageReference:
    """Test common average referencing."""
    
    def test_car(self):
        """Test CAR application."""
        car = CommonAverageReference()
        
        # Generate data with common noise
        common_noise = np.random.randn(1000)
        data = np.random.randn(4, 1000) + common_noise[np.newaxis, :]
        
        # Apply CAR
        referenced = car.apply(data)
        
        assert referenced.shape == data.shape
        # After CAR, mean across channels should be close to 0
        assert np.abs(np.mean(referenced, axis=0)).mean() < np.abs(np.mean(data, axis=0)).mean()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
