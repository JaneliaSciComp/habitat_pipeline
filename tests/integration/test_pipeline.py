"""Integration tests for complete pipeline workflows."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from habitat_pipeline import (
    SignalProcessor,
    QualityMetrics,
    QualityAssessor,
    SpikeDetector,
    SpikeSorter,
    MultiAnimalCoordinator
)
from habitat_pipeline.ingestion.formats import BinaryLoader


class TestBasicPipeline:
    """Test complete single-animal pipeline."""
    
    def test_full_pipeline(self):
        """Test full pipeline from data loading to spike sorting."""
        # Generate synthetic data
        sampling_rate = 30000.0
        n_channels = 16
        duration = 1.0
        n_samples = int(sampling_rate * duration)
        
        # Create data with spikes
        data = np.random.randn(n_channels, n_samples) * 10
        
        # Add synthetic spikes
        for _ in range(20):
            ch = np.random.randint(0, n_channels)
            t = np.random.randint(1000, n_samples - 1000)
            spike = -100 * np.exp(-0.5 * ((np.arange(60) - 30) / 5)**2)
            data[ch, t:t + 60] += spike
        
        # 1. Preprocessing
        processor = SignalProcessor(sampling_rate)
        filtered = processor.bandpass_filter(data, lowcut=300, highcut=6000)
        assert filtered.shape == data.shape
        
        # 2. Quality control
        qc = QualityMetrics(sampling_rate)
        metrics = qc.compute_all_metrics(filtered)
        assert 'snr' in metrics
        assert 'noise_level' in metrics
        
        assessor = QualityAssessor()
        assessment = assessor.assess_all(metrics)
        assert 'overall_passed' in assessment
        
        # 3. Spike detection
        detector = SpikeDetector(sampling_rate, threshold_factor=4.0)
        spike_times = detector.detect_spikes(filtered)
        assert len(spike_times) > 0
        
        # 4. Extract waveforms
        waveforms = detector.extract_waveforms(filtered, spike_times)
        assert len(waveforms) > 0
        
        # 5. Spike sorting
        sorter = SpikeSorter(method='kmeans')
        labels = sorter.sort_all_channels(waveforms)
        assert len(labels) > 0


class TestMultiAnimalPipeline:
    """Test multi-animal analysis pipeline."""
    
    def test_multi_animal_coordinator(self):
        """Test multi-animal coordinator workflow."""
        # Initialize coordinator
        coordinator = MultiAnimalCoordinator(n_jobs=2)
        
        # Register test animals
        animals = ['test_001', 'test_002', 'test_003']
        
        # Create temp directories
        temp_dirs = []
        for animal_id in animals:
            temp_dir = tempfile.mkdtemp()
            temp_dirs.append(temp_dir)
            
            # Create dummy data file
            data_file = Path(temp_dir) / 'data.bin'
            dummy_data = np.random.randint(-1000, 1000, size=(1000, 4), dtype=np.int16)
            dummy_data.tofile(str(data_file))
            
            # Register animal
            coordinator.register_animal(
                animal_id=animal_id,
                data_path=str(data_file),
                metadata={'animal_id': animal_id}
            )
        
        # Check registration
        assert len(coordinator.animals) == 3
        
        # Define simple processing function
        def process_func(animal_id, data_path, metadata, **kwargs):
            # Simple analysis
            loader = BinaryLoader(
                str(data_path),
                num_channels=4,
                sampling_rate=30000.0
            )
            data, fs = loader.load_ephys()
            
            return {
                'animal_id': animal_id,
                'n_channels': data.shape[0],
                'n_samples': data.shape[1]
            }
        
        # Process all animals
        results = coordinator.process_all_animals(
            process_func,
            parallel=False  # Use sequential for testing
        )
        
        # Verify results
        assert len(results) == 3
        for animal_id in animals:
            assert animal_id in results
            assert results[animal_id]['n_channels'] == 4
        
        # Check status
        status = coordinator.get_animal_status()
        assert all(s == 'processed' for s in status.values())
        
        # Cleanup
        for temp_dir in temp_dirs:
            import shutil
            shutil.rmtree(temp_dir)


class TestSynchronization:
    """Test synchronization workflow."""
    
    def test_timestamp_alignment(self):
        """Test timestamp alignment workflow."""
        from habitat_pipeline.synchronization import TimestampAligner, SyncValidator
        
        # Create synthetic timestamps with offset and drift
        n_samples = 1000
        reference_ts = np.arange(n_samples) / 30000.0  # 30 kHz
        
        # Create stream with offset and small drift
        offset = 0.1  # 100 ms offset
        drift = 1.0001  # 0.01% drift
        stream_ts = reference_ts * drift + offset
        
        # Align
        aligner = TimestampAligner(reference_stream='reference')
        streams = {'reference': reference_ts, 'stream1': stream_ts}
        aligned = aligner.align_streams(streams)
        
        assert 'reference' in aligned
        assert 'stream1' in aligned
        assert 'stream1' in aligner.transforms
        
        # Validate
        validator = SyncValidator(max_drift_ms=10.0)
        validation = validator.validate_alignment(
            stream_ts,
            reference_ts,
            aligner.transforms['stream1']
        )
        
        assert 'passed' in validation
        assert 'mean_error_ms' in validation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
