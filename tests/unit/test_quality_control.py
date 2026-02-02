"""Unit tests for quality control module."""

import pytest
import numpy as np

from habitat_pipeline.quality_control import (
    QualityMetrics,
    QualityAssessor,
    QualityReport
)


class TestQualityMetrics:
    """Test quality metrics computation."""
    
    def test_init(self):
        """Test quality metrics initialization."""
        metrics = QualityMetrics(sampling_rate=30000.0)
        assert metrics.sampling_rate == 30000.0
    
    def test_compute_snr(self):
        """Test SNR computation."""
        metrics = QualityMetrics(sampling_rate=30000.0)
        
        # Generate test data
        data = np.random.randn(4, 30000)
        
        snr = metrics.compute_snr(data)
        
        assert len(snr) == 4
        assert all(np.isfinite(snr))
    
    def test_compute_noise_level(self):
        """Test noise level computation."""
        metrics = QualityMetrics(sampling_rate=30000.0)
        
        # Generate test data with different noise levels
        data = np.random.randn(4, 30000)
        data[0, :] *= 10  # Higher noise on first channel
        
        noise_level = metrics.compute_noise_level(data)
        
        assert len(noise_level) == 4
        assert noise_level[0] > noise_level[1]  # First channel should be noisier


class TestQualityAssessor:
    """Test quality assessment."""
    
    def test_init(self):
        """Test quality assessor initialization."""
        assessor = QualityAssessor(snr_threshold=3.0)
        assert assessor.snr_threshold == 3.0
    
    def test_assess_snr(self):
        """Test SNR assessment."""
        assessor = QualityAssessor(snr_threshold=5.0)
        
        # Good SNR
        snr_good = np.array([10.0, 8.0, 12.0])
        passed, bad_channels = assessor.assess_snr(snr_good)
        assert passed is True
        assert len(bad_channels) == 0
        
        # Bad SNR
        snr_bad = np.array([10.0, 2.0, 12.0])
        passed, bad_channels = assessor.assess_snr(snr_bad)
        assert passed is False
        assert 1 in bad_channels
    
    def test_assess_all(self):
        """Test comprehensive assessment."""
        assessor = QualityAssessor()
        
        metrics = {
            'snr': np.array([10.0, 8.0, 12.0]),
            'noise_level': np.array([50.0, 60.0, 55.0]),
            'drift': np.array([20.0, 25.0, 22.0]),
            'correlation_matrix': np.eye(3)
        }
        
        results = assessor.assess_all(metrics)
        
        assert 'overall_passed' in results
        assert 'snr_passed' in results
        assert 'noise_passed' in results


class TestQualityReport:
    """Test quality report generation."""
    
    def test_init(self):
        """Test quality report initialization."""
        report = QualityReport()
        assert 'timestamp' in report.report_data
    
    def test_add_metrics(self):
        """Test adding metrics to report."""
        report = QualityReport()
        
        metrics = {
            'snr': np.array([10.0, 8.0]),
            'noise_level': np.array([50.0, 60.0])
        }
        
        report.add_metrics(metrics)
        
        assert 'snr' in report.report_data['metrics']
        assert len(report.report_data['metrics']['snr']) == 2
    
    def test_generate_summary(self):
        """Test summary generation."""
        report = QualityReport()
        
        assessment = {
            'overall_passed': True,
            'snr_passed': True,
            'noise_passed': True,
            'drift_passed': True,
            'snr_bad_channels': [],
            'noise_bad_channels': [],
            'drift_bad_channels': [],
            'bridged_pairs': []
        }
        
        report.add_assessment(assessment)
        summary = report.generate_summary()
        
        assert summary['overall_quality'] == 'PASS'
        assert summary['num_snr_bad_channels'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
