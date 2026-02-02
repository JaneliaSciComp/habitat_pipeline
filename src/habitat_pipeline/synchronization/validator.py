"""Validation of synchronization quality."""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SyncValidator:
    """Validate quality of synchronization between streams."""
    
    def __init__(self, max_drift_ms: float = 1.0, max_jitter_ms: float = 0.5):
        """
        Initialize sync validator.
        
        Parameters
        ----------
        max_drift_ms : float
            Maximum allowed drift in milliseconds
        max_jitter_ms : float
            Maximum allowed jitter in milliseconds
        """
        self.max_drift_ms = max_drift_ms
        self.max_jitter_ms = max_jitter_ms
    
    def validate_alignment(
        self,
        source_timestamps: np.ndarray,
        target_timestamps: np.ndarray,
        transform: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Validate timestamp alignment quality.
        
        Parameters
        ----------
        source_timestamps : np.ndarray
            Source timestamps
        target_timestamps : np.ndarray
            Target timestamps
        transform : dict
            Alignment transform
            
        Returns
        -------
        dict
            Validation results
        """
        # Apply transform
        aligned = transform['slope'] * source_timestamps + transform['intercept']
        
        # Compute residuals
        residuals = target_timestamps - aligned
        
        # Compute metrics
        mean_error = np.mean(residuals)
        std_error = np.std(residuals)
        max_error = np.max(np.abs(residuals))
        
        # Convert to milliseconds (assuming timestamps in seconds)
        mean_error_ms = mean_error * 1000
        std_error_ms = std_error * 1000
        max_error_ms = max_error * 1000
        
        # Check against thresholds
        passed = (
            abs(mean_error_ms) < self.max_drift_ms and
            std_error_ms < self.max_jitter_ms
        )
        
        results = {
            'passed': passed,
            'mean_error_ms': mean_error_ms,
            'std_error_ms': std_error_ms,
            'max_error_ms': max_error_ms,
            'rmse': transform.get('rmse', np.sqrt(np.mean(residuals**2)))
        }
        
        logger.info(f"Sync validation: passed={passed}, mean_error={mean_error_ms:.3f}ms, std={std_error_ms:.3f}ms")
        
        return results
    
    def compute_sync_quality(
        self,
        sync_signal1: np.ndarray,
        sync_signal2: np.ndarray,
        sampling_rate: float
    ) -> float:
        """
        Compute synchronization quality score.
        
        Parameters
        ----------
        sync_signal1 : np.ndarray
            First sync signal
        sync_signal2 : np.ndarray
            Second sync signal
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        float
            Quality score (0-1, higher is better)
        """
        # Ensure same length
        min_len = min(len(sync_signal1), len(sync_signal2))
        signal1 = sync_signal1[:min_len]
        signal2 = sync_signal2[:min_len]
        
        # Compute cross-correlation
        correlation = np.corrcoef(signal1, signal2)[0, 1]
        
        # Quality score based on correlation
        quality = max(0, correlation)
        
        logger.info(f"Sync quality score: {quality:.3f}")
        
        return quality
