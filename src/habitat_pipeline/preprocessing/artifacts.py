"""Artifact detection and removal."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ArtifactDetector:
    """
    Detect artifacts in electrophysiology data.
    
    Uses multiple methods to identify bad segments and channels.
    """
    
    def __init__(self, threshold_std: float = 5.0, window_size: float = 1.0):
        """
        Initialize artifact detector.
        
        Parameters
        ----------
        threshold_std : float
            Threshold in standard deviations for artifact detection
        window_size : float
            Window size in seconds for artifact detection
        """
        self.threshold_std = threshold_std
        self.window_size = window_size
    
    def detect_amplitude_artifacts(
        self,
        data: np.ndarray,
        sampling_rate: float
    ) -> np.ndarray:
        """
        Detect artifacts based on amplitude threshold.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        np.ndarray
            Boolean mask indicating artifact samples (n_samples,)
        """
        logger.info("Detecting amplitude artifacts")
        
        # Calculate threshold based on robust statistics
        median_abs = np.median(np.abs(data), axis=1, keepdims=True)
        threshold = self.threshold_std * median_abs
        
        # Find samples exceeding threshold in any channel
        artifacts = np.any(np.abs(data) > threshold, axis=0)
        
        # Expand artifacts to surrounding samples
        window_samples = int(self.window_size * sampling_rate)
        artifacts_expanded = np.convolve(
            artifacts.astype(float),
            np.ones(window_samples) / window_samples,
            mode='same'
        ) > 0
        
        logger.info(f"Detected {np.sum(artifacts_expanded)} artifact samples ({100*np.mean(artifacts_expanded):.2f}%)")
        
        return artifacts_expanded
    
    def detect_noisy_channels(
        self,
        data: np.ndarray,
        threshold: float = 3.0
    ) -> List[int]:
        """
        Detect noisy channels based on signal statistics.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        threshold : float
            Threshold in standard deviations
            
        Returns
        -------
        list of int
            Indices of noisy channels
        """
        logger.info("Detecting noisy channels")
        
        # Calculate RMS for each channel
        rms = np.sqrt(np.mean(data**2, axis=1))
        
        # Find outlier channels
        median_rms = np.median(rms)
        mad = np.median(np.abs(rms - median_rms))
        threshold_value = median_rms + threshold * mad * 1.4826  # MAD to std conversion
        
        noisy_channels = np.where(rms > threshold_value)[0].tolist()
        
        logger.info(f"Detected {len(noisy_channels)} noisy channels: {noisy_channels}")
        
        return noisy_channels
    
    def detect_flat_channels(
        self,
        data: np.ndarray,
        threshold: float = 1e-6
    ) -> List[int]:
        """
        Detect flat (dead) channels.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        threshold : float
            Minimum variance threshold
            
        Returns
        -------
        list of int
            Indices of flat channels
        """
        logger.info("Detecting flat channels")
        
        # Calculate variance for each channel
        variance = np.var(data, axis=1)
        
        # Find channels with low variance
        flat_channels = np.where(variance < threshold)[0].tolist()
        
        logger.info(f"Detected {len(flat_channels)} flat channels: {flat_channels}")
        
        return flat_channels


class ArtifactRemover:
    """
    Remove artifacts from electrophysiology data.
    
    Provides methods to interpolate, remove, or mask artifact periods.
    """
    
    def __init__(self, detector: Optional[ArtifactDetector] = None):
        """
        Initialize artifact remover.
        
        Parameters
        ----------
        detector : ArtifactDetector, optional
            Artifact detector instance
        """
        self.detector = detector or ArtifactDetector()
    
    def remove_artifacts(
        self,
        data: np.ndarray,
        sampling_rate: float,
        method: str = 'interpolate'
    ) -> np.ndarray:
        """
        Remove artifacts from data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        sampling_rate : float
            Sampling rate in Hz
        method : str
            Removal method ('interpolate', 'zero', or 'mask')
            
        Returns
        -------
        np.ndarray
            Data with artifacts removed
        """
        # Detect artifacts
        artifact_mask = self.detector.detect_amplitude_artifacts(data, sampling_rate)
        
        if not np.any(artifact_mask):
            logger.info("No artifacts detected")
            return data
        
        cleaned_data = data.copy()
        
        if method == 'interpolate':
            # Interpolate over artifact periods
            for ch in range(data.shape[0]):
                if np.any(artifact_mask):
                    # Find good samples
                    good_samples = ~artifact_mask
                    if np.sum(good_samples) > 0:
                        # Linear interpolation
                        x_good = np.where(good_samples)[0]
                        x_bad = np.where(artifact_mask)[0]
                        cleaned_data[ch, x_bad] = np.interp(
                            x_bad, x_good, data[ch, x_good]
                        )
        
        elif method == 'zero':
            # Set artifact periods to zero
            cleaned_data[:, artifact_mask] = 0
        
        elif method == 'mask':
            # Set artifact periods to NaN
            cleaned_data[:, artifact_mask] = np.nan
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Removed artifacts using {method} method")
        
        return cleaned_data
    
    def remove_bad_channels(
        self,
        data: np.ndarray,
        bad_channel_indices: List[int],
        method: str = 'interpolate'
    ) -> np.ndarray:
        """
        Remove or interpolate bad channels.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        bad_channel_indices : list of int
            Indices of bad channels
        method : str
            Removal method ('interpolate' or 'zero')
            
        Returns
        -------
        np.ndarray
            Data with bad channels removed
        """
        if not bad_channel_indices:
            return data
        
        cleaned_data = data.copy()
        
        if method == 'interpolate':
            # Interpolate from neighboring channels
            for bad_ch in bad_channel_indices:
                # Find neighboring good channels
                neighbors = []
                if bad_ch > 0 and bad_ch - 1 not in bad_channel_indices:
                    neighbors.append(bad_ch - 1)
                if bad_ch < data.shape[0] - 1 and bad_ch + 1 not in bad_channel_indices:
                    neighbors.append(bad_ch + 1)
                
                if neighbors:
                    cleaned_data[bad_ch] = np.mean(data[neighbors], axis=0)
                else:
                    cleaned_data[bad_ch] = 0
        
        elif method == 'zero':
            cleaned_data[bad_channel_indices] = 0
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Removed {len(bad_channel_indices)} bad channels using {method} method")
        
        return cleaned_data
