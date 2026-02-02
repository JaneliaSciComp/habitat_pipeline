"""Quality metrics computation for electrophysiology data."""

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class QualityMetrics:
    """
    Compute quality metrics for electrophysiology data.
    
    Provides various metrics to assess data quality including SNR, noise levels,
    and channel quality.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize quality metrics calculator.
        
        Parameters
        ----------
        sampling_rate : float
            Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.metrics = {}
    
    def compute_snr(self, data: np.ndarray, signal_band: tuple = (300, 3000)) -> np.ndarray:
        """
        Compute signal-to-noise ratio for each channel.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        signal_band : tuple
            Frequency band for signal estimation (low, high) in Hz
            
        Returns
        -------
        np.ndarray
            SNR for each channel in dB
        """
        logger.info("Computing signal-to-noise ratio")
        
        from scipy.signal import welch
        
        snr = np.zeros(data.shape[0])
        
        for ch in range(data.shape[0]):
            # Compute power spectral density
            freqs, psd = welch(data[ch], fs=self.sampling_rate, nperseg=1024)
            
            # Signal power in specified band
            signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
            signal_power = np.mean(psd[signal_mask])
            
            # Noise power outside signal band
            noise_mask = ~signal_mask
            noise_power = np.mean(psd[noise_mask])
            
            # SNR in dB
            snr[ch] = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        self.metrics['snr'] = snr
        return snr
    
    def compute_noise_level(self, data: np.ndarray) -> np.ndarray:
        """
        Compute noise level for each channel.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
            
        Returns
        -------
        np.ndarray
            RMS noise level for each channel
        """
        logger.info("Computing noise levels")
        
        # Use robust estimate of noise (MAD)
        mad = np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1)
        noise_level = mad * 1.4826  # Convert MAD to std
        
        self.metrics['noise_level'] = noise_level
        return noise_level
    
    def compute_drift(self, data: np.ndarray, window_size: int = 10000) -> np.ndarray:
        """
        Compute baseline drift for each channel.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        window_size : int
            Window size for drift calculation in samples
            
        Returns
        -------
        np.ndarray
            Drift metric for each channel
        """
        logger.info("Computing baseline drift")
        
        drift = np.zeros(data.shape[0])
        
        for ch in range(data.shape[0]):
            # Calculate moving average
            n_windows = data.shape[1] // window_size
            if n_windows > 1:
                windowed_means = []
                for i in range(n_windows):
                    start = i * window_size
                    end = (i + 1) * window_size
                    windowed_means.append(np.mean(data[ch, start:end]))
                
                # Drift is std of windowed means
                drift[ch] = np.std(windowed_means)
        
        self.metrics['drift'] = drift
        return drift
    
    def compute_channel_correlation(self, data: np.ndarray) -> np.ndarray:
        """
        Compute correlation matrix between channels.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
            
        Returns
        -------
        np.ndarray
            Correlation matrix (n_channels, n_channels)
        """
        logger.info("Computing channel correlations")
        
        correlation_matrix = np.corrcoef(data)
        
        self.metrics['correlation_matrix'] = correlation_matrix
        return correlation_matrix
    
    def compute_all_metrics(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute all available quality metrics.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
            
        Returns
        -------
        dict
            Dictionary containing all computed metrics
        """
        logger.info("Computing all quality metrics")
        
        self.compute_snr(data)
        self.compute_noise_level(data)
        self.compute_drift(data)
        self.compute_channel_correlation(data)
        
        return self.metrics
    
    def get_metrics(self) -> Dict[str, np.ndarray]:
        """Get computed metrics."""
        return self.metrics
