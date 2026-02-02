"""Spike detection algorithms."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class SpikeDetector:
    """
    Detect neural spikes in electrophysiology data.
    
    Uses threshold-based detection with various threshold estimation methods.
    """
    
    def __init__(
        self,
        sampling_rate: float,
        threshold_method: str = 'mad',
        threshold_factor: float = 4.0,
        peak_sign: str = 'negative'
    ):
        """
        Initialize spike detector.
        
        Parameters
        ----------
        sampling_rate : float
            Sampling rate in Hz
        threshold_method : str
            Method for threshold estimation ('mad', 'std', or 'absolute')
        threshold_factor : float
            Threshold factor (multiplied by noise estimate)
        peak_sign : str
            Sign of spikes to detect ('negative', 'positive', or 'both')
        """
        self.sampling_rate = sampling_rate
        self.threshold_method = threshold_method
        self.threshold_factor = threshold_factor
        self.peak_sign = peak_sign
    
    def estimate_threshold(self, data: np.ndarray) -> float:
        """
        Estimate detection threshold from data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (1D array)
            
        Returns
        -------
        float
            Detection threshold
        """
        if self.threshold_method == 'mad':
            # Median absolute deviation method
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            threshold = self.threshold_factor * mad * 1.4826
        
        elif self.threshold_method == 'std':
            # Standard deviation method
            threshold = self.threshold_factor * np.std(data)
        
        elif self.threshold_method == 'absolute':
            # Use threshold factor as absolute value
            threshold = self.threshold_factor
        
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        
        return threshold
    
    def detect_spikes(
        self,
        data: np.ndarray,
        min_isi_ms: float = 1.0
    ) -> Dict[int, np.ndarray]:
        """
        Detect spikes in multi-channel data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        min_isi_ms : float
            Minimum inter-spike interval in milliseconds
            
        Returns
        -------
        dict
            Dictionary mapping channel index to spike times (in samples)
        """
        logger.info(f"Detecting spikes using {self.threshold_method} method")
        
        min_isi_samples = int(min_isi_ms * self.sampling_rate / 1000)
        spike_times = {}
        
        for ch in range(data.shape[0]):
            # Estimate threshold for this channel
            threshold = self.estimate_threshold(data[ch])
            
            # Detect threshold crossings
            if self.peak_sign == 'negative':
                crossings = data[ch] < -threshold
            elif self.peak_sign == 'positive':
                crossings = data[ch] > threshold
            else:  # both
                crossings = np.abs(data[ch]) > threshold
            
            # Find peaks
            if np.any(crossings):
                # Get crossing indices
                crossing_indices = np.where(crossings)[0]
                
                # Group nearby crossings and find peak within each group
                peaks = []
                i = 0
                while i < len(crossing_indices):
                    # Find consecutive crossings
                    group_start = crossing_indices[i]
                    group_end = group_start
                    
                    while i < len(crossing_indices) - 1 and crossing_indices[i + 1] - crossing_indices[i] <= min_isi_samples:
                        group_end = crossing_indices[i + 1]
                        i += 1
                    
                    # Find peak within group
                    group_indices = np.arange(group_start, group_end + 1)
                    if self.peak_sign == 'negative':
                        peak_idx = group_indices[np.argmin(data[ch, group_indices])]
                    else:
                        peak_idx = group_indices[np.argmax(np.abs(data[ch, group_indices]))]
                    
                    peaks.append(peak_idx)
                    i += 1
                
                spike_times[ch] = np.array(peaks)
            else:
                spike_times[ch] = np.array([])
        
        total_spikes = sum(len(times) for times in spike_times.values())
        logger.info(f"Detected {total_spikes} total spikes across {len(spike_times)} channels")
        
        return spike_times
    
    def extract_waveforms(
        self,
        data: np.ndarray,
        spike_times: Dict[int, np.ndarray],
        window_ms: Tuple[float, float] = (1.0, 2.0)
    ) -> Dict[int, np.ndarray]:
        """
        Extract spike waveforms around detected spike times.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        spike_times : dict
            Dictionary mapping channel to spike times
        window_ms : tuple
            Window around spike (before, after) in milliseconds
            
        Returns
        -------
        dict
            Dictionary mapping channel to waveforms (n_spikes, n_samples_waveform)
        """
        logger.info("Extracting spike waveforms")
        
        window_before = int(window_ms[0] * self.sampling_rate / 1000)
        window_after = int(window_ms[1] * self.sampling_rate / 1000)
        waveform_length = window_before + window_after
        
        waveforms = {}
        
        for ch, times in spike_times.items():
            if len(times) == 0:
                waveforms[ch] = np.array([]).reshape(0, waveform_length)
                continue
            
            ch_waveforms = []
            for spike_time in times:
                start = int(spike_time - window_before)
                end = int(spike_time + window_after)
                
                # Check bounds
                if start >= 0 and end < data.shape[1]:
                    waveform = data[ch, start:end]
                    ch_waveforms.append(waveform)
            
            waveforms[ch] = np.array(ch_waveforms)
        
        return waveforms
