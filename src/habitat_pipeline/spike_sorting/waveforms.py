"""Waveform analysis utilities."""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WaveformAnalyzer:
    """Analyze spike waveform properties."""
    
    def __init__(self, sampling_rate: float):
        """
        Initialize waveform analyzer.
        
        Parameters
        ----------
        sampling_rate : float
            Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
    
    def compute_mean_waveform(self, waveforms: np.ndarray) -> np.ndarray:
        """Compute mean waveform."""
        return np.mean(waveforms, axis=0)
    
    def compute_std_waveform(self, waveforms: np.ndarray) -> np.ndarray:
        """Compute standard deviation of waveforms."""
        return np.std(waveforms, axis=0)
    
    def compute_peak_to_trough(self, waveform: np.ndarray) -> Tuple[float, float]:
        """
        Compute peak-to-trough duration.
        
        Parameters
        ----------
        waveform : np.ndarray
            Spike waveform
            
        Returns
        -------
        duration_ms : float
            Peak-to-trough duration in milliseconds
        amplitude : float
            Peak-to-trough amplitude
        """
        trough_idx = np.argmin(waveform)
        peak_idx = np.argmax(waveform[trough_idx:]) + trough_idx
        
        duration_samples = peak_idx - trough_idx
        duration_ms = duration_samples / self.sampling_rate * 1000
        
        amplitude = waveform[peak_idx] - waveform[trough_idx]
        
        return duration_ms, amplitude
    
    def compute_snr(self, waveforms: np.ndarray) -> float:
        """
        Compute signal-to-noise ratio of waveforms.
        
        Parameters
        ----------
        waveforms : np.ndarray
            Spike waveforms (n_spikes, n_samples)
            
        Returns
        -------
        float
            SNR in dB
        """
        mean_waveform = self.compute_mean_waveform(waveforms)
        noise = waveforms - mean_waveform
        
        signal_power = np.mean(mean_waveform**2)
        noise_power = np.mean(noise**2)
        
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return snr_db
    
    def analyze_unit(
        self,
        waveforms: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze all properties of a unit.
        
        Parameters
        ----------
        waveforms : np.ndarray
            Spike waveforms for a unit
            
        Returns
        -------
        dict
            Dictionary of waveform properties
        """
        mean_waveform = self.compute_mean_waveform(waveforms)
        duration, amplitude = self.compute_peak_to_trough(mean_waveform)
        snr = self.compute_snr(waveforms)
        
        return {
            'n_spikes': len(waveforms),
            'peak_to_trough_ms': duration,
            'amplitude': amplitude,
            'snr_db': snr,
        }
