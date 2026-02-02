"""Signal filtering and processing functions."""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Main signal processing class for electrophysiology data.
    
    Provides filtering, downsampling, and common signal processing operations.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize signal processor.
        
        Parameters
        ----------
        sampling_rate : float
            Sampling rate of the data in Hz
        """
        self.sampling_rate = sampling_rate
    
    def bandpass_filter(
        self,
        data: np.ndarray,
        lowcut: float,
        highcut: float,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply bandpass filter to data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        lowcut : float
            Low cutoff frequency in Hz
        highcut : float
            High cutoff frequency in Hz
        order : int
            Filter order
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        logger.info(f"Applying bandpass filter: {lowcut}-{highcut} Hz, order={order}")
        
        nyquist = self.sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design Butterworth filter
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = signal.sosfiltfilt(sos, data[i])
        
        return filtered_data
    
    def notch_filter(
        self,
        data: np.ndarray,
        freq: float,
        quality: float = 30.0
    ) -> np.ndarray:
        """
        Apply notch filter to remove line noise.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        freq : float
            Frequency to notch out (typically 50 or 60 Hz)
        quality : float
            Quality factor (higher = narrower notch)
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        logger.info(f"Applying notch filter at {freq} Hz, Q={quality}")
        
        # Design notch filter
        b, a = signal.iirnotch(freq, quality, self.sampling_rate)
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = signal.filtfilt(b, a, data[i])
        
        return filtered_data
    
    def downsample(
        self,
        data: np.ndarray,
        target_rate: float,
        method: str = 'decimate'
    ) -> Tuple[np.ndarray, float]:
        """
        Downsample data to target sampling rate.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        target_rate : float
            Target sampling rate in Hz
        method : str
            Downsampling method ('decimate' or 'resample')
            
        Returns
        -------
        downsampled_data : np.ndarray
            Downsampled data
        new_rate : float
            Actual new sampling rate
        """
        if target_rate >= self.sampling_rate:
            logger.warning("Target rate >= current rate, no downsampling applied")
            return data, self.sampling_rate
        
        downsample_factor = int(self.sampling_rate / target_rate)
        new_rate = self.sampling_rate / downsample_factor
        
        logger.info(f"Downsampling from {self.sampling_rate} to {new_rate} Hz (factor={downsample_factor})")
        
        downsampled_data = np.zeros((data.shape[0], data.shape[1] // downsample_factor))
        
        if method == 'decimate':
            for i in range(data.shape[0]):
                downsampled_data[i] = signal.decimate(data[i], downsample_factor, zero_phase=True)
        elif method == 'resample':
            for i in range(data.shape[0]):
                downsampled_data[i] = signal.resample(data[i], data.shape[1] // downsample_factor)
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
        
        return downsampled_data, new_rate
    
    def detrend(self, data: np.ndarray, method: str = 'linear') -> np.ndarray:
        """
        Remove trend from data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
        method : str
            Detrending method ('linear' or 'constant')
            
        Returns
        -------
        np.ndarray
            Detrended data
        """
        logger.info(f"Detrending data using {method} method")
        
        detrended_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            if method == 'linear':
                detrended_data[i] = signal.detrend(data[i], type='linear')
            elif method == 'constant':
                detrended_data[i] = signal.detrend(data[i], type='constant')
        
        return detrended_data


class BandpassFilter:
    """Standalone bandpass filter."""
    
    def __init__(self, lowcut: float, highcut: float, sampling_rate: float, order: int = 4):
        """Initialize bandpass filter."""
        self.lowcut = lowcut
        self.highcut = highcut
        self.sampling_rate = sampling_rate
        self.order = order
        
        nyquist = sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        self.sos = signal.butter(order, [low, high], btype='band', output='sos')
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply filter to data."""
        if data.ndim == 1:
            return signal.sosfiltfilt(self.sos, data)
        else:
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i] = signal.sosfiltfilt(self.sos, data[i])
            return filtered_data


class NotchFilter:
    """Standalone notch filter."""
    
    def __init__(self, freq: float, sampling_rate: float, quality: float = 30.0):
        """Initialize notch filter."""
        self.freq = freq
        self.sampling_rate = sampling_rate
        self.quality = quality
        self.b, self.a = signal.iirnotch(freq, quality, sampling_rate)
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply filter to data."""
        if data.ndim == 1:
            return signal.filtfilt(self.b, self.a, data)
        else:
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i] = signal.filtfilt(self.b, self.a, data[i])
            return filtered_data
