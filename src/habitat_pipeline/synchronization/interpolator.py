"""Temporal interpolation utilities."""

import logging
from typing import Optional

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


class TemporalInterpolator:
    """Interpolate data to common time base."""
    
    def __init__(self, method: str = 'linear'):
        """
        Initialize temporal interpolator.
        
        Parameters
        ----------
        method : str
            Interpolation method ('linear', 'cubic', or 'nearest')
        """
        self.method = method
    
    def interpolate(
        self,
        timestamps: np.ndarray,
        data: np.ndarray,
        target_timestamps: np.ndarray,
        fill_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Interpolate data to target timestamps.
        
        Parameters
        ----------
        timestamps : np.ndarray
            Original timestamps
        data : np.ndarray
            Original data values
        target_timestamps : np.ndarray
            Target timestamps for interpolation
        fill_value : float, optional
            Value for extrapolation (if None, uses 'extrapolate')
            
        Returns
        -------
        np.ndarray
            Interpolated data
        """
        if len(timestamps) != len(data):
            raise ValueError("Timestamps and data must have same length")
        
        # Create interpolator
        if self.method == 'linear':
            kind = 'linear'
        elif self.method == 'cubic':
            kind = 'cubic'
        elif self.method == 'nearest':
            kind = 'nearest'
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")
        
        # Handle fill value
        if fill_value is None:
            bounds_error = False
            fill_value = 'extrapolate'
        else:
            bounds_error = False
        
        # Perform interpolation
        f = interpolate.interp1d(
            timestamps,
            data,
            kind=kind,
            bounds_error=bounds_error,
            fill_value=fill_value
        )
        
        interpolated_data = f(target_timestamps)
        
        logger.info(f"Interpolated data from {len(data)} to {len(target_timestamps)} samples")
        
        return interpolated_data
    
    def resample_to_common_timebase(
        self,
        timestamps_list: list,
        data_list: list,
        sampling_rate: Optional[float] = None
    ) -> tuple:
        """
        Resample multiple data streams to common timebase.
        
        Parameters
        ----------
        timestamps_list : list
            List of timestamp arrays
        data_list : list
            List of data arrays
        sampling_rate : float, optional
            Target sampling rate. If None, uses minimum rate.
            
        Returns
        -------
        common_timestamps : np.ndarray
            Common timestamp array
        resampled_data_list : list
            List of resampled data arrays
        """
        # Find common time range
        t_start = max(ts[0] for ts in timestamps_list)
        t_end = min(ts[-1] for ts in timestamps_list)
        
        # Determine sampling rate
        if sampling_rate is None:
            # Use minimum sampling rate
            rates = [len(ts) / (ts[-1] - ts[0]) for ts in timestamps_list]
            sampling_rate = min(rates)
        
        # Create common timebase
        n_samples = int((t_end - t_start) * sampling_rate)
        common_timestamps = np.linspace(t_start, t_end, n_samples)
        
        # Resample all data
        resampled_data_list = []
        for timestamps, data in zip(timestamps_list, data_list):
            resampled = self.interpolate(timestamps, data, common_timestamps)
            resampled_data_list.append(resampled)
        
        logger.info(f"Resampled {len(data_list)} streams to common timebase ({n_samples} samples)")
        
        return common_timestamps, resampled_data_list
