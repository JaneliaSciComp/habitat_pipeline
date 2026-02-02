"""Timestamp alignment for multimodal data streams."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


class TimestampAligner:
    """
    Align timestamps across multiple data modalities.
    
    Handles clock drift and offset between different recording systems.
    """
    
    def __init__(self, reference_stream: str = 'ephys'):
        """
        Initialize timestamp aligner.
        
        Parameters
        ----------
        reference_stream : str
            Name of reference stream for alignment
        """
        self.reference_stream = reference_stream
        self.transforms = {}
    
    def compute_transform(
        self,
        source_timestamps: np.ndarray,
        target_timestamps: np.ndarray,
        sync_events: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute linear transform from source to target timestamps.
        
        Parameters
        ----------
        source_timestamps : np.ndarray
            Source timestamps
        target_timestamps : np.ndarray
            Target timestamps (same events as source)
        sync_events : np.ndarray, optional
            Indices of synchronization events in both streams
            
        Returns
        -------
        dict
            Transform parameters (slope, intercept)
        """
        if sync_events is not None:
            source_sync = source_timestamps[sync_events]
            target_sync = target_timestamps[sync_events]
        else:
            source_sync = source_timestamps
            target_sync = target_timestamps
        
        # Fit linear model: target = slope * source + intercept
        coeffs = np.polyfit(source_sync, target_sync, 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        # Calculate residuals
        predicted = slope * source_sync + intercept
        residuals = target_sync - predicted
        rmse = np.sqrt(np.mean(residuals**2))
        
        transform = {
            'slope': slope,
            'intercept': intercept,
            'rmse': rmse
        }
        
        logger.info(f"Computed transform: slope={slope:.6f}, intercept={intercept:.3f}, RMSE={rmse:.3f}")
        
        return transform
    
    def apply_transform(
        self,
        timestamps: np.ndarray,
        transform: Dict[str, float]
    ) -> np.ndarray:
        """
        Apply transform to timestamps.
        
        Parameters
        ----------
        timestamps : np.ndarray
            Input timestamps
        transform : dict
            Transform parameters
            
        Returns
        -------
        np.ndarray
            Aligned timestamps
        """
        return transform['slope'] * timestamps + transform['intercept']
    
    def align_streams(
        self,
        streams: Dict[str, np.ndarray],
        sync_signals: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Align multiple streams to reference.
        
        Parameters
        ----------
        streams : dict
            Dictionary of stream names to timestamps
        sync_signals : dict, optional
            Dictionary of stream names to synchronization signals
            
        Returns
        -------
        dict
            Dictionary of aligned timestamps
        """
        logger.info(f"Aligning streams to reference: {self.reference_stream}")
        
        aligned_streams = {}
        reference_ts = streams[self.reference_stream]
        aligned_streams[self.reference_stream] = reference_ts
        
        for stream_name, timestamps in streams.items():
            if stream_name == self.reference_stream:
                continue
            
            # Find sync events if sync signals provided
            sync_events = None
            if sync_signals is not None and stream_name in sync_signals:
                # Simple peak detection for sync pulses
                ref_sync = sync_signals[self.reference_stream]
                stream_sync = sync_signals[stream_name]
                
                # Detect peaks
                ref_peaks = self._detect_sync_peaks(ref_sync)
                stream_peaks = self._detect_sync_peaks(stream_sync)
                
                # Match peaks
                n_events = min(len(ref_peaks), len(stream_peaks))
                sync_events = np.arange(n_events)
                
                transform = self.compute_transform(
                    timestamps[stream_peaks[:n_events]],
                    reference_ts[ref_peaks[:n_events]],
                    sync_events
                )
            else:
                # Assume timestamps correspond directly
                n_events = min(len(timestamps), len(reference_ts))
                transform = self.compute_transform(
                    timestamps[:n_events],
                    reference_ts[:n_events]
                )
            
            # Store transform
            self.transforms[stream_name] = transform
            
            # Apply transform
            aligned_streams[stream_name] = self.apply_transform(timestamps, transform)
        
        return aligned_streams
    
    def _detect_sync_peaks(self, signal: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Detect synchronization peaks in signal."""
        # Threshold crossing
        threshold_value = np.mean(signal) + threshold * np.std(signal)
        above_threshold = signal > threshold_value
        
        # Find rising edges
        peaks = np.where(np.diff(above_threshold.astype(int)) > 0)[0]
        
        return peaks
