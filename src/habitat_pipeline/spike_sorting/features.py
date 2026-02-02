"""Feature extraction for spike sorting."""

import logging
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from spike waveforms for clustering."""
    
    def __init__(self, method: str = 'pca', n_components: int = 3):
        """
        Initialize feature extractor.
        
        Parameters
        ----------
        method : str
            Feature extraction method ('pca', 'peak', or 'combined')
        n_components : int
            Number of components/features
        """
        self.method = method
        self.n_components = n_components
        self.transformer = None
    
    def extract_pca_features(self, waveforms: np.ndarray) -> np.ndarray:
        """Extract PCA features."""
        self.transformer = PCA(n_components=self.n_components)
        return self.transformer.fit_transform(waveforms)
    
    def extract_peak_features(self, waveforms: np.ndarray) -> np.ndarray:
        """Extract peak-based features."""
        features = []
        
        for waveform in waveforms:
            # Peak amplitude
            peak_amp = np.min(waveform)
            
            # Peak time
            peak_time = np.argmin(waveform)
            
            # Trough-to-peak time
            trough_idx = np.argmin(waveform)
            peak_idx = np.argmax(waveform[trough_idx:]) + trough_idx
            trough_to_peak = peak_idx - trough_idx
            
            features.append([peak_amp, peak_time, trough_to_peak])
        
        return np.array(features)
    
    def extract(self, waveforms: np.ndarray) -> np.ndarray:
        """
        Extract features from waveforms.
        
        Parameters
        ----------
        waveforms : np.ndarray
            Spike waveforms (n_spikes, n_samples)
            
        Returns
        -------
        np.ndarray
            Feature matrix (n_spikes, n_features)
        """
        if len(waveforms) == 0:
            return np.array([]).reshape(0, self.n_components)
        
        if self.method == 'pca':
            return self.extract_pca_features(waveforms)
        elif self.method == 'peak':
            return self.extract_peak_features(waveforms)
        elif self.method == 'combined':
            pca_features = self.extract_pca_features(waveforms)
            peak_features = self.extract_peak_features(waveforms)
            return np.hstack([pca_features, peak_features])
        else:
            raise ValueError(f"Unknown method: {self.method}")
