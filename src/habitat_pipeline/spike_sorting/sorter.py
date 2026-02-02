"""Spike sorting and clustering."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class SpikeSorter:
    """
    Sort detected spikes into units using clustering algorithms.
    
    Supports multiple clustering methods including K-means and GMM.
    """
    
    def __init__(
        self,
        method: str = 'kmeans',
        n_clusters: Optional[int] = None,
        n_features: int = 3
    ):
        """
        Initialize spike sorter.
        
        Parameters
        ----------
        method : str
            Clustering method ('kmeans' or 'gmm')
        n_clusters : int, optional
            Number of clusters. If None, automatically determined.
        n_features : int
            Number of PCA features to use
        """
        self.method = method
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.pca = None
        self.clusterer = None
    
    def extract_features(self, waveforms: np.ndarray) -> np.ndarray:
        """
        Extract features from waveforms using PCA.
        
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
            return np.array([]).reshape(0, self.n_features)
        
        # Adjust n_features if needed
        n_components = min(self.n_features, waveforms.shape[0], waveforms.shape[1])
        
        # Initialize PCA
        self.pca = PCA(n_components=n_components)
        
        # Extract features
        features = self.pca.fit_transform(waveforms)
        
        logger.info(f"Extracted {n_components} PCA features from {len(waveforms)} waveforms")
        
        return features
    
    def cluster(self, features: np.ndarray) -> np.ndarray:
        """
        Cluster features into units.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_spikes, n_features)
            
        Returns
        -------
        np.ndarray
            Cluster labels for each spike
        """
        if len(features) == 0:
            return np.array([])
        
        # Determine number of clusters if not specified
        n_clusters = self.n_clusters
        if n_clusters is None:
            # Simple heuristic: sqrt(n_spikes/2)
            n_clusters = max(2, int(np.sqrt(len(features) / 2)))
            n_clusters = min(n_clusters, 10)  # Cap at 10
        
        # Cannot cluster if n_samples < n_clusters
        if len(features) < n_clusters:
            n_clusters = max(1, len(features))
        
        logger.info(f"Clustering {len(features)} spikes into {n_clusters} clusters using {self.method}")
        
        # Perform clustering
        if self.method == 'kmeans':
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = self.clusterer.fit_predict(features)
        
        elif self.method == 'gmm':
            self.clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = self.clusterer.fit_predict(features)
        
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        return labels
    
    def sort_channel(
        self,
        waveforms: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sort spikes from a single channel.
        
        Parameters
        ----------
        waveforms : np.ndarray
            Spike waveforms (n_spikes, n_samples)
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels
        features : np.ndarray
            Feature matrix
        """
        if len(waveforms) == 0:
            return np.array([]), np.array([]).reshape(0, self.n_features)
        
        # Extract features
        features = self.extract_features(waveforms)
        
        # Cluster
        labels = self.cluster(features)
        
        return labels, features
    
    def sort_all_channels(
        self,
        waveforms_dict: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Sort spikes from all channels.
        
        Parameters
        ----------
        waveforms_dict : dict
            Dictionary mapping channel to waveforms
            
        Returns
        -------
        dict
            Dictionary mapping channel to cluster labels
        """
        logger.info("Sorting spikes from all channels")
        
        labels_dict = {}
        
        for ch, waveforms in waveforms_dict.items():
            if len(waveforms) > 0:
                labels, _ = self.sort_channel(waveforms)
                labels_dict[ch] = labels
            else:
                labels_dict[ch] = np.array([])
        
        total_units = sum(len(np.unique(labels)) for labels in labels_dict.values() if len(labels) > 0)
        logger.info(f"Identified {total_units} total units across {len(labels_dict)} channels")
        
        return labels_dict
