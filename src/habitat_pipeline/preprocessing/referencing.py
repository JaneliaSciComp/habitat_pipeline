"""Reference schemes for electrophysiology data."""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CommonAverageReference:
    """
    Common Average Reference (CAR) for removing common noise.
    
    Subtracts the average across channels from each channel.
    """
    
    def __init__(self, exclude_channels: Optional[List[int]] = None):
        """
        Initialize CAR.
        
        Parameters
        ----------
        exclude_channels : list of int, optional
            Channels to exclude from average calculation
        """
        self.exclude_channels = exclude_channels or []
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply common average reference.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
            
        Returns
        -------
        np.ndarray
            Referenced data
        """
        logger.info("Applying common average reference")
        
        # Create mask for channels to include
        mask = np.ones(data.shape[0], dtype=bool)
        mask[self.exclude_channels] = False
        
        # Calculate common average
        common_avg = np.mean(data[mask], axis=0)
        
        # Subtract from all channels
        referenced_data = data - common_avg[np.newaxis, :]
        
        return referenced_data


class MedianReference:
    """
    Median Reference for robust noise removal.
    
    Subtracts the median across channels from each channel.
    """
    
    def __init__(self, exclude_channels: Optional[List[int]] = None):
        """
        Initialize median reference.
        
        Parameters
        ----------
        exclude_channels : list of int, optional
            Channels to exclude from median calculation
        """
        self.exclude_channels = exclude_channels or []
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply median reference.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
            
        Returns
        -------
        np.ndarray
            Referenced data
        """
        logger.info("Applying median reference")
        
        # Create mask for channels to include
        mask = np.ones(data.shape[0], dtype=bool)
        mask[self.exclude_channels] = False
        
        # Calculate common median
        common_median = np.median(data[mask], axis=0)
        
        # Subtract from all channels
        referenced_data = data - common_median[np.newaxis, :]
        
        return referenced_data


class LocalAverageReference:
    """
    Local Average Reference using spatial neighbors.
    
    References each channel to the average of its spatial neighbors.
    """
    
    def __init__(self, channel_positions: np.ndarray, radius: float = 100.0):
        """
        Initialize local average reference.
        
        Parameters
        ----------
        channel_positions : np.ndarray
            Channel positions (n_channels, 2 or 3)
        radius : float
            Radius for neighbor selection in same units as positions
        """
        self.channel_positions = channel_positions
        self.radius = radius
        self._compute_neighbors()
    
    def _compute_neighbors(self):
        """Compute neighbor list for each channel."""
        n_channels = self.channel_positions.shape[0]
        self.neighbors = []
        
        for i in range(n_channels):
            # Calculate distances to all other channels
            distances = np.sqrt(
                np.sum((self.channel_positions - self.channel_positions[i])**2, axis=1)
            )
            # Find channels within radius (excluding self)
            neighbor_indices = np.where((distances < self.radius) & (distances > 0))[0]
            self.neighbors.append(neighbor_indices.tolist())
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply local average reference.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (n_channels, n_samples)
            
        Returns
        -------
        np.ndarray
            Referenced data
        """
        logger.info("Applying local average reference")
        
        referenced_data = np.zeros_like(data)
        
        for i, neighbors in enumerate(self.neighbors):
            if neighbors:
                # Average of neighbors
                local_avg = np.mean(data[neighbors], axis=0)
                referenced_data[i] = data[i] - local_avg
            else:
                # No neighbors, keep original
                referenced_data[i] = data[i]
        
        return referenced_data
