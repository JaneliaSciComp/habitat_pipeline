"""Base data loader classes and interfaces."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    All data loaders should inherit from this class and implement the required methods.
    """
    
    def __init__(self, data_path: str, **kwargs):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        data_path : str
            Path to data file or directory
        **kwargs
            Additional loader-specific parameters
        """
        self.data_path = Path(data_path)
        self.metadata = {}
        self.config = kwargs
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
    
    @abstractmethod
    def load_ephys(self, channel_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """
        Load electrophysiology data.
        
        Parameters
        ----------
        channel_ids : list of int, optional
            Specific channels to load. If None, load all channels.
            
        Returns
        -------
        data : np.ndarray
            Electrophysiology data array (n_channels, n_samples)
        sampling_rate : float
            Sampling rate in Hz
        """
        pass
    
    @abstractmethod
    def load_behavior(self) -> Dict[str, np.ndarray]:
        """
        Load behavioral data.
        
        Returns
        -------
        dict
            Dictionary containing behavioral data streams
        """
        pass
    
    @abstractmethod
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from the data file.
        
        Returns
        -------
        dict
            Metadata dictionary
        """
        pass
    
    def get_sampling_rate(self) -> float:
        """Get the sampling rate of the electrophysiology data."""
        if not self.metadata:
            self.metadata = self.load_metadata()
        return self.metadata.get("sampling_rate", 30000.0)
    
    def get_num_channels(self) -> int:
        """Get the number of electrophysiology channels."""
        if not self.metadata:
            self.metadata = self.load_metadata()
        return self.metadata.get("num_channels", 0)
    
    def get_duration(self) -> float:
        """Get the duration of the recording in seconds."""
        if not self.metadata:
            self.metadata = self.load_metadata()
        return self.metadata.get("duration", 0.0)


class DataLoader:
    """
    Main data loader class that automatically detects and loads data from various formats.
    """
    
    SUPPORTED_FORMATS = {
        ".nwb": "NWBLoader",
        ".continuous": "OpenEphysLoader",
        ".rhd": "IntanLoader",
        ".rhs": "IntanLoader",
        ".dat": "BinaryLoader",
        ".bin": "BinaryLoader",
    }
    
    def __init__(self, data_path: str, format: Optional[str] = None, **kwargs):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        data_path : str
            Path to data file or directory
        format : str, optional
            Explicit format specification. If None, auto-detect from file extension.
        **kwargs
            Additional loader-specific parameters
        """
        self.data_path = Path(data_path)
        self.format = format
        self.config = kwargs
        self.loader = None
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        self._initialize_loader()
    
    def _initialize_loader(self):
        """Initialize the appropriate loader based on data format."""
        if self.format is None:
            # Auto-detect format
            self.format = self._detect_format()
        
        # Import and instantiate the appropriate loader
        loader_class_name = self.SUPPORTED_FORMATS.get(self.format)
        
        if loader_class_name is None:
            raise ValueError(f"Unsupported format: {self.format}")
        
        # Dynamic import of loader class
        from habitat_pipeline.ingestion import formats
        loader_class = getattr(formats, loader_class_name)
        self.loader = loader_class(str(self.data_path), **self.config)
        
        logger.info(f"Initialized {loader_class_name} for {self.data_path}")
    
    def _detect_format(self) -> str:
        """Detect data format from file extension or directory structure."""
        if self.data_path.is_file():
            suffix = self.data_path.suffix.lower()
            if suffix in self.SUPPORTED_FORMATS:
                return suffix
        elif self.data_path.is_dir():
            # Check for common file patterns
            if list(self.data_path.glob("*.nwb")):
                return ".nwb"
            elif list(self.data_path.glob("*.continuous")):
                return ".continuous"
            elif list(self.data_path.glob("*.rhd")) or list(self.data_path.glob("*.rhs")):
                return ".rhd"
        
        raise ValueError(f"Could not detect format for: {self.data_path}")
    
    def load_ephys(self, channel_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """Load electrophysiology data using the appropriate loader."""
        return self.loader.load_ephys(channel_ids)
    
    def load_behavior(self) -> Dict[str, np.ndarray]:
        """Load behavioral data using the appropriate loader."""
        return self.loader.load_behavior()
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata using the appropriate loader."""
        return self.loader.load_metadata()
    
    def get_sampling_rate(self) -> float:
        """Get sampling rate."""
        return self.loader.get_sampling_rate()
    
    def get_num_channels(self) -> int:
        """Get number of channels."""
        return self.loader.get_num_channels()
    
    def get_duration(self) -> float:
        """Get recording duration."""
        return self.loader.get_duration()
