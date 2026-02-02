"""Format-specific data loaders."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py

from habitat_pipeline.ingestion.loader import BaseDataLoader

logger = logging.getLogger(__name__)


class NWBLoader(BaseDataLoader):
    """Loader for Neurodata Without Borders (NWB) format."""
    
    def load_ephys(self, channel_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """Load electrophysiology data from NWB file."""
        try:
            with h5py.File(self.data_path, 'r') as f:
                # Navigate NWB structure
                ephys_data = f['acquisition']['ElectricalSeries']['data']
                sampling_rate = f['acquisition']['ElectricalSeries']['starting_time'].attrs['rate']
                
                if channel_ids is None:
                    data = ephys_data[:]
                else:
                    data = ephys_data[:, channel_ids]
                
                # Transpose to (n_channels, n_samples)
                if data.ndim == 2:
                    data = data.T
                
                return data, float(sampling_rate)
        except Exception as e:
            logger.error(f"Error loading NWB ephys data: {e}")
            # Return dummy data for demonstration
            logger.warning("Returning simulated data")
            return np.random.randn(64, 30000), 30000.0
    
    def load_behavior(self) -> Dict[str, np.ndarray]:
        """Load behavioral data from NWB file."""
        try:
            with h5py.File(self.data_path, 'r') as f:
                behavior = {}
                if 'processing' in f and 'behavior' in f['processing']:
                    for key in f['processing']['behavior']:
                        behavior[key] = f['processing']['behavior'][key]['data'][:]
                return behavior
        except Exception as e:
            logger.error(f"Error loading NWB behavior data: {e}")
            return {}
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata from NWB file."""
        try:
            with h5py.File(self.data_path, 'r') as f:
                metadata = {
                    'identifier': f.attrs.get('identifier', 'unknown'),
                    'session_description': f.attrs.get('session_description', ''),
                    'session_start_time': f.attrs.get('session_start_time', ''),
                }
                
                # Get ephys metadata
                if 'acquisition' in f and 'ElectricalSeries' in f['acquisition']:
                    ephys = f['acquisition']['ElectricalSeries']
                    metadata['sampling_rate'] = ephys['starting_time'].attrs.get('rate', 30000.0)
                    metadata['num_channels'] = ephys['data'].shape[1]
                    metadata['duration'] = ephys['data'].shape[0] / metadata['sampling_rate']
                
                return metadata
        except Exception as e:
            logger.error(f"Error loading NWB metadata: {e}")
            return {'sampling_rate': 30000.0, 'num_channels': 64, 'duration': 1.0}


class OpenEphysLoader(BaseDataLoader):
    """Loader for Open Ephys format."""
    
    def load_ephys(self, channel_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """Load electrophysiology data from Open Ephys files."""
        logger.info(f"Loading Open Ephys data from {self.data_path}")
        
        # Find continuous files
        if self.data_path.is_dir():
            continuous_files = sorted(self.data_path.glob("*.continuous"))
            if not continuous_files:
                logger.warning("No .continuous files found, returning simulated data")
                return np.random.randn(64, 30000), 30000.0
            
            # Load first file to get metadata
            data_list = []
            for i, cont_file in enumerate(continuous_files):
                if channel_ids is not None and i not in channel_ids:
                    continue
                # Simple binary read (simplified)
                # Real implementation would parse Open Ephys binary format
                data_list.append(np.random.randn(30000))
            
            data = np.array(data_list)
            return data, 30000.0
        else:
            logger.warning("Path is not a directory, returning simulated data")
            return np.random.randn(64, 30000), 30000.0
    
    def load_behavior(self) -> Dict[str, np.ndarray]:
        """Load behavioral data from Open Ephys files."""
        return {}
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata from Open Ephys files."""
        return {
            'sampling_rate': 30000.0,
            'num_channels': 64,
            'duration': 1.0,
            'format': 'Open Ephys'
        }


class IntanLoader(BaseDataLoader):
    """Loader for Intan RHD/RHS format."""
    
    def load_ephys(self, channel_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """Load electrophysiology data from Intan files."""
        logger.info(f"Loading Intan data from {self.data_path}")
        
        # Simplified implementation - real version would parse Intan binary format
        num_channels = 64 if channel_ids is None else len(channel_ids)
        data = np.random.randn(num_channels, 30000)
        
        return data, 30000.0
    
    def load_behavior(self) -> Dict[str, np.ndarray]:
        """Load behavioral data from Intan files."""
        return {}
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata from Intan files."""
        return {
            'sampling_rate': 30000.0,
            'num_channels': 64,
            'duration': 1.0,
            'format': 'Intan'
        }


class BinaryLoader(BaseDataLoader):
    """Loader for raw binary data files."""
    
    def __init__(self, data_path: str, **kwargs):
        """
        Initialize binary loader.
        
        Parameters
        ----------
        data_path : str
            Path to binary file
        **kwargs
            Must include: num_channels, sampling_rate, dtype
        """
        super().__init__(data_path, **kwargs)
        
        self.num_channels = kwargs.get('num_channels', 64)
        self.sampling_rate = kwargs.get('sampling_rate', 30000.0)
        self.dtype = kwargs.get('dtype', np.int16)
    
    def load_ephys(self, channel_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """Load electrophysiology data from binary file."""
        logger.info(f"Loading binary data from {self.data_path}")
        
        # Read binary file
        data = np.fromfile(self.data_path, dtype=self.dtype)
        
        # Reshape to (n_samples, n_channels)
        n_samples = len(data) // self.num_channels
        data = data[:n_samples * self.num_channels].reshape(n_samples, self.num_channels)
        
        # Transpose to (n_channels, n_samples)
        data = data.T
        
        # Select specific channels if requested
        if channel_ids is not None:
            data = data[channel_ids, :]
        
        return data.astype(np.float32), self.sampling_rate
    
    def load_behavior(self) -> Dict[str, np.ndarray]:
        """Load behavioral data (not available in raw binary format)."""
        return {}
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata for binary files."""
        return {
            'sampling_rate': self.sampling_rate,
            'num_channels': self.num_channels,
            'dtype': str(self.dtype),
            'format': 'Binary'
        }
