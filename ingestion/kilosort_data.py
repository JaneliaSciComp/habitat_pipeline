"""
Kilosort 4 Data Import and Storage Module

This module provides a data class for importing and storing electrophysiology data
from Kilosort 4 output, designed for the habitat_pipeline multi-animal analysis system.

The class supports:
- Loading all standard Kilosort 4 outputs
- Efficient data access and filtering by animal, session, channel, firing rate
- Integration with behavioral data analysis pipeline
- Memory-efficient spike time binning for continuous behavioral features
- Event-aligned spike extraction for discrete behavioral events
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import json
import warnings
from datetime import datetime


@dataclass
class KilosortData:
    """
    Data class for importing and storing Kilosort 4 electrophysiology data.
    
    This class handles the complete Kilosort 4 output structure and provides
    efficient methods for data access, filtering, and integration with 
    behavioral analysis pipelines.
    
    Attributes:
        data_path: Path to the Kilosort 4 output directory
        animal_id: Identifier for the animal
        session_id: Identifier for the recording session
        sample_rate: Sampling rate in Hz
        metadata: Additional metadata dictionary
    """
    
    # Input parameters
    data_path: Union[str, Path]
    animal_id: str
    session_id: str
    sample_rate: float = 30000.0  # Default Neuropixels sampling rate
    
    def __post_init__(self):
        """Initialize the data class and load Kilosort output."""
        # Convert to Path object
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Kilosort data path does not exist: {self.data_path}")
        
        # Initialize data attributes
        self.spike_times = None
        self.spike_clusters = None
        self.cluster_info = None
        self.templates = None
        self.channel_map = None
        self.amplitudes = None
        self.pc_features = None
        self.pc_feature_ind = None
        self.whitening_mat = None
        self.winv = None
        self.channel_positions = None
        
        # Computed properties (cached for efficiency)
        self._firing_rates = None
        self._spike_times_binned = {}
        
        # Metadata
        self.metadata = {
            'load_time': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'animal_id': self.animal_id,
            'session_id': self.session_id
        }
        
        # Load all Kilosort 4 output files
        self._load_kilosort_data()
    
    def _load_kilosort_data(self) -> None:
        """Load all standard Kilosort 4 output files."""
        
        # Required files - these must exist
        required_files = {
            'spike_times.npy': 'spike_times',
            'spike_clusters.npy': 'spike_clusters',
        }
        
        # Load required files
        for filename, attr_name in required_files.items():
            file_path = self.data_path / filename
            if file_path.exists():
                setattr(self, attr_name, np.load(file_path))
            else:
                raise FileNotFoundError(f"Required Kilosort file not found: {file_path}")
        
        # Optional files with their default handling
        optional_files = {
            #'amplitudes.npy': 'amplitudes',
            #'templates.npy': 'templates',
            'channel_map.npy': 'channel_map',
            'channel_positions.npy': 'channel_positions',
            #'pc_features.npy': 'pc_features',
            #'pc_feature_ind.npy': 'pc_feature_ind',
            #'whitening_mat.npy': 'whitening_mat',
            #'whitening_mat_inv.npy': 'winv'
        }
        
        # Load optional files
        for filename, attr_name in optional_files.items():
            file_path = self.data_path / filename
            if file_path.exists():
                setattr(self, attr_name, np.load(file_path))
            else:
                warnings.warn(f"Optional Kilosort file not found: {file_path}")
        
        # Convert spike times to seconds
        if self.spike_times is not None:
            self.spike_times = self.spike_times.flatten() / self.sample_rate
        
        # Load cluster information
        self._load_cluster_info()
        
        # Load sampling rate from params if available
        self._load_params()
    
    def _load_cluster_info(self) -> None:
        """Load cluster information from cluster_info.tsv or cluster_group.tsv or create from basic data."""
        cluster_info_path = self.data_path / 'cluster_info.tsv'
        cluster_group_path = self.data_path / 'cluster_group.tsv'
        
        if cluster_info_path.exists():
            self.cluster_info = pd.read_csv(cluster_info_path, sep='\t')
        elif cluster_group_path.exists():
            # Load cluster_group.tsv as fallback
            cluster_group_df = pd.read_csv(cluster_group_path, sep='\t')
            
            if self.spike_clusters is not None:
                unique_clusters = np.unique(self.spike_clusters)
                
                # Create cluster_info from cluster_group.tsv and basic data
                self.cluster_info = pd.DataFrame({
                    'cluster_id': unique_clusters,
                    'channel': self._get_cluster_channels(unique_clusters),
                })
                
                # Add group information from cluster_group.tsv
                if 'cluster_id' in cluster_group_df.columns and 'group' in cluster_group_df.columns:
                    # Merge group information
                    group_mapping = dict(zip(cluster_group_df['cluster_id'], cluster_group_df['group']))
                    self.cluster_info['group'] = [group_mapping.get(cid, 'good') for cid in unique_clusters]
                else:
                    # Fallback if cluster_group.tsv format is unexpected
                    self.cluster_info['group'] = ['good'] * len(unique_clusters)
        else:
            # Create basic cluster info from available data
            if self.spike_clusters is not None:
                unique_clusters = np.unique(self.spike_clusters)
                self.cluster_info = pd.DataFrame({
                    'cluster_id': unique_clusters,
                    'channel': self._get_cluster_channels(unique_clusters),
                    'group': ['good'] * len(unique_clusters)  # Default assumption
                })
        
        # Ensure cluster_id is the index
        if self.cluster_info is not None and 'cluster_id' in self.cluster_info.columns:
            self.cluster_info.set_index('cluster_id', inplace=True)
    
    def _get_cluster_channels(self, cluster_ids: np.ndarray) -> np.ndarray:
        """Get the primary channel for each cluster based on template amplitude."""
        if self.templates is None:
            return np.zeros(len(cluster_ids), dtype=int)
        
        channels = []
        for cluster_id in cluster_ids:
            if cluster_id < len(self.templates):
                # Find channel with maximum amplitude for this template
                template_amps = np.max(np.abs(self.templates[cluster_id]), axis=0)
                max_channel = np.argmax(template_amps)
                channels.append(max_channel)
            else:
                channels.append(0)
        
        return np.array(channels)
    
    def _load_params(self) -> None:
        """Load parameters from params.py if available."""
        params_path = self.data_path / 'params.py'
        if params_path.exists():
            try:
                # Simple parsing for sample_rate
                with open(params_path, 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if 'sample_rate' in line and '=' in line:
                            rate_str = line.split('=')[1].strip().rstrip(',')
                            self.sample_rate = float(rate_str)
                            break
            except Exception as e:
                warnings.warn(f"Could not parse params.py: {e}")
    
    @property
    def firing_rates(self) -> pd.Series:
        """
        Calculate and cache firing rates for all clusters.
        
        Returns:
            Series with cluster_id as index and firing rate (Hz) as values
        """
        if self._firing_rates is None:
            if self.spike_times is None or self.spike_clusters is None:
                raise ValueError("Spike data not loaded")
            
            duration = self.spike_times.max() - self.spike_times.min()
            cluster_counts = pd.Series(self.spike_clusters).value_counts()
            self._firing_rates = cluster_counts / duration
            self._firing_rates.name = 'firing_rate_hz'
        
        return self._firing_rates
    
    def get_clusters(self, 
                    animal_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    channels: Optional[List[int]] = None,
                    min_firing_rate: Optional[float] = None,
                    max_firing_rate: Optional[float] = None,
                    cluster_group: Optional[str] = None) -> List[int]:
        """
        Get cluster IDs based on filtering criteria.
        
        Args:
            animal_id: Filter by animal ID (for multi-animal datasets)
            session_id: Filter by session ID
            channels: Filter by channel numbers
            min_firing_rate: Minimum firing rate in Hz
            max_firing_rate: Maximum firing rate in Hz
            cluster_group: Filter by cluster group ('good', 'mua', 'noise')
        
        Returns:
            List of cluster IDs matching the criteria
        """
        if self.cluster_info is None:
            raise ValueError("Cluster info not available")
        
        mask = pd.Series(True, index=self.cluster_info.index)
        
        # Filter by current animal/session if specified
        if animal_id is not None and animal_id != self.animal_id:
            return []
        if session_id is not None and session_id != self.session_id:
            return []
        
        # Filter by channels
        if channels is not None:
            if 'channel' in self.cluster_info.columns:
                mask &= self.cluster_info['channel'].isin(channels)
        
        # Filter by firing rate
        firing_rates = self.firing_rates
        if min_firing_rate is not None:
            mask &= firing_rates >= min_firing_rate
        if max_firing_rate is not None:
            mask &= firing_rates <= max_firing_rate
        
        # Filter by cluster group
        if cluster_group is not None and 'group' in self.cluster_info.columns:
            mask &= self.cluster_info['group'] == cluster_group
        
        return mask[mask].index.tolist()
    
    def get_spike_times(self, cluster_ids: Union[int, List[int]]) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Get spike times for specified cluster(s).
        
        Args:
            cluster_ids: Single cluster ID or list of cluster IDs
        
        Returns:
            Array of spike times (single cluster) or dict mapping cluster_id to spike times
        """
        if self.spike_times is None or self.spike_clusters is None:
            raise ValueError("Spike data not loaded")
            
        if isinstance(cluster_ids, int):
            mask = self.spike_clusters == cluster_ids
            return self.spike_times[mask]
        else:
            result = {}
            for cluster_id in cluster_ids:
                mask = self.spike_clusters == cluster_id
                result[cluster_id] = self.spike_times[mask]
            return result
    
    def bin_spike_times(self, 
                       cluster_ids: Union[int, List[int]],
                       bin_size: float = 0.025,  # 25ms bins for 40Hz behavioral data
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Bin spike times for continuous behavioral feature analysis.
        
        Args:
            cluster_ids: Single cluster ID or list of cluster IDs
            bin_size: Bin size in seconds (default: 25ms for 40Hz)
            start_time: Start time in seconds (default: first spike)
            end_time: End time in seconds (default: last spike)
        
        Returns:
            Binned spike counts as array(s)
        """
        if self.spike_times is None:
            raise ValueError("Spike times not loaded")
            
        if start_time is None:
            start_time = self.spike_times.min()
        if end_time is None:
            end_time = self.spike_times.max()
        
        # Create time bins
        bins = np.arange(start_time, end_time + bin_size, bin_size)
        
        if isinstance(cluster_ids, int):
            spike_times = self.get_spike_times(cluster_ids)
            if isinstance(spike_times, np.ndarray):
                counts, _ = np.histogram(spike_times, bins=bins)
                return counts
        else:
            result = {}
            spike_times_dict = self.get_spike_times(cluster_ids)
            if isinstance(spike_times_dict, dict):
                for cluster_id, spike_times in spike_times_dict.items():
                    counts, _ = np.histogram(spike_times, bins=bins)
                    result[cluster_id] = counts
                return result
        
        raise ValueError("Invalid cluster_ids or spike_times format")
    
    def get_event_aligned_spikes(self, 
                               cluster_ids: Union[int, List[int]],
                               event_times: np.ndarray,
                               window_pre: float = 1.0,
                               window_post: float = 1.0) -> Dict[int, List[np.ndarray]]:
        """
        Extract spikes aligned to behavioral events.
        
        Args:
            cluster_ids: Single cluster ID or list of cluster IDs
            event_times: Array of event times in seconds
            window_pre: Time window before event (seconds)
            window_post: Time window after event (seconds)
        
        Returns:
            Dictionary with aligned spike times for each cluster and event
        """
        if isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]
        
        result = {}
        spike_times_dict = self.get_spike_times(cluster_ids)
        
        if isinstance(spike_times_dict, dict):
            for cluster_id, spike_times in spike_times_dict.items():
                cluster_events = []
                
                for event_time in event_times:
                    # Find spikes in window around event
                    start_time = event_time - window_pre
                    end_time = event_time + window_post
                    
                    mask = (spike_times >= start_time) & (spike_times <= end_time)
                    event_spikes = spike_times[mask] - event_time  # Align to event
                    cluster_events.append(event_spikes)
                
                result[cluster_id] = cluster_events
        
        return result
    
    def get_cluster_waveform(self, cluster_id: int, channel: Optional[int] = None) -> np.ndarray:
        """
        Get the average waveform for a cluster.
        
        Args:
            cluster_id: Cluster ID
            channel: Specific channel (default: primary channel)
        
        Returns:
            Average waveform
        """
        if self.templates is None:
            raise ValueError("Templates not available")
        
        if cluster_id >= len(self.templates):
            raise ValueError(f"Cluster {cluster_id} not found in templates")
        
        template = self.templates[cluster_id]
        
        if channel is None:
            # Return waveform from primary channel
            if self.cluster_info is not None and 'channel' in self.cluster_info.columns:
                channel = int(self.cluster_info.loc[cluster_id, 'channel'])
            else:
                # Find channel with maximum amplitude
                channel_amps = np.max(np.abs(template), axis=0)
                channel = int(np.argmax(channel_amps))
        
        return template[:, channel]
    
    def save_processed_data(self, output_path: Union[str, Path]) -> None:
        """
        Save processed data for quick loading.
        
        Args:
            output_path: Path to save processed data
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save core data
        if self.spike_times is not None:
            np.save(output_path / 'spike_times.npy', self.spike_times)
        if self.spike_clusters is not None:
            np.save(output_path / 'spike_clusters.npy', self.spike_clusters)
        
        # Save cluster info
        if self.cluster_info is not None:
            self.cluster_info.to_csv(output_path / 'cluster_info.csv')
        
        # Save firing rates
        if self._firing_rates is not None:
            self.firing_rates.to_csv(output_path / 'firing_rates.csv')
        
        # Save metadata
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def __repr__(self) -> str:
        """String representation of the KilosortData object."""
        n_spikes = len(self.spike_times) if self.spike_times is not None else 0
        n_clusters = len(self.cluster_info) if self.cluster_info is not None else 0
        duration = 0
        if self.spike_times is not None:
            duration = self.spike_times.max() - self.spike_times.min()
        
        return (f"KilosortData(animal={self.animal_id}, session={self.session_id}, "
                f"n_spikes={n_spikes}, n_clusters={n_clusters}, duration={duration:.1f}s)")


def load_kilosort_session(data_path: Union[str, Path], 
                         animal_id: str, 
                         session_id: str) -> KilosortData:
    """
    Convenience function to load a Kilosort session.
    
    Args:
        data_path: Path to Kilosort output directory
        animal_id: Animal identifier
        session_id: Session identifier
    
    Returns:
        KilosortData object with loaded data
    """
    return KilosortData(data_path=data_path, animal_id=animal_id, session_id=session_id)


def load_multiple_sessions(session_configs: List[Dict[str, Any]]) -> List[KilosortData]:
    """
    Load multiple Kilosort sessions.
    
    Args:
        session_configs: List of dictionaries with keys: 'data_path', 'animal_id', 'session_id'
    
    Returns:
        List of KilosortData objects
    """
    sessions = []
    for config in session_configs:
        try:
            session = KilosortData(**config)
            sessions.append(session)
        except Exception as e:
            warnings.warn(f"Failed to load session {config}: {e}")
    
    return sessions


# Example usage and demonstration
if __name__ == "__main__":
    # Example of how to use the KilosortData class
    
    # Load a single session
    try:
        # Replace with actual path to your Kilosort output
        ks_data = load_kilosort_session(
            data_path="path/to/kilosort/output",
            animal_id="rat001",
            session_id="session001"
        )
        
        print(ks_data)
        
        # Get good clusters with firing rate between 1-50 Hz
        good_clusters = ks_data.get_clusters(
            min_firing_rate=1.0,
            max_firing_rate=50.0,
            cluster_group='good'
        )
        
        print(f"Found {len(good_clusters)} good clusters")
        
        # Get spike times for the first few clusters
        if good_clusters:
            spike_times = ks_data.get_spike_times(good_clusters[:3])
            print(f"Loaded spike times for {len(spike_times)} clusters")
            
            # Bin spike times for behavioral analysis
            binned_spikes = ks_data.bin_spike_times(
                good_clusters[0], 
                bin_size=0.025  # 25ms bins
            )
            print(f"Binned spikes shape: {binned_spikes.shape}")
            
    except FileNotFoundError:
        print("Demo: Kilosort data path not found. Please provide valid path for actual usage.")
    except Exception as e:
        print(f"Demo: Error loading data: {e}")
    
    # Load multiple sessions
    session_configs = [
        {"data_path": "path/to/session1", "animal_id": "rat001", "session_id": "day1"},
        {"data_path": "path/to/session2", "animal_id": "rat001", "session_id": "day2"},
        {"data_path": "path/to/session3", "animal_id": "rat002", "session_id": "day1"},
    ]
    
    sessions = load_multiple_sessions(session_configs)
    print(f"Loaded {len(sessions)} sessions successfully")