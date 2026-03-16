import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

SAMPLE_RATE = 30000.0


class KilosortData:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.cluster_info = None
        self.locate_KS_folder()
        self.extract_ids_from_path()
        self.load_spike_data()
        self.select_clusters()
        self.extract_cluster_properties()
        self.get_cluster_spikes_fast()
        self.metadata = {
            'data_path': str(self.KSfolder),
            'animal_id': self.animal_id,
            'session_id': self.session_id,
            'n_clusters': len(self.ks_ids),
            'n_spikes': len(self.spike_times),
        }

    def locate_KS_folder(self):
        # Check if data_dir already ends with "kilosort4"
        if self.data_dir.name.lower() == "kilosort4":
            self.KSfolder = self.data_dir
            return
        
        # locate Kilosort output folder: look for first subfolder matching '*kilosort*'
        try:
            kilosort_parents = [p for p in self.data_dir.iterdir() if p.is_dir() and "kilosort" in p.name.lower()]
        except Exception:
            kilosort_parents = []

        if kilosort_parents:
            ks_parent = kilosort_parents[0]
            ks_candidate = ks_parent / "kilosort4"
            self.KSfolder = ks_candidate if ks_candidate.exists() else None
        else:
            self.KSfolder = None

    def extract_ids_from_path(self):
        """
        Extract animal_id and session_id from the directory path.
        
        Expected path format: .../animalID/sessionID_merged.kilosort/kilosort4
        """
        path_parts = self.KSfolder.parts
        
        # Default values
        self.animal_id = "unknown_animal"
        self.session_id = "unknown_session"
        
        # Navigate up the path to find the expected structure
        current_path = self.KSfolder
        
        # If we're in kilosort4 folder, go up one level to sessionID_merged.kilosort
        if current_path.name.lower() == "kilosort4":
            session_folder = current_path.parent
            if session_folder.name.endswith("_merged.kilosort"):
                # Extract session ID by removing "_merged.kilosort" suffix
                self.session_id = session_folder.name.replace("_merged.kilosort", "")
                
                # Go up one more level to get animal ID
                animal_folder = session_folder.parent
                if animal_folder.name:
                    self.animal_id = animal_folder.name
        else:
            # If not in kilosort4 folder, try to find the structure in the path
            for i, part in enumerate(reversed(path_parts)):
                if part.endswith("_merged.kilosort"):
                    # Found session folder
                    self.session_id = part.replace("_merged.kilosort", "")
                    
                    # Animal ID should be the parent folder
                    animal_index = len(path_parts) - i - 2
                    if animal_index >= 0:
                        self.animal_id = path_parts[animal_index]
                    break

    def load_spike_data(self):
        """Load spike times and cluster assignments from Kilosort output files."""
        spike_times_path = self.KSfolder / 'spike_times.npy'
        spike_clusters_path = self.KSfolder / 'spike_clusters.npy'
        cluster_info_path = self.KSfolder / 'cluster_info.tsv'
        channel_map_path = self.KSfolder / 'channel_map.npy'

        if not spike_times_path.exists() or not spike_clusters_path.exists():
            raise FileNotFoundError("Spike times or clusters file not found in the specified directory.")

        print("Loading spike data...")
        self.spike_times = np.load(spike_times_path) - 31 # to align with the middle of the template
        self.spike_clusters = np.load(spike_clusters_path)
        self.channel_map = np.load(channel_map_path) if channel_map_path.exists() else None

        if cluster_info_path.exists():
            cluster_info = pd.read_csv(cluster_info_path, sep='\t')
            cluster_info = cluster_info.sort_values(by=["depth","cluster_id"], ascending=[False, True]).reset_index(drop=True)
            self.cluster_info = cluster_info
        else:
            print("Cluster info file not found. Using KS labels.")
            self.cluster_info = None
        self.ks_labels = pd.read_csv(self.KSfolder / 'cluster_KSLabel.tsv',sep = '\t')

    def select_clusters(self):
        """Select specific clusters to load based on provided cluster IDs."""
        print("Selecting clusters...")
        if self.cluster_info is None:
            ci = self.ks_labels
        else:
            ci = self.cluster_info
        mask = pd.Series(False, index=ci.index)
        if "KSLabel" in ci.columns:
            mask = mask | (ci["KSLabel"].astype(str).str.lower() == "good")
        if "group" in ci.columns:
            mask = mask | (ci["group"].astype(str).str.lower() == "good")
            ci2 = ci.loc[mask, ["group"]]
            mask2 = (ci2["group"].astype(str).str.lower() == "good")
            self.curated_cells = mask2.to_numpy(dtype=bool)
        # store as numpy boolean array for downstream code expecting array
        self.to_load = mask.to_numpy(dtype=bool)

    def extract_cluster_properties(self):
        """Extract properties like channel, amplitude, firing rate for selected clusters."""
        channel_map = self.channel_map
        to_load = self.to_load
        print("Extracting cluster properties...")
        if self.cluster_info is None:
            print("The session is not curated! Using KS labels.")
            ci = self.ks_labels
            ks_ids = ci["cluster_id"].tolist()
            ks_ids = [ks_ids[i] for i, load in enumerate(to_load) if load]
            channel = self.waveform2channel()
            channel = np.array([channel[i] for i, load in enumerate(to_load) if load])
            ks_channel = channel_map[channel]
            amplitude_df = pd.read_csv(self.KSfolder / 'cluster_Amplitude.tsv', sep='\t') 
            amplitude = amplitude_df.loc[amplitude_df["cluster_id"].isin(ks_ids), ["Amplitude"]]
            fr = []
            amp = []
        else:
            ci = self.cluster_info
            ks_ids = ci["cluster_id"].tolist()
            ks_ids = [ks_ids[i] for i, load in enumerate(to_load) if load]
            ks_channel = ci.loc[ci["cluster_id"].isin(ks_ids), ["ch"]]
            amplitude = ci.loc[ci["cluster_id"].isin(ks_ids), ["Amplitude"]]
            amp = ci.loc[ci["cluster_id"].isin(ks_ids), ["amp"]]
            fr = ci.loc[ci["cluster_id"].isin(ks_ids), ["fr"]]
            channel_map = np.array(channel_map)   # ensure numpy array
            ks_channel = np.array(ks_channel)
            channel = np.array([np.where(channel_map == a)[0][0] for a in ks_channel])

        channel_positions = np.load(self.KSfolder / 'channel_positions.npy') 
        DV = channel_positions[tuple(channel),1]
        XX = channel_positions[tuple(channel),0]
        nn = len(ks_ids)
        i_shank = np.round(XX / 100.0).astype(int)
        cell_numbers = np.column_stack((i_shank, np.arange(0, nn, dtype=int)))
        self.cell_numbers = cell_numbers

        self.channel = ks_channel.squeeze()
        self.amplitude = np.array(amplitude).squeeze()
        self.fr = np.array(fr).squeeze()
        self.amp = np.array(amp).squeeze()
        self.ks_ids = ks_ids
        self.DV = DV
        self.XX = XX

    def waveform2channel(self):
        """
        Find the channel with the largest peak-to-peak amplitude for each template.
        """
        p = Path(self.KSfolder)
        templates_path = p / "templates.npy"
        templates = np.load(templates_path) 

        n_templates = templates.shape[0]
        channels = np.empty(n_templates, dtype=int)

        for i in range(n_templates):
            T = templates[i]  # 2D array for this template
            ptp_per_channel = T.max(axis=0) - T.min(axis=0)
            # pick channel with largest peak-to-peak; argmax returns first on ties
            ch_idx = int(np.argmax(ptp_per_channel))
            channels[i] = ch_idx

        return channels

    def read_timestamps(self):
        """
        Locate the first '*.timestamps.dat' file in the parent directory of KSfolder
        and return the sample indices as a numpy array (dtype=np.uint64).
        """
        ks = Path(self.KSfolder)
        parent = ks.parent
        try:
            fpath = next(parent.glob("*.timestamps.dat"))
        except StopIteration:
            raise FileNotFoundError(f"No '*.timestamps.dat' found in {parent}")
        
        n_header = 25
        with open(fpath, "rb") as fid:
            if fpath.name != "sd_in_env.timestamps.dat":
                for _ in range(n_header):
                    fid.readline()
            rest = fid.read()

        if len(rest) % 4 != 0:
            raise ValueError("Timestamps file length (after header) is not a multiple of 4 bytes")

        # Interpret bytes as little-endian unsigned 32-bit integers
        samples = np.frombuffer(rest, dtype="<u4").astype(np.uint64)
        return samples

    def get_cluster_spikes(self):
        sample_indices = self.read_timestamps()
        spike_SI = sample_indices[self.spike_times]
        allSpikeSI = [spike_SI[self.spike_clusters == int(c)] for c in self.ks_ids]
        return allSpikeSI
    
    def get_cluster_spikes_fast(self):
        """Return a list of numpy arrays, each containing spike sample indices for a cluster."""
        print("Grouping spikes by cluster...")
        sample_indices = self.read_timestamps()
        spike_times = sample_indices[self.spike_times].astype(float) / SAMPLE_RATE  # convert to seconds
        spike_clusters = self.spike_clusters
        ks_ids = self.ks_ids
        order = np.argsort(spike_clusters)
        sorted_clusters = spike_clusters[order]
        sorted_spike_times = spike_times[order]

        # find boundaries for unique cluster ids
        unique_ids, start_idx, counts = np.unique(sorted_clusters, return_index=True, return_counts=True)

        # map cluster id -> array slice
        grouped = {}
        for uid, s, cnt in zip(unique_ids, start_idx, counts):
            # sort spikes within group to ensure ascending order
            grouped[int(uid)] = np.sort(sorted_spike_times[s : s + cnt])

        # now collect for ks_ids (missing ids will not be present in grouped)
        spike_times_by_cell = [grouped.get(int(c), np.array([], dtype=spike_times.dtype)) for c in ks_ids]
        self.spike_times_by_cell = spike_times_by_cell
        return spike_times_by_cell
    
    @property
    def duration_seconds(self):
        """Calculate total recording duration in seconds"""
        if self.spike_times is None or len(self.spike_times) == 0:
            return 0.0
        return (self.spike_times.max() - self.spike_times.min()) / SAMPLE_RATE
    
    def get_firing_rates(self, bin_size_sec=1.0):
        """Calculate firing rates for all clusters"""
        rates = {}
        duration = self.duration_seconds
        for i, cluster_id in enumerate(self.ks_ids):
            spike_times = self.spike_times_by_cell[i]
            rates[cluster_id] = len(spike_times) / duration
        return rates

    def get_isi_statistics(self):
        """Calculate inter-spike interval statistics"""
        isi_stats = {}
        for i, cluster_id in enumerate(self.ks_ids):
            spike_times = self.spike_times_by_cell[i] 
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                isi_stats[cluster_id] = {
                    'mean_isi': np.mean(isis),
                    'median_isi': np.median(isis),
                    'cv_isi': np.std(isis) / np.mean(isis)
                }
        return isi_stats

    def calculate_firing_pattern_metrics(self, time_bin_sec=60.0):
        """
        Calculate firing pattern quality metrics (7-9):
        - Firing Rate: Average spikes per second
        - Presence Ratio: Fraction of recording time with detectable activity
        - Coefficient of Variation: Variability in interspike intervals
        
        Parameters:
        -----------
        time_bin_sec : float, default=60.0
            Time bin size in seconds for calculating presence ratio
            
        Returns:
        --------
        dict : Dictionary with cluster_id as keys and metrics as values
        """
        metrics = {}
        duration = self.duration_seconds
        
        if duration <= 0:
            print("Warning: Recording duration is 0, cannot calculate metrics")
            return metrics
            
        for i, cluster_id in enumerate(self.ks_ids):
            spike_times = self.spike_times_by_cell[i]
            
            if len(spike_times) == 0:
                metrics[cluster_id] = {
                    'firing_rate': 0.0,
                    'presence_ratio': 0.0,
                    'cv_isi': float('inf')
                }
                continue
                
            # 1. Firing Rate (spikes/second)
            firing_rate = len(spike_times) / duration
            
            # 2. Presence Ratio (fraction of time bins with spikes)
            n_bins = int(np.ceil(duration / time_bin_sec))
            if n_bins > 0:
                bin_edges = np.linspace(spike_times[0] if len(spike_times) > 0 else 0, 
                                      spike_times[0] + duration if len(spike_times) > 0 else duration, 
                                      n_bins + 1)
                spike_counts, _ = np.histogram(spike_times, bins=bin_edges)
                presence_ratio = np.sum(spike_counts > 0) / n_bins
            else:
                presence_ratio = 0.0
            
            # 3. Coefficient of Variation of ISI
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                mean_isi = np.mean(isis)
                cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else float('inf')
            else:
                cv_isi = float('inf')
                
            metrics[cluster_id] = {
                'firing_rate': firing_rate,
                'presence_ratio': presence_ratio,
                'cv_isi': cv_isi
            }
            
        return metrics
    
    def filter_cells_by_firing_patterns(self, 
                                       min_firing_rate=0.5,
                                       max_firing_rate=100.0,
                                       min_presence_ratio=0.8,
                                       max_cv_isi=10.0,
                                       time_bin_sec=60.0):
        """
        Filter cells based on firing pattern quality metrics.
        
        Parameters:
        -----------
        min_firing_rate : float, default=0.5
            Minimum acceptable firing rate (Hz)
        max_firing_rate : float, default=100.0
            Maximum acceptable firing rate (Hz)
        min_presence_ratio : float, default=0.8
            Minimum fraction of time bins with activity (0-1)
        max_cv_isi : float, default=10.0
            Maximum coefficient of variation for ISI
        time_bin_sec : float, default=60.0
            Time bin size for presence ratio calculation
            
        Returns:
        --------
        dict : Dictionary with filtering results
            - 'passed_clusters': list of cluster IDs that passed all criteria
            - 'failed_clusters': dict with cluster IDs and reasons for failure
            - 'metrics': dict with calculated metrics for all clusters
            - 'summary': dict with counts and percentages
        """
        # Calculate metrics
        metrics = self.calculate_firing_pattern_metrics(time_bin_sec=time_bin_sec)
        
        passed_clusters = []
        failed_clusters = {}
        
        for cluster_id, cluster_metrics in metrics.items():
            reasons = []
            
            # Check firing rate
            fr = cluster_metrics['firing_rate']
            if fr < min_firing_rate:
                reasons.append(f"firing_rate_too_low ({fr:.2f} < {min_firing_rate})")
            elif fr > max_firing_rate:
                reasons.append(f"firing_rate_too_high ({fr:.2f} > {max_firing_rate})")
            
            # Check presence ratio
            pr = cluster_metrics['presence_ratio']
            if pr < min_presence_ratio:
                reasons.append(f"presence_ratio_too_low ({pr:.3f} < {min_presence_ratio})")
            
            # Check CV ISI
            cv = cluster_metrics['cv_isi']
            if cv > max_cv_isi:
                reasons.append(f"cv_isi_too_high ({cv:.3f} > {max_cv_isi})")
            
            if len(reasons) == 0:
                passed_clusters.append(cluster_id)
            else:
                failed_clusters[cluster_id] = reasons
        
        # Summary statistics
        total_clusters = len(metrics)
        passed_count = len(passed_clusters)
        failed_count = len(failed_clusters)
        
        summary = {
            'total_clusters': total_clusters,
            'passed_count': passed_count,
            'failed_count': failed_count,
            'pass_rate': passed_count / total_clusters if total_clusters > 0 else 0.0
        }
        
        return {
            'passed_clusters': passed_clusters,
            'failed_clusters': failed_clusters,
            'metrics': metrics,
            'summary': summary
        }
    
    def print_firing_pattern_summary(self, filter_results=None, **filter_kwargs):
        """
        Print a summary of firing pattern quality metrics and filtering results.
        
        Parameters:
        -----------
        filter_results : dict, optional
            Results from filter_cells_by_firing_patterns(). If None, will run filtering.
        **filter_kwargs : keyword arguments
            Parameters to pass to filter_cells_by_firing_patterns() if filter_results is None
        """
        if filter_results is None:
            filter_results = self.filter_cells_by_firing_patterns(**filter_kwargs)
        
        metrics = filter_results['metrics']
        summary = filter_results['summary']
        
        print(f"\n=== Firing Pattern Quality Metrics Summary ===")
        print(f"Total clusters: {summary['total_clusters']}")
        print(f"Passed quality checks: {summary['passed_count']} ({summary['pass_rate']:.1%})")
        print(f"Failed quality checks: {summary['failed_count']}")
        
        if len(metrics) > 0:
            # Calculate overall statistics
            firing_rates = [m['firing_rate'] for m in metrics.values()]
            presence_ratios = [m['presence_ratio'] for m in metrics.values()]
            cv_isis = [m['cv_isi'] for m in metrics.values() if np.isfinite(m['cv_isi'])]
            
            print(f"\nOverall Statistics:")
            print(f"  Firing Rate: {np.mean(firing_rates):.2f} ± {np.std(firing_rates):.2f} Hz")
            print(f"  Presence Ratio: {np.mean(presence_ratios):.3f} ± {np.std(presence_ratios):.3f}")
            if len(cv_isis) > 0:
                print(f"  CV ISI: {np.mean(cv_isis):.3f} ± {np.std(cv_isis):.3f}")
            
            # Show failure reasons
            if filter_results['failed_clusters']:
                print(f"\nFailure reasons:")
                failure_counts = {}
                for reasons in filter_results['failed_clusters'].values():
                    for reason in reasons:
                        failure_type = reason.split('(')[0].strip()
                        failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
                
                for reason, count in sorted(failure_counts.items()):
                    print(f"  {reason}: {count} clusters")

    def save_processed_data(self, cache_dir=None):
        """Save processed data for faster subsequent loading"""
        cache_path = cache_dir or self.KSfolder / 'processed_cache.pkl'
        cache_data = {
            'spike_times_by_cell': self.spike_times_by_cell,
            'ks_ids': self.ks_ids,
            'metadata': self.metadata
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def __repr__(self) -> str:
        """String representation of the KilosortData object."""
        n_spikes = len(self.spike_times) if self.spike_times is not None else 0
        n_clusters = len(self.ks_ids) if self.ks_ids is not None else 0
        duration = 0
        if self.spike_times is not None:
            duration = (self.spike_times.max() - self.spike_times.min()) / SAMPLE_RATE
        
        return (f"KilosortData(animal={self.animal_id}, session={self.session_id}, "
                f"n_spikes={n_spikes}, n_clusters={n_clusters}, duration={duration:.1f}s)")