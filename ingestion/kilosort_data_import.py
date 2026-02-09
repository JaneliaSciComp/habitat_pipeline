import os
from pathlib import Path

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
    #    self.get_cluster_spikes_fast()
    #    self.find_buggy_periods()
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
        spike_SI = sample_indices[self.spike_times]
        spike_clusters = self.spike_clusters
        ks_ids = self.ks_ids
        order = np.argsort(spike_clusters)
        sorted_clusters = spike_clusters[order]
        sorted_spike_SI = spike_SI[order]

        # find boundaries for unique cluster ids
        unique_ids, start_idx, counts = np.unique(sorted_clusters, return_index=True, return_counts=True)

        # map cluster id -> array slice
        grouped = {}
        for uid, s, cnt in zip(unique_ids, start_idx, counts):
            # sort spikes within group to ensure ascending order
            grouped[int(uid)] = np.sort(sorted_spike_SI[s : s + cnt])

        # now collect for ks_ids (missing ids will not be present in grouped)
        allSpikeSI_fast = [grouped.get(int(c), np.array([], dtype=spike_SI.dtype)) for c in ks_ids]
        self.allSpikeSI = allSpikeSI_fast
        return allSpikeSI_fast

    def find_buggy_periods(self):
        """Identify and return periods of buggy performance."""
        def cluster_fr(c):
            if len(c) < 2:
                return np.nan
            denom = c[-1] - c[0]
            return (len(c) / denom * SAMPLE_RATE) if denom > 0 else np.nan

        # Ensure we have per-cluster spike lists
        allSpikeSI = getattr(self, 'allSpikeSI', None)
        if allSpikeSI is None:
            allSpikeSI = self.get_cluster_spikes_fast()
            self.allSpikeSI = allSpikeSI

        results = {}
        # Compute firing rates (fr) if missing
        fr = getattr(self, 'fr', None)
        if fr is None or (hasattr(fr, '__len__') and len(fr) == 0):
            fr = np.array([cluster_fr(c) for c in allSpikeSI], dtype=float)
            self.fr = fr

        # detect dropped packets in SampleIndices (diff > 1)
        SampleIndices = self.read_timestamps()
        i_drop_start = np.where(np.diff(SampleIndices) > 1)[0]
        if i_drop_start.size > 0:
            dropped_SI = np.vstack((SampleIndices[i_drop_start], SampleIndices[i_drop_start + 1])).T
        else:
            dropped_SI = np.empty((0, 2), dtype=SampleIndices.dtype)
        results['dropped_SI'] = dropped_SI

        # select units for stripe detection: ks_ids where fr < 50
        # units with fr below 50Hz are selected. The ones that go above are prone to be artifacts
        ks_ids = getattr(self, 'ks_ids', [])
        fr_arr = np.asarray(fr)
        ks_ids_array = np.asarray(ks_ids)
        if ks_ids_array.size and fr_arr.size == ks_ids_array.size:
            ks_ids_stripe_detection = ks_ids_array[fr_arr < 50].tolist()
        else:
            # fallback: try length-based mapping
            ks_ids_stripe_detection = ks_ids_array[fr_arr < 50].tolist() if ks_ids_array.size else []
        results['ks_ids_stripe_detection'] = ks_ids_stripe_detection

        # decide the shortest stripe duration in ms based on sum of low-FR rates
        sum_low_fr = float(fr_arr[fr_arr < 50].sum()) if fr_arr.size else 0.0
        if sum_low_fr > 2000:
            th = 10
        elif sum_low_fr > 1000:
            th = 30
        else:
            th = 50
        results['th'] = th

        # collect spike sample indices for detection of stripes
        spike_SI = SampleIndices[self.spike_times]
        # if len(ks_ids_stripe_detection) > 0:
        #     mask = np.isin(self.spike_clusters, ks_ids_stripe_detection)
        #     spikes_for_detection = spike_SI[mask]
        # else:
        #     spikes_for_detection = np.array([], dtype=spike_SI.dtype)
        spikes_for_detection = spike_SI

        # find recording stripes (large gaps) in spikes_for_detection
        #  using the decided threshold th
        diffs = np.diff(spikes_for_detection) / (SAMPLE_RATE / 1000.0)
        i_stripes = np.where(diffs > th)[0]
        stripe_SI = np.vstack((spikes_for_detection[i_stripes], spikes_for_detection[i_stripes + 1])).T
        results['stripe_SI'] = stripe_SI

        # merge stripe_SI and dropped_SI into non-overlapping intervals
        # both arrays have shape (N,2) with [start, end]
        combined = None
        if dropped_SI.size and stripe_SI.size:
            combined = np.vstack((dropped_SI, stripe_SI))
        elif dropped_SI.size:
            combined = dropped_SI.copy()
        elif stripe_SI.size:
            combined = stripe_SI.copy()
        else:
            combined = np.empty((0, 2), dtype=SampleIndices.dtype)

        if combined.size == 0:
            buggy_SI = combined
        else:
            # sort by start time
            order = np.argsort(combined[:, 0])
            rows = combined[order]
            merged = []
            cur_start, cur_end = int(rows[0, 0]), int(rows[0, 1])
            for s, e in rows[1:]:
                s, e = int(s), int(e)
                if s <= cur_end:  # overlap or contiguous
                    cur_end = max(cur_end, e)
                else:
                    merged.append((cur_start, cur_end))
                    cur_start, cur_end = s, e
            merged.append((cur_start, cur_end))
            buggy_SI = np.array(merged, dtype=SampleIndices.dtype)

        results['buggy_SI'] = buggy_SI
        buggy_samples = np.diff(buggy_SI).sum()

        print(f"Found {len(buggy_SI)} buggy periods with total duration {buggy_samples / SAMPLE_RATE:.2f} s")
        self.buggy_periods = results
        return results

    def get_spike_data(self):
        """Return spike times and cluster assignments."""
        if self.spike_times is None or self.spike_clusters is None:
            raise ValueError("Spike data not loaded. Call load_spike_data() first.")
        
        return self.spike_times, self.spike_clusters, self.cluster_info
    
    def __repr__(self) -> str:
        """String representation of the KilosortData object."""
        n_spikes = len(self.spike_times) if self.spike_times is not None else 0
        n_clusters = len(self.ks_ids) if self.ks_ids is not None else 0
        duration = 0
        if self.spike_times is not None:
            duration = (self.spike_times.max() - self.spike_times.min()) / SAMPLE_RATE
        
        return (f"KilosortData(animal={self.animal_id}, session={self.session_id}, "
                f"n_spikes={n_spikes}, n_clusters={n_clusters}, duration={duration:.1f}s)")