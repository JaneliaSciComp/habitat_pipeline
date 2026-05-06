"""
Kilosort Data Import Module

Provides KilosortData (a dataclass holding spike-sorted electrophysiology data)
and load_kilosort_data() to load it from disk or a DataStorageManager.
"""

import datetime
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 30000.0


# ---------------------------------------------------------------------------
# KilosortData dataclass
# ---------------------------------------------------------------------------

@dataclass
class KilosortData:
    """Pure data container for Kilosort spike-sorting results.

    All I/O is handled by the standalone ``load_kilosort_data()`` function.
    This class only stores data and provides analysis methods that operate
    on in-memory arrays.
    """

    # Identity
    animal_id: str
    session_id: str

    # Core spike data
    spike_times: np.ndarray                  # raw sample indices from Kilosort
    spike_clusters: np.ndarray               # cluster assignment per spike
    spike_times_by_cell: List[np.ndarray]    # spike times (seconds) per cluster

    # Cluster properties
    ks_ids: List[int]
    channel: np.ndarray
    amplitude: np.ndarray
    fr: np.ndarray
    amp: np.ndarray
    DV: np.ndarray
    XX: np.ndarray
    cell_numbers: np.ndarray
    to_load: np.ndarray                      # bool mask: which clusters are "good"

    # Optional
    curated_cells: Optional[np.ndarray] = None
    cluster_info: Optional[pd.DataFrame] = None
    ks_labels: Optional[pd.DataFrame] = None
    channel_map: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    filter_results: Optional[Dict] = None

    # --- analysis methods (pure computation, no I/O) -----------------------

    @property
    def duration_seconds(self) -> float:
        """Total recording duration in seconds."""
        if self.spike_times is None or len(self.spike_times) == 0:
            return 0.0
        return (self.spike_times.max() - self.spike_times.min()) / SAMPLE_RATE

    def get_firing_rates(self, bin_size_sec: float = 1.0) -> Dict[int, float]:
        """Calculate mean firing rate (Hz) for every cluster."""
        duration = self.duration_seconds
        return {
            cid: len(self.spike_times_by_cell[i]) / duration
            for i, cid in enumerate(self.ks_ids)
        }

    def get_isi_statistics(self) -> Dict[int, Dict]:
        """Calculate inter-spike-interval statistics per cluster."""
        isi_stats: Dict[int, Dict] = {}
        for i, cid in enumerate(self.ks_ids):
            st = self.spike_times_by_cell[i]
            if len(st) > 1:
                isis = np.diff(st)
                isi_stats[cid] = {
                    'mean_isi': np.mean(isis),
                    'median_isi': np.median(isis),
                    'cv_isi': np.std(isis) / np.mean(isis),
                }
        return isi_stats

    def calculate_firing_pattern_metrics(self, time_bin_sec: float = 60.0) -> Dict[int, Dict]:
        """Calculate firing-rate, presence-ratio, and CV-ISI per cluster."""
        metrics: Dict[int, Dict] = {}
        duration = self.duration_seconds
        if duration <= 0:
            logger.warning("Recording duration is 0 — cannot calculate metrics")
            return metrics

        for i, cid in enumerate(self.ks_ids):
            st = self.spike_times_by_cell[i]
            if len(st) == 0:
                metrics[cid] = {'firing_rate': 0.0, 'presence_ratio': 0.0, 'cv_isi': float('inf')}
                continue

            firing_rate = len(st) / duration

            n_bins = int(np.ceil(duration / time_bin_sec))
            if n_bins > 0:
                bin_edges = np.linspace(st[0], st[0] + duration, n_bins + 1)
                spike_counts, _ = np.histogram(st, bins=bin_edges)
                presence_ratio = np.sum(spike_counts > 0) / n_bins
            else:
                presence_ratio = 0.0

            if len(st) > 1:
                isis = np.diff(st)
                mean_isi = np.mean(isis)
                cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else float('inf')
            else:
                cv_isi = float('inf')

            metrics[cid] = {
                'firing_rate': firing_rate,
                'presence_ratio': presence_ratio,
                'cv_isi': cv_isi,
            }
        return metrics

    def filter_cells_by_firing_patterns(
        self,
        min_firing_rate: float = 0.5,
        max_firing_rate: float = 100.0,
        min_presence_ratio: float = 0.8,
        max_cv_isi: float = 10.0,
        time_bin_sec: float = 60.0,
    ) -> Dict:
        """Filter clusters by firing-pattern quality metrics.

        Returns a dict with keys: passed_clusters, failed_clusters, metrics, summary.
        Also stores the result in ``self.filter_results``.
        """
        metrics = self.calculate_firing_pattern_metrics(time_bin_sec=time_bin_sec)
        passed, failed = [], {}

        for cid, m in metrics.items():
            reasons = []
            if m['firing_rate'] < min_firing_rate:
                reasons.append(f"firing_rate_too_low ({m['firing_rate']:.2f} < {min_firing_rate})")
            elif m['firing_rate'] > max_firing_rate:
                reasons.append(f"firing_rate_too_high ({m['firing_rate']:.2f} > {max_firing_rate})")
            if m['presence_ratio'] < min_presence_ratio:
                reasons.append(f"presence_ratio_too_low ({m['presence_ratio']:.3f} < {min_presence_ratio})")
            if m['cv_isi'] > max_cv_isi:
                reasons.append(f"cv_isi_too_high ({m['cv_isi']:.3f} > {max_cv_isi})")
            if reasons:
                failed[cid] = reasons
            else:
                passed.append(cid)

        total = len(metrics)
        self.filter_results = {
            'passed_clusters': passed,
            'failed_clusters': failed,
            'metrics': metrics,
            'summary': {
                'total_clusters': total,
                'passed_count': len(passed),
                'failed_count': len(failed),
                'pass_rate': len(passed) / total if total > 0 else 0.0,
            },
        }
        return self.filter_results

    def get_filtered_cells_spike_times(self, **filter_kwargs) -> List[np.ndarray]:
        """Return spike times (seconds) for clusters that pass filter_cells_by_firing_patterns."""
        results = self.filter_cells_by_firing_patterns(**filter_kwargs)
        passed = set(results['passed_clusters'])
        return [
            self.spike_times_by_cell[i]
            for i, cid in enumerate(self.ks_ids)
            if cid in passed
        ]

    def bin_spike_times(
        self,
        bin_size_sec: float = 1.0,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        filtered_only: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bin spike times into a firing-rate matrix (n_cells x n_bins).

        Parameters
        ----------
        bin_size_sec : float
            Width of each time bin in seconds.
        t_start : float, optional
            Start time in seconds. Defaults to the earliest spike.
        t_end : float, optional
            End time in seconds. Defaults to the latest spike.
        filtered_only : bool
            If True (default), use only cells that pass
            ``filter_cells_by_firing_patterns()``.  Runs it with default
            parameters if ``filter_results`` is not yet populated.

        Returns
        -------
        matrix : np.ndarray, shape (n_cells, n_bins)
            Firing-rate matrix in Hz (spike count / bin_size_sec).
        bin_centers : np.ndarray, shape (n_bins,)
            Time of each bin centre in seconds.
        """
        if filtered_only:
            if self.filter_results is None:
                self.filter_cells_by_firing_patterns()
            spike_times_list = self.get_filtered_cells_spike_times()
        else:
            spike_times_list = self.spike_times_by_cell

        if t_start is None:
            t_start = min(
                (st[0] for st in spike_times_list if len(st) > 0), default=0.0
            )
        if t_end is None:
            t_end = max(
                (st[-1] for st in spike_times_list if len(st) > 0), default=t_start + 1.0
            )

        bin_edges = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)
        n_bins = len(bin_edges) - 1
        n_cells = len(spike_times_list)

        matrix = np.zeros((n_cells, n_bins), dtype=np.float64)
        for i, st in enumerate(spike_times_list):
            if len(st) > 0:
                counts, _ = np.histogram(st, bins=bin_edges)
                matrix[i] = counts / bin_size_sec

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return matrix, bin_centers

    def print_firing_pattern_summary(self, filter_results: Optional[Dict] = None, **filter_kwargs):
        """Print a human-readable summary of quality metrics."""
        if filter_results is None:
            filter_results = self.filter_cells_by_firing_patterns(**filter_kwargs)

        metrics = filter_results['metrics']
        summary = filter_results['summary']

        print(f"\n=== Firing Pattern Quality Metrics Summary ===")
        print(f"Total clusters: {summary['total_clusters']}")
        print(f"Passed quality checks: {summary['passed_count']} ({summary['pass_rate']:.1%})")
        print(f"Failed quality checks: {summary['failed_count']}")

        if metrics:
            frs = [m['firing_rate'] for m in metrics.values()]
            prs = [m['presence_ratio'] for m in metrics.values()]
            cvs = [m['cv_isi'] for m in metrics.values() if np.isfinite(m['cv_isi'])]
            print(f"\nOverall Statistics:")
            print(f"  Firing Rate: {np.mean(frs):.2f} +/- {np.std(frs):.2f} Hz")
            print(f"  Presence Ratio: {np.mean(prs):.3f} +/- {np.std(prs):.3f}")
            if cvs:
                print(f"  CV ISI: {np.mean(cvs):.3f} +/- {np.std(cvs):.3f}")

            if filter_results['failed_clusters']:
                print(f"\nFailure reasons:")
                counts: Dict[str, int] = {}
                for reasons in filter_results['failed_clusters'].values():
                    for r in reasons:
                        key = r.split('(')[0].strip()
                        counts[key] = counts.get(key, 0) + 1
                for reason, cnt in sorted(counts.items()):
                    print(f"  {reason}: {cnt} clusters")

    def __repr__(self) -> str:
        n_spikes = len(self.spike_times) if self.spike_times is not None else 0
        n_clusters = len(self.ks_ids) if self.ks_ids is not None else 0
        return (
            f"KilosortData(animal={self.animal_id}, session={self.session_id}, "
            f"n_spikes={n_spikes}, n_clusters={n_clusters}, "
            f"duration={self.duration_seconds:.1f}s)"
        )


# ---------------------------------------------------------------------------
# I/O helper functions
# ---------------------------------------------------------------------------

def _locate_ks_folder(data_dir: Path) -> Optional[Path]:
    """Find the kilosort4 output folder starting from *data_dir*."""
    if data_dir.name.lower() == "kilosort4":
        return data_dir
    try:
        ks_parents = [p for p in data_dir.iterdir() if p.is_dir() and "kilosort" in p.name.lower()]
    except Exception:
        ks_parents = []
    if ks_parents:
        candidate = ks_parents[0] / "kilosort4"
        return candidate if candidate.exists() else None
    return None


def _extract_ids_from_path(ks_folder: Path) -> tuple:
    """Extract (animal_id, session_id) from the kilosort directory path."""
    animal_id, session_id = "unknown_animal", "unknown_session"
    if ks_folder.name.lower() == "kilosort4":
        session_folder = ks_folder.parent
        if session_folder.name.endswith("_merged.kilosort"):
            session_id = session_folder.name.replace("_merged.kilosort", "")
            animal_folder = session_folder.parent
            if animal_folder.name:
                animal_id = animal_folder.name
    else:
        parts = ks_folder.parts
        for i, part in enumerate(reversed(parts)):
            if part.endswith("_merged.kilosort"):
                session_id = part.replace("_merged.kilosort", "")
                idx = len(parts) - i - 2
                if idx >= 0:
                    animal_id = parts[idx]
                break
    return animal_id, session_id


def _read_timestamps(ks_folder: Path) -> np.ndarray:
    """Read sample indices from the first ``*.timestamps.dat`` in the parent directory."""
    parent = ks_folder.parent
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

    return np.frombuffer(rest, dtype="<u4").astype(np.uint64)


def _waveform_to_channel(ks_folder: Path) -> np.ndarray:
    """Find the channel with the largest peak-to-peak amplitude for each template."""
    templates = np.load(ks_folder / "templates.npy")
    n_templates = templates.shape[0]
    channels = np.empty(n_templates, dtype=int)
    for i in range(n_templates):
        ptp = templates[i].max(axis=0) - templates[i].min(axis=0)
        channels[i] = int(np.argmax(ptp))
    return channels


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _find_cached_file(ks_folder: Path) -> Optional[Path]:
    """Find the most recent cached pkl in *ks_folder*, preferring processed over full."""
    cached = sorted(ks_folder.glob("kilosort_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cached:
        return None
    processed = [f for f in cached if f.name.startswith("kilosort_processed_")]
    return processed[0] if processed else cached[0]


def _load_cached(filepath: Path) -> dict:
    """Load a cached pkl and return its raw dict."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _dict_to_kilosort_data(saved: dict) -> KilosortData:
    """Reconstruct a KilosortData from a saved dict."""
    def _get(key, default=None):
        v = saved.get(key, default)
        if key in ('KSfolder', 'data_dir') and isinstance(v, str):
            return Path(v)
        return v

    return KilosortData(
        animal_id=_get('animal_id', 'unknown_animal'),
        session_id=_get('session_id', 'unknown_session'),
        spike_times=_get('spike_times', np.array([])),
        spike_clusters=_get('spike_clusters', np.array([])),
        spike_times_by_cell=_get('spike_times_by_cell', []),
        ks_ids=_get('ks_ids', []),
        channel=_get('channel', np.array([])),
        amplitude=_get('amplitude', np.array([])),
        fr=_get('fr', np.array([])),
        amp=_get('amp', np.array([])),
        DV=_get('DV', np.array([])),
        XX=_get('XX', np.array([])),
        cell_numbers=_get('cell_numbers', np.array([])),
        to_load=_get('to_load', np.array([], dtype=bool)),
        curated_cells=_get('curated_cells'),
        cluster_info=_get('cluster_info'),
        ks_labels=_get('ks_labels'),
        channel_map=_get('channel_map'),
        metadata=_get('metadata', {}),
        filter_results=_get('filter_results'),
    )


def save_kilosort_data(
    ks_data: KilosortData,
    ks_folder: Path,
    filename: Optional[str] = None,
    exclude_large_arrays: bool = False,
) -> str:
    """Save a KilosortData instance to a pkl file in *ks_folder*.

    Returns the path to the saved file.
    """
    if filename is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "kilosort_processed" if exclude_large_arrays else "kilosort_full"
        filename = f"{prefix}_{ks_data.animal_id}_{ks_data.session_id}_{ts}.pkl"
    if not filename.endswith('.pkl'):
        filename += '.pkl'

    save_path = Path(ks_folder) / filename

    save_obj: Dict = {
        'animal_id': ks_data.animal_id,
        'session_id': ks_data.session_id,
        'ks_ids': ks_data.ks_ids,
        'channel': ks_data.channel,
        'amplitude': ks_data.amplitude,
        'fr': ks_data.fr,
        'amp': ks_data.amp,
        'DV': ks_data.DV,
        'XX': ks_data.XX,
        'cell_numbers': ks_data.cell_numbers,
        'to_load': ks_data.to_load,
        'curated_cells': ks_data.curated_cells,
        'spike_times_by_cell': ks_data.spike_times_by_cell,
        'metadata': ks_data.metadata,
        'cluster_info': ks_data.cluster_info,
        'ks_labels': ks_data.ks_labels,
        'KSfolder': str(ks_folder),
        'excluded_large_arrays': exclude_large_arrays,
    }
    if not exclude_large_arrays:
        save_obj['spike_times'] = ks_data.spike_times
        save_obj['spike_clusters'] = ks_data.spike_clusters
        save_obj['channel_map'] = ks_data.channel_map

    with open(save_path, 'wb') as f:
        pickle.dump(save_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = save_path.stat().st_size / (1024 * 1024)
    logger.info("Saved KilosortData to %s (%.2f MB)", save_path, size_mb)
    return str(save_path)


def load_kilosort_from_file(filepath: Union[str, Path]) -> KilosortData:
    """Load a previously saved KilosortData from a pkl file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Save file not found: {filepath}")
    saved = _load_cached(filepath)
    ks_data = _dict_to_kilosort_data(saved)
    logger.info(
        "Loaded KilosortData from %s (animal=%s, session=%s, clusters=%d)",
        filepath.name, ks_data.animal_id, ks_data.session_id, len(ks_data.ks_ids),
    )
    return ks_data


# ---------------------------------------------------------------------------
# Raw data loading pipeline (module-level functions)
# ---------------------------------------------------------------------------

def _load_spike_data(ks_folder: Path):
    """Load spike times, clusters, cluster_info, and ks_labels from Kilosort output."""
    spike_times_path = ks_folder / 'spike_times.npy'
    spike_clusters_path = ks_folder / 'spike_clusters.npy'
    cluster_info_path = ks_folder / 'cluster_info.tsv'
    channel_map_path = ks_folder / 'channel_map.npy'

    if not spike_times_path.exists() or not spike_clusters_path.exists():
        raise FileNotFoundError("Spike times or clusters file not found in the specified directory.")

    logger.info("Loading spike data...")
    spike_times = np.load(spike_times_path) - 31  # align with template center
    spike_clusters = np.load(spike_clusters_path)
    channel_map = np.load(channel_map_path) if channel_map_path.exists() else None

    if cluster_info_path.exists():
        cluster_info = pd.read_csv(cluster_info_path, sep='\t')
        cluster_info = cluster_info.sort_values(
            by=["depth", "cluster_id"], ascending=[False, True]
        ).reset_index(drop=True)
    else:
        logger.info("Cluster info file not found — session not curated, using KS labels.")
        cluster_info = None

    ks_labels = pd.read_csv(ks_folder / 'cluster_KSLabel.tsv', sep='\t')

    return spike_times, spike_clusters, channel_map, cluster_info, ks_labels


def _select_clusters(cluster_info, ks_labels):
    """Return (to_load, curated_cells) boolean arrays selecting 'good' clusters."""
    ci = cluster_info if cluster_info is not None else ks_labels
    mask = pd.Series(False, index=ci.index)
    curated_cells = None

    if "group" in ci.columns:
        mask = mask | (ci["group"].astype(str).str.lower() == "good")
        ci2 = ci.loc[mask, ["group"]]
        curated_cells = (ci2["group"].astype(str).str.lower() == "good").to_numpy(dtype=bool)

    elif "KSLabel" in ci.columns:
        mask = mask | (ci["KSLabel"].astype(str).str.lower() == "good")
    

    return mask.to_numpy(dtype=bool), curated_cells


def _extract_cluster_properties(ks_folder, to_load, cluster_info, ks_labels, channel_map):
    """Extract per-cluster properties (channel, amplitude, position, etc.)."""
    logger.info("Extracting cluster properties...")
    if cluster_info is None:
        ci = ks_labels
        ks_ids = ci["cluster_id"].tolist()
        ks_ids = [ks_ids[i] for i, load in enumerate(to_load) if load]
        channel = _waveform_to_channel(ks_folder)
        channel = np.array([channel[i] for i, load in enumerate(to_load) if load])
        ks_channel = channel_map[channel]
        amplitude_df = pd.read_csv(ks_folder / 'cluster_Amplitude.tsv', sep='\t')
        amplitude = amplitude_df.loc[amplitude_df["cluster_id"].isin(ks_ids), ["Amplitude"]]
        fr, amp = [], []
    else:
        ci = cluster_info
        ks_ids = ci["cluster_id"].tolist()
        ks_ids = [ks_ids[i] for i, load in enumerate(to_load) if load]
        ks_channel = ci.loc[ci["cluster_id"].isin(ks_ids), ["ch"]]
        amplitude = ci.loc[ci["cluster_id"].isin(ks_ids), ["Amplitude"]]
        amp = ci.loc[ci["cluster_id"].isin(ks_ids), ["amp"]]
        fr = ci.loc[ci["cluster_id"].isin(ks_ids), ["fr"]]
        channel_map_arr = np.array(channel_map)
        ks_channel = np.array(ks_channel)
        channel = np.array([np.where(channel_map_arr == a)[0][0] for a in ks_channel])

    channel_positions = np.load(ks_folder / 'channel_positions.npy')
    DV = channel_positions[tuple(channel), 1]
    XX = channel_positions[tuple(channel), 0]
    nn = len(ks_ids)
    i_shank = np.round(XX / 100.0).astype(int)
    cell_numbers = np.column_stack((i_shank, np.arange(nn, dtype=int)))

    return dict(
        ks_ids=ks_ids,
        channel=np.array(ks_channel).squeeze(),
        amplitude=np.array(amplitude).squeeze(),
        fr=np.array(fr).squeeze(),
        amp=np.array(amp).squeeze(),
        DV=DV,
        XX=XX,
        cell_numbers=cell_numbers,
    )


def _group_spikes_by_cluster(ks_folder, spike_times, spike_clusters, ks_ids):
    """Group spikes by cluster and convert to seconds."""
    logger.info("Grouping spikes by cluster...")
    sample_indices = _read_timestamps(ks_folder)
    spike_sec = sample_indices[spike_times].astype(float) / SAMPLE_RATE

    order = np.argsort(spike_clusters)
    sorted_clusters = spike_clusters[order]
    sorted_times = spike_sec[order]

    unique_ids, start_idx, counts = np.unique(sorted_clusters, return_index=True, return_counts=True)
    grouped = {}
    for uid, s, cnt in zip(unique_ids, start_idx, counts):
        grouped[int(uid)] = np.sort(sorted_times[s : s + cnt])

    return [grouped.get(int(c), np.array([], dtype=spike_sec.dtype)) for c in ks_ids]


# ---------------------------------------------------------------------------
# Main loader function
# ---------------------------------------------------------------------------

def load_kilosort_data(
    data_input: Union[str, Path],
    force_reload: bool = False,
) -> KilosortData:
    """Load Kilosort data from a path.

    Parameters
    ----------
    data_input : str or Path
        Path to the kilosort data directory.
    force_reload : bool
        If True, skip cached pkl files and load raw data from scratch.

    Returns
    -------
    KilosortData
    """
    # Resolve path and IDs
    data_dir = Path(data_input)
    ks_folder = _locate_ks_folder(data_dir)
    if ks_folder is None:
        raise FileNotFoundError(f"Could not locate kilosort4 folder under {data_dir}")
    animal_id, session_id = _extract_ids_from_path(ks_folder)

    # Try cache
    if not force_reload:
        cached = _find_cached_file(ks_folder)
        if cached is not None:
            saved = _load_cached(cached)
            ks_data = _dict_to_kilosort_data(saved)
            ks_data.animal_id = animal_id
            ks_data.session_id = session_id
            logger.info("Loaded cached KilosortData from %s", cached.name)
            return ks_data

    # Load raw data
    spike_times, spike_clusters, channel_map, cluster_info, ks_labels = _load_spike_data(ks_folder)
    to_load, curated_cells = _select_clusters(cluster_info, ks_labels)
    props = _extract_cluster_properties(ks_folder, to_load, cluster_info, ks_labels, channel_map)
    spike_times_by_cell = _group_spikes_by_cluster(ks_folder, spike_times, spike_clusters, props['ks_ids'])

    ks_data = KilosortData(
        animal_id=animal_id,
        session_id=session_id,
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        spike_times_by_cell=spike_times_by_cell,
        ks_ids=props['ks_ids'],
        channel=props['channel'],
        amplitude=props['amplitude'],
        fr=props['fr'],
        amp=props['amp'],
        DV=props['DV'],
        XX=props['XX'],
        cell_numbers=props['cell_numbers'],
        to_load=to_load,
        curated_cells=curated_cells,
        cluster_info=cluster_info,
        ks_labels=ks_labels,
        channel_map=channel_map,
        metadata={
            'data_path': str(ks_folder),
            'animal_id': animal_id,
            'session_id': session_id,
            'n_clusters': len(props['ks_ids']),
            'n_spikes': len(spike_times),
        },
    )
    return ks_data
