# KilosortData - Electrophysiology Data Import

A Python dataclass and loader for Kilosort 4 spike-sorted electrophysiology
data, used by the habitat_pipeline multi-animal analysis system.

The module is split into two parts:

- `KilosortData` — a pure data container with analysis methods that operate
  on already-loaded arrays (no I/O).
- Module-level functions (`load_kilosort_data`, `save_kilosort_data`,
  `load_kilosort_from_file`) that handle disk I/O and caching.

## Quick Start

```python
from ingestion.kilosort_data_import import load_kilosort_data

# Load from a session/animal directory or directly from a kilosort4 folder.
# animal_id and session_id are inferred from the path layout.
ks_data = load_kilosort_data("path/to/animal/session_merged.kilosort/kilosort4")

print(ks_data)
# KilosortData(animal=..., session=..., n_spikes=..., n_clusters=..., duration=...s)

# Access spikes per cluster (already in seconds, aligned via timestamps file)
spikes_cluster_0 = ks_data.spike_times_by_cell[0]

# Quality-filter clusters and bin into a firing-rate matrix
matrix, bin_centers = ks_data.bin_spike_times(bin_size_sec=0.025)
```

## KilosortData

Dataclass fields:

**Identity**
- `animal_id: str`
- `session_id: str`

**Core spike data**
- `spike_times: np.ndarray` — raw sample indices from Kilosort
- `spike_clusters: np.ndarray` — cluster assignment per spike
- `spike_times_by_cell: List[np.ndarray]` — spike times in seconds, one
  array per cluster in `ks_ids`

**Cluster properties** (one entry per kept cluster, parallel to `ks_ids`)
- `ks_ids: List[int]` — cluster IDs that passed `_select_clusters`
- `channel: np.ndarray` — peak channel per cluster
- `amplitude: np.ndarray` — template amplitude
- `fr: np.ndarray` — Kilosort-reported firing rate (only when curated)
- `amp: np.ndarray` — Kilosort-reported amp (only when curated)
- `DV: np.ndarray` — dorsoventral channel position (channel_positions[:, 1])
- `XX: np.ndarray` — mediolateral channel position (channel_positions[:, 0])
- `cell_numbers: np.ndarray` — `(shank_index, within-shank index)` per cluster
- `to_load: np.ndarray` — bool mask over the full cluster table marking
  "good" clusters

**Optional**
- `curated_cells: Optional[np.ndarray]` — bool array, present when manual
  curation is available
- `cluster_info: Optional[pd.DataFrame]` — `cluster_info.tsv` (post-curation)
- `ks_labels: Optional[pd.DataFrame]` — `cluster_KSLabel.tsv`
- `channel_map: Optional[np.ndarray]`
- `metadata: Dict`
- `filter_results: Optional[Dict]` — populated by
  `filter_cells_by_firing_patterns`

### Methods

All methods are pure computation over the in-memory arrays.

#### `duration_seconds` (property)

Total recording duration: `(spike_times.max() - spike_times.min()) / 30000`.

#### `get_firing_rates(bin_size_sec=1.0) -> Dict[int, float]`

Mean firing rate (Hz) for every cluster, keyed by cluster ID.

#### `get_isi_statistics() -> Dict[int, Dict]`

Per-cluster ISI stats: `mean_isi`, `median_isi`, `cv_isi`. Clusters with
fewer than 2 spikes are omitted.

#### `calculate_firing_pattern_metrics(time_bin_sec=60.0) -> Dict[int, Dict]`

Per-cluster quality metrics: `firing_rate`, `presence_ratio` (fraction of
`time_bin_sec`-wide bins containing at least one spike), and `cv_isi`.

#### `filter_cells_by_firing_patterns(...)`

```python
ks_data.filter_cells_by_firing_patterns(
    min_firing_rate=0.5,
    max_firing_rate=100.0,
    min_presence_ratio=0.8,
    max_cv_isi=10.0,
    time_bin_sec=60.0,
)
```

Computes metrics and applies thresholds. Returns (and stores in
`self.filter_results`) a dict with:

- `passed_clusters: List[int]`
- `failed_clusters: Dict[int, List[str]]` — cluster ID → reasons
- `metrics: Dict[int, Dict]`
- `summary: {total_clusters, passed_count, failed_count, pass_rate}`

#### `get_filtered_cells_spike_times(**filter_kwargs) -> List[np.ndarray]`

Runs `filter_cells_by_firing_patterns` and returns the spike-time arrays
(seconds) for clusters that passed.

#### `bin_spike_times(bin_size_sec=1.0, t_start=None, t_end=None, filtered_only=True)`

Bins spike times into a firing-rate matrix.

- If `filtered_only=True` (default), only clusters that pass
  `filter_cells_by_firing_patterns` are included; if no filter has been
  run, defaults are applied.
- `t_start` / `t_end` default to the earliest / latest spike across the
  selected clusters.

Returns:
- `matrix: np.ndarray, shape (n_cells, n_bins)` — rates in Hz
- `bin_centers: np.ndarray, shape (n_bins,)` — centre of each bin in seconds

#### `print_firing_pattern_summary(filter_results=None, **filter_kwargs)`

Prints a human-readable summary of metrics and per-reason failure counts.
Runs the filter if `filter_results` is not provided.

## I/O Functions

### `load_kilosort_data(data_input, force_reload=False) -> KilosortData`

Top-level loader.

```python
from ingestion.kilosort_data_import import load_kilosort_data

ks_data = load_kilosort_data("path/to/data", force_reload=False)
```

Behaviour:

1. Resolves the kilosort4 folder. `data_input` may be the kilosort4 folder
   itself or a parent directory containing exactly one `*kilosort*`
   subfolder with a `kilosort4` child.
2. Infers `animal_id` and `session_id` from the path. The expected layout
   is `.../<animal_id>/<session_id>_merged.kilosort/kilosort4/`.
3. Unless `force_reload=True`, returns the most recent cached pkl found in
   the kilosort4 folder (preferring `kilosort_processed_*.pkl` over
   `kilosort_full_*.pkl`).
4. Otherwise loads raw data: spike times/clusters, `cluster_info.tsv` (if
   present, otherwise falls back to `cluster_KSLabel.tsv`),
   `channel_map.npy`, `channel_positions.npy`, and the
   `*.timestamps.dat` file in the parent directory; selects "good"
   clusters; and groups spikes by cluster (converted to seconds).

Notes on raw loading:
- `spike_times.npy` is shifted by `-31` samples to align with the template
  centre.
- Spike sample indices are remapped through the timestamps file before
  conversion to seconds (sample rate = 30 kHz).
- "Good" cluster selection uses `cluster_info["group"] == "good"` when
  curated, otherwise `cluster_KSLabel["KSLabel"] == "good"`.

### `save_kilosort_data(ks_data, ks_folder, filename=None, exclude_large_arrays=False) -> str`

Pickles a `KilosortData` to `ks_folder`.

- If `filename` is omitted, names the file
  `kilosort_full_<animal>_<session>_<timestamp>.pkl` (or
  `kilosort_processed_...` when `exclude_large_arrays=True`).
- With `exclude_large_arrays=True`, omits `spike_times`, `spike_clusters`,
  and `channel_map` to produce a smaller file suitable for downstream
  analysis that only needs `spike_times_by_cell`.

### `load_kilosort_from_file(filepath) -> KilosortData`

Loads a previously saved pkl directly, bypassing path-based ID inference.

## Expected On-Disk Layout

```
<animal_id>/
└── <session_id>_merged.kilosort/
    ├── <something>.timestamps.dat        # required (sample-index remap)
    └── kilosort4/
        ├── spike_times.npy               # required
        ├── spike_clusters.npy            # required
        ├── cluster_KSLabel.tsv           # required
        ├── channel_map.npy               # required for property extraction
        ├── channel_positions.npy         # required for DV/XX
        ├── templates.npy                 # required when no cluster_info.tsv
        ├── cluster_info.tsv              # optional (post-curation)
        ├── cluster_Amplitude.tsv         # required when no cluster_info.tsv
        └── kilosort_*.pkl                # optional cache files
```

When `cluster_info.tsv` is absent, peak channels are recomputed from
`templates.npy` (max peak-to-peak across channels).

## Sample Rate

`SAMPLE_RATE = 30000.0` is hard-coded at the top of the module. Change it
there if your acquisition rate differs.
