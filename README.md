# Habitat Pipeline

Analysis pipeline for multi-animal electrophysiology and behavioral data. Handles data ingestion, ephys–behavior synchronization, spike sorting output processing, neural decoding, and visualization for freely behaving animals recorded simultaneously.

## Quick Start

```python
from ingestion.data_paths import DataStorageManager
from ingestion.kilosort_data_import import KilosortData
from video.behavioral_events import BehavioralEventsData
from ingestion.ephys_sync import DataSyncManager
from ephys.decode_opponent_identity import decode_opponent_identity_population

# Initialize — discovers all session data files automatically
data_manager = DataStorageManager("631", "20251216", auto_load=True)

# Load electrophysiology and behavioral data
ks_data = KilosortData(data_manager)
behavior_data = BehavioralEventsData(data_manager)

# Synchronize behavioral timestamps with ephys clock
sync_manager = DataSyncManager(data_manager, dio_channel=1)
behavior_data.synchronize_with_ephys(sync_manager, create_new_columns=True)

# Decode opponent identity from neural activity
results = decode_opponent_identity_population(
    ks_data=ks_data,
    behavior_data=behavior_data,
    animal_of_interest="631",
    behavior_type='EC',  # Encounter events
    use_quality_cells=True
)
```

## Project Structure

```
habitat_pipeline/
├── config/                         # Path configuration (JSON)
│   └── default_paths.json
├── ingestion/                      # Data loading & preprocessing
│   ├── data_paths.py              # Centralized path management (DataStorageManager)
│   ├── kilosort_data_import.py    # Kilosort 4 data loading & quality metrics
│   ├── kilosort_data.py           # Low-level Kilosort file parsing
│   ├── ephys_sync.py              # Ephys ↔ behavior synchronization
│   └── trodes_to_python.py        # SpikeGadgets Trodes binary reader
├── video/                          # Behavioral tracking & events
│   ├── tracking_import.py         # Position tracking data loader
│   ├── behavioral_events.py       # Behavioral event management
│   ├── behavioral_visualization.py # Event heatmaps & timelines
│   └── plot_trajectory.py         # Trajectory & occupancy plots
├── ephys/                          # Electrophysiology analysis
│   ├── decode_opponent_identity.py # LDA-based opponent decoding
│   ├── population_geometry.py     # Population dynamics & dimensionality reduction
│   └── plot_ephys_qa_stats.py     # Quality assessment visualization
└── workflow.py                     # CLI workflow orchestration
```

## Modules

### ingestion

**`data_paths.py`** — Centralized path management and session discovery.

| Symbol | Description |
|--------|-------------|
| `DataStorageManager` | Discovers and manages all data file paths for a given animal/session. Supports `auto_load` for automatic path resolution. |
| `get_kilosort_path()` | Locate Kilosort output directory. Returns `Path` or `List[Path]` for multi-session. |
| `get_dio_path()` | Get DIO (digital I/O) file path. Returns `Path` or `List[Path]` for multi-session. |
| `get_pulse_log_path()` | Get pulse log file for synchronization. |
| `get_video_files_by_date()` | Find video files by session date. |
| `get_tracking_files_by_date()` | Find tracking data files by session date. |
| `get_event_files_by_date()` | Find behavioral event CSV files by session date. |
| `get_animals_and_sessions()` | Scan ephys directory to list all available animal/session pairs. |
| `verify_kilosort_path()` | Validate a Kilosort directory has required files. |

**`kilosort_data_import.py`** — High-level Kilosort 4 data loading with quality filtering.

| Symbol | Description |
|--------|-------------|
| `KilosortData` | Main data class. Accepts `DataStorageManager` or a file path. Supports `session_index` for multi-session recordings. |
| `.load_spike_data()` | Load spike times and cluster assignments. |
| `.get_cluster_spikes_fast()` | Efficiently split spikes by cluster using stable sort + `np.split`. |
| `.select_clusters()` | Filter clusters by quality label (good/mua). |
| `.calculate_firing_pattern_metrics()` | Compute firing rate, presence ratio, CV ISI per cluster. |
| `.filter_cells_by_firing_patterns()` | Select cells passing quality thresholds. |
| `.get_event_aligned_spikes()` | Extract spikes aligned to behavioral event times. |
| `.bin_spike_times()` | Bin spikes into fixed time intervals. |

**`kilosort_data.py`** — Low-level Kilosort file parser (direct path interface).


**`ephys_sync.py`** — Ephys–behavior timestamp synchronization via inter-pulse interval matching.

| Symbol | Description |
|--------|-------------|
| `DataSyncManager` | Manages sync state for a session. Provides `ephys_to_behavior(t)` and `behavior_to_ephys(t)` conversion. Supports `session_index` for multi-DIO paths. |
| `load_ephys_sync()` | Load DIO and pulse log data. Handles list DIO paths via `session_index`. |
| `find_sync_mapping()` | Primary sync via linear regression on matched inter-pulse intervals. |
| `find_sync_mapping_new()` | Advanced sync with interval matching and subset search. |
| `plot_sync_results()` | Visualize sync quality (residuals, matched pulses). |

**`trodes_to_python.py`** — Read SpikeGadgets Trodes binary files.

| Symbol | Description |
|--------|-------------|
| `readTrodesExtractedDataFile()` | Parse binary Trodes data files with settings header. |

### video

**`behavioral_events.py`** — Load and query behavioral event data.

| Symbol | Description |
|--------|-------------|
| `BehavioralEventsData` | Loads behavioral event CSVs. Filter by type, rat, role. Provides `synchronize_with_ephys()` for timestamp alignment. |
| `.get_events_by_type()` | Filter events by behavior abbreviation (F, EC, etc.). |
| `.get_events_by_rat()` | Filter events involving a specific animal as initiator or victim. |
| `.extract_opponent_labels()` | Extract event times and opponent identity labels for decoding. |

**`behavioral_visualization.py`** — Standalone behavioral event plots.

| Symbol | Description |
|--------|-------------|
| `plot_rat_interaction_heatmap()` | Heatmap of rat-pair interaction counts by event type. |
| `plot_rat_behavior_heatmap()` | Heatmap of behavior types for a specific rat vs. all partners. |
| `plot_behavioral_event_timeline()` | Timeline with Y=animal, X=event index, colored lines connecting initiator/victim. Center-outward Y-axis reordering by interaction frequency. |

**`tracking_import.py`** — Load position tracking data.

| Symbol | Description |
|--------|-------------|
| `load_tracking_data()` | Load tracking CSV/TSV via DataStorageManager. |
| `load_timestamps()` | Load corresponding .npy timestamp file. |
| `parse_tracking()` | Organize DataFrame by animal/object name. |

**`plot_trajectory.py`** — Trajectory visualization.

| Symbol | Description |
|--------|-------------|
| `plot_animal_path()` | Plot single animal trajectory with optional start/end markers. |
| `plot_multiple_paths()` | Overlay trajectories of multiple animals. |
| `plot_path_heatmap()` | 2D positional occupancy heatmap. |
| `calculate_path_statistics()` | Compute distance, speed, and occupancy metrics. |

### ephys

**`decode_opponent_identity.py`** — Opponent identity decoding from neural activity using cross-validated LDA.

| Symbol | Description |
|--------|-------------|
| `decode_opponent_identity_single_cell()` | Single-cell LDA decoding. Supports `selected_opponents` to restrict to specific opponent labels. |
| `decode_opponent_identity_population()` | Population-level decoding across quality-filtered cells. Passes `selected_opponents` through to per-cell calls. |
| `align_spikes_to_events()` | Align spike times to behavioral event times within a time window. |
| `extract_firing_rate_features()` | Bin aligned spikes into firing rate feature vectors. |
| `plot_decoding_accuracy_distribution()` | Histogram + boxplot of cross-validated accuracies across cells. |
| `plot_best_cells_decoding()` | Bar plot of top cells + confusion matrix of best cell. |
| `plot_decoding_summary()` | Comprehensive multi-panel summary figure. |
| `plot_top_cells_firing_rates()` | Peri-event firing rate traces split by opponent class. |

**`population_geometry.py`** — Neural population dynamics and state-space analysis.

| Symbol | Description |
|--------|-------------|
| `PopulationGeometryAnalyzer` | Construct population firing rate matrices, apply PCA/UMAP, compare conditions. |
| `.construct_population_matrix()` | Build (cells × time bins × trials) tensor from event-aligned spikes. |
| `.apply_dimensionality_reduction()` | PCA or UMAP on population activity. |
| `.plot_population_state_space()` | Visualize reduced-dimensionality trajectories by condition. |

**`plot_ephys_qa_stats.py`** — Quality assessment plots.

| Symbol | Description |
|--------|-------------|
| `plot_firing_pattern_histograms()` | Distribution plots for firing rate, presence ratio, CV ISI. |
| `plot_pass_fail_histograms()` | Pass/fail comparison overlay for quality thresholds. |

### workflow

**`workflow.py`** — End-to-end CLI pipeline.

```bash
python workflow.py --animal_id 631 --session_id 20251216 --save_plots --output_dir ./results
```

Orchestrates: Kilosort data loading → tracking processing → ephys-video synchronization → visualization generation.

## Configuration

Data paths are defined in JSON files under `config/`:

```json
{
    "ephys_base_path": "/path/to/ephys",
    "video_base_path": "/path/to/video",
    "tracking_base_path": "/path/to/tracking",
    "events_base_path": "/path/to/events",
    "pulse_log_path": "/path/to/pulse_log.csv"
}
```

`DataStorageManager` loads configuration automatically, searching the `config/` directory first and falling back to the module directory.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `pipeline_demo.ipynb` | Full workflow: data loading, behavioral analysis, population decoding, PCA |
| `clock_sync_demo.ipynb` | Simulated sync pulse timing, interval-based synchronization, error analysis |
| `ephys/LDA_demo.ipynb` | Single-cell opponent identity decoding walkthrough |
| `video/trajectory_plots_examples.ipynb` | Trajectory visualization and spatial occupancy examples |

## Installation

```bash
git clone https://github.com/your-org/habitat_pipeline.git
cd habitat_pipeline
pip install -e .
```

Or with pixi:

```bash
pixi install
```

### Dependencies

numpy, pandas, matplotlib, seaborn, scikit-learn, scipy. Optional: umap-learn.
