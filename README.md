# Habitat Pipeline

Analysis pipeline for multi-animal electrophysiology and behavioral data (RatCity cohorts at Janelia). Handles data ingestion, ephys–behavior clock synchronization, Kilosort 4 spike-sorting output processing, neural decoding (opponent identity, event outcome, location), population geometry, and visualization for freely behaving animals recorded simultaneously.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a deeper module-by-module reference.

## Quick Start

```python
from ingestion.data_paths import DataStorageManager
from ingestion.kilosort_data_import import load_kilosort_data
from ingestion.ephys_sync import DataSyncManager
from video.behavioral_events import load_behavioral_events
from ephys.decode_opponent_identity import decode_opponent_identity_population

# Discover all session data files automatically
dsm = DataStorageManager("631", "20251216", auto_load=True)

# Load electrophysiology and behavioral data
ks_data = load_kilosort_data(dsm.get_kilosort_path())
behavior_data = load_behavioral_events(
    dsm.get_behavioral_event_files(),
    session_id=dsm.session_id,
)

# Synchronize behavioral timestamps with the ephys clock
sync_manager = DataSyncManager(dsm, dio_channel=1)
behavior_data.synchronize_with_ephys(sync_manager, create_new_columns=True)

# Decode opponent identity from neural activity
results = decode_opponent_identity_population(
    ks_data=ks_data,
    behavior_data=behavior_data,
    animal_of_interest="631",
    behavior_type="EC",   # Encounter events
    use_quality_cells=True,
)
```

## Project Structure

```
habitat_pipeline/
├── config/                            # Cohort path configuration (JSON)
│   ├── default_paths.json             # Cohort 7
│   └── cohort5_paths.json             # Cohort 5
├── ingestion/                         # Data loading and preprocessing
│   ├── data_paths.py                  # DataStorageManager + path discovery
│   ├── kilosort_data_import.py        # Kilosort 4 loader + quality metrics (dataclass)
│   ├── kilosort_data.py               # Low-level Kilosort file parser
│   ├── ephys_sync.py                  # Ephys ↔ behavior clock synchronization
│   └── trodes_to_python.py            # SpikeGadgets Trodes binary reader
├── video/                             # Behavioral tracking and events
│   ├── tracking_import.py             # VideoTrackingData + loader
│   ├── behavioral_events.py           # BehavioralEventsData + opponent/group/outcome labels
│   ├── behavioral_visualization.py    # Event heatmaps & timelines
│   └── plot_trajectory.py             # Trajectory, occupancy, Voronoi, proximity
├── ephys/                             # Electrophysiology analysis
│   ├── _lda_decoding.py               # Shared LDA / CV / feature core (label-agnostic)
│   ├── decoding_plots.py              # Shared decoding plots
│   ├── decode_opponent_identity.py    # Opponent identity / ID-group LDA decoding
│   ├── decode_event_outcome.py        # Winner/loser LDA decoding
│   ├── decode_location.py             # Bayesian location decoding
│   ├── population_geometry.py         # PCA/UMAP population dynamics
│   ├── rastermap_viz.py               # Rastermap visualization
│   └── plot_ephys_qa_stats.py         # Quality-metric plots + CLI
├── gui/                               # Streamlit + Panel dashboards
│   ├── app.py                         # Streamlit entry
│   ├── interactive_app.py             # Panel + Bokeh + Plotly entry
│   ├── tabs/                          # Tracking / Behavioral / Decoding / Population
│   ├── loaders.py, runners.py, state.py, widgets.py, cache.py, plotting.py
├── database/                          # Optional SQLite session/animal metadata
└── workflow.py                        # Legacy CLI orchestrator
```

## Modules

### ingestion

**`data_paths.py`** — Centralized path management and session discovery.

| Symbol | Description |
|--------|-------------|
| `DataStorageManager(animal_id, session_id, config_path=None, auto_load=True)` | Discovers and validates all data file paths for an animal/session. Passes to every downstream loader. |
| `.get_kilosort_path()` / `.get_dio_path(channel)` / `.get_pulse_log_path()` | Ephys-side path accessors. |
| `.get_video_files()` / `.get_tracking_files()` / `.get_behavioral_event_files()` | Behavior-side path accessors. |
| `get_animals_and_sessions(config_path=None)` | Scan ephys root and return a DataFrame of available `(session, animal, kilosort_path)`. |
| `verify_kilosort_path(path)` | Validate a Kilosort directory has the required files. |

**`kilosort_data_import.py`** — High-level Kilosort 4 loader (dataclass + function), with on-disk pickle cache.

| Symbol | Description |
|--------|-------------|
| `KilosortData` | Pure-data dataclass: `spike_times_by_cell`, `ks_ids`, `cluster_info`, channel/amplitude/etc., metadata. |
| `.duration_seconds` | Recording duration (with cache-trim-safe fallback). |
| `.get_firing_rates()` / `.get_isi_statistics()` | Per-cluster firing-rate and ISI stats. |
| `.calculate_firing_pattern_metrics()` | Firing rate, presence ratio, CV ISI per cluster. |
| `.filter_cells_by_firing_patterns(...)` | Apply quality thresholds. |
| `.get_filtered_cells_spike_times(...)` / `.bin_spike_times(...)` | Cached filtered spike lists and binned firing-rate matrices. |
| `load_kilosort_data(path, force_reload=False)` | Disk loader with auto-cache. |
| `save_kilosort_data(...)` / `load_kilosort_from_file(...)` | Persist or reload an instance. |

**`ephys_sync.py`** — Ephys–behavior timestamp synchronization via inter-pulse interval matching.

| Symbol | Description |
|--------|-------------|
| `DataSyncManager(data_manager, dio_channel=1)` | Sync state for a session. Exposes `convert_ephys_to_behavior(t)` and `convert_behavior_to_ephys(t)` plus `slope`/`intercept`. |
| `load_ephys_sync(data_manager, dio_channel=1)` | Load DIO pulses and pulse-log timestamps. |
| `find_sync_mapping(TSESync, TSBSync, system_time)` | Linear regression on matched IPIs; returns `{slope, intercept, r_squared, matched_ephys, matched_behavior, ...}`. |
| `plot_sync_results(...)` | 4-panel sync quality diagnostic. |

**`trodes_to_python.py`** — `readTrodesExtractedDataFile()` to parse SpikeGadgets Trodes binary files.

### video

**`behavioral_events.py`** — Behavioral event records and label extraction.

| Symbol | Description |
|--------|-------------|
| `BehavioralEventsData` | Dataclass holding `events_data` (DataFrame), `BEHAVIOR_TYPES` mapping, sync state. |
| `.synchronize_with_ephys(sync_manager)` | Add `ts_*_ephys` columns to `events_data`. |
| `.extract_opponent_labels(animal, behavior_type, ...)` | Per-event opponent identity labels. |
| `.extract_group_labels(animal, behavior_type, ...)` | Pool opponents into `low`/`high` halves by numeric ID. |
| `.extract_outcome_labels(animal, behavior_type, ...)` | Winner/loser labels for the focal animal (defaults to "any aggressive event"). |
| `.get_events_by_type()` / `.get_events_by_rat()` / `.get_available_event_types()` / `.get_available_rats()` | Filtering and inventory helpers. |
| `load_behavioral_events(files, session_id=...)` | Load CSV(s) into a `BehavioralEventsData`. |

**`behavioral_visualization.py`** — Standalone event plots: `plot_rat_interaction_heatmap`, `plot_rat_behavior_heatmap`, `plot_behavioral_event_timeline`, `plot_events_on_trajectory`.

**`tracking_import.py`** — Position tracking loader.

| Symbol | Description |
|--------|-------------|
| `VideoTrackingData` | Dataclass with `parsed_data: Dict[str, DataFrame]`, `timestamps`, helpers. |
| `load_tracking_data(source, file_index=0, load_ts=True)` | Accepts a `DataStorageManager` or a path. |
| `parse_tracking(df)` | Split a combined frame into per-animal DataFrames. |
| `load_timestamps(path)` | Locate the paired `*_ts.npy` file. |

**`plot_trajectory.py`** — Trajectory and occupancy plots: `plot_animal_path`, `plot_multiple_paths`, `plot_path_heatmap`, `plot_territorial_occupancy`, `plot_voronoi_territories`, `plot_proximity_network` (+ `compute_proximity_interactions`), `calculate_path_statistics`, `save_visualization`.

### ephys

The decoding modules share a single label-agnostic core ([ephys/_lda_decoding.py](ephys/_lda_decoding.py)) and a shared plotting module ([ephys/decoding_plots.py](ephys/decoding_plots.py)). Wrappers add label-specific extraction and identical plot calls.

**`decode_opponent_identity.py`** — Opponent identity (and ID-group) LDA decoding.

| Symbol | Description |
|--------|-------------|
| `decode_opponent_identity_single_cell(...)` | Single-cell decode wrapper. |
| `decode_opponent_identity_population(ks_data, behavior_data, animal_of_interest, behavior_type=None, label_mode='opponent'|'group', ...)` | Per-cell population sweep. |
| `decode_opponent_identity_time_resolved(...)` | Population LDA per time bin with optional shuffle null. |

**`decode_event_outcome.py`** — Winner vs loser decoding (mirrors the opponent module). `behavior_type=None` includes every event with both `winner` and `loser` populated.

**`decode_location.py`** — Bayesian decoding of `(x, y)` for tracked objects: `build_binned_data`, `decode_location_single_cell`, `decode_location_population`, `decode_all_locations`, `plot_decoding_results`, `plot_all_decoding_summary`.

**`population_geometry.py`** — PCA/UMAP on event-aligned population activity.

| Symbol | Description |
|--------|-------------|
| `PopulationGeometryAnalyzer(ks_data, behavior_data)` | Build (cells × bins × trials) tensors, run PCA/UMAP. |
| `.construct_population_matrix(...)` / `.apply_dimensionality_reduction(...)` | Core methods. |
| `.plot_population_dynamics(...)` / `.plot_pca_summary(...)` / `.plot_normalized_population_matrix(...)` | Built-in plots. |
| `run_population_analysis_pipeline(ks, be, ...)` | End-to-end wrapper. |

**`rastermap_viz.py`** — `run_rastermap`, `plot_rastermap`, `plot_rastermap_interactive`, `plot_rastermap_with_events`, `bin_spikes_matrix`. Requires `pip install rastermap`.

**`plot_ephys_qa_stats.py`** — Quality-assessment plots: `plot_firing_pattern_histograms`, `plot_pass_fail_histograms`, `test_threshold_combinations`, `load_and_analyze_data` (+ CLI).

**`decoding_plots.py`** — Shared plots for both opponent and outcome decoders. Result-dict-driven titles (`results['parameters']['class_label']` / `['analysis_title']`):

- `plot_decoding_accuracy_distribution` — histogram + boxplot of per-cell CV accuracies
- `plot_best_cells_decoding` — top-N bar chart + best-cell confusion matrix
- `plot_decoding_summary` — 6-panel dashboard
- `plot_time_resolved_decoding` — accuracy curve vs time + best-bin confusion matrix
- `plot_top_cells_firing_rates` — peri-event firing-rate curves split by class
- `plot_top_cells_rasters` — spike rasters for top cells, sorted by class

### gui

Two parallel apps share the same loaders.

| App | Command | Best for |
|---|---|---|
| Streamlit explorer | `streamlit run gui/app.py` | Browsing all analyses across tabs (Tracking, Behavioral, Decoding, Population) |
| Panel explorer | `panel serve gui/interactive_app.py --show` | Linked timeline ↔ Rastermap ↔ 3D PCA |

The Streamlit app caches heavy results to `.gui_cache/` (git-ignored). Decoding and population-geometry runs are persisted to disk and invalidated automatically when parameters change. See [gui/README.md](gui/README.md) for details.

### workflow

**`workflow.py`** — Legacy end-to-end CLI. Useful as a scaffold; per-module CLIs and the GUIs are the recommended entry points.

```bash
python workflow.py --animal_id 631 --session_id 20251216 --save_plots --output_dir ./results
```

Orchestrates: Kilosort data loading → tracking processing → ephys-video synchronization → visualization generation.

### database (optional)

SQLAlchemy ORM (`Animal`, `ExperimentSession`, `DataFile`, `HabitatDatabase`) over `habitat_pipeline.db`, plus `database/database_cli.py` for `init_database`, `scan_directory`, `add_*`, `show_status`, `export_summary`. Bridge helpers in `database/database_integration.py` (`PipelineIntegration`, `quick_setup`). See [database/README.md](database/README.md).

## Configuration

Data paths are defined in JSON files under `config/`:

```json
{
  "ephys":    "/path/to/ephys",
  "video":    "/path/to/video",
  "tracking": "/path/to/tracking",
  "events":   "/path/to/events"
}
```

`DataStorageManager` loads the active config automatically: passing `config_path=None` reads `config/default_paths.json` (Cohort 7); passing just a filename (`"cohort5_paths.json"`) looks under `config/` first, otherwise treats it as an absolute path.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `pipeline_demo.ipynb` | Full workflow: data loading, behavioral analysis, population decoding, PCA. |
| `test.ipynb` | Working scratchpad covering the most recent refactors. |
| `db_example.ipynb` | Database setup and queries. |
| `ephys/LDA_demo.ipynb` | Single-cell opponent identity decoding walkthrough. |
| `video/trajectory_plots_examples.ipynb` | Trajectory visualization and spatial occupancy examples. |
| `tests/KilosortData_test_demo.ipynb` | KilosortData loader demo. |

## Installation

```bash
git clone https://github.com/JaneliaSciComp/habitat_pipeline.git
cd habitat_pipeline
pip install -e .
```

For the dashboards:

```bash
pip install -e ".[gui]"
pip install rastermap  # required for rastermap_viz and the Panel app
```

Or with pixi:

```bash
pixi install
```

Or conda:

```bash
conda env create -f environment.yml
conda activate habitat-pipeline
```

### Dependencies

Core: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, h5py.
GUI extras: streamlit, panel, bokeh, plotly, param.
Optional: rastermap, umap-learn, sqlalchemy, opencv-python.

## Tests

```bash
cd tests
python run_tests.py          # all tests
python run_tests.py loading  # subset
pytest -v                    # via pytest directly
```

See [tests/README.md](tests/README.md) for coverage details.
