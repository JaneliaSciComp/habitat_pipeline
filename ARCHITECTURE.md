# Habitat Pipeline - Data Processing Flow Architecture

## Block Diagram Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                  HABITAT PIPELINE                                    │
│                            Data Processing Flow Architecture                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│     USER INPUT      │    │   CONFIGURATION     │    │    FILE SYSTEM      │
├─────────────────────┤    ├─────────────────────┤    ├─────────────────────┤
│ • animal_id         │    │ • default_paths.json│    │ • Kilosort Files    │
│ • session_id        │    │ • ephys path        │    │ • Tracking Files    │
│ • config_path       │    │ • video path        │    │ • Timestamp Files   │
│ • output_dir        │    │ • tracking path     │    │ • Sync Files        │
│ • processing flags  │    │                     │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
          │                          │                          │
          │                          │                          │
          ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              WORKFLOW.PY (Main Orchestrator)                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • parse_arguments()                                                                 │
│ • setup_output_directory()                                                          │
│ • main() - coordinates all processing steps                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
          │
          ├─────────────────┬─────────────────┬─────────────────┬─────────────────┐
          ▼                 ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ PATH MANAGEMENT │ │ EPHYS PROCESSING│ │VIDEO PROCESSING │ │ VISUALIZATION   │ │ SYNCHRONIZATION │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    DETAILED MODULE BREAKDOWN
═══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          1. PATH MANAGEMENT MODULE                                   │
│                        ingestion/data_paths.py                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: animal_id, session_id, config_path
  │
  ▼
┌─────────────────────────┐
│ get_kilosort_path()     │ ──────┐
├─────────────────────────┤       │
│ • Reads config file     │       │
│ • Partial ID matching   │       │ 
│ • Directory traversal   │       │
│ • Path construction     │       │
└─────────────────────────┘       │
  │                               │
  ▼                               │
kilosort_path ────────────────────┤
  │                               │
  ▼                               │
┌─────────────────────────┐       │
│ verify_kilosort_path()  │ ◄─────┘
├─────────────────────────┤
│ • Path existence check  │
│ • Required files check  │
│ • Detailed error msgs   │
└─────────────────────────┘
  │
  ▼
OUTPUT: validated_kilosort_path, validation_message

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        2. EPHYS PROCESSING MODULE                                   │
│                    ingestion/kilosort_data_import.py                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: kilosort_path
  │
  ▼
┌─────────────────────────┐
│ KilosortData.__init__() │
├─────────────────────────┤
│ • locate_KS_folder()    │ ──┐
│ • extract_ids_from_path()│  │
│ • load_spike_data()     │  │
│ • select_clusters()     │  │
│ • extract_cluster_props()│  │
│ • get_cluster_spikes_fast│  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ LOADED DATA:            │  │
│ • spike_times           │  │
│ • spike_clusters        │  │
│ • cluster_info          │  │
│ • ks_ids                │  │
│ • channel info          │  │
│ • amplitudes            │  │
│ • firing rates          │  │
│ • cell_numbers          │  │
│ • allSpikeSI            │  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ UTILITY METHODS:        │  │
│ • get_spike_data()      │  │
│ • read_timestamps()     │  │
│ • waveform2channel()    │  │
│ • __repr__()           │ ◄┘
└─────────────────────────┘
  │
  ▼
OUTPUT: KilosortData object with all ephys data and metadata

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        3. VIDEO PROCESSING MODULE                                   │
│                        video/tracking_import.py                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: tracking_file_path
  │
  ▼
┌─────────────────────────┐
│ load_tracking_data()    │
├─────────────────────────┤
│ • Multi-format support  │ ──┐
│ • CSV/TSV/TXT parsing   │   │
│ • Encoding detection    │   │
│ • Error handling        │   │
└─────────────────────────┘   │
  │                           │
  ▼                           │
tracking_DataFrame ───────────┤
  │                           │
  ▼                           │
┌─────────────────────────┐   │
│ parse_tracking()        │   │
├─────────────────────────┤   │
│ • Group by object_name  │   │
│ • Remove ID columns     │   │
│ • Validate data        │ ◄─┘
│ • Reset indices        │
└─────────────────────────┘
  │
  ▼
animals_dict: Dict[animal_name, DataFrame]
  │
  ├─────────────────────────────────┐
  │                                 │
  ▼                                 ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│ load_timestamps()       │ │ Per-animal DataFrames:  │
├─────────────────────────┤ ├─────────────────────────┤
│ • Pattern matching      │ │ • frame                 │
│ • _mask_metrics → _ts   │ │ • center_x, center_y    │
│ • Fuzzy file search     │ │ • area, perimeter       │
│ • Load .npy arrays      │ │ • circularity          │
└─────────────────────────┘ │ • orientation          │
  │                         │ • bbox coordinates     │
  ▼                         └─────────────────────────┘
timestamps_array
  │
  ▼
OUTPUT: tracking_path, tracking_df, animals_dict, timestamps

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        4. VISUALIZATION MODULE                                      │
│                      video/path_visualization.py                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: animals_dict, animal_name, parameters
  │
  ├─────────────────┬─────────────────┬─────────────────┬─────────────────┐
  ▼                 ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│plot_animal_path()│ │plot_multiple_   │ │plot_path_       │ │calculate_path_  │ │save_           │
├─────────────────┤ │paths()          │ │heatmap()        │ │statistics()     │ │visualization() │
│• Extract coords │ ├─────────────────┤ ├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│• Plot trajectory│ │• Multi-animal   │ │• 2D histogram   │ │• Total distance │ │• File output    │
│• Mark start/end │ │  comparison     │ │• Time density   │ │• Speed analysis │ │• Multiple       │
│• Add statistics │ │• Color coding   │ │• Heatmap colors │ │• Position stats │ │  formats        │
│• Distance calc  │ │• Legend/labels  │ │• Territory map  │ │• Movement metrics│ │• High DPI       │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
  │                 │                 │                 │                 │
  ▼                 ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            VISUALIZATION OUTPUTS                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • Individual path plots with statistics                                             │
│ • Multi-animal comparison plots                                                     │
│ • Position heatmaps and territory analysis                                         │
│ • Movement statistics and metrics                                                  │
│ • High-quality saved plots (PNG/PDF/SVG)                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      5. SYNCHRONIZATION MODULE                                      │
│                      ingestion/ephys_sync.py                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: animal_id, session_id, channel, timestamps
  │
  ▼
┌─────────────────────────┐
│ load_ephys_sync()       │
├─────────────────────────┤
│ • Load sync channels    │ ──┐
│ • Extract timestamps    │   │
│ • System time reference │   │
└─────────────────────────┘   │
  │                           │
  ▼                           │
TSESync, TSBSync, sys_time ───┤
  │                           │
  ▼                           │
┌─────────────────────────┐   │
│ find_sync_mapping()     │   │
├─────────────────────────┤   │
│ • Cross-correlation     │ ◄─┘
│ • Time alignment        │
│ • Mapping generation    │
└─────────────────────────┘
  │
  ▼
OUTPUT: sync_mapping (ephys ↔ video time alignment)

═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    DATA FLOW SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              INTEGRATED WORKFLOW                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

USER INPUTS (animal_id, session_id, flags)
    │
    ▼
PATH RESOLUTION (config → validated paths)
    │
    ├─────────────────────┬─────────────────────┐
    ▼                     ▼                     ▼
EPHYS DATA              VIDEO DATA            SYNC DATA
    │                     │                     │
    ▼                     ▼                     │
KilosortData           tracking_df,             │
object                 animals_dict,            │
    │                  timestamps               │
    │                     │                     │
    │                     ▼                     │
    │               VISUALIZATIONS              │
    │               (plots, statistics)         │
    │                     │                     │
    └─────────────────────┼─────────────────────┘
                          ▼
                   SYNCHRONIZED ANALYSIS
                   (ephys ↔ video alignment)
                          │
                          ▼
                    OUTPUT DIRECTORY
                   ├── plots/
                   │   ├── animal_paths.png
                   │   ├── heatmaps.png
                   │   └── comparisons.png
                   └── data/
                       ├── processed_data.*
                       └── sync_mapping.*

═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                  KEY DATA STRUCTURES
═══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ CONFIG DATA                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • ephys: base path to electrophysiology data                                       │
│ • video: base path to video files                                                  │
│ • tracking: base path to tracking analysis results                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ EPHYS DATA STRUCTURES                                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • spike_times: numpy array of spike timestamps                                     │
│ • spike_clusters: numpy array of cluster assignments                               │
│ • cluster_info: pandas DataFrame with cluster metadata                             │
│ • ks_ids: list of selected cluster IDs                                            │
│ • allSpikeSI: list of spike sample indices per cluster                            │
│ • channel/amplitude/firing_rate: cluster properties                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ VIDEO DATA STRUCTURES                                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • tracking_df: raw DataFrame with all tracking data                                │
│ • animals_dict: Dict[animal_name, DataFrame] parsed by animal                      │
│ • timestamps: numpy array of video frame timestamps                                │
│ • coordinate columns: center_x, center_y, bbox_*, area, perimeter                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ VISUALIZATION OUTPUTS                                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • matplotlib Figure objects with animal trajectories                               │
│ • Path statistics dictionaries (distance, speed, territory)                       │
│ • Saved plot files in multiple formats (PNG/PDF/SVG)                              │
│ • Interactive and static visualization options                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ SYNCHRONIZATION DATA                                                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • TSESync/TSBSync: ephys synchronization signals                                   │
│ • sync_mapping: time alignment between ephys and video                             │
│ • system_time_at_creation: absolute time reference                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘