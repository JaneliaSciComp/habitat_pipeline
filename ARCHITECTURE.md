# Habitat Pipeline - Architecture Overview

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                  HABITAT PIPELINE                                    │
│                        Multi-Animal Neurobehavioral Analysis System                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│    USER INTERFACE   │    │   CONFIGURATION     │    │    DATA SOURCES     │
├─────────────────────┤    ├─────────────────────┤    ├─────────────────────┤
│ • Jupyter Notebooks │    │ • default_paths.json│    │ • Kilosort Output   │
│ • Command Line Tools│    │ • Analysis Parameters│   │ • Video Tracking    │
│ • Interactive Plots │    │ • Quality Thresholds│    │ • Behavioral Events │
│ • Python API        │    │ • ML Hyperparams    │    │ • DIO Sync Channels │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
          │                          │                          │
          │                          │                          │
          ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         DATA STORAGE MANAGER (Core Hub)                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • Centralized path resolution and data discovery                                    │
│ • Auto-detection of ephys, video, tracking, and behavioral files                   │
│ • Session metadata management and validation                                        │
│ • Standardized data access APIs across all modules                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
          │
          ├─────────────────┬─────────────────┬─────────────────┬─────────────────┐
          ▼                 ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ EPHYS ANALYSIS  │ │BEHAVIORAL EVENTS│ │ VIDEO TRACKING  │ │ SYNCHRONIZATION │ │MACHINE LEARNING │
│     MODULE      │ │     MODULE      │ │     MODULE      │ │     MODULE      │ │     MODULE      │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
         │                  │                  │                  │                  │
         ▼                  ▼                  ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ QUALITY CONTROL │ │   VISUALIZATION │ │   DATABASE      │ │     WORKFLOWS   │ │     OUTPUTS     │
│     MODULE      │ │     MODULE      │ │     MODULE      │ │     MODULE      │ │     MODULE      │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    DETAILED MODULE BREAKDOWN
═══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          1. DATA STORAGE MANAGER                                    │
│                           ingestion/data_paths.py                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: animal_id, session_id, config_path
  │
  ▼
┌─────────────────────────┐
│ DataStorageManager      │ ──────┐
├─────────────────────────┤       │
│ • Auto-load all paths   │       │
│ • Path validation       │       │ 
│ • Metadata collection   │       │
│ • Error handling        │       │
└─────────────────────────┘       │
  │                               │
  ▼                               ▼
┌─────────────────────────┐ ┌──────────────────────────┐
│ DATA DISCOVERY:         │ │ INTEGRATED ACCESS:       │
│ • Kilosort ephys data   │ │ • get_kilosort_path()   │
│ • Video tracking files  │ │ • get_video_files()     │
│ • Behavioral events     │ │ • get_tracking_files()  │
│ • DIO sync channels     │ │ • get_dio_path()        │
│ • Session metadata      │ │ • Session validation    │
└─────────────────────────┘ └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        2. ENHANCED EPHYS ANALYSIS MODULE                            │
│                      ingestion/kilosort_data_import.py                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: DataStorageManager OR kilosort_path (backward compatible)
  │
  ▼
┌─────────────────────────┐
│ KilosortData.__init__() │
├─────────────────────────┤
│ • DataManager detection │ ──┐
│ • Spike data loading    │  │
│ • Cluster selection     │  │
│ • Quality computation   │  │
│ • Fast spike extraction │  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ QUALITY METRICS:        │  │
│ • Firing rate stats     │  │
│ • Presence ratio        │  │
│ • ISI statistics        │  │
│ • CV coefficient        │  │
│ • Quality filtering     │  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ ANALYSIS METHODS:       │  │
│ • calculate_firing_     │  │
│   pattern_metrics()     │  │
│ • filter_cells_by_      │  │
│   firing_patterns()     │  │
│ • print_firing_pattern_ │ ◄┘
│   summary()             │
└─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                       3. BEHAVIORAL EVENTS ANALYSIS MODULE                          │
│                           video/behavioral_events.py                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: DataStorageManager
  │
  ▼
┌─────────────────────────┐
│ BehavioralEventsData    │
├─────────────────────────┤
│ • CSV event loading     │ ──┐
│ • Multi-animal support  │  │
│ • Event classification  │  │
│ • Temporal alignment    │  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ EVENT PROCESSING:       │  │
│ • Initiator/victim      │  │
│   identification        │  │
│ • Event type filtering  │  │
│ • Opponent extraction   │  │
│ • Ephys synchronization │  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ VISUALIZATION:          │  │
│ • Interaction heatmaps  │  │
│ • Behavior timelines    │ ◄┘
│ • Animal-specific plots │
└─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                       4. VIDEO TRACKING MODULE                                      │
│                           video/tracking_import.py                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: DataStorageManager
  │
  ▼
┌─────────────────────────┐
│ VideoTrackingData       │
├─────────────────────────┤
│ • Multi-format support  │ ──┐
│ • Object name parsing   │  │
│ • Trajectory extraction │  │
│ • Timestamp loading     │  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ TRACKING FEATURES:      │  │
│ • Position coordinates  │  │
│ • Movement metrics      │  │
│ • Object properties     │  │
│ • Temporal trajectories │ ◄┘
└─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         5. NEURAL DECODING MODULE                                   │
│                          ephys/decode_opponent_identity.py                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: KilosortData, BehavioralEventsData, analysis_parameters
  │
  ▼
┌─────────────────────────┐
│ SINGLE CELL ANALYSIS:   │
├─────────────────────────┤
│ • Spike alignment       │ ──┐
│ • Feature extraction    │  │
│ • Time bin analysis     │  │
│ • LDA classification    │  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ POPULATION ANALYSIS:    │  │
│ • Cross-validation      │  │
│ • Quality cell filtering│  │
│ • Opponent identity     │  │
│   decoding              │  │
│ • Performance metrics   │ ◄┘
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ ADVANCED VISUALIZATION: │
├─────────────────────────┤
│ • Accuracy distributions│
│ • Confusion matrices    │
│ • PETH plots           │
│ • Best cell analysis   │
│ • Summary dashboards   │
└─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      6. QUALITY ASSESSMENT MODULE                                   │
│                         ephys/plot_ephys_qa_stats.py                               │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: KilosortData metrics and filtering results
  │
  ▼
┌─────────────────────────┐
│ QUALITY VISUALIZATIONS: │
├─────────────────────────┤
│ • Firing rate histograms│ ──┐
│ • Presence ratio plots  │  │
│ • ISI distributions     │  │
│ • Quality threshold     │  │
│   visualizations        │  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ INTERACTIVE FEATURES:   │  │
│ • Parameter adjustment  │ ◄┘
│ • Real-time filtering   │
│ • Statistical summaries │
└─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        7. SYNCHRONIZATION MODULE                                    │
│                           ingestion/ephys_sync.py                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: DataStorageManager, DIO_channel, timestamps
  │
  ▼
┌─────────────────────────┐
│ DataSyncManager         │
├─────────────────────────┤
│ • Multi-modal sync      │ ──┐
│ • Timestamp alignment   │  │
│ • Cross-correlation     │  │
│ • Time conversion       │  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ SYNC CAPABILITIES:      │  │
│ • Ephys ↔ Video         │ ◄┘
│ • Behavioral ↔ Neural   │
│ • Multi-animal coords   │
└─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          8. DATABASE MODULE                                         │
│                            database/*.py                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: Analysis results, session metadata
  │
  ▼
┌─────────────────────────┐
│ DATABASE OPERATIONS:    │
├─────────────────────────┤
│ • Session storage       │ ──┐
│ • Result archiving      │  │
│ • Metadata indexing     │  │
│ • Query interface       │  │
└─────────────────────────┘  │
  │                          │
  ▼                          │
┌─────────────────────────┐  │
│ DATA MANAGEMENT:        │ ◄┘
│ • SQLite backend       │
│ • CLI interface        │
│ • Integration APIs     │
└─────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    INTEGRATED WORKFLOW
═══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              TYPICAL ANALYSIS PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

USER INPUTS (animal_id, session_id)
    │
    ▼
DATA STORAGE MANAGER (centralized data discovery)
    │
    ├─────────────────────┬─────────────────────┬─────────────────────┐
    ▼                     ▼                     ▼                     ▼
KILOSORT DATA          VIDEO TRACKING        BEHAVIORAL EVENTS     SYNC MANAGER
(enhanced w/ QA)       (multi-animal)        (interactions)        (alignment)
    │                     │                     │                     │
    ▼                     ▼                     ▼                     │
QUALITY FILTERING      TRAJECTORY PLOTS      EVENT ANALYSIS          │
    │                     │                     │                     │
    └─────────────────────┼─────────────────────┼─────────────────────┘
                          ▼                     ▼
                   SYNCHRONIZED NEURAL-BEHAVIORAL ANALYSIS
                          │
                          ├─────────────────┬─────────────────┐
                          ▼                 ▼                 ▼
                   DECODING ANALYSIS    VISUALIZATION     DATABASE STORAGE
                   (LDA, cross-val)     (dashboards)      (results)
                          │                 │                 │
                          └─────────────────┼─────────────────┘
                                            ▼
                                    ANALYSIS OUTPUTS
                                   ├── quality_plots/
                                   │   ├── firing_patterns.png
                                   │   └── cell_filtering.png
                                   ├── behavioral_analysis/
                                   │   ├── interaction_heatmaps.png
                                   │   └── event_timelines.png
                                   ├── neural_decoding/
                                   │   ├── accuracy_distributions.png
                                   │   ├── best_cells_peth.png
                                   │   └── decoding_summary.png
                                   └── results_database/
                                       └── session_results.db

═══════════════════════════════════════════════════════════════════════════════════════════════════════
                                  KEY DATA STRUCTURES
═══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ DATA STORAGE MANAGER                                                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • kilosort_path: Path to ephys data                                               │
│ • video_files: List of video file paths                                           │
│ • tracking_files: List of tracking analysis results                               │
│ • behavioral_event_files: List of event CSV files                                 │
│ • dio_paths: Dict of DIO channel paths                                            │
│ • metadata: Session information and validation results                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ ENHANCED EPHYS DATA                                                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • spike_times_by_cell: List[np.array] of spike times per cell                     │
│ • ks_ids: Selected cluster IDs                                                    │
│ • cluster_info: Enhanced metadata with quality metrics                            │
│ • firing_rates: Hz per cell                                                       │
│ • presence_ratios: Session coverage per cell                                      │
│ • cv_isi: Coefficient of variation of inter-spike intervals                       │
│ • quality_thresholds: Filtering parameters and results                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ BEHAVIORAL EVENT DATA                                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • events_df: DataFrame with initiator/victim/type/timestamps                      │
│ • event_types: Available behavior classifications                                  │
│ • animal_ids: Participating animals                                               │
│ • ephys_timestamps: Synchronized neural timestamps                                 │
│ • opponent_mappings: Role-based interaction analysis                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ NEURAL DECODING RESULTS                                                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ • cell_results: Dict[cluster_id, decoding_performance]                            │
│ • population_accuracy: Cross-validated performance metrics                         │
│ • confusion_matrices: Classification results per cell                             │
│ • behavioral_summary: Event counts and opponent statistics                        │
│ • best_cells: Top performing neural decoders                                      │
│ • visualization_data: Plotting data for comprehensive analysis                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════════
                               ANALYSIS CAPABILITIES MATRIX
═══════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ FEATURE                    │ STATUS │ MODULES INVOLVED                              │
├────────────────────────────┼────────┼──────────────────────────────────────────────────┤
│ Automated Data Discovery   │   ✅   │ DataStorageManager                           │
│ Ephys Quality Assessment   │   ✅   │ KilosortData, plot_ephys_qa_stats           │
│ Multi-Animal Tracking      │   ✅   │ VideoTrackingData, plot_trajectory          │
│ Behavioral Event Analysis  │   ✅   │ BehavioralEventsData                        │
│ Multi-Modal Synchronization│   ✅   │ DataSyncManager, ephys_sync                 │
│ Neural Decoding (LDA)      │   ✅   │ decode_opponent_identity                    │
│ Population Analysis        │   ✅   │ decode_opponent_identity, visualization     │
│ Interactive Visualization  │   ✅   │ All plotting modules                        │
│ Database Integration       │   ✅   │ database_core, database_integration         │
│ Command Line Interface     │   ✅   │ decode_opponent_identity CLI                │
│ Jupyter Integration        │   ✅   │ test.ipynb, examples                        │
│ Reproducible Workflows     │   ✅   │ DataStorageManager, parameterized analysis  │
└─────────────────────────────────────────────────────────────────────────────────────┘