# Habitat Pipeline
**Integrated Multi-Animal Electrophysiology and Behavior Analysis Pipeline**

A modular, scalable software platform for processing and analyzing large-scale electrophysiology and behavioral data from multiple freely behaving animals recorded simultaneously. The system supports the full experimental lifecycle: raw data ingestion, preprocessing, quality control, spike sorting, multimodal synchronization, advanced multi-animal analysis, and visualization.

## 🚀 Quick Start

```python
# Basic usage example
from ingestion.data_paths import DataStorageManager
from ingestion.kilosort_data_import import KilosortData
from video.behavioral_events import BehavioralEventsData
from ingestion.ephys_sync import DataSyncManager
from ephys.decode_opponent_identity import decode_opponent_identity_population

# Initialize data manager
data_manager = DataStorageManager("631", "20251216", auto_load=True)

# Load electrophysiology data
ks_data = KilosortData(data_manager)
print(f"Loaded {len(ks_data.ks_ids)} neural clusters")

# Load behavioral data  
behavior_data = BehavioralEventsData(data_manager)
print(f"Loaded {len(behavior_data.events_df)} behavioral events")

# Synchronize behavioral events with ephys timestamps
sync_manager = DataSyncManager(data_manager, dio_channel=1)
behavior_data.synchronize_with_ephys(sync_manager, create_new_columns=True)
print("✓ Synchronized behavioral events with neural timestamps")

# Run opponent identity decoding
results = decode_opponent_identity_population(
    ks_data=ks_data,
    behavior_data=behavior_data,
    animal_of_interest="631",
    behavior_type='EC',  # Encounter events
    use_quality_cells=True
)
```

## 📁 Project Structure

```
habitat_pipeline/
├── config/                    # Configuration files
│   └── default_paths.json     # Data path configurations
├── ingestion/                 # Data loading and preprocessing
│   ├── data_paths.py         # Centralized path management
│   ├── kilosort_data_import.py # Electrophysiology data processing
│   ├── ephys_sync.py         # Multi-modal synchronization
│   └── trodes_to_python.py   # SpikeGadgets integration
├── video/                     # Video and behavioral analysis
│   ├── tracking_import.py     # Animal position tracking
│   ├── behavioral_events.py   # Behavioral event detection
│   └── plot_trajectory.py     # Trajectory visualization
├── ephys/                     # Electrophysiology analysis
│   ├── plot_ephys_qa_stats.py # Quality assessment plots
│   └── decode_opponent_identity.py # Neural decoding analysis
├── database/                  # Data management and storage
│   ├── database_core.py      # Database schema and operations
│   └── database_integration.py # Pipeline integration
└── workflow.py               # Automated pipeline workflows
```

## 🔧 Core Modules

### Data Management
- **DataStorageManager**: Centralized path management and data discovery
- **KilosortData**: Electrophysiology spike data processing with quality metrics
- **VideoTrackingData**: Animal position and movement analysis
- **BehavioralEventsData**: Event detection and classification

### Analysis Features
- **✅ Electrophysiology Quality Assessment**: Firing rate, presence ratio, ISI statistics
- **✅ Multi-Modal Synchronization**: Align ephys, video, and behavioral timestamps  
- **✅ Behavioral Event Analysis**: Interaction detection, opponent identification
- **✅ Neural Decoding**: Machine learning-based opponent identity decoding
- **✅ Visualization Tools**: Comprehensive plotting for QA and analysis results

### Advanced Analytics
- **Linear Discriminant Analysis (LDA)**: Cross-validated opponent identity decoding
- **Peri-Event Time Histograms (PETHs)**: Neural activity around behavioral events
- **Population Analysis**: Multi-cell decoding performance assessment
- **Quality Filtering**: Automatic cell selection based on firing patterns

## 🎯 Key Features

### 1. Automated Data Discovery
```python
# Automatic path resolution and data loading
data_manager = DataStorageManager("animal_id", "session_id", auto_load=True)
# Finds electrophysiology, video, tracking, and behavioral event files
```

### 2. Quality-Controlled Analysis
```python
# Automatic quality assessment
ks_data = KilosortData(data_manager)
metrics = ks_data.calculate_firing_pattern_metrics()
quality_cells = ks_data.filter_cells_by_firing_patterns(
    min_firing_rate=0.5,      # Hz
    min_presence_ratio=0.8,   # Session coverage
    max_cv_isi=5.0           # ISI variability
)
```

### 3. Behavioral Event Processing
```python
# Load and analyze behavioral interactions
behavior_data = BehavioralEventsData(data_manager)
behavior_data.synchronize_with_ephys(sync_manager)

# Visualize interaction patterns
behavior_data.plot_rat_interaction_heatmap(event_type='F')  # Fights
behavior_data.plot_rat_behavior_heatmap('631')  # Animal-specific
```

### 4. Neural Decoding Pipeline
```python
# Opponent identity decoding from neural activity
results = decode_opponent_identity_population(
    ks_data=ks_data,
    behavior_data=behavior_data,
    animal_of_interest="631",
    behavior_type='EC',
    time_window=(-0.5, 1.0),   # 500ms pre to 1s post-event
    cv_folds=5                 # Cross-validation
)

# Comprehensive visualization
from ephys.decode_opponent_identity import plot_decoding_summary
plot_decoding_summary(results)
```

### 5. Multi-Modal Visualization
```python
# Comprehensive quality assessment plots
from ephys.plot_ephys_qa_stats import plot_firing_pattern_histograms
plot_firing_pattern_histograms(metrics, quality_cells)

# Trajectory visualization
from video.plot_trajectory import plot_animal_path
plot_animal_path(tracking_data, 'rat631')

# Peri-event firing rate analysis
from ephys.decode_opponent_identity import plot_top_cells_firing_rates
plot_top_cells_firing_rates(ks_data, behavior_data, results)
```

## 📊 Analysis Capabilities

### Electrophysiology Analysis
- **Spike Sorting Integration**: Kilosort 4 output processing
- **Quality Metrics**: Firing rate, ISI statistics, presence ratio
- **Population Analysis**: Multi-cell decoding and visualization
- **Temporal Dynamics**: Peri-event time histograms (PETHs)

### Behavioral Analysis  
- **Event Detection**: Automated interaction classification
- **Opponent Identification**: Role-based analysis (initiator/victim)
- **Temporal Alignment**: Precise ephys-behavior synchronization
- **Social Networks**: Inter-animal interaction patterns

### Machine Learning
- **Cross-Validated LDA**: Robust opponent identity decoding
- **Performance Assessment**: Accuracy distributions, confusion matrices
- **Feature Selection**: Automatic quality-based cell filtering
- **Statistical Validation**: Proper chance-level calculations

## 🔄 Workflow Integration

The pipeline supports both interactive analysis and automated workflows:

### Interactive Analysis (Jupyter)
See `test.ipynb` for complete examples of:
- Data loading and quality assessment
- Behavioral event analysis
- Neural decoding workflows
- Visualization generation

### Command Line Interface
```bash
# Run opponent decoding analysis
python ephys/decode_opponent_identity.py \
    --animal_id 631 \
    --session_id 20251216 \
    --behavior_type EC \
    --save_plots \
    --output_dir ./results
```

### Database Integration
```python
# Store and retrieve analysis results
from database.database_integration import store_session_data
store_session_data(data_manager, results)
```

## 📈 Performance & Scalability

- **Large Dataset Support**: Optimized for Neuropixels data (hundreds of channels)
- **Multi-Animal Processing**: Supports up to 12 simultaneous animals
- **Efficient Memory Usage**: Lazy loading and data streaming
- **Parallel Processing**: Multi-core analysis for population statistics
- **Caching System**: Preprocessed data storage for rapid analysis

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/habitat_pipeline.git
   cd habitat_pipeline
   ```

2. **Configure data paths**
   ```json
   // config/default_paths.json  
   {
     "ephys": "/path/to/ephys/data",
     "video": "/path/to/video/data",
     "tracking": "/path/to/tracking/data"
   }
   ```

3. **Run example analysis**
   ```python
   # Open and execute test.ipynb for full pipeline demonstration
   ```

## 📚 Architecture Principles

- **Modularity**: Independent, composable analysis modules
- **Scalability**: Designed for large-scale multi-animal experiments  
- **Reproducibility**: Versioned transformations and parameterized analyses
- **Interoperability**: Clear APIs for integration with lab tools
- **Data Integrity**: Comprehensive validation and error handling

## 🧪 Experimental Requirements

The pipeline addresses key experimental analysis needs:

### Neural-Behavioral Relationships
- Interactive GUI for examining spike trains with video overlay
- Fast cell selection based on animal, session, anatomy, firing rate
- Temporal alignment of spikes with continuous behavioral features
- Event-based analysis around discrete behavioral interactions

### Data Organization
- Hierarchical cell selection (single/multi-session, single/multi-animal)
- Efficient spike time storage and retrieval
- Behavioral feature extraction from video and audio
- Synchronized multi-modal data streams

### Analysis Outputs
- Population encoding of behavioral features
- Individual cell characterization plots
- Statistical and machine learning-based decoding
- Reproducible visualization and reporting

---

**Status**: ✅ Active Development | **Version**: 2.0 | **Last Updated**: March 2026

For detailed usage examples, see the Jupyter notebooks in the repository. For technical documentation, refer to `ARCHITECTURE.md`. 
