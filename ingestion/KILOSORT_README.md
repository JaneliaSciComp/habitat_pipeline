# KilosortData - Electrophysiology Data Import and Analysis

A Python data class for importing, storing, and analyzing Kilosort 4 electrophysiology data, specifically designed for the habitat_pipeline multi-animal analysis system.

## Features

- **Complete Kilosort 4 Support**: Loads all standard Kilosort output files
- **Efficient Data Access**: Fast filtering and querying of spike data by multiple criteria
- **Behavioral Integration**: Tools for aligning spikes with behavioral events and continuous features
- **Multi-Animal Support**: Designed for large-scale studies with multiple animals and sessions
- **Memory Efficient**: Lazy loading and caching for large datasets
- **Quality Control**: Built-in firing rate and cluster quality filtering

## Installation

Place `kilosort_data.py` in your project directory and install dependencies:

```bash
pip install numpy pandas pathlib
```

## Quick Start

```python
from kilosort_data import load_kilosort_session

# Load a single session
ks_data = load_kilosort_session(
    data_path="path/to/kilosort/output",
    animal_id="rat001",
    session_id="session001"
)

# Get high-quality clusters
good_clusters = ks_data.get_clusters(
    min_firing_rate=1.0,
    max_firing_rate=50.0,
    cluster_group='good'
)

# Extract spike times for analysis
spike_times = ks_data.get_spike_times(good_clusters[0])
```

## Core Classes and Functions

### KilosortData Class

Main data container for a single Kilosort session.

**Key Attributes:**
- `spike_times`: Array of spike times in seconds
- `spike_clusters`: Array of cluster IDs for each spike  
- `cluster_info`: DataFrame with cluster metadata
- `templates`: Spike waveform templates
- `firing_rates`: Computed firing rates for all clusters

**Key Methods:**

#### Data Loading
- `__init__(data_path, animal_id, session_id)`: Initialize and load Kilosort data
- Automatically loads all standard Kilosort 4 files (.npy format)
- Converts spike times to seconds using sampling rate

#### Cluster Filtering
```python
get_clusters(
    animal_id=None,           # Filter by animal ID
    session_id=None,          # Filter by session ID  
    channels=None,            # List of channel numbers
    min_firing_rate=None,     # Minimum firing rate (Hz)
    max_firing_rate=None,     # Maximum firing rate (Hz)
    cluster_group=None        # 'good', 'mua', 'noise'
)
```

#### Spike Time Extraction
```python
# Single cluster
spike_times = ks_data.get_spike_times(cluster_id)

# Multiple clusters
spike_dict = ks_data.get_spike_times([cluster1, cluster2, cluster3])
```

#### Behavioral Analysis Integration

**Continuous Features** (position, velocity, etc.):
```python
# Bin spikes for 40 Hz behavioral data
binned_spikes = ks_data.bin_spike_times(
    cluster_ids=[1, 2, 3],
    bin_size=0.025,  # 25ms bins
    start_time=0,
    end_time=300     # 5 minutes
)
```

**Discrete Events** (interactions, vocalizations, etc.):
```python
# Align spikes to behavioral events
event_times = np.array([10.5, 25.3, 45.1])  # Event timestamps
aligned_spikes = ks_data.get_event_aligned_spikes(
    cluster_ids=[1, 2, 3],
    event_times=event_times,
    window_pre=1.0,   # 1s before event
    window_post=2.0   # 2s after event
)
```

### Convenience Functions

#### Single Session Loading
```python
from kilosort_data import load_kilosort_session

ks_data = load_kilosort_session(
    data_path="path/to/kilosort/output",
    animal_id="rat001", 
    session_id="day1"
)
```

#### Multi-Session Loading
```python
from kilosort_data import load_multiple_sessions

session_configs = [
    {"data_path": "path/to/rat001/day1", "animal_id": "rat001", "session_id": "day1"},
    {"data_path": "path/to/rat001/day2", "animal_id": "rat001", "session_id": "day2"},
    {"data_path": "path/to/rat002/day1", "animal_id": "rat002", "session_id": "day1"},
]

sessions = load_multiple_sessions(session_configs)
```

## Expected Kilosort File Structure

The class expects the standard Kilosort 4 output directory structure:

```
kilosort_output/
├── spike_times.npy          # Required: Spike times in samples
├── spike_clusters.npy       # Required: Cluster ID for each spike
├── amplitudes.npy           # Optional: Spike amplitudes
├── templates.npy            # Optional: Spike templates/waveforms
├── channel_map.npy          # Optional: Channel mapping
├── channel_positions.npy    # Optional: Physical channel positions
├── cluster_info.tsv         # Optional: Cluster metadata
├── pc_features.npy          # Optional: PC features
├── pc_feature_ind.npy       # Optional: PC feature indices  
├── whitening_mat.npy        # Optional: Whitening matrix
├── whitening_mat_inv.npy    # Optional: Inverse whitening matrix
└── params.py               # Optional: Parameters file
```

**Required files**: `spike_times.npy`, `spike_clusters.npy`  
**Optional files**: All others (warnings issued if missing)

## Integration with Habitat Pipeline

This class is specifically designed to support the habitat pipeline requirements:

### 1. Interactive GUI Support
- Fast cluster filtering enables responsive GUI interactions
- Efficient spike time extraction for real-time visualization
- Waveform access for cluster quality assessment

### 2. Behavioral Analysis Pipeline
- **Continuous features**: Use `bin_spike_times()` with behavioral sampling rate
- **Discrete events**: Use `get_event_aligned_spikes()` for event-triggered analysis
- **Multi-animal studies**: Filter by animal, session, anatomical location

### 3. Data Organization
- Hierarchical organization: Animal → Session → Clusters
- Fast filtering by firing rate, anatomical location, quality metrics
- Cached computations for repeated analysis

### 4. Scalability
- Memory-efficient loading of large Neuropixels datasets
- Supports up to 12 animals as specified in pipeline requirements
- Parallel processing ready (stateless methods)

## Example Workflows

### Quality Control Analysis
```python
# Load session
ks_data = load_kilosort_session("path/to/data", "rat001", "day1")

# Get firing rate statistics
rates = ks_data.firing_rates
print(f"Mean firing rate: {rates.mean():.2f} Hz")
print(f"Clusters: {len(rates)} total")

# Filter high-quality clusters
good_clusters = ks_data.get_clusters(
    min_firing_rate=0.5,      # Exclude very quiet cells
    max_firing_rate=100.0,    # Exclude potential artifacts
    cluster_group='good'       # Only manually curated good clusters
)
print(f"Good clusters: {len(good_clusters)}")
```

### Position Encoding Analysis
```python
# Get hippocampal clusters (example channel range)
hipp_clusters = ks_data.get_clusters(
    channels=list(range(100, 200)),  # Hippocampal channels
    min_firing_rate=1.0,
    cluster_group='good'
)

# Bin spikes to match 40 Hz position tracking
binned_spikes = ks_data.bin_spike_times(
    hipp_clusters,
    bin_size=0.025,  # 25ms bins for 40 Hz
    start_time=0,
    end_time=3600    # 1 hour session
)

# Now correlate with position data (loaded separately)
# correlations = analyze_position_encoding(binned_spikes, position_data)
```

### Social Interaction Analysis  
```python
# Load interaction timestamps (example)
interaction_times = np.array([45.2, 156.7, 278.3, 445.1])

# Extract spikes around interactions
social_spikes = ks_data.get_event_aligned_spikes(
    cluster_ids=good_clusters,
    event_times=interaction_times,
    window_pre=2.0,   # 2s before interaction
    window_post=5.0   # 5s after interaction
)

# Analyze pre/post interaction activity
for cluster_id, events in social_spikes.items():
    pre_counts = [len(spikes[spikes < 0]) for spikes in events]
    post_counts = [len(spikes[spikes > 0]) for spikes in events] 
    print(f"Cluster {cluster_id}: Pre={np.mean(pre_counts):.1f}, Post={np.mean(post_counts):.1f}")
```

## Performance Notes

- **Loading**: ~1-2 seconds for typical Neuropixels session
- **Filtering**: Sub-millisecond for most filter combinations
- **Spike extraction**: ~10-100ms depending on cluster count and session length
- **Memory usage**: Scales with session length, efficient for multi-hour recordings
- **Caching**: Firing rates and binned spikes are cached after first computation

## Error Handling

The class provides informative error messages for common issues:
- Missing required files
- Invalid file formats
- Empty spike trains
- Invalid time ranges
- Missing cluster IDs

## Extending the Class

The modular design allows easy extension:

```python
class ExtendedKilosortData(KilosortData):
    def custom_analysis_method(self):
        # Add your custom analysis methods
        pass
    
    def _load_custom_data(self):
        # Override to load additional file formats
        pass
```

## See Also

- `example_usage.py`: Complete usage examples and demonstrations
- Pipeline documentation: Integration with behavioral analysis pipeline
- Kilosort documentation: Understanding the source data format