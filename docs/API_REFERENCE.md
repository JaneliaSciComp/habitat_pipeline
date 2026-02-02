# API Reference

## Core Modules

### Data Ingestion

#### DataLoader
Main class for loading electrophysiology data from various formats.

```python
from habitat_pipeline import DataLoader

loader = DataLoader("path/to/data", format="nwb")
data, sampling_rate = loader.load_ephys()
behavior = loader.load_behavior()
metadata = loader.load_metadata()
```

**Supported formats:**
- NWB (Neurodata Without Borders)
- Open Ephys
- Intan (RHD/RHS)
- Raw binary

#### MetadataParser
Parse and validate experimental metadata.

```python
from habitat_pipeline.ingestion import MetadataParser

parser = MetadataParser("metadata.yaml")
parser.set("key", "value")
parser.save("output_metadata.yaml")
```

### Preprocessing

#### SignalProcessor
Signal processing operations for electrophysiology data.

```python
from habitat_pipeline import SignalProcessor

processor = SignalProcessor(sampling_rate=30000.0)

# Bandpass filter
filtered = processor.bandpass_filter(data, lowcut=300, highcut=6000)

# Notch filter for line noise
filtered = processor.notch_filter(data, freq=60)

# Downsample
downsampled, new_rate = processor.downsample(data, target_rate=1000.0)
```

#### ArtifactRemover
Detect and remove artifacts from recordings.

```python
from habitat_pipeline.preprocessing import ArtifactDetector, ArtifactRemover

detector = ArtifactDetector(threshold_std=5.0)
remover = ArtifactRemover(detector)

cleaned_data = remover.remove_artifacts(data, sampling_rate)
```

### Quality Control

#### QualityMetrics
Compute quality metrics for data assessment.

```python
from habitat_pipeline import QualityMetrics

qc = QualityMetrics(sampling_rate)
metrics = qc.compute_all_metrics(data)

# Individual metrics
snr = qc.compute_snr(data)
noise = qc.compute_noise_level(data)
drift = qc.compute_drift(data)
```

#### QualityAssessor
Assess data quality against thresholds.

```python
from habitat_pipeline import QualityAssessor

assessor = QualityAssessor(snr_threshold=3.0)
results = assessor.assess_all(metrics)

if results['overall_passed']:
    print("Data passed quality control")
```

### Spike Sorting

#### SpikeDetector
Detect neural spikes in electrophysiology data.

```python
from habitat_pipeline import SpikeDetector

detector = SpikeDetector(
    sampling_rate=30000.0,
    threshold_method='mad',
    threshold_factor=4.0
)

spike_times = detector.detect_spikes(data)
waveforms = detector.extract_waveforms(data, spike_times)
```

#### SpikeSorter
Sort spikes into putative units.

```python
from habitat_pipeline import SpikeSorter

sorter = SpikeSorter(method='kmeans', n_clusters=3)
labels = sorter.sort_all_channels(waveforms)
```

### Synchronization

#### TimestampAligner
Align timestamps across multiple data streams.

```python
from habitat_pipeline.synchronization import TimestampAligner

aligner = TimestampAligner(reference_stream='ephys')
aligned_streams = aligner.align_streams(streams, sync_signals)
```

#### SyncValidator
Validate synchronization quality.

```python
from habitat_pipeline.synchronization import SyncValidator

validator = SyncValidator(max_drift_ms=1.0)
results = validator.validate_alignment(source_ts, target_ts, transform)
```

### Multi-Animal Analysis

#### MultiAnimalCoordinator
Coordinate analysis across multiple animals.

```python
from habitat_pipeline import MultiAnimalCoordinator

coordinator = MultiAnimalCoordinator(n_jobs=4)

# Register animals
coordinator.register_animal('Mouse_001', '/data/mouse_001')

# Synchronize
coordinator.synchronize_animals()

# Process all
results = coordinator.process_all_animals(my_processing_func)
```

### Visualization

#### Plotter
Create publication-quality plots.

```python
from habitat_pipeline import Plotter

plotter = Plotter(output_dir='./plots')

# Plot traces
plotter.plot_traces(data, sampling_rate)

# Plot raster
plotter.plot_raster(spike_times, sampling_rate)

# Plot waveforms
plotter.plot_waveforms(waveforms, sampling_rate)

# Plot PSD
plotter.plot_psd(data, sampling_rate)
```

#### InteractivePlotter
Create interactive visualizations.

```python
from habitat_pipeline.visualization import InteractivePlotter

plotter = InteractivePlotter()
fig = plotter.plot_traces_interactive(data, sampling_rate)
plotter.show()
```

## Utilities

#### setup_logging
Configure logging for the pipeline.

```python
from habitat_pipeline.utils import setup_logging

logger = setup_logging(level='INFO', log_file='pipeline.log')
```

#### load_config
Load configuration from YAML file.

```python
from habitat_pipeline.utils import load_config

config = load_config('config/default_config.yaml')
```
