# User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Concepts](#basic-concepts)
4. [Workflows](#workflows)
5. [Configuration](#configuration)
6. [Advanced Usage](#advanced-usage)

## Introduction

Habitat Pipeline is designed to process and analyze large-scale electrophysiology and behavioral data from multiple freely behaving animals recorded simultaneously. This guide will walk you through the main features and typical workflows.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Quick Install

```bash
pip install git+https://github.com/M-Proskurin/habitat_pipeline.git
```

### From Source

```bash
git clone https://github.com/M-Proskurin/habitat_pipeline.git
cd habitat_pipeline
pip install -e .
```

### Docker Installation

```bash
docker pull habitat_pipeline:latest
```

## Basic Concepts

### Pipeline Architecture

The Habitat Pipeline is organized into modular components:

1. **Ingestion**: Load data from various formats
2. **Preprocessing**: Filter and clean signals
3. **Quality Control**: Assess data quality
4. **Spike Sorting**: Detect and cluster spikes
5. **Synchronization**: Align multi-modal data
6. **Multi-Animal**: Coordinate across animals
7. **Visualization**: Create plots and figures

### Data Flow

```
Raw Data → Ingestion → Preprocessing → Quality Control → 
Spike Sorting → Analysis → Visualization
```

## Workflows

### Single Animal Analysis

```python
from habitat_pipeline import *

# 1. Load data
loader = DataLoader("data.nwb")
data, fs = loader.load_ephys()

# 2. Preprocess
processor = SignalProcessor(fs)
filtered = processor.bandpass_filter(data, 300, 6000)

# 3. Quality control
qc = QualityMetrics(fs)
metrics = qc.compute_all_metrics(filtered)

# 4. Spike detection
detector = SpikeDetector(fs)
spikes = detector.detect_spikes(filtered)

# 5. Visualization
plotter = Plotter("./output")
plotter.plot_raster(spikes, fs)
```

### Multi-Animal Analysis

```python
from habitat_pipeline import MultiAnimalCoordinator

# Setup coordinator
coordinator = MultiAnimalCoordinator(n_jobs=-1)

# Register animals
for animal_id in animal_list:
    coordinator.register_animal(animal_id, f"data/{animal_id}")

# Process in parallel
results = coordinator.process_all_animals(
    my_analysis_func,
    parallel=True
)
```

### Quality Control Pipeline

```python
from habitat_pipeline import QualityMetrics, QualityAssessor, QualityReport

# Compute metrics
qc = QualityMetrics(sampling_rate)
metrics = qc.compute_all_metrics(data)

# Assess quality
assessor = QualityAssessor()
assessment = assessor.assess_all(metrics)

# Generate report
report = QualityReport()
report.add_metrics(metrics)
report.add_assessment(assessment)
report.save_json("qc_report.json")
report.save_text("qc_report.txt")
```

## Configuration

### YAML Configuration

Create a configuration file (e.g., `my_config.yaml`):

```yaml
preprocessing:
  filters:
    bandpass:
      enabled: true
      lowcut: 300
      highcut: 6000
    notch:
      enabled: true
      freq: 60

spike_sorting:
  detection:
    threshold_factor: 4.0
  clustering:
    method: kmeans
    n_clusters: 3
```

Load and use:

```python
from habitat_pipeline.utils import load_config

config = load_config("my_config.yaml")

# Use in pipeline
processor = SignalProcessor(sampling_rate)
filtered = processor.bandpass_filter(
    data,
    lowcut=config['preprocessing']['filters']['bandpass']['lowcut'],
    highcut=config['preprocessing']['filters']['bandpass']['highcut']
)
```

## Advanced Usage

### Custom Processing Functions

Define custom processing for multi-animal analysis:

```python
def my_custom_analysis(animal_id, data_path, metadata, **kwargs):
    """Custom analysis function."""
    # Load data
    loader = DataLoader(data_path)
    data, fs = loader.load_ephys()
    
    # Your custom analysis
    results = analyze(data)
    
    return results

# Use with coordinator
results = coordinator.process_all_animals(my_custom_analysis)
```

### Parallel Processing

```python
from habitat_pipeline.multi_animal import ParallelProcessor

processor = ParallelProcessor(n_jobs=4)

# Process multiple items in parallel
results = processor.map(my_func, data_list)
```

### Interactive Visualization

```python
from habitat_pipeline.visualization import InteractivePlotter

plotter = InteractivePlotter()
fig = plotter.plot_traces_interactive(data, sampling_rate)
plotter.show()  # Opens interactive window
```

### Artifact Removal

```python
from habitat_pipeline.preprocessing import ArtifactDetector, ArtifactRemover

# Configure detector
detector = ArtifactDetector(threshold_std=5.0, window_size=1.0)

# Create remover
remover = ArtifactRemover(detector)

# Remove artifacts
cleaned = remover.remove_artifacts(data, sampling_rate, method='interpolate')
```

### Synchronization

```python
from habitat_pipeline.synchronization import (
    TimestampAligner,
    SyncValidator
)

# Align streams
aligner = TimestampAligner(reference_stream='ephys')
aligned = aligner.align_streams(streams, sync_signals)

# Validate
validator = SyncValidator(max_drift_ms=1.0)
validation = validator.validate_alignment(source, target, transform)
```

## Tips and Best Practices

1. **Always run quality control** before analysis
2. **Use configuration files** for reproducibility
3. **Save intermediate results** for debugging
4. **Leverage parallel processing** for multi-animal experiments
5. **Document your analysis** with comprehensive logging

## Troubleshooting

### Common Issues

**Issue**: Import errors
**Solution**: Make sure package is installed: `pip install -e .`

**Issue**: Memory errors with large datasets
**Solution**: Process data in chunks or use downsampling

**Issue**: Poor spike detection
**Solution**: Adjust threshold_factor or try different methods

## Getting Help

- Check the [API Reference](API_REFERENCE.md)
- See [examples/](../examples/) for complete workflows
- Open an issue on GitHub
