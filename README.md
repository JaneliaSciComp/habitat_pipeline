# Habitat Pipeline

**Integrated Multi-Animal Electrophysiology and Behavior Analysis Pipeline**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/M-Proskurin/habitat_pipeline/workflows/Habitat%20Pipeline%20CI/badge.svg)](https://github.com/M-Proskurin/habitat_pipeline/actions)

A modular, scalable, and standards-compliant software platform designed to process and analyze large-scale electrophysiology and behavioral data from multiple freely behaving animals recorded simultaneously.

## Features

### 🔬 Complete Analysis Pipeline
- **Raw Data Ingestion**: Support for NWB, Open Ephys, Intan, and raw binary formats
- **Preprocessing**: Advanced filtering, artifact removal, and signal conditioning
- **Quality Control**: Automated quality metrics and assessment
- **Spike Sorting**: Detection, clustering, and waveform analysis
- **Synchronization**: Multi-modal temporal alignment
- **Multi-Animal Analysis**: Parallel processing and cross-animal coordination
- **Visualization**: Publication-quality and interactive plots

### 🚀 Key Capabilities
- **Modular Design**: Independent, reusable components
- **Scalable Architecture**: Parallel processing for multi-animal experiments
- **Standards Compliant**: Support for NWB and other neuroscience data formats
- **Reproducible**: Configuration-driven analysis with comprehensive logging
- **Docker Ready**: Containerized deployment for reproducibility

## Installation

### From Source

```bash
git clone https://github.com/M-Proskurin/habitat_pipeline.git
cd habitat_pipeline
pip install -e .
```

### With Docker

```bash
docker build -t habitat_pipeline .
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output habitat_pipeline
```

### Using Docker Compose

```bash
docker-compose up
```

## Quick Start

### Basic Pipeline Example

```python
from habitat_pipeline import (
    DataLoader,
    SignalProcessor,
    QualityMetrics,
    SpikeDetector,
    Plotter
)

# Load data
loader = DataLoader("path/to/data.nwb")
data, sampling_rate = loader.load_ephys()

# Preprocess
processor = SignalProcessor(sampling_rate)
filtered_data = processor.bandpass_filter(data, lowcut=300, highcut=6000)

# Quality control
qc = QualityMetrics(sampling_rate)
metrics = qc.compute_all_metrics(filtered_data)

# Spike detection
detector = SpikeDetector(sampling_rate)
spike_times = detector.detect_spikes(filtered_data)

# Visualize
plotter = Plotter(output_dir="./output")
plotter.plot_traces(filtered_data, sampling_rate)
plotter.plot_raster(spike_times, sampling_rate)
```

### Multi-Animal Analysis

```python
from habitat_pipeline import MultiAnimalCoordinator

# Initialize coordinator
coordinator = MultiAnimalCoordinator(n_jobs=4)

# Register animals
for animal_id in ['Mouse_001', 'Mouse_002', 'Mouse_003']:
    coordinator.register_animal(
        animal_id=animal_id,
        data_path=f"./data/{animal_id}"
    )

# Synchronize
coordinator.synchronize_animals()

# Process all animals in parallel
results = coordinator.process_all_animals(
    processing_func=my_analysis_function,
    parallel=True
)
```

## Configuration

Pipeline behavior can be customized through YAML configuration files:

```yaml
# config/default_config.yaml

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
    peak_sign: negative
  clustering:
    method: kmeans
    n_clusters: null  # auto-detect

quality_control:
  metrics:
    snr_threshold: 3.0
    noise_threshold: 100.0
```

## Project Structure

```
habitat_pipeline/
├── src/habitat_pipeline/       # Main package source
│   ├── ingestion/              # Data loading and parsing
│   ├── preprocessing/          # Signal processing and filtering
│   ├── quality_control/        # QC metrics and assessment
│   ├── spike_sorting/          # Spike detection and clustering
│   ├── synchronization/        # Multi-modal alignment
│   ├── multi_animal/           # Multi-animal coordination
│   ├── visualization/          # Plotting and visualization
│   └── utils/                  # Utilities and helpers
├── tests/                      # Unit and integration tests
├── examples/                   # Example scripts and workflows
├── config/                     # Configuration files
├── docs/                       # Documentation
├── Dockerfile                  # Docker image definition
└── docker-compose.yml          # Docker Compose configuration
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=habitat_pipeline --cov-report=html
```

## Examples

See the `examples/` directory for complete workflows:

- `basic_pipeline.py`: Single-animal processing pipeline
- `multi_animal_analysis.py`: Multi-animal parallel analysis

## License

This project is licensed under the MIT License.

## Citation

If you use Habitat Pipeline in your research, please cite:

```bibtex
@software{habitat_pipeline,
  title = {Habitat Pipeline: Multi-Animal Electrophysiology Analysis Platform},
  author = {Habitat Pipeline Contributors},
  year = {2026},
  url = {https://github.com/M-Proskurin/habitat_pipeline}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/M-Proskurin/habitat_pipeline/issues)
- **Examples**: [examples/](examples/)

---

**Habitat Pipeline** - Making multi-animal electrophysiology analysis reproducible, scalable, and accessible
