"""
Example: Basic Pipeline Usage

This example demonstrates the basic usage of the Habitat Pipeline
for processing single-animal electrophysiology data.
"""

import numpy as np
from habitat_pipeline import (
    DataLoader,
    SignalProcessor,
    QualityMetrics,
    QualityAssessor,
    SpikeDetector,
    SpikeSorter,
    Plotter
)
from habitat_pipeline.utils import setup_logging, load_config

# Setup logging
logger = setup_logging(level="INFO")

def main():
    """Run basic pipeline example."""
    
    print("=" * 80)
    print("Habitat Pipeline - Basic Example")
    print("=" * 80)
    
    # 1. Generate synthetic data for demonstration
    print("\n1. Generating synthetic data...")
    sampling_rate = 30000.0  # 30 kHz
    duration = 1.0  # 1 second
    n_channels = 64
    n_samples = int(sampling_rate * duration)
    
    # Generate synthetic neural data with noise and spikes
    data = np.random.randn(n_channels, n_samples) * 10  # Background noise
    
    # Add some synthetic spikes
    n_spikes = 50
    for _ in range(n_spikes):
        spike_channel = np.random.randint(0, n_channels)
        spike_time = np.random.randint(1000, n_samples - 1000)
        spike_waveform = -100 * np.exp(-0.5 * ((np.arange(60) - 30) / 5)**2)  # Gaussian
        data[spike_channel, spike_time:spike_time + 60] += spike_waveform
    
    print(f"  Generated data: {n_channels} channels, {n_samples} samples ({duration}s)")
    
    # 2. Preprocessing
    print("\n2. Preprocessing data...")
    processor = SignalProcessor(sampling_rate)
    
    # Bandpass filter
    filtered_data = processor.bandpass_filter(data, lowcut=300, highcut=6000)
    print("  Applied bandpass filter (300-6000 Hz)")
    
    # Notch filter for line noise
    filtered_data = processor.notch_filter(filtered_data, freq=60)
    print("  Applied notch filter (60 Hz)")
    
    # 3. Quality Control
    print("\n3. Quality control...")
    qc_metrics = QualityMetrics(sampling_rate)
    metrics = qc_metrics.compute_all_metrics(filtered_data)
    print(f"  Computed {len(metrics)} quality metrics")
    
    assessor = QualityAssessor()
    assessment = assessor.assess_all(metrics)
    print(f"  Quality assessment: {'PASSED' if assessment['overall_passed'] else 'FAILED'}")
    
    # 4. Spike Detection
    print("\n4. Detecting spikes...")
    detector = SpikeDetector(sampling_rate, threshold_factor=4.0)
    spike_times = detector.detect_spikes(filtered_data)
    total_spikes = sum(len(times) for times in spike_times.values())
    print(f"  Detected {total_spikes} spikes across {len(spike_times)} channels")
    
    # Extract waveforms
    waveforms = detector.extract_waveforms(filtered_data, spike_times)
    
    # 5. Spike Sorting
    print("\n5. Sorting spikes...")
    sorter = SpikeSorter(method='kmeans', n_clusters=3)
    labels = sorter.sort_all_channels(waveforms)
    total_units = sum(len(np.unique(ch_labels)) for ch_labels in labels.values() if len(ch_labels) > 0)
    print(f"  Identified {total_units} putative units")
    
    # 6. Visualization
    print("\n6. Creating visualizations...")
    plotter = Plotter(output_dir="./output/plots")
    
    # Plot raw traces
    fig1 = plotter.plot_traces(
        filtered_data,
        sampling_rate,
        channels=list(range(8)),
        time_range=(0, 0.1)
    )
    print("  Created trace plot")
    
    # Plot spike raster
    fig2 = plotter.plot_raster(
        spike_times,
        sampling_rate,
        time_range=(0, duration)
    )
    print("  Created raster plot")
    
    # Plot waveforms for first channel with spikes
    for ch, ch_waveforms in waveforms.items():
        if len(ch_waveforms) > 0:
            fig3 = plotter.plot_waveforms(
                ch_waveforms[:100],  # Plot first 100 waveforms
                sampling_rate,
                title=f"Spike Waveforms - Channel {ch}"
            )
            print(f"  Created waveform plot for channel {ch}")
            break
    
    # Plot power spectral density
    fig4 = plotter.plot_psd(
        filtered_data,
        sampling_rate,
        channels=list(range(4))
    )
    print("  Created PSD plot")
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("Results saved to: ./output/plots")
    print("=" * 80)


if __name__ == "__main__":
    main()
