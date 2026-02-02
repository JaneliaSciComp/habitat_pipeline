"""
Example: Multi-Animal Analysis

This example demonstrates multi-animal data processing and analysis
using the Habitat Pipeline.
"""

import numpy as np
from pathlib import Path

from habitat_pipeline import (
    SignalProcessor,
    SpikeDetector,
    MultiAnimalCoordinator
)
from habitat_pipeline.multi_animal import ResultAggregator
from habitat_pipeline.utils import setup_logging

# Setup logging
logger = setup_logging(level="INFO")


def process_animal_data(animal_id, data_path, metadata, **kwargs):
    """
    Process data for a single animal.
    
    Parameters
    ----------
    animal_id : str
        Animal identifier
    data_path : Path
        Path to animal data
    metadata : dict
        Animal metadata
    **kwargs
        Additional processing parameters
        
    Returns
    -------
    dict
        Processing results
    """
    print(f"\nProcessing animal: {animal_id}")
    
    # Generate synthetic data for demonstration
    sampling_rate = 30000.0
    duration = 1.0
    n_channels = 64
    n_samples = int(sampling_rate * duration)
    
    # Synthetic data with different characteristics per animal
    np.random.seed(hash(animal_id) % 2**32)  # Reproducible but different per animal
    data = np.random.randn(n_channels, n_samples) * 10
    
    # Preprocessing
    processor = SignalProcessor(sampling_rate)
    filtered_data = processor.bandpass_filter(data, lowcut=300, highcut=6000)
    
    # Spike detection
    detector = SpikeDetector(sampling_rate, threshold_factor=4.0)
    spike_times = detector.detect_spikes(filtered_data)
    
    # Compute results
    total_spikes = sum(len(times) for times in spike_times.values())
    firing_rate = total_spikes / duration
    
    results = {
        'animal_id': animal_id,
        'n_spikes': total_spikes,
        'firing_rate': firing_rate,
        'n_active_channels': len([ch for ch, times in spike_times.items() if len(times) > 0])
    }
    
    print(f"  Spikes: {total_spikes}, Firing rate: {firing_rate:.2f} Hz")
    
    return results


def main():
    """Run multi-animal analysis example."""
    
    print("=" * 80)
    print("Habitat Pipeline - Multi-Animal Analysis Example")
    print("=" * 80)
    
    # 1. Initialize coordinator
    print("\n1. Initializing multi-animal coordinator...")
    coordinator = MultiAnimalCoordinator(n_jobs=4)
    
    # 2. Register animals
    print("\n2. Registering animals...")
    animals = ['Mouse_001', 'Mouse_002', 'Mouse_003', 'Mouse_004']
    
    for animal_id in animals:
        coordinator.register_animal(
            animal_id=animal_id,
            data_path=f"./data/{animal_id}",
            metadata={'strain': 'C57BL/6', 'age': '8 weeks'}
        )
    
    print(f"  Registered {len(animals)} animals")
    
    # 3. Synchronize animals (optional)
    print("\n3. Synchronizing animals...")
    # In real scenario, would use actual sync signals
    sync_transforms = coordinator.synchronize_animals()
    print("  Animals synchronized")
    
    # 4. Process all animals in parallel
    print("\n4. Processing all animals in parallel...")
    results = coordinator.process_all_animals(
        processing_func=process_animal_data,
        parallel=True
    )
    
    # 5. Aggregate results
    print("\n5. Aggregating results...")
    aggregator = ResultAggregator()
    
    for animal_id, result in results.items():
        aggregator.add_result(animal_id, result)
    
    # Compute summary statistics
    summary = aggregator.compute_summary_statistics()
    
    print("\n" + "=" * 80)
    print("Summary Statistics Across Animals:")
    print("=" * 80)
    
    for metric_name, stats in summary.items():
        print(f"\n{metric_name}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.2f}")
    
    # 6. Check status
    print("\n" + "=" * 80)
    print("Processing Status:")
    print("=" * 80)
    
    status = coordinator.get_animal_status()
    for animal_id, animal_status in status.items():
        print(f"  {animal_id}: {animal_status}")
    
    print("\n" + "=" * 80)
    print("Multi-animal analysis completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
