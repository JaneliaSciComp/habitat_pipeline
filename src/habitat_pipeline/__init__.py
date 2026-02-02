"""
Habitat Pipeline: Integrated Multi-Animal Electrophysiology and Behavior Analysis Pipeline

A modular, scalable, and standards-compliant software platform designed to process and analyze
large-scale electrophysiology and behavioral data from multiple freely behaving animals recorded
simultaneously.
"""

__version__ = "0.1.0"
__author__ = "Habitat Pipeline Contributors"

from habitat_pipeline.ingestion import DataLoader, MetadataParser
from habitat_pipeline.preprocessing import SignalProcessor, ArtifactRemover
from habitat_pipeline.quality_control import QualityMetrics, QualityAssessor
from habitat_pipeline.spike_sorting import SpikeDetector, SpikeSorter
from habitat_pipeline.synchronization import TimestampAligner, SyncValidator
from habitat_pipeline.multi_animal import MultiAnimalCoordinator
from habitat_pipeline.visualization import Plotter, InteractivePlotter

__all__ = [
    "DataLoader",
    "MetadataParser",
    "SignalProcessor",
    "ArtifactRemover",
    "QualityMetrics",
    "QualityAssessor",
    "SpikeDetector",
    "SpikeSorter",
    "TimestampAligner",
    "SyncValidator",
    "MultiAnimalCoordinator",
    "Plotter",
    "InteractivePlotter",
]
