"""Multi-animal analysis module for coordinating analysis across animals."""

from habitat_pipeline.multi_animal.coordinator import MultiAnimalCoordinator
from habitat_pipeline.multi_animal.processor import ParallelProcessor
from habitat_pipeline.multi_animal.aggregator import ResultAggregator

__all__ = [
    "MultiAnimalCoordinator",
    "ParallelProcessor",
    "ResultAggregator",
]
