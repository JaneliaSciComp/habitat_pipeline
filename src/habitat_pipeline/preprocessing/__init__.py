"""Preprocessing module for signal filtering, artifact removal, and data cleaning."""

from habitat_pipeline.preprocessing.filters import SignalProcessor, BandpassFilter, NotchFilter
from habitat_pipeline.preprocessing.artifacts import ArtifactRemover, ArtifactDetector
from habitat_pipeline.preprocessing.referencing import CommonAverageReference, MedianReference

__all__ = [
    "SignalProcessor",
    "BandpassFilter",
    "NotchFilter",
    "ArtifactRemover",
    "ArtifactDetector",
    "CommonAverageReference",
    "MedianReference",
]
