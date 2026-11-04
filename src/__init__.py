"""Nocturne source package metadata."""

from .contracts import (
    EpochRecord,
    ExperimentConfig,
    RecordingRecord,
    SignalMetadata,
    SubjectRecord,
)
from .pipeline import PipelineConfig, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "EpochRecord",
    "ExperimentConfig",
    "PipelineConfig",
    "RecordingRecord",
    "SignalMetadata",
    "SubjectRecord",
    "__version__",
    "run_pipeline",
]
