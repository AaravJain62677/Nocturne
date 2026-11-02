"""Typed record contracts for the Nocturne research pipeline.

These dataclasses define the explicit data structures used between ingestion,
preprocessing, modeling, and evaluation stages. They intentionally remain light
weight and do not implement any training or inference logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import (
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
    SUPPORTED_SENSOR_SETS,
    SUPPORTED_TAXONOMIES,
)


@dataclass(frozen=True, slots=True)
class SignalMetadata:
    """Metadata describing one source signal or modality."""

    name: str
    source: str
    units: str | None = None
    sampling_rate_hz: float | None = None
    channel: str | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class SubjectRecord:
    """Explicit subject-level summary for dataset partitioning."""

    subject_id: str
    dataset: str
    recording_ids: tuple[str, ...] = ()
    wearable_signals: tuple[str, ...] = ()
    psg_signals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecordingRecord:
    """Container for one subject's recording and associated metadata."""

    subject_id: str
    recording_id: str
    dataset: str
    source_dir: Path | None = None
    available_modalities: tuple[str, ...] = ()
    signal_metadata: tuple[SignalMetadata, ...] = ()
    modality_data: dict[str, tuple[tuple[str, float | str], ...]] = field(default_factory=dict)
    stage_labels: tuple[str, ...] = ()
    timestamps: tuple[str, ...] = ()
    sampling_rates: dict[str, float | None] = field(default_factory=dict)
    sensor_set: str = "combined"


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """One epoch of aligned wearable and optionally PSG reference data."""

    subject_id: str
    recording_id: str
    epoch_index: int
    timestamp: str
    stage_label: str
    wearable_features: dict[str, float] = field(default_factory=dict)
    psg_features: dict[str, float] = field(default_factory=dict)
    excluded: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Typed contract for future experiment execution."""

    dataset: str
    sensor_set: str
    taxonomy: str
    model_name: str
    mode: str = "validate"
    seed: int = 42
    repo_root: Path | None = None

    def __post_init__(self) -> None:
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{self.dataset}'. "
                f"Supported values: {', '.join(SUPPORTED_DATASETS)}."
            )
        if self.sensor_set not in SUPPORTED_SENSOR_SETS:
            raise ValueError(
                f"Unsupported sensor set '{self.sensor_set}'. "
                f"Supported values: {', '.join(SUPPORTED_SENSOR_SETS)}."
            )
        if self.taxonomy not in SUPPORTED_TAXONOMIES:
            raise ValueError(
                f"Unsupported taxonomy '{self.taxonomy}'. "
                f"Supported values: {', '.join(SUPPORTED_TAXONOMIES)}."
            )
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{self.model_name}'. "
                f"Supported values: {', '.join(SUPPORTED_MODELS)}."
            )
        if self.seed < 0:
            raise ValueError(f"Seed must be non-negative; got {self.seed!r}.")


__all__ = [
    "EpochRecord",
    "ExperimentConfig",
    "RecordingRecord",
    "SignalMetadata",
    "SubjectRecord",
]
