"""Dataset ingestion interfaces and metadata contracts.

This module defines the dataset registry and validation contracts for the
sleep-staging research repository. It intentionally does not download or parse raw
files, which remain a later-stage task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import (
    PROCESSING_DIR,
    RAW_DATA_DIR,
    SPLIT_DIR,
    SUPPORTED_DATASETS,
    SUPPORTED_SENSOR_SETS,
)
from src.contracts import EpochRecord, RecordingRecord, SignalMetadata, SubjectRecord
from src.ingest.load_dreamt import DreamtRecord, load_dreamt_dataset


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata contract for a supported dataset."""

    name: str
    source: str
    description: str
    wearable_signals: tuple[str, ...]
    psg_signals: tuple[str, ...]
    expected_signal_names: tuple[str, ...]
    epoch_seconds: int = 30


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "dreamt": DatasetSpec(
        name="dreamt",
        source="physionet",
        description="DREAMT wearable/PSG paired study.",
        wearable_signals=("accel_x", "accel_y", "accel_z", "hr", "hrv"),
        psg_signals=("eeg_fpz_cz", "eeg_pz_oz", "emg", "eog"),
        expected_signal_names=(
            "accel_x",
            "accel_y",
            "accel_z",
            "hr",
            "hrv",
            "eeg_fpz_cz",
            "eeg_pz_oz",
            "emg",
            "eog",
        ),
        epoch_seconds=30,
    ),
    "sleep_edf": DatasetSpec(
        name="sleep_edf",
        source="public",
        description="Sleep-EDF public sleep dataset used as a reference benchmark.",
        wearable_signals=("hr", "hrv"),
        psg_signals=("eeg_fpz_cz", "eeg_pz_oz", "emg", "eog"),
        expected_signal_names=("hr", "hrv", "eeg_fpz_cz", "eeg_pz_oz", "emg", "eog"),
        epoch_seconds=30,
    ),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    """Return metadata for a supported dataset identifier."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            f"Supported values: {', '.join(SUPPORTED_DATASETS)}."
        )
    return DATASET_REGISTRY[dataset]


def validate_sensor_set(dataset: str, sensor_set: str) -> tuple[str, ...]:
    """Validate that the requested sensor set is supported for the dataset.

    Returns the canonical signal list for that sensor set. This keeps the project
    explicit about wearable-only, PSG-only, and combined setups before real data
    is loaded.
    """
    if sensor_set not in SUPPORTED_SENSOR_SETS:
        raise ValueError(
            f"Unsupported sensor set '{sensor_set}'. "
            f"Supported values: {', '.join(SUPPORTED_SENSOR_SETS)}."
        )

    spec = get_dataset_spec(dataset)
    if sensor_set == "wearable":
        if not spec.wearable_signals:
            raise ValueError(f"Dataset '{dataset}' does not define a wearable sensor set.")
        return spec.wearable_signals
    if sensor_set == "psg":
        if not spec.psg_signals:
            raise ValueError(f"Dataset '{dataset}' does not define a PSG sensor set.")
        return spec.psg_signals
    if not spec.expected_signal_names:
        raise ValueError(f"Dataset '{dataset}' does not define a combined sensor set.")
    return spec.expected_signal_names


def validate_dataset_layout(dataset: str, sensor_set: str = "combined") -> dict[str, Path | str]:
    """Validate the canonical directory layout for a dataset's repository path.

    This contract defines the expected raw, processed, and split directories without
    requiring dataset files to exist yet. It also validates the sensor-set choice
    for the dataset to prevent unsupported signal combinations from being used.
    """
    spec = get_dataset_spec(dataset)
    validate_sensor_set(dataset, sensor_set)
    base_dir = Path(__file__).resolve().parents[2] / "data" / spec.name
    layout = {
        "dataset": spec.name,
        "base": base_dir,
        "raw": base_dir / "raw",
        "processed": base_dir / "processed",
        "splits": base_dir / "splits",
        "sensor_set": sensor_set,
        "repo_raw": RAW_DATA_DIR,
        "repo_processed": PROCESSING_DIR,
        "repo_splits": SPLIT_DIR,
    }
    for key in ("raw", "processed", "splits"):
        path = layout[key]
        if not path.is_absolute():
            raise ValueError(f"Expected an absolute path for dataset layout key '{key}'.")
    return layout


def load_dataset(dataset: str, *args: Any, **kwargs: Any) -> Any:
    """Load a dataset according to its registry entry.

    The real DREAMT implementation is available through the specialized loader,
    while the generic entry point keeps the repository contract explicit and
    avoids hidden data-loading assumptions.
    """
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    sensor_set = kwargs.get("sensor_set", "combined")
    if sensor_set is not None:
        validate_sensor_set(dataset, sensor_set)

    taxonomy = kwargs.get("taxonomy")
    if taxonomy is not None and taxonomy not in ("standard", "secondary"):
        raise ValueError(
            f"Unsupported taxonomy '{taxonomy}'. Supported values: standard, secondary."
        )

    if dataset == "dreamt":
        data_dir = kwargs.get("data_dir")
        if data_dir is None:
            raise NotImplementedError(
                f"Dataset loading for '{dataset}' is not implemented yet; "
                "set data_dir to a local DREAMT dataset root to enable discovery."
            )
        data_path = Path(data_dir)
        if not data_path.exists():
            raise NotImplementedError(
                f"Dataset loading for '{dataset}' is not implemented yet; "
                f"DREAMT data directory not found at {data_path}."
            )
        return load_dreamt_dataset(data_path, sensor_set=sensor_set)

    raise NotImplementedError(
        f"Dataset loading for '{dataset}' is not implemented yet; "
        "this workflow is intentionally unavailable in this phase."
    )


__all__ = [
    "DATASET_REGISTRY",
    "DatasetSpec",
    "DreamtRecord",
    "EpochRecord",
    "RecordingRecord",
    "SignalMetadata",
    "SubjectRecord",
    "get_dataset_spec",
    "load_dataset",
    "validate_dataset_layout",
    "validate_sensor_set",
]
