"""Preprocessing helpers and taxonomy definitions for sleep staging."""

from __future__ import annotations

from src.config import SECONDARY_SLEEP_STAGE_MAP, canonical_sleep_stage
from src.preprocessing.eeg_features import compute_eeg_band_stats
from src.preprocessing.epoch_segmentation import (
    align_recording_to_epochs,
    canonicalize_stage_labels,
)
from src.preprocessing.watch_features import (
    build_wearable_feature_table,
    compute_hrv_features,
    extract_wearable_epoch_features,
)

STANDARD_STAGE_ORDER: tuple[str, ...] = ("WAKE", "N1", "N2", "N3", "REM")
SECONDARY_STAGE_ORDER: tuple[str, ...] = ("WAKE", "LIGHT", "DEEP", "REM")
PREPROCESSING_VERSION: str = "synthetic-v0.1.0"


def resolve_stage_taxonomy(taxonomy: str) -> tuple[str, ...]:
    """Resolve the requested label taxonomy.

    The function validates arguments and raises a meaningful error rather than
    silently accepting unsupported values or converting unknown labels.
    """
    if taxonomy == "standard":
        return STANDARD_STAGE_ORDER
    if taxonomy == "secondary":
        return SECONDARY_STAGE_ORDER
    raise ValueError(
        f"Unsupported taxonomy '{taxonomy}'. Supported values: standard, secondary."
    )


def apply_stage_taxonomy(labels: list[str], taxonomy: str) -> list[str]:
    """Apply a taxonomy mapping to a sequence of stage labels.

    Unknown and unscored labels remain explicit and are never silently converted
    to WAKE. This function only transforms known labels to the chosen taxonomy.
    """
    if taxonomy == "standard":
        return [canonical_sleep_stage(label) for label in labels]

    if taxonomy == "secondary":
        mapped: list[str] = []
        for label in labels:
            canonical = canonical_sleep_stage(label)
            mapped.append(SECONDARY_SLEEP_STAGE_MAP.get(canonical, canonical))
        return mapped

    raise ValueError(
        f"Unsupported taxonomy '{taxonomy}'. Supported values: standard, secondary."
    )


def build_epoch_dataset(
    record: dict,
    epoch_seconds: int = 30,
    time_origin: float | None = None,
) -> list[dict[str, object]]:
    """Construct a synthetic but deterministic epoch dataset from a single record."""
    aligned = align_recording_to_epochs(
        record,
        epoch_seconds=epoch_seconds,
        time_origin=time_origin,
    )
    dataset: list[dict[str, object]] = []
    for epoch in aligned["epochs"]:
        features = extract_wearable_epoch_features(
            record,
            epoch_index=epoch["epoch_index"],
            epoch_seconds=epoch_seconds,
            time_origin=aligned["time_origin"],
            epoch_start=epoch["epoch_start"],
            epoch_end=epoch["epoch_end"],
        )
        dataset.append(
            {
                "subject_id": aligned["subject_id"],
                "recording_id": aligned["recording_id"],
                "epoch_start": epoch["epoch_start"],
                "epoch_end": epoch["epoch_end"],
                "reference_stage": epoch["reference_stage"],
                "wearable_features": {
                    key: value
                    for key, value in features.items()
                    if key not in {"modality_availability", "missingness"}
                },
                "modality_availability": features["modality_availability"],
                "missingness": features["missingness"],
                "excluded": epoch["excluded"],
                "exclusion_reason": epoch["exclusion_reason"],
                "preprocessing_version": PREPROCESSING_VERSION,
            }
        )
    return dataset


__all__ = [
    "SECONDARY_STAGE_ORDER",
    "STANDARD_STAGE_ORDER",
    "PREPROCESSING_VERSION",
    "align_recording_to_epochs",
    "apply_stage_taxonomy",
    "build_epoch_dataset",
    "build_wearable_feature_table",
    "canonicalize_stage_labels",
    "compute_eeg_band_stats",
    "compute_hrv_features",
    "extract_wearable_epoch_features",
    "resolve_stage_taxonomy",
]
