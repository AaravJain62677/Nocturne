"""Epoch alignment and label processing for DREAMT-style recordings."""

from __future__ import annotations

from typing import Any

from src.config import EPOCH_SECONDS, canonical_sleep_stage


def canonicalize_stage_labels(
    labels: list[str] | tuple[str, ...],
    taxonomy: str = "standard",
) -> list[str]:
    """Normalize stage labels to the repository's canonical taxonomy.

    Unknown, movement and unscored labels remain explicit and are never silently
    remapped to Wake. The taxonomy is validated but not silently transformed.
    """
    normalized: list[str] = []
    for label in labels:
        normalized.append(canonical_sleep_stage(label))

    if taxonomy == "standard":
        return normalized
    if taxonomy == "secondary":
        from src.preprocessing import SECONDARY_SLEEP_STAGE_MAP

        return [SECONDARY_SLEEP_STAGE_MAP.get(label, label) for label in normalized]
    raise ValueError(f"Unsupported taxonomy '{taxonomy}'. Supported values: standard, secondary.")


def align_recording_to_epochs(
    record: dict[str, Any],
    epoch_seconds: int = EPOCH_SECONDS,
) -> dict[str, Any]:
    """Align a record's modality streams and labels into epoch-level rows.

    This function intentionally keeps the preprocessing contract explicit and does
    not interpolate missing samples or shift timestamps silently.
    """
    if epoch_seconds <= 0:
        raise ValueError(f"Epoch length must be positive; got {epoch_seconds!r}.")

    subject_id = record.get("subject_id", "unknown_subject")
    recording_id = record.get("recording_id", "unknown_record")
    modality_data = record.get("modality_data", {})
    stage_labels = tuple(record.get("stage_labels", ()))
    timestamps = tuple(record.get("timestamps", ()))

    canonical_labels = tuple(canonical_sleep_stage(label) for label in stage_labels)
    excluded_mask = tuple(
        label in {"UNKNOWN", "UNSCORED", "MOVEMENT"}
        for label in canonical_labels
    )

    epoch_starts = tuple(str(ts) for ts in timestamps)
    aligned = {
        "subject_id": subject_id,
        "recording_id": recording_id,
        "epoch_count": len(epoch_starts),
        "epoch_starts": epoch_starts,
        "labels": canonical_labels,
        "excluded_mask": excluded_mask,
        "epoch_seconds": epoch_seconds,
        "modality_keys": tuple(sorted(modality_data)),
    }
    return aligned


__all__ = [
    "align_recording_to_epochs",
    "canonicalize_stage_labels",
]
