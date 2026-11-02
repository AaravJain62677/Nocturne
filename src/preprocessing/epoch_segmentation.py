"""Epoch alignment and label processing for DREAMT-style recordings."""

from __future__ import annotations

import math
from typing import Any

from src.config import EPOCH_SECONDS, canonical_sleep_stage


def _to_float(value: object) -> float | None:
    """Convert a timestamp-like value to a float or return None when invalid."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _epoch_value(value: float) -> int | float:
    """Render a timestamp-like epoch boundary cleanly as an int when possible."""
    if math.isfinite(value) and float(value).is_integer():
        return int(value)
    return value


def _render_epoch_boundary(
    raw_origin: float | int | str,
    value: float,
    prefer_strings: bool,
) -> str | int | float:
    """Render an epoch boundary using the source timestamp representation."""
    normalized = _epoch_value(value)
    if prefer_strings:
        return str(normalized)
    return normalized


def _safe_canonical_stage(label: object) -> str:
    """Normalize a stage label, downgrading malformed entries to UNKNOWN."""
    try:
        return canonical_sleep_stage(label)
    except ValueError:
        return "UNKNOWN"


def _make_stage_rows(record: dict[str, Any]) -> list[tuple[str | None, str]]:
    """Normalize stage data into (timestamp, label) rows."""
    stage_labels = record.get("stage_labels", ())
    stage_rows: list[tuple[str | None, str]] = []
    if not stage_labels:
        return stage_rows

    timestamps = tuple(record.get("timestamps", ()))
    if stage_labels and isinstance(stage_labels[0], tuple):
        for item in stage_labels:
            if len(item) != 2:
                continue
            raw_ts, label = item
            stage_rows.append((str(raw_ts) if raw_ts is not None else None, str(label)))
        return stage_rows

    for index, label in enumerate(stage_labels):
        raw_ts = timestamps[index] if index < len(timestamps) else None
        stage_rows.append((str(raw_ts) if raw_ts is not None else None, str(label)))
    return stage_rows


def canonicalize_stage_labels(
    labels: list[str] | tuple[str, ...],
    taxonomy: str = "standard",
) -> list[str]:
    """Normalize stage labels to the repository's canonical taxonomy."""
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
    record: dict[str, Any] | Any,
    epoch_seconds: int = EPOCH_SECONDS,
    time_origin: float | None = None,
) -> dict[str, Any]:
    """Align a record's stage labels into explicit 30-second epochs.

    Each returned epoch keeps the original timestamp semantics, records ambiguous
    label overlaps, and keeps unscored or invalid epochs explicit instead of
    silently assigning a label.
    """
    if epoch_seconds <= 0:
        raise ValueError(f"Epoch length must be positive; got {epoch_seconds!r}.")

    if hasattr(record, "subject_id"):
        subject_id = getattr(record, "subject_id")
        recording_id = getattr(record, "recording_id")
        timestamps = tuple(getattr(record, "timestamps", ()))
        stage_labels = tuple(getattr(record, "stage_labels", ()))
        modality_data = getattr(record, "modality_data", {})
    else:
        subject_id = record.get("subject_id", "unknown_subject")
        recording_id = record.get("recording_id", "unknown_record")
        timestamps = tuple(record.get("timestamps", ()))
        stage_labels = tuple(record.get("stage_labels", ()))
        modality_data = record.get("modality_data", {})

    numeric_timestamps = [
        value for value in (_to_float(ts) for ts in timestamps) if value is not None
    ]
    if time_origin is not None:
        origin = float(time_origin)
    elif numeric_timestamps:
        origin = min(numeric_timestamps)
    else:
        origin = 0.0

    duplicate_count = 0
    seen: set[float] = set()
    for value in numeric_timestamps:
        if value in seen:
            duplicate_count += 1
        seen.add(value)

    if numeric_timestamps:
        last_timestamp = max(numeric_timestamps)
        epoch_count = max(1, int((last_timestamp - origin) / epoch_seconds) + 1)
    elif stage_labels:
        epoch_count = max(1, len(stage_labels))
    else:
        epoch_count = 1

    prefer_strings = bool(timestamps) and all(
        isinstance(ts, str) or (ts is not None and not isinstance(ts, (int, float)))
        for ts in timestamps
    )
    if time_origin is not None:
        prefer_strings = False
    epoch_starts = tuple(
        _render_epoch_boundary(origin, origin + index * epoch_seconds, prefer_strings)
        for index in range(epoch_count)
    )
    stage_rows = _make_stage_rows({"stage_labels": stage_labels, "timestamps": timestamps})

    epochs: list[dict[str, Any]] = []
    reference_labels: list[str] = []
    excluded_mask: list[bool] = []
    for index, epoch_start in enumerate(epoch_starts):
        numeric_start = _to_float(epoch_start)
        epoch_start_value = float(numeric_start if numeric_start is not None else 0.0)
        epoch_end_value = epoch_start_value + epoch_seconds
        epoch_end = _render_epoch_boundary(origin, epoch_end_value, prefer_strings)
        matches: list[tuple[float, str]] = []
        for raw_ts, label in stage_rows:
            numeric_ts = _to_float(raw_ts)
            if numeric_ts is None:
                continue
            if epoch_start_value <= numeric_ts < epoch_end_value:
                matches.append((numeric_ts, str(label)))

        if not matches:
            reference_stage = "UNKNOWN"
            excluded = True
            reason = "missing_label"
        else:
            matches.sort(key=lambda item: item[0])
            unique_labels = {_safe_canonical_stage(item[1]) for item in matches}
            if len(unique_labels) > 1:
                reference_stage = _safe_canonical_stage(matches[0][1])
                excluded = reference_stage in {"UNKNOWN", "UNSCORED", "MOVEMENT"}
                reason = "ambiguous_label_overlap"
            else:
                reference_stage = _safe_canonical_stage(matches[0][1])
                excluded = reference_stage in {"UNKNOWN", "UNSCORED", "MOVEMENT"}
                reason = None if not excluded else "reference_excluded"

        epochs.append(
            {
                "epoch_index": index,
                "epoch_start": epoch_start,
                "epoch_end": epoch_end,
                "reference_stage": reference_stage,
                "excluded": excluded,
                "exclusion_reason": reason,
                "label_matches": tuple(matches),
            }
        )
        reference_labels.append(reference_stage)
        excluded_mask.append(excluded)

    epoch_ends = tuple(
        _render_epoch_boundary(
            origin,
            float(_to_float(epoch_start) if _to_float(epoch_start) is not None else 0.0)
            + epoch_seconds,
            prefer_strings,
        )
        for epoch_start in epoch_starts
    )

    return {
        "subject_id": subject_id,
        "recording_id": recording_id,
        "epoch_count": len(epoch_starts),
        "epoch_starts": epoch_starts,
        "epoch_ends": epoch_ends,
        "labels": tuple(reference_labels),
        "excluded_mask": tuple(excluded_mask),
        "epoch_seconds": epoch_seconds,
        "modality_keys": tuple(sorted(modality_data)),
        "time_origin": origin,
        "time_unit": "seconds",
        "duplicate_timestamps": duplicate_count,
        "epochs": tuple(epochs),
    }


__all__ = [
    "align_recording_to_epochs",
    "canonicalize_stage_labels",
]
