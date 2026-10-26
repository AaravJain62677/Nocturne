"""DREAMT dataset discovery and metadata inspection utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from src.config import canonical_sleep_stage

KNOWN_MODALITIES: tuple[str, ...] = (
    "accel_x",
    "accel_y",
    "accel_z",
    "hr",
    "hrv",
    "eeg_fpz_cz",
    "eeg_pz_oz",
    "emg",
    "eog",
)


@dataclass(frozen=True)
class DreamtRecord:
    """Metadata and parsed content for one DREAMT recording directory."""

    subject_id: str
    recording_id: str
    source_dir: Path
    available_modalities: tuple[str, ...]
    modality_data: dict[str, tuple[tuple[str, float | str], ...]] = field(default_factory=dict)
    stage_labels: tuple[str, ...] = ()
    timestamps: tuple[str, ...] = ()
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    sampling_rates: dict[str, float | None] = field(default_factory=dict)


def _normalize_name(name: str) -> str:
    """Normalize file basename into a canonical signal identifier."""
    normalized = name.lower().strip()
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace(".csv", "")
    normalized = normalized.replace(".tsv", "")
    normalized = normalized.replace(".txt", "")
    return normalized


def _contains_stage_or_signal_files(path: Path) -> bool:
    """Detect whether a directory looks like a DREAMT record directory."""
    if not path.is_dir():
        return False

    names = {_normalize_name(child.name) for child in path.iterdir() if child.is_file()}
    if not names:
        return False

    stage_keywords = {"stage", "stages", "label", "labels", "annotation", "annotations"}
    if names and any(keyword in name for name in names for keyword in stage_keywords):
        return True

    return any(name in KNOWN_MODALITIES or name.startswith("accel_") for name in names)


def discover_dreamt_records(data_dir: str | Path) -> list[Path]:
    """Return all record directories under the DREAMT dataset root."""
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"DREAMT data directory does not exist: {root}")

    candidates: set[Path] = set()
    if root.is_dir() and _contains_stage_or_signal_files(root):
        candidates.add(root)

    for path in sorted(root.rglob("*")):
        if path.is_dir() and _contains_stage_or_signal_files(path):
            candidates.add(path)

    if not candidates:
        raise FileNotFoundError(
            f"No DREAMT record directories were found under {root}. Expected wearable or PSG files."
        )

    return sorted(candidates, key=lambda p: str(p))


def _guess_field(fieldnames: list[str], *candidates: str) -> str | None:
    """Return the first matching CSV column name from common timestamp/label patterns."""
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV/TSV text file into a list of row dictionaries."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Malformed recording file '{path}': missing CSV header row.")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Recording file '{path}' contains no rows to inspect.")
    return rows


def _parse_modality_series(path: Path) -> tuple[tuple[str, float | str], ...]:
    """Read a modality file and keep timestamp-aligned values as tuples."""
    rows = _read_csv_rows(path)
    fieldnames = list(rows[0].keys())
    timestamp_field = _guess_field(fieldnames, "timestamp", "time", "t", "seconds")
    value_field = _guess_field(fieldnames, "value", "signal", "reading", "amplitude")

    if timestamp_field is None:
        raise ValueError(
            f"Malformed modality file '{path}': missing a timestamp column. "
            f"Found columns: {', '.join(fieldnames)}."
        )
    if value_field is None:
        for candidate in fieldnames:
            if candidate.lower() not in {timestamp_field.lower(), "stage", "label", "annotation"}:
                value_field = candidate
                break
    if value_field is None:
        raise ValueError(
            f"Malformed modality file '{path}': unable to identify the value column."
        )

    series: list[tuple[str, float | str]] = []
    for row in rows:
        ts = str(row.get(timestamp_field, "")).strip()
        raw_value = row.get(value_field, "")
        if not ts:
            continue
        try:
            parsed_value: float | str = float(raw_value)
        except (TypeError, ValueError):
            parsed_value = str(raw_value).strip()
        series.append((ts, parsed_value))

    if not series:
        raise ValueError(f"Modality file '{path}' contains no valid timestamp/value rows.")
    return tuple(series)


def _parse_stage_series(path: Path) -> tuple[tuple[str, str], ...]:
    """Read stage annotation rows and preserve labels without coercing unknown values."""
    rows = _read_csv_rows(path)
    fieldnames = list(rows[0].keys())
    timestamp_field = _guess_field(fieldnames, "timestamp", "time", "t", "seconds")
    label_field = _guess_field(fieldnames, "stage", "label", "sleep_stage", "annotation")

    if timestamp_field is None:
        raise ValueError(
            f"Malformed stage file '{path}': missing a timestamp column. "
            f"Found columns: {', '.join(fieldnames)}."
        )
    if label_field is None:
        raise ValueError(
            f"Malformed stage file '{path}': missing a stage/label column. "
            f"Found columns: {', '.join(fieldnames)}."
        )

    parsed: list[tuple[str, str]] = []
    for row in rows:
        timestamp = str(row.get(timestamp_field, "")).strip()
        label = str(row.get(label_field, "")).strip()
        if not timestamp or not label:
            continue
        canonical = canonical_sleep_stage(label)
        parsed.append((timestamp, canonical))
    if not parsed:
        raise ValueError(f"Stage file '{path}' contains no valid stage annotations.")
    return tuple(parsed)


def _resolve_subject_and_record(record_dir: Path) -> tuple[str, str]:
    """Infer subject and recording IDs from the directory structure."""
    parts = [part for part in record_dir.parts if part not in {"/", "."}]
    if not parts:
        return ("unknown_subject", record_dir.name)

    subject_id = parts[-2] if len(parts) >= 2 else "unknown_subject"
    recording_id = record_dir.name
    return (subject_id, recording_id)


def validate_dreamt_layout(data_dir: str | Path) -> list[Path]:
    """Validate the DREAMT layout and return record directories in a deterministic order."""
    return discover_dreamt_records(data_dir)


def load_dreamt_record(
    record_dir: str | Path,
    sensor_set: str = "combined",
) -> DreamtRecord:
    """Load and validate a single DREAMT recording directory.

    This implementation keeps the record as a metadata container and explicit
    modality series. It does not resample or train on the data.
    """
    path = Path(record_dir).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"DREAMT record directory does not exist: {path}")

    files = sorted(path.iterdir())
    if not files:
        raise ValueError(f"DREAMT record directory is empty: {path}")

    stage_files = [
        candidate
        for candidate in files
        if candidate.is_file() and "stage" in candidate.name.lower()
    ]
    if not stage_files:
        stage_files = [
            candidate
            for candidate in files
            if candidate.is_file()
            and any(
                keyword in candidate.name.lower()
                for keyword in ("label", "annotation")
            )
        ]
    if not stage_files:
        raise ValueError(f"DREAMT record '{path.name}' is missing a stage annotation file.")

    modality_files = [
        candidate
        for candidate in files
        if candidate.is_file()
        and candidate.suffix.lower() in {".csv", ".tsv", ".txt"}
        and candidate not in stage_files
    ]
    if not modality_files:
        raise ValueError(f"DREAMT record '{path.name}' does not contain any modality files.")

    modality_data: dict[str, tuple[tuple[str, float | str], ...]] = {}
    available: set[str] = set()
    for signal_file in modality_files:
        modality_name = _normalize_name(signal_file.name)
        known_modalities = {
            "accel_x",
            "accel_y",
            "accel_z",
            "hr",
            "hrv",
            "eeg_fpz_cz",
            "eeg_pz_oz",
            "emg",
            "eog",
        }
        if modality_name in known_modalities:
            modality_data[modality_name] = _parse_modality_series(signal_file)
            available.add(modality_name)

    if not available:
        raise ValueError(
            f"DREAMT record '{path.name}' does not include any recognized modality files."
        )

    stage_rows = _parse_stage_series(stage_files[0])
    stages = tuple(label for _, label in stage_rows)
    timestamps = tuple(ts for ts, _ in stage_rows)
    start_timestamp = timestamps[0] if timestamps else None
    end_timestamp = timestamps[-1] if timestamps else None

    subject_id, recording_id = _resolve_subject_and_record(path)

    if sensor_set == "wearable":
        required = {"accel_x", "accel_y", "accel_z", "hr"}
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"DREAMT record '{path.name}' is missing wearable modalities: {', '.join(missing)}."
            )
    elif sensor_set == "psg":
        required = {"eeg_fpz_cz", "eeg_pz_oz", "emg", "eog"}
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"DREAMT record '{path.name}' is missing PSG modalities: {', '.join(missing)}."
            )
    elif sensor_set == "combined":
        pass
    else:
        raise ValueError(f"Unsupported sensor set '{sensor_set}'.")

    return DreamtRecord(
        subject_id=subject_id,
        recording_id=recording_id,
        source_dir=path,
        available_modalities=tuple(sorted(available)),
        modality_data=modality_data,
        stage_labels=stages,
        timestamps=timestamps,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )


def load_dreamt_dataset(
    data_dir: str | Path,
    sensor_set: str = "combined",
) -> list[DreamtRecord]:
    """Load all valid DREAMT record directories under a raw-data root."""
    records: list[DreamtRecord] = []
    for record_dir in validate_dreamt_layout(data_dir):
        records.append(load_dreamt_record(record_dir, sensor_set=sensor_set))
    return records


__all__ = [
    "DreamtRecord",
    "discover_dreamt_records",
    "load_dreamt_dataset",
    "load_dreamt_record",
    "validate_dreamt_layout",
]
