"""DREAMT dataset discovery and metadata inspection utilities."""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from src.config import RAW_DATA_DIR, REPO_ROOT, canonical_sleep_stage
from src.contracts import RecordingRecord, SignalMetadata, SubjectRecord

KNOWN_MODALITIES: tuple[str, ...] = (
    "accel_x",
    "accel_y",
    "accel_z",
    "bvp",
    "eda",
    "temp",
    "hr",
    "ibi",
    "hrv",
    "eeg_fpz_cz",
    "eeg_pz_oz",
    "emg",
    "eog",
)

SIGNAL_UNITS: dict[str, str] = {
    "accel_x": "g",
    "accel_y": "g",
    "accel_z": "g",
    "bvp": "AU",
    "eda": "uS",
    "temp": "C",
    "hr": "bpm",
    "ibi": "ms",
    "sleep_stage": "stage",
}

SIGNAL_SOURCES: dict[str, str] = {
    "accel_x": "wearable",
    "accel_y": "wearable",
    "accel_z": "wearable",
    "bvp": "wearable",
    "eda": "wearable",
    "temp": "wearable",
    "hr": "wearable",
    "ibi": "wearable",
    "hrv": "wearable",
    "eeg_fpz_cz": "psg",
    "eeg_pz_oz": "psg",
    "emg": "psg",
    "eog": "psg",
}

SIGNAL_ALIASES: dict[str, str] = {
    "bvp": "bvp",
    "acc_x": "accel_x",
    "acc_y": "accel_y",
    "acc_z": "accel_z",
    "accel_x": "accel_x",
    "accel_y": "accel_y",
    "accel_z": "accel_z",
    "eda": "eda",
    "temp": "temp",
    "temperature": "temp",
    "hr": "hr",
    "heart_rate": "hr",
    "ibi": "ibi",
    "inter_beat_interval": "ibi",
    "sleep_stage": "sleep_stage",
    "stage": "sleep_stage",
    "label": "sleep_stage",
    "annotation": "sleep_stage",
    "timestamp": "timestamp",
    "time": "timestamp",
    "seconds": "timestamp",
    "eeg_fpz_cz": "eeg_fpz_cz",
    "eeg_pz_oz": "eeg_pz_oz",
    "emg": "emg",
    "eog": "eog",
}

RECORD_LIKE_FIELDS: tuple[str, ...] = (
    "bvp",
    "accel_x",
    "accel_y",
    "accel_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "hr",
    "ibi",
    "eda",
    "temp",
    "eeg_fpz_cz",
    "eeg_pz_oz",
    "emg",
    "eog",
    "sleep_stage",
    "stage",
    "label",
    "annotation",
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
    signal_metadata: tuple[SignalMetadata, ...] = ()


def _normalize_name(name: str) -> str:
    """Normalize a file name into a canonical signal identifier."""
    normalized = name.lower().strip()
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace(".csv", "")
    normalized = normalized.replace(".tsv", "")
    normalized = normalized.replace(".txt", "")
    return normalized


def _normalize_column_name(name: str) -> str:
    """Normalize a CSV column name for matching against official DREAMT fields."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return normalized


def _canonical_signal_name(raw_name: str) -> str:
    """Map a raw field name to a canonical DREAMT modality or label column."""
    normalized = _normalize_column_name(raw_name)
    return SIGNAL_ALIASES.get(normalized, normalized)


def _coerce_float(value: object) -> float | None:
    """Convert a value to float, returning None for missing or malformed entries."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _estimate_sampling_rate(timestamps: Iterable[str]) -> float | None:
    """Estimate the sampling frequency from a monotonic timestamp series in seconds."""
    values: list[float] = []
    for value in timestamps:
        numeric = _coerce_float(value)
        if numeric is not None:
            values.append(float(numeric))
    if len(values) < 2:
        return None

    deltas = [
        current - previous
        for previous, current in zip(values, values[1:])
        if current > previous and math.isfinite(current) and math.isfinite(previous)
    ]
    if not deltas:
        return None

    step = median(deltas)
    if not math.isfinite(step) or step <= 0:
        return None
    return 1.0 / float(step)


def _guess_field(fieldnames: Iterable[str], *candidates: str) -> str | None:
    """Find the first matching CSV column using a case-insensitive comparison."""
    lowered = {str(name).lower(): str(name) for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV/TSV file into a list of row dictionaries."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"Malformed recording file '{path}': missing CSV header row.")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Recording file '{path}' contains no rows to inspect.")
    return rows


def _parse_signal_file(path: Path) -> tuple[tuple[str, float | str], ...]:
    """Parse a modality file that stores a timestamped signal series."""
    rows = _read_csv_rows(path)
    fieldnames = list(rows[0].keys())
    timestamp_field = _guess_field(fieldnames, "timestamp", "time", "t", "seconds", "sec")
    if timestamp_field is None:
        raise ValueError(
            f"Malformed modality file '{path}': missing a timestamp column. "
            f"Found columns: {', '.join(fieldnames)}."
        )

    value_field: str | None = None
    for column in fieldnames:
        if column.lower() == timestamp_field.lower():
            continue
        if _canonical_signal_name(column) in {"sleep_stage", "stage", "label", "annotation"}:
            continue
        value_field = column
        break
    if value_field is None:
        raise ValueError(
            f"Malformed modality file '{path}': unable to identify the value column."
        )

    series: list[tuple[str, float | str]] = []
    for row in rows:
        timestamp = str(row.get(timestamp_field, "")).strip()
        raw_value = row.get(value_field, "")
        if not timestamp:
            continue
        numeric_value = _coerce_float(raw_value)
        parsed_value: float | str = (
            numeric_value if numeric_value is not None else str(raw_value).strip()
        )
        series.append((timestamp, parsed_value))
    if not series:
        raise ValueError(f"Modality file '{path}' contains no valid timestamp/value rows.")
    return tuple(series)


def _parse_stage_file(path: Path) -> tuple[tuple[str, str], ...]:
    """Parse a stage annotation file while preserving explicit unknown values."""
    rows = _read_csv_rows(path)
    fieldnames = list(rows[0].keys())
    timestamp_field = _guess_field(fieldnames, "timestamp", "time", "t", "seconds", "sec")
    label_field = _guess_field(
        fieldnames,
        "sleep_stage",
        "stage",
        "label",
        "annotation",
        "sleepstage",
    )
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
        parsed.append((timestamp, canonical_sleep_stage(label)))
    if not parsed:
        raise ValueError(f"Stage file '{path}' contains no valid stage annotations.")
    return tuple(parsed)


def _contains_record_like_files(path: Path) -> bool:
    """Check whether a path contains DREAMT-like record files or a single combined CSV."""
    if not path.exists():
        return False
    if path.is_file():
        try:
            rows = _read_csv_rows(path)
        except ValueError:
            return False
        headers = list(rows[0].keys())
        normalized = {_normalize_column_name(header) for header in headers}
        if any(column in normalized for column in RECORD_LIKE_FIELDS):
            return True
        return False

    files = sorted(path.iterdir())
    for child in files:
        if child.is_file() and child.suffix.lower() in {".csv", ".tsv", ".txt"}:
            try:
                rows = _read_csv_rows(child)
            except ValueError:
                continue
            headers = list(rows[0].keys())
            normalized = {_normalize_column_name(header) for header in headers}
            if any(column in normalized for column in RECORD_LIKE_FIELDS):
                return True
    return False


def _subject_and_record_id_for_path(path: Path) -> tuple[str, str]:
    """Infer subject and recording identifiers from a record path."""
    if path.is_file():
        parent = path.parent.name
        return (parent or "unknown_subject", path.stem or path.name)

    parts = [part for part in path.parts if part not in {"/", "."}]
    if len(parts) >= 2:
        return (parts[-2], path.name)
    return ("unknown_subject", path.name)


def _build_signal_metadata(signal_name: str, sampling_rate_hz: float | None) -> SignalMetadata:
    """Create typed metadata for one signal channel from its canonical name."""
    source = SIGNAL_SOURCES.get(signal_name, "unknown")
    return SignalMetadata(
        name=signal_name,
        source=source,
        units=SIGNAL_UNITS.get(signal_name),
        sampling_rate_hz=sampling_rate_hz,
        description=(
            "PSG reference annotation"
            if signal_name == "sleep_stage"
            else f"{source} modality"
        ),
    )


def _build_record_from_content(
    source_dir: Path,
    subject_id: str,
    recording_id: str,
    modality_data: dict[str, tuple[tuple[str, float | str], ...]],
    stage_rows: tuple[tuple[str, str], ...],
    sensor_set: str,
    dataset: str = "dreamt",
) -> RecordingRecord:
    """Construct a typed RecordingRecord from a parsed DREAMT record."""
    stage_labels = tuple(label for _, label in stage_rows)
    timestamps = tuple(timestamp for timestamp, _ in stage_rows)
    if not timestamps and modality_data:
        first_signal = next(iter(modality_data.values()))
        timestamps = tuple(timestamp for timestamp, _ in first_signal)

    sampling_rates: dict[str, float | None] = {}
    for signal_name, series in modality_data.items():
        sampled_timestamps = tuple(timestamp for timestamp, _ in series)
        sampling_rates[signal_name] = _estimate_sampling_rate(sampled_timestamps)

    metadata = [
        _build_signal_metadata(signal_name, sampling_rates.get(signal_name))
        for signal_name in sorted(modality_data)
    ]

    available = tuple(sorted(modality_data))
    required = {
        "wearable": {"accel_x", "accel_y", "accel_z", "hr"},
        "psg": {"eeg_fpz_cz", "eeg_pz_oz", "emg", "eog"},
        "combined": set(),
    }[sensor_set]
    missing = sorted(required - set(modality_data)) if required else []
    if missing:
        raise ValueError(
            f"DREAMT record '{recording_id}' is missing required {sensor_set} modalities: "
            f"{', '.join(missing)}."
        )

    return RecordingRecord(
        subject_id=subject_id,
        recording_id=recording_id,
        dataset=dataset,
        source_dir=source_dir,
        available_modalities=available,
        signal_metadata=tuple(metadata),
        modality_data=modality_data,
        stage_labels=stage_labels,
        timestamps=timestamps,
        sampling_rates=sampling_rates,
        sensor_set=sensor_set,
    )


def inspect_dreamt_recording(
    record_dir: str | Path,
    sensor_set: str = "combined",
) -> RecordingRecord:
    """Inspect a single DREAMT recording and return typed metadata without loading all data."""
    path = Path(record_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"DREAMT record directory/file does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            raise ValueError(f"Unsupported DREAMT record file type: {path}")
        rows = _read_csv_rows(path)
        headers = list(rows[0].keys())
        normalized = {_normalize_column_name(header) for header in headers}
        if not any(column in normalized for column in RECORD_LIKE_FIELDS):
            raise ValueError(
                f"DREAMT record file '{path}' does not contain recognizable "
                "DREAMT signal columns."
            )

        modality_data: dict[str, tuple[tuple[str, float | str], ...]] = {}
        stage_rows: tuple[tuple[str, str], ...] = ()
        timestamp_field = _guess_field(headers, "timestamp", "time", "t", "seconds", "sec")
        if timestamp_field is None:
            raise ValueError(f"DREAMT record file '{path}' is missing a timestamp column.")

        for column in headers:
            canonical = _canonical_signal_name(column)
            if canonical in {"sleep_stage", "stage", "label", "annotation"}:
                stage_rows = tuple(
                    (
                        str(row.get(timestamp_field, "")).strip(),
                        canonical_sleep_stage(str(row.get(column, "")).strip()),
                    )
                    for row in rows
                    if str(row.get(timestamp_field, "")).strip()
                    and str(row.get(column, "")).strip()
                )
                continue
            if canonical in {"timestamp", "time", "seconds"}:
                continue
            if canonical in {
                "accel_x",
                "accel_y",
                "accel_z",
                "bvp",
                "eda",
                "temp",
                "hr",
                "ibi",
                "eeg_fpz_cz",
                "eeg_pz_oz",
                "emg",
                "eog",
            }:
                modality_data[canonical] = tuple(
                    (
                        str(row.get(timestamp_field, "")).strip(),
                        _coerce_float(row.get(column))
                        if _coerce_float(row.get(column)) is not None
                        else str(row.get(column, "")).strip(),
                    )
                    for row in rows
                    if str(row.get(timestamp_field, "")).strip()
                    and str(row.get(column, "")).strip()
                )

        if not modality_data and not stage_rows:
            raise ValueError(
                f"DREAMT record file '{path}' does not contain recognized "
                "wearable or PSG content."
            )

        subject_id, recording_id = _subject_and_record_id_for_path(path)
        return _build_record_from_content(
            source_dir=path.parent,
            subject_id=subject_id,
            recording_id=recording_id,
            modality_data=modality_data,
            stage_rows=stage_rows,
            sensor_set=sensor_set,
        )

    if not path.is_dir():
        raise ValueError(f"Unsupported DREAMT record target: {path}")

    files = sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in {".csv", ".tsv", ".txt"}
    )
    if not files:
        raise ValueError(f"DREAMT record directory is empty: {path}")

    modality_data: dict[str, tuple[tuple[str, float | str], ...]] = {}
    stage_rows: tuple[tuple[str, str], ...] = ()
    for file in files:
        try:
            rows = _read_csv_rows(file)
        except ValueError:
            continue
        headers = list(rows[0].keys())
        normalized = {_normalize_column_name(header) for header in headers}
        if not any(column in normalized for column in RECORD_LIKE_FIELDS):
            continue

        timestamp_field = _guess_field(headers, "timestamp", "time", "t", "seconds", "sec")
        if timestamp_field is None:
            continue

        for column in headers:
            canonical = _canonical_signal_name(column)
            if canonical in {"sleep_stage", "stage", "label", "annotation"}:
                stage_rows = tuple(
                    (
                        str(row.get(timestamp_field, "")).strip(),
                        canonical_sleep_stage(str(row.get(column, "")).strip()),
                    )
                    for row in rows
                    if str(row.get(timestamp_field, "")).strip()
                    and str(row.get(column, "")).strip()
                )
            elif canonical in {
                "accel_x",
                "accel_y",
                "accel_z",
                "bvp",
                "eda",
                "temp",
                "hr",
                "ibi",
                "eeg_fpz_cz",
                "eeg_pz_oz",
                "emg",
                "eog",
            }:
                modality_data[canonical] = tuple(
                    (
                        str(row.get(timestamp_field, "")).strip(),
                        _coerce_float(row.get(column))
                        if _coerce_float(row.get(column)) is not None
                        else str(row.get(column, "")).strip(),
                    )
                    for row in rows
                    if str(row.get(timestamp_field, "")).strip()
                    and str(row.get(column, "")).strip()
                )

    if not modality_data and not stage_rows:
        raise ValueError(
            f"DREAMT record directory '{path}' does not contain recognizable "
            "DREAMT content."
        )

    subject_id, recording_id = _subject_and_record_id_for_path(path)
    return _build_record_from_content(
        source_dir=path,
        subject_id=subject_id,
        recording_id=recording_id,
        modality_data=modality_data,
        stage_rows=stage_rows,
        sensor_set=sensor_set,
    )


def resolve_dreamt_root(data_dir: str | Path | None = None) -> Path:
    """Resolve the dataset root from an explicit path or the repository's known data directories."""
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()

    candidates = [
        RAW_DATA_DIR,
        RAW_DATA_DIR / "dreamt",
        REPO_ROOT / "data" / "dreamt",
        REPO_ROOT / "data" / "raw",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return RAW_DATA_DIR


def discover_dreamt_records(data_dir: str | Path | None = None) -> list[Path]:
    """Return candidate record directories/files under a DREAMT data root."""
    root = resolve_dreamt_root(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"DREAMT data directory does not exist: {root}")

    candidates: set[Path] = set()
    if root.is_file():
        if _contains_record_like_files(root):
            candidates.add(root)
    else:
        for child in sorted(root.rglob("*")):
            if child.is_dir() and _contains_record_like_files(child):
                candidates.add(child)
                continue
            if child.is_file() and child.parent == root and _contains_record_like_files(child):
                candidates.add(child)

    if not candidates:
        raise FileNotFoundError(
            f"No DREAMT record files or directories were found under {root}. "
            "Install the dataset or point the loader at a valid DREAMT root."
        )
    return sorted(candidates, key=lambda item: str(item))


def discover_dreamt_subjects(data_dir: str | Path) -> dict[str, SubjectRecord]:
    """Discover subject-level records without loading all signals into memory at once."""
    subjects: dict[str, SubjectRecord] = {}
    for record_path in discover_dreamt_records(data_dir):
        if record_path.is_dir():
            subject_id, recording_id = _subject_and_record_id_for_path(record_path)
        else:
            subject_id, recording_id = _subject_and_record_id_for_path(record_path)
        record_entry = subjects.setdefault(
            subject_id,
            SubjectRecord(
                subject_id=subject_id,
                dataset="dreamt",
                recording_ids=(),
            ),
        )
        recording_ids = list(record_entry.recording_ids)
        if recording_id not in recording_ids:
            recording_ids.append(recording_id)
        subjects[subject_id] = SubjectRecord(
            subject_id=subject_id,
            dataset="dreamt",
            recording_ids=tuple(sorted(recording_ids)),
            wearable_signals=("accel_x", "accel_y", "accel_z", "bvp", "eda", "temp", "hr", "ibi"),
            psg_signals=("eeg_fpz_cz", "eeg_pz_oz", "emg", "eog"),
        )
    return subjects


def validate_dreamt_layout(data_dir: str | Path) -> list[Path]:
    """Compatibility wrapper for the repository's existing dataset layout contract."""
    return discover_dreamt_records(data_dir)


def load_dreamt_record(
    record_dir: str | Path,
    sensor_set: str = "combined",
) -> DreamtRecord:
    """Load a single DREAMT record and return the legacy metadata container used
    by existing tests.
    """
    recording = inspect_dreamt_recording(record_dir, sensor_set=sensor_set)
    return DreamtRecord(
        subject_id=recording.subject_id,
        recording_id=recording.recording_id,
        source_dir=recording.source_dir or Path(record_dir),
        available_modalities=recording.available_modalities,
        modality_data=recording.modality_data,
        stage_labels=recording.stage_labels,
        timestamps=recording.timestamps,
        start_timestamp=recording.timestamps[0] if recording.timestamps else None,
        end_timestamp=recording.timestamps[-1] if recording.timestamps else None,
        sampling_rates=recording.sampling_rates,
        signal_metadata=recording.signal_metadata,
    )


def load_dreamt_dataset(
    data_dir: str | Path,
    sensor_set: str = "combined",
) -> list[RecordingRecord]:
    """Load DREAMT records one at a time without requiring the full dataset in memory."""
    records: list[RecordingRecord] = []
    for record_path in discover_dreamt_records(data_dir):
        records.append(inspect_dreamt_recording(record_path, sensor_set=sensor_set))
    return records


__all__ = [
    "DreamtRecord",
    "discover_dreamt_records",
    "discover_dreamt_subjects",
    "inspect_dreamt_recording",
    "load_dreamt_dataset",
    "load_dreamt_record",
    "resolve_dreamt_root",
    "validate_dreamt_layout",
]
