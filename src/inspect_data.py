"""Dataset inspection CLI and DREAMT audit utilities for the Nocturne project.

This module intentionally focuses on dataset discovery, integrity checks, and
metadata inspection, not feature extraction or model fitting. It is designed to
support DREAMT validation with direct local data roots while preserving clear,
defensive error handling.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from src.config import SUPPORTED_DATASETS, SUPPORTED_SENSOR_SETS
from src.ingest.load_dreamt import (
    discover_dreamt_records,
    inspect_dreamt_recording,
    resolve_dreamt_root,
)

_DREAMT_REQUIRED_MODALITIES: tuple[str, ...] = (
    "accel_x",
    "accel_y",
    "accel_z",
    "bvp",
    "eda",
    "temp",
    "hr",
    "ibi",
)


def _canonical_modality_name(path: Path) -> str | None:
    """Infer a canonical DREAMT modality or label name from a file name."""
    name = path.stem.lower().replace("-", "_").replace(" ", "_")
    if name in {"stage_labels", "sleep_stage", "sleep_stage_label", "label", "labels"}:
        return "sleep_stage"
    if name.startswith("accel"):
        if "x" in name:
            return "accel_x"
        if "y" in name:
            return "accel_y"
        if "z" in name:
            return "accel_z"
    if name.startswith("acc_"):
        if name.endswith("x"):
            return "accel_x"
        if name.endswith("y"):
            return "accel_y"
        if name.endswith("z"):
            return "accel_z"
    if name in {"bvp", "hr", "eda", "temp", "ibi", "eeg_fpz_cz", "eeg_pz_oz", "emg", "eog"}:
        return name
    if "stage" in name and "sleep" not in name:
        return "sleep_stage"
    if "sleep" in name and "stage" in name:
        return "sleep_stage"
    return None


def _is_corrupt_file(path: Path) -> bool:
    """Return True when a supposed signal file is empty or malformed."""
    if not path.is_file():
        return False
    if path.stat().st_size == 0:
        return True
    if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
    except (OSError, ValueError, csv.Error):
        return True
    return row is None and path.name.lower() not in {"metadata.txt"}


def _record_file_summary(root: Path, record_path: Path) -> dict[str, object]:
    """Collect file-level details for a single DREAMT record."""
    record_files = sorted(p for p in record_path.iterdir() if p.is_file())
    summary = {
        "subject_id": record_path.parent.name if record_path.parent.name else record_path.name,
        "recording_id": record_path.name,
        "files": [],
        "modalities": [],
        "label_files": [],
    }
    for file_path in record_files:
        file_name = str(file_path.relative_to(root))
        summary["files"].append(file_name)
        modality_name = _canonical_modality_name(file_path)
        if modality_name is not None:
            summary["modalities"].append(modality_name)
        if modality_name == "sleep_stage" or "stage" in file_path.stem.lower():
            summary["label_files"].append(file_name)
    return summary


def audit_dreamt_dataset(data_dir: str | Path | None = None) -> dict[str, object]:
    """Audit a DREAMT dataset root and report subject, recording, and file-level health."""
    root = resolve_dreamt_root(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"DREAMT data directory does not exist: {root}")

    record_paths = discover_dreamt_records(root)
    subjects: set[str] = set()
    modalities: set[str] = set()
    label_files: list[str] = []
    file_count = 0
    total_size_bytes = 0
    missing_files: list[str] = []
    corrupt_files: list[str] = []
    timestamp_coverage: list[dict[str, object]] = []
    record_summaries: list[dict[str, object]] = []

    for record_path in record_paths:
        record = inspect_dreamt_recording(record_path, sensor_set="combined")
        subjects.add(record.subject_id)
        record_summary = _record_file_summary(root, record_path)
        record_summaries.append(record_summary)

        for file_path in sorted(p for p in record_path.iterdir() if p.is_file()):
            file_count += 1
            total_size_bytes += file_path.stat().st_size
            modality_name = _canonical_modality_name(file_path)
            if modality_name is not None:
                modalities.add(modality_name)
            if modality_name == "sleep_stage" or "stage" in file_path.stem.lower():
                label_files.append(str(file_path.relative_to(root)))
            if _is_corrupt_file(file_path):
                corrupt_files.append(str(file_path.relative_to(root)))

        required_missing = [
            f"{record.subject_id}/{record.recording_id}/{field}.csv"
            for field in _DREAMT_REQUIRED_MODALITIES
            if field not in record.available_modalities
        ]
        missing_files.extend(required_missing)
        timestamp_coverage.append(
            {
                "subject_id": record.subject_id,
                "recording_id": record.recording_id,
                "start_timestamp": record.timestamps[0] if record.timestamps else None,
                "end_timestamp": record.timestamps[-1] if record.timestamps else None,
                "stage_count": len(record.stage_labels),
            }
        )

    return {
        "dataset": "dreamt",
        "data_dir": str(root),
        "subject_count": len(subjects),
        "record_count": len(record_paths),
        "subjects": sorted(subjects),
        "recordings": [p.name for p in record_paths],
        "file_count": file_count,
        "total_size_bytes": total_size_bytes,
        "modalities": sorted(modalities),
        "label_files": sorted(set(label_files)),
        "missing_files": sorted(set(missing_files)),
        "corrupt_files": sorted(set(corrupt_files)),
        "timestamp_coverage": timestamp_coverage,
        "records": record_summaries,
    }


def inspect_dataset(
    dataset: str,
    data_dir: str | Path | None = None,
    sensor_set: str = "combined",
) -> dict[str, object]:
    """Inspect a supported dataset and return a machine-readable summary."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported values: {', '.join(SUPPORTED_DATASETS)}."
        )

    if sensor_set not in SUPPORTED_SENSOR_SETS:
        raise ValueError(
            f"Unsupported sensor set '{sensor_set}'. "
            f"Supported values: {', '.join(SUPPORTED_SENSOR_SETS)}."
        )

    if dataset != "dreamt":
        raise NotImplementedError(
            f"Dataset inspection for '{dataset}' is not implemented yet; "
            "only the DREAMT loader is available in this milestone."
        )

    root = resolve_dreamt_root(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"DREAMT data directory does not exist: {root}")

    record_paths = discover_dreamt_records(root)
    records: list[dict[str, object]] = []
    modalities: set[str] = set()
    subjects: set[str] = set()

    for record_path in record_paths:
        record = inspect_dreamt_recording(record_path, sensor_set=sensor_set)
        record_summary = {
            "subject_id": record.subject_id,
            "recording_id": record.recording_id,
            "source_dir": str(record.source_dir),
            "available_modalities": list(record.available_modalities),
            "stage_count": len(record.stage_labels),
            "start_timestamp": record.timestamps[0] if record.timestamps else None,
            "end_timestamp": record.timestamps[-1] if record.timestamps else None,
        }
        records.append(record_summary)
        subjects.add(record.subject_id)
        modalities.update(record.available_modalities)

    return {
        "dataset": dataset,
        "data_dir": str(root),
        "record_count": len(records),
        "subject_count": len(subjects),
        "modalities": sorted(modalities),
        "sensor_set": sensor_set,
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the dataset-inspection CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a supported sleep dataset and report discovery metadata without "
            "performing model training or feature extraction."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Dataset identifier to inspect.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Optional explicit dataset root to inspect. Defaults to the "
            "repository data directory."
        ),
    )
    parser.add_argument(
        "--sensor-set",
        choices=SUPPORTED_SENSOR_SETS,
        default="combined",
        help="Sensor set to validate while inspecting the dataset.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help=(
            "Run the DREAMT integrity audit and report subjects, files, "
            "modalities, missing data, and timestamp coverage."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output instead of plain text.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset-inspection CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = (
            audit_dreamt_dataset(args.data_dir)
            if args.dataset == "dreamt" and args.audit
            else inspect_dataset(
                dataset=args.dataset,
                data_dir=args.data_dir,
                sensor_set=args.sensor_set,
            )
        )
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        parser.exit(status=2, message=f"Inspection error: {exc}\n")

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"dataset={summary['dataset']}")
        print(f"data_dir={summary['data_dir']}")
        print(f"record_count={summary['record_count']}")
        print(f"subject_count={summary['subject_count']}")
        print(f"sensor_set={summary.get('sensor_set', 'combined')}")
        if summary.get("modalities"):
            print(f"modalities={', '.join(summary['modalities'])}")
        if args.audit:
            print(f"file_count={summary['file_count']}")
            print(f"missing_files={len(summary['missing_files'])}")
            print(f"corrupt_files={len(summary['corrupt_files'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_dreamt_dataset", "build_parser", "inspect_dataset", "main"]
