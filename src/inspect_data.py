"""Dataset inspection CLI for the Nocturne project.

This module intentionally focuses on dataset discovery and metadata inspection, not
feature extraction or model fitting. It is designed to support DREAMT validation
with direct local data roots while preserving clear, defensive error handling.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.config import SUPPORTED_DATASETS, SUPPORTED_SENSOR_SETS
from src.ingest.load_dreamt import (
    discover_dreamt_records,
    inspect_dreamt_recording,
    resolve_dreamt_root,
)


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
        summary = inspect_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            sensor_set=args.sensor_set,
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
        print(f"sensor_set={summary['sensor_set']}")
        if summary["modalities"]:
            print(f"modalities={', '.join(summary['modalities'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "inspect_dataset", "main"]
