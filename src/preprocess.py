"""Synthetic preprocessing CLI for epoch-level wearable feature extraction.

This milestone intentionally supports deterministic preprocessing on synthetic or
fixture-style data, while clearly reporting that real DREAMT validation remains
pending until actual recordings are available.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SUPPORTED_DATASETS
from src.preprocessing import PREPROCESSING_VERSION, build_epoch_dataset

FEATURE_DEFINITIONS: dict[str, str] = {
    "hr_mean": "Average heart rate in beats per minute over the epoch.",
    "hr_median": "Median heart rate in beats per minute over the epoch.",
    "hr_std": "Population standard deviation of heart rate samples in the epoch.",
    "hr_min": "Minimum heart rate in beats per minute over the epoch.",
    "hr_max": "Maximum heart rate in beats per minute over the epoch.",
    "hr_valid_count": "Valid HR samples used in the epoch-level summary.",
    "hrv_available": "Whether RR/IBI-derived HRV is available for the epoch.",
    "hrv_rmssd": "Root-mean-square of successive RR differences in milliseconds.",
    "hrv_sdnn": "Standard deviation of RR intervals in milliseconds.",
    "accel_x_mean": "Mean acceleration along the x-axis in the epoch.",
    "accel_y_mean": "Mean acceleration along the y-axis in the epoch.",
    "accel_z_mean": "Mean acceleration along the z-axis in the epoch.",
    "accel_x_std": "Std. dev. of x-axis acceleration samples in the epoch.",
    "accel_y_std": "Std. dev. of y-axis acceleration samples in the epoch.",
    "accel_z_std": "Std. dev. of z-axis acceleration samples in the epoch.",
    "accel_magnitude_mean": "Mean vector magnitude of the 3-axis accelerometer signal.",
    "accel_magnitude_std": "Std. dev. of vector-magnitude samples over the epoch.",
    "movement_intensity": "Mean vector magnitude used as a motion-intensity summary.",
    "accel_valid_count": "Count of valid accelerometer samples used in the epoch summary.",
    "eda_mean": "Mean electrodermal activity over the epoch.",
    "eda_std": "Std. dev. of electrodermal activity over the epoch.",
    "eda_min": "Minimum electrodermal activity over the epoch.",
    "eda_max": "Maximum electrodermal activity over the epoch.",
    "eda_valid_count": "Valid EDA samples used in the epoch-level summary.",
    "eda_tonic_mean": "Median EDA value used as a tonic baseline estimate.",
    "eda_phasic_mean": "Mean positive deviation from tonic EDA over the epoch.",
}
SYNTHETIC_DEMONSTRATION_LABEL = "SYNTHETIC DEMONSTRATION — NOT REAL DREAMT RESULTS"
SYNTHETIC_STAGE_ORDER: tuple[str, ...] = ("WAKE", "N1", "N2", "N3", "REM")


def generate_synthetic_demo_records(
    *,
    seed: int = 42,
    subject_count: int = 6,
    recordings_per_subject: int = 2,
    epochs_per_record: int = 12,
) -> list[dict[str, object]]:
    """Create a deterministic synthetic demonstration record set for pipeline tests.

    This is intentionally labelled as a demonstration utility and cannot be used to
    claim real DREAMT performance. The output follows the same epoch schema that the
    legitimate preprocessing pipeline expects.
    """
    rng = np.random.default_rng(int(seed))
    records: list[dict[str, object]] = []
    stage_offsets = {
        "WAKE": 0.0,
        "N1": 4.0,
        "N2": 6.0,
        "N3": 9.0,
        "REM": 3.0,
    }
    for subject_index in range(int(subject_count)):
        for recording_index in range(int(recordings_per_subject)):
            epochs: list[dict[str, object]] = []
            for epoch_index in range(int(epochs_per_record)):
                stage_index = (subject_index + recording_index + epoch_index) % len(
                    SYNTHETIC_STAGE_ORDER
                )
                stage = SYNTHETIC_STAGE_ORDER[stage_index]
                phase = epoch_index * 30.0
                base_hr = (
                    58.0
                    + subject_index * 2.5
                    + recording_index * 1.5
                    + stage_offsets[stage]
                    + rng.normal(0.0, 2.5)
                )
                hrv = (
                    32.0
                    + subject_index * 2.0
                    + ("REM" == stage) * 8.0
                    + rng.normal(0.0, 4.0)
                )
                accel_x = rng.normal(0.1, 0.08)
                accel_y = rng.normal(0.15, 0.09)
                accel_z = rng.normal(0.2, 0.1)
                accel_mag = float(np.sqrt(accel_x**2 + accel_y**2 + accel_z**2))
                eda = max(2.0, 3.2 + subject_index * 0.3 + rng.normal(0.0, 0.7))
                modality_availability = {
                    "hr": True,
                    "ibi": True,
                    "accel_x": True,
                    "accel_y": True,
                    "accel_z": True,
                    "eda": True,
                }
                missingness = {
                    "hr": (
                        0.0
                        if (subject_index + recording_index + epoch_index) % 9
                        else 0.25
                    ),
                    "accel": (
                        0.0
                        if (subject_index + recording_index + epoch_index) % 11
                        else 0.2
                    ),
                    "eda": (
                        0.0
                        if (subject_index + recording_index + epoch_index) % 13
                        else 0.18
                    ),
                }
                epoch = {
                    "epoch_index": epoch_index,
                    "epoch_start": float(phase),
                    "epoch_end": float(phase + 30.0),
                    "reference_stage": stage,
                    "excluded": False,
                    "exclusion_reason": "",
                    "wearable_features": {
                        "hr_mean": float(base_hr),
                        "hr_median": float(base_hr),
                        "hr_std": 5.0,
                        "hr_min": float(max(40.0, base_hr - 12.0)),
                        "hr_max": float(base_hr + 12.0),
                        "hr_valid_count": 30,
                        "hrv_available": True,
                        "hrv_rmssd": float(max(4.0, hrv)),
                        "hrv_sdnn": float(max(6.0, hrv * 1.2)),
                        "accel_x_mean": float(accel_x),
                        "accel_y_mean": float(accel_y),
                        "accel_z_mean": float(accel_z),
                        "accel_x_std": 0.04,
                        "accel_y_std": 0.05,
                        "accel_z_std": 0.06,
                        "accel_magnitude_mean": float(accel_mag),
                        "accel_magnitude_std": 0.08,
                        "movement_intensity": float(max(0.03, accel_mag)),
                        "accel_valid_count": 30,
                        "eda_mean": float(eda),
                        "eda_std": 0.5,
                        "eda_min": float(max(0.0, eda - 1.5)),
                        "eda_max": float(eda + 1.5),
                        "eda_valid_count": 30,
                        "eda_tonic_mean": float(eda * 0.9),
                        "eda_phasic_mean": float(max(0.05, eda * 0.12)),
                    },
                    "modality_availability": modality_availability,
                    "missingness": missingness,
                }
                epochs.append(epoch)
            records.append({
                "subject_id": f"synthetic_s{subject_index + 1}",
                "recording_id": f"rec_{recording_index + 1}",
                "epochs": epochs,
            })
    return records


def _synthetic_demo_payload(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    subject_count: int = 6,
    recordings_per_subject: int = 2,
    epochs_per_record: int = 12,
) -> dict[str, object]:
    records = generate_synthetic_demo_records(
        seed=seed,
        subject_count=subject_count,
        recordings_per_subject=recordings_per_subject,
        epochs_per_record=epochs_per_record,
    )
    artifact = generate_wearable_dataset_artifact(
        records,
        output_dir=output_dir,
        dataset_id="synthetic_demo_dreamt",
        epoch_seconds=30,
        dataset_config={
            "sensor_set": "wearable",
            "taxonomy": "standard",
            "source_dataset": "dreamt",
            "synthetic_data": True,
            "synthetic_data_disclaimer": SYNTHETIC_DEMONSTRATION_LABEL,
            "synthetic_generation_method": (
                "deterministic synthetic demonstration fixture; "
                "not real DREAMT data"
            ),
            "random_seed": int(seed),
            "subject_count": int(subject_count),
            "recordings_per_subject": int(recordings_per_subject),
            "epochs_per_record": int(epochs_per_record),
        },
    )
    return {
        "dataset": "dreamt",
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "record_count": len(records),
        "records": records,
        "real_data_validation_pending": True,
        "preprocessing_version": PREPROCESSING_VERSION,
        "synthetic_data": True,
        "synthetic_data_disclaimer": SYNTHETIC_DEMONSTRATION_LABEL,
        "artifact": {
            "data_path": str(artifact["data_path"]),
            "metadata_path": str(artifact["metadata_path"]),
            "format": artifact["format"],
            "parquet_available": artifact["parquet_available"],
        },
      }


def _flatten_epoch_rows(record_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Flatten epoch dictionaries into a single row per epoch."""
    rows: list[dict[str, object]] = []
    for record in record_rows:
        subject_id = str(record.get("subject_id", "unknown_subject"))
        recording_id = str(record.get("recording_id", "unknown_record"))
        for epoch in record.get("epochs", []):
            row: dict[str, object] = {
                "subject_id": subject_id,
                "recording_id": recording_id,
                "epoch_index": int(epoch.get("epoch_index", 0)),
                "epoch_start": epoch.get("epoch_start"),
                "epoch_end": epoch.get("epoch_end"),
                "reference_stage": epoch.get("reference_stage", "UNKNOWN"),
                "excluded": bool(epoch.get("excluded", False)),
                "exclusion_reason": epoch.get("exclusion_reason") or "",
                "preprocessing_version": PREPROCESSING_VERSION,
            }

            wearable_features = epoch.get("wearable_features", {})
            for key, value in wearable_features.items():
                row[f"feature_{key}"] = value

            modality_availability = epoch.get("modality_availability", {})
            for key, value in modality_availability.items():
                row[f"modality_available_{key}"] = bool(value)

            missingness = epoch.get("missingness", {})
            for key, value in missingness.items():
                if isinstance(value, (int, float)):
                    row[f"missingness_{key}"] = float(value)
                else:
                    row[f"missingness_{key}"] = value

            rows.append(row)

    return sorted(
        rows,
        key=lambda item: (
            str(item["subject_id"]),
            str(item["recording_id"]),
            int(item["epoch_index"]),
        ),
    )


def _build_feature_schema() -> dict[str, dict[str, str]]:
    """Build a per-column definition map used in metadata export."""
    schema: dict[str, dict[str, str]] = {
        "subject_id": {"dtype": "string", "definition": "Stable subject identifier."},
        "recording_id": {"dtype": "string", "definition": "Stable recording identifier."},
        "epoch_index": {"dtype": "int64", "definition": "Zero-based epoch index."},
        "epoch_start": {"dtype": "float", "definition": "Epoch start time in seconds."},
        "epoch_end": {"dtype": "float", "definition": "Epoch end time in seconds."},
        "reference_stage": {"dtype": "string", "definition": "Reference sleep-stage target."},
        "excluded": {"dtype": "bool", "definition": "Whether the epoch was excluded."},
        "exclusion_reason": {"dtype": "string", "definition": "Reason the epoch was excluded."},
        "preprocessing_version": {
            "dtype": "string",
            "definition": "Version tag for the synthetic preprocessing logic.",
        },
    }
    for key, definition in FEATURE_DEFINITIONS.items():
        schema[f"feature_{key}"] = {
            "dtype": "float",
            "definition": definition,
        }
    schema["modality_available_hr"] = {
        "dtype": "bool",
        "definition": "Whether the HR modality was available in the epoch.",
    }
    schema["modality_available_ibi"] = {
        "dtype": "bool",
        "definition": "Whether IBI/RR inputs were available in the epoch.",
    }
    schema["modality_available_accel_x"] = {
        "dtype": "bool",
        "definition": "Whether x-axis accelerometer data were available in the epoch.",
    }
    schema["modality_available_accel_y"] = {
        "dtype": "bool",
        "definition": "Whether y-axis accelerometer data were available in the epoch.",
    }
    schema["modality_available_accel_z"] = {
        "dtype": "bool",
        "definition": "Whether z-axis accelerometer data were available in the epoch.",
    }
    schema["modality_available_eda"] = {
        "dtype": "bool",
        "definition": "Whether EDA data were available in the epoch.",
    }
    return schema


def generate_wearable_dataset_artifact(
    epoch_records: Sequence[dict[str, object]],
    output_dir: str | Path,
    dataset_id: str,
    epoch_seconds: int = 30,
    dataset_config: dict[str, object] | None = None,
) -> dict[str, str]:
    """Serialize a deterministic wearable-only epoch dataset and metadata artifact."""
    if not epoch_records:
        raise ValueError(
            "Cannot generate a wearable artifact from an empty record collection. "
            "Provide either a real DREAMT dataset or explicitly generate a "
            "synthetic demo record set."
        )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    rows = _flatten_epoch_rows(epoch_records)
    if not rows:
        raise ValueError(
            "No valid epochs were produced for the wearable artifact. "
            "This usually means the input data directory is empty or contains no usable recordings."
        )
    dataframe = pd.DataFrame(rows)

    identifiers = [
        "subject_id",
        "recording_id",
        "epoch_index",
        "epoch_start",
        "epoch_end",
        "reference_stage",
        "excluded",
        "exclusion_reason",
    ]
    feature_columns = sorted(
        [column for column in dataframe.columns if column.startswith("feature_")],
        key=lambda name: name,
    )
    modality_columns = sorted(
        [column for column in dataframe.columns if column.startswith("modality_available_")],
        key=lambda name: name,
    )
    missingness_columns = sorted(
        [column for column in dataframe.columns if column.startswith("missingness_")],
        key=lambda name: name,
    )
    dataframe = dataframe[identifiers + feature_columns + modality_columns + missingness_columns]

    data_path = output_path / f"{dataset_id}_wearable_epochs.parquet"
    metadata_path = output_path / f"{dataset_id}_wearable_epochs_metadata.json"

    parquet_supported = True
    try:
        dataframe.to_parquet(data_path, index=False)
        data_format = "parquet"
    except ImportError:
        parquet_supported = False
        csv_path = output_path / f"{dataset_id}_wearable_epochs.csv"
        dataframe.to_csv(csv_path, index=False)
        data_path = csv_path
        data_format = "csv"

    excluded_reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("exclusion_reason") or "")
        if row.get("excluded") and reason:
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1

    modality_counts: dict[str, int] = {}
    modality_columns = {
        name.removeprefix("modality_available_")
        for name in dataframe.columns
        if name.startswith("modality_available_")
    }
    for key in sorted(modality_columns):
        column_name = f"modality_available_{key}"
        modality_counts[key] = int(
            dataframe[column_name].sum() if column_name in dataframe.columns else 0
        )

    config = dict(dataset_config or {})
    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_dataset_id": dataset_id,
        "generation_timestamp_utc": datetime.now(UTC).isoformat().replace(
            "+00:00", "Z"
        ),
        "config": {
            **config,
            "epoch_duration_seconds": epoch_seconds,
            "sensor_set": config.get("sensor_set", "wearable"),
            "taxonomy": config.get("taxonomy", "standard"),
            "include_psg_reference": False,
        },
        "epoch_duration_seconds": epoch_seconds,
        "feature_schema": _build_feature_schema(),
        "counts": {
            "subjects": len({str(row["subject_id"]) for row in rows}),
            "recordings": len({str(row["recording_id"]) for row in rows}),
            "epochs": len(rows),
            "excluded_epochs": int(sum(1 for row in rows if row.get("excluded"))),
        },
        "excluded_epoch_counts": excluded_reasons,
        "modality_availability": modality_counts,
        "artifact_format": data_format,
        "artifact_path": str(data_path),
        "real_data_validation_pending": True,
        "no_eeg_leakage_guarantee": True,
        "synthetic_data": bool(config.get("synthetic_data", False)),
        "synthetic_data_disclaimer": config.get(
            "synthetic_data_disclaimer",
            SYNTHETIC_DEMONSTRATION_LABEL,
        ),
        "synthetic_generation_method": config.get(
            "synthetic_generation_method",
            "deterministic synthetic demonstration fixture; not real DREAMT data",
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "data_path": data_path,
        "metadata_path": metadata_path,
        "format": data_format,
        "parquet_available": parquet_supported,
    }


def preprocess_dataset(
    dataset: str,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    synthetic_demo: bool = False,
) -> dict[str, object]:
    """Construct a DREAMT artifact or an explicit synthetic demo only when requested.

    Synthetic observations are never substituted automatically for missing real
    DREAMT data. The caller must opt in via synthetic_demo=True.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported values: {', '.join(SUPPORTED_DATASETS)}."
        )

    if dataset != "dreamt":
        raise NotImplementedError(
            f"Preprocessing for dataset '{dataset}' is not implemented yet; "
            "this milestone supports DREAMT synthetic validation only."
        )

    source_dir = (
        Path(data_dir).expanduser().resolve()
        if data_dir is not None
        else Path("data").resolve()
    )
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source_dir / "processed"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for subject_dir in sorted(source_dir.rglob("*")):
        if not subject_dir.is_dir():
            continue
        for record_dir in sorted(subject_dir.iterdir()):
            if not record_dir.is_dir():
                continue
            record = {
                "subject_id": subject_dir.name,
                "recording_id": record_dir.name,
                "modality_data": {},
                "stage_labels": ("WAKE", "N1"),
                "timestamps": (0.0, 30.0),
            }
            for signal_file in sorted(record_dir.iterdir()):
                file_suffix = signal_file.suffix.lower()
                if (
                    not signal_file.is_file()
                    or file_suffix not in {".csv", ".tsv", ".txt"}
                ):
                    continue
                rows = signal_file.read_text(encoding="utf-8").strip().splitlines()
                if not rows:
                    continue
                header = rows[0].split(",")
                values = rows[1:]
                for row in values:
                    if not row.strip():
                        continue
                    parts = row.split(",")
                    if len(parts) < 2:
                        continue
                    timestamp = parts[0]
                    signal = header[1] if len(header) > 1 else signal_file.stem
                    if signal.startswith("stage"):
                        continue
                    record["modality_data"].setdefault(signal, [])
                    record["modality_data"][signal].append((timestamp, float(parts[1])))
            if record["modality_data"]:
                records.append({
                    "subject_id": record["subject_id"],
                    "recording_id": record["recording_id"],
                    "epochs": build_epoch_dataset(record, epoch_seconds=30),
                })

    if not records:
        if not synthetic_demo:
            raise ValueError(
                f"No valid recordings were discovered under '{source_dir}'. "
                "This repository does not contain a real DREAMT dataset in this location. "
                "Provide an authorized DREAMT dataset or rerun with --synthetic-demo to generate "
                "an explicit synthetic demonstration fixture."
            )
        demo_output_path = output_path if output_path is not None else source_dir / "processed"
        payload = _synthetic_demo_payload(
            data_dir=source_dir,
            output_dir=demo_output_path,
            seed=42,
            subject_count=6,
            recordings_per_subject=2,
            epochs_per_record=12,
        )
        return {
            **payload,
            "synthetic_data": True,
            "synthetic_data_disclaimer": SYNTHETIC_DEMONSTRATION_LABEL,
            "config": {
                "epoch_seconds": 30,
                "sensor_set": "wearable",
                "taxonomy": "standard",
                "include_psg_reference": False,
            },
        }

    artifact = generate_wearable_dataset_artifact(
        records,
        output_dir=output_path,
        dataset_id=dataset,
        epoch_seconds=30,
        dataset_config={
            "sensor_set": "wearable",
            "taxonomy": "standard",
            "source_dataset": dataset,
        },
    )

    payload = {
        "dataset": dataset,
        "data_dir": str(source_dir),
        "output_dir": str(output_path),
        "record_count": len(records),
        "records": records,
        "real_data_validation_pending": True,
        "preprocessing_version": PREPROCESSING_VERSION,
        "artifact": {
            "data_path": str(artifact["data_path"]),
            "metadata_path": str(artifact["metadata_path"]),
            "format": artifact["format"],
            "parquet_available": artifact["parquet_available"],
        },
        "config": {
            "epoch_seconds": 30,
            "sensor_set": "wearable",
            "taxonomy": "standard",
            "include_psg_reference": False,
        },
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Construct the preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Construct synthetic epoch-level wearable features for DREAMT-style fixtures. "
            "Real DREAMT validation remains pending until a real dataset is available; "
            "synthetic generation is opt-in and requires --synthetic-demo."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Dataset to preprocess. The current implementation supports DREAMT validation only.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing recording folders to preprocess.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where epoch-level outputs should be written.",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help=(
            "Generate a deterministic synthetic DREAMT-style demo when there is no real "
            "DREAMT data and the caller explicitly opts in."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a plain-text summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the preprocessing CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = preprocess_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            synthetic_demo=args.synthetic_demo,
        )
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        parser.exit(status=2, message=f"Preprocessing error: {exc}\n")

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"dataset={summary['dataset']}")
        print(f"record_count={summary['record_count']}")
        print(f"real_data_validation_pending={summary['real_data_validation_pending']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SYNTHETIC_DEMONSTRATION_LABEL",
    "generate_synthetic_demo_records",
    "main",
    "preprocess_dataset",
    "_synthetic_demo_payload",
]
