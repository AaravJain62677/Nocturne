from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from src.config import canonical_sleep_stage
from src.preprocess import generate_wearable_dataset_artifact
from src.preprocessing import (
    align_recording_to_epochs,
    build_epoch_dataset,
    build_wearable_feature_table,
    canonicalize_stage_labels,
    compute_hrv_features,
    extract_wearable_epoch_features,
)


@pytest.fixture
def sample_record() -> dict:
    return {
        "subject_id": "subject_01",
        "recording_id": "record_001",
        "modality_data": {
            "accel_x": (("0", 0.1), ("30", 0.2), ("60", 0.3)),
            "accel_y": (("0", 0.2), ("30", 0.1), ("60", 0.25)),
            "accel_z": (("0", 0.3), ("30", 0.2), ("60", 0.4)),
            "hr": (("0", 60.0), ("30", 64.0), ("60", 62.0)),
            "eeg_fpz_cz": (("0", 0.1), ("30", 0.2), ("60", 0.3)),
        },
        "stage_labels": ("WAKE", "N1", "REM"),
        "timestamps": ("0", "30", "60"),
    }


def test_epoch_alignment_preserves_timestamps_and_subject_ids(sample_record: dict) -> None:
    aligned = align_recording_to_epochs(sample_record, epoch_seconds=30)

    assert aligned["subject_id"] == "subject_01"
    assert aligned["recording_id"] == "record_001"
    assert aligned["epoch_count"] == 3
    assert aligned["epoch_starts"] == ("0", "30", "60")
    assert aligned["labels"] == ("WAKE", "N1", "REM")


def test_unknown_labels_are_excluded_but_preserved(sample_record: dict) -> None:
    record = {
        **sample_record,
        "stage_labels": ("WAKE", "UNKNOWN", "REM", "UNSCORED"),
        "timestamps": ("0", "30", "60", "90"),
    }

    aligned = align_recording_to_epochs(record, epoch_seconds=30)

    assert aligned["labels"] == ("WAKE", "UNKNOWN", "REM", "UNSCORED")
    assert aligned["excluded_mask"] == (False, True, False, True)


def test_build_wearable_feature_table_excludes_psg_columns(sample_record: dict) -> None:
    features = build_wearable_feature_table(sample_record)

    assert "hr_mean" in features.columns
    assert "accel_magnitude_mean" in features.columns
    assert "eeg_fpz_cz" not in features.columns


def test_hrv_is_flagged_when_beat_to_beat_is_insufficient() -> None:
    values = {"hr": (("0", 60.0), ("30", 60.5), ("60", 61.0))}
    result = compute_hrv_features(values)

    assert result["hrv_available"] is False
    assert result["rmssd"] is None
    assert result["sdnn"] is None


def test_canonical_stage_labels_retain_unknown_and_movement_values() -> None:
    labels = ["W", "N2", "movement", "unknown", "rem"]
    mapped = canonicalize_stage_labels(labels, taxonomy="standard")

    assert mapped == ["WAKE", "N2", "MOVEMENT", "UNKNOWN", "REM"]
    assert canonical_sleep_stage("Unknown") == "UNKNOWN"


def test_epoch_alignment_handles_offsets_and_ambiguous_labels() -> None:
    record = {
        "subject_id": "subject_01",
        "recording_id": "record_001",
        "timestamps": ("10", "40", "50", "90"),
        "stage_labels": ("WAKE", "N1", "REM", "N2"),
        "modality_data": {},
    }

    aligned = align_recording_to_epochs(record, epoch_seconds=30, time_origin=10)

    assert aligned["epoch_starts"] == (10, 40, 70)
    assert aligned["labels"][0] == "WAKE"
    assert aligned["excluded_mask"][0] is False
    assert aligned["epochs"][0]["epoch_start"] == 10


def test_extract_wearable_features_handles_hrv_and_eda() -> None:
    record = {
        "subject_id": "subject_01",
        "recording_id": "record_001",
        "modality_data": {
            "hr": (("0", 60.0), ("10", 62.0), ("20", 58.0), ("30", 61.0)),
            "ibi": (("0", 1000.0), ("10", 990.0), ("20", 1005.0), ("30", 1010.0)),
            "eda": (("0", 1.0), ("10", 1.2), ("20", 1.4), ("30", 1.3)),
            "accel_x": (("0", 0.1), ("10", 0.2), ("20", 0.15), ("30", 0.1)),
            "accel_y": (("0", 0.2), ("10", 0.1), ("20", 0.1), ("30", 0.2)),
            "accel_z": (("0", 0.3), ("10", 0.2), ("20", 0.2), ("30", 0.3)),
        },
        "timestamps": ("0", "10", "20", "30"),
    }

    features = extract_wearable_epoch_features(record, epoch_index=0, epoch_seconds=30)

    assert features["hr_mean"] > 0
    assert features["hrv_rmssd"] is not None
    assert features["eda_mean"] > 0
    assert features["accel_magnitude_mean"] > 0
    assert features["modality_availability"]["hr"] is True


def test_missing_and_duplicate_timestamps_are_handled_explicitly() -> None:
    record = {
        "subject_id": "subject_01",
        "recording_id": "record_001",
        "timestamps": ("0", "0", "10", "30"),
        "stage_labels": ("WAKE", "WAKE", "N1", "REM"),
        "modality_data": {
            "hr": (("0", 60.0), ("0", 60.0), ("10", 62.0), ("30", 64.0)),
        },
    }

    aligned = align_recording_to_epochs(record, epoch_seconds=30)
    assert aligned["duplicate_timestamps"] == 1
    assert aligned["epochs"][0]["reference_stage"] == "WAKE"


def test_build_epoch_dataset_keeps_label_and_predictor_separation() -> None:
    record = {
        "subject_id": "subject_01",
        "recording_id": "record_001",
        "timestamps": ("0", "15", "30", "45"),
        "stage_labels": ("WAKE", "N1", "HEAVY", "REM"),
        "modality_data": {
            "hr": (("0", 60.0), ("15", 62.0), ("30", 64.0), ("45", 63.0)),
            "accel_x": (("0", 0.1), ("15", 0.2), ("30", 0.1), ("45", 0.2)),
            "accel_y": (("0", 0.1), ("15", 0.1), ("30", 0.2), ("45", 0.1)),
            "accel_z": (("0", 0.2), ("15", 0.1), ("30", 0.1), ("45", 0.2)),
        },
    }

    dataset = build_epoch_dataset(record, epoch_seconds=30)

    assert len(dataset) == 2
    assert "reference_stage" in dataset[0]
    assert "hr_mean" in dataset[0]["wearable_features"]
    assert "eeg_fpz_cz" not in dataset[0]["wearable_features"]


def test_preprocessing_output_metadata_includes_version_and_config() -> None:
    record = {
        "subject_id": "subject_01",
        "recording_id": "record_001",
        "modality_data": {
            "hr": (("0", 60.0), ("15", 62.0), ("30", 64.0)),
            "accel_x": (("0", 0.1), ("15", 0.2), ("30", 0.1)),
            "accel_y": (("0", 0.1), ("15", 0.1), ("30", 0.2)),
            "accel_z": (("0", 0.2), ("15", 0.2), ("30", 0.1)),
        },
        "stage_labels": ("WAKE", "N1"),
        "timestamps": ("0", "30"),
    }

    dataset = build_epoch_dataset(record, epoch_seconds=30)

    assert dataset[0]["preprocessing_version"]
    assert dataset[0]["reference_stage"] in {"WAKE", "N1"}


def test_generate_wearable_dataset_artifact_serializes_and_preserves_schema(tmp_path: Path) -> None:
    record = {
        "subject_id": "subject_01",
        "recording_id": "record_001",
        "modality_data": {
            "hr": (("0", 60.0), ("15", 62.0), ("30", 64.0)),
            "accel_x": (("0", 0.1), ("15", 0.2), ("30", 0.1)),
            "accel_y": (("0", 0.1), ("15", 0.1), ("30", 0.2)),
            "accel_z": (("0", 0.2), ("15", 0.2), ("30", 0.1)),
            "eeg_fpz_cz": (("0", 0.1), ("15", 0.2), ("30", 0.1)),
        },
        "stage_labels": ("WAKE", "N1"),
        "timestamps": ("0", "30"),
    }

    artifact = generate_wearable_dataset_artifact(
        [{
            "subject_id": record["subject_id"],
            "recording_id": record["recording_id"],
            "epochs": build_epoch_dataset(record, epoch_seconds=30),
        }],
        output_dir=tmp_path,
        dataset_id="synthetic_dreamt",
        epoch_seconds=30,
        dataset_config={"sensor_set": "wearable", "taxonomy": "standard"},
    )

    assert artifact["data_path"].suffix == ".parquet"
    dataframe = pd.read_parquet(artifact["data_path"])
    assert "subject_id" in dataframe.columns
    assert "recording_id" in dataframe.columns
    assert "reference_stage" in dataframe.columns
    assert "feature_hr_mean" in dataframe.columns
    assert "feature_eeg_mean" not in dataframe.columns
    assert "eeg_fpz_cz" not in dataframe.columns


def test_generate_wearable_dataset_artifact_metadata_tracks_counts_and_exclusions(
    tmp_path: Path,
) -> None:
    record = {
        "subject_id": "subject_01",
        "recording_id": "record_001",
        "modality_data": {
            "hr": (("0", 60.0), ("15", 62.0), ("30", 64.0)),
            "accel_x": (("0", 0.1), ("15", 0.2), ("30", 0.1)),
            "accel_y": (("0", 0.1), ("15", 0.1), ("30", 0.2)),
            "accel_z": (("0", 0.2), ("15", 0.2), ("30", 0.1)),
        },
        "stage_labels": ("WAKE", "UNKNOWN"),
        "timestamps": ("0", "30"),
    }

    artifact = generate_wearable_dataset_artifact(
        [{
            "subject_id": record["subject_id"],
            "recording_id": record["recording_id"],
            "epochs": build_epoch_dataset(record, epoch_seconds=30),
        }],
        output_dir=tmp_path,
        dataset_id="synthetic_dreamt",
        epoch_seconds=30,
        dataset_config={"sensor_set": "wearable", "taxonomy": "standard"},
    )

    metadata = json.loads(Path(artifact["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["counts"]["subjects"] == 1
    assert metadata["counts"]["recordings"] == 1
    assert metadata["counts"]["epochs"] == 2
    assert metadata["counts"]["excluded_epochs"] >= 1
    assert metadata["excluded_epoch_counts"]
    assert metadata["preprocessing_version"]


def test_generate_wearable_dataset_artifact_is_deterministic_for_same_inputs(
    tmp_path: Path,
) -> None:
    record = {
        "subject_id": "subject_01",
        "recording_id": "record_001",
        "modality_data": {
            "hr": (("0", 60.0), ("15", 62.0), ("30", 64.0)),
            "accel_x": (("0", 0.1), ("15", 0.2), ("30", 0.1)),
            "accel_y": (("0", 0.1), ("15", 0.1), ("30", 0.2)),
            "accel_z": (("0", 0.2), ("15", 0.2), ("30", 0.1)),
        },
        "stage_labels": ("WAKE", "N1"),
        "timestamps": ("0", "30"),
    }

    input_records = [{
        "subject_id": record["subject_id"],
        "recording_id": record["recording_id"],
        "epochs": build_epoch_dataset(record, epoch_seconds=30),
    }]

    first = generate_wearable_dataset_artifact(
        input_records,
        output_dir=tmp_path / "first",
        dataset_id="synthetic_dreamt",
        epoch_seconds=30,
        dataset_config={"sensor_set": "wearable", "taxonomy": "standard"},
    )
    second = generate_wearable_dataset_artifact(
        input_records,
        output_dir=tmp_path / "second",
        dataset_id="synthetic_dreamt",
        epoch_seconds=30,
        dataset_config={"sensor_set": "wearable", "taxonomy": "standard"},
    )

    first_df = pd.read_parquet(first["data_path"])
    second_df = pd.read_parquet(second["data_path"])
    pd.testing.assert_frame_equal(first_df, second_df)


def test_preprocessing_pipeline_uses_synthetic_record_fixture(tmp_path) -> None:
    record_dir = tmp_path / "subject_01" / "record_001"
    record_dir.mkdir(parents=True)
    (record_dir / "hr.csv").write_text(
        "timestamp,hr\n0,60\n15,62\n30,64\n45,63\n",
        encoding="utf-8",
    )
    (record_dir / "accel_x.csv").write_text(
        "timestamp,accel_x\n0,0.1\n15,0.2\n30,0.1\n45,0.2\n",
        encoding="utf-8",
    )
    (record_dir / "accel_y.csv").write_text(
        "timestamp,accel_y\n0,0.2\n15,0.3\n30,0.2\n45,0.2\n",
        encoding="utf-8",
    )
    (record_dir / "accel_z.csv").write_text(
        "timestamp,accel_z\n0,0.1\n15,0.2\n30,0.2\n45,0.1\n",
        encoding="utf-8",
    )
    (record_dir / "stage_labels.csv").write_text(
        "timestamp,stage\n0,W\n30,N2\n",
        encoding="utf-8",
    )

    features = build_epoch_dataset(
        {
            "subject_id": "subject_01",
            "recording_id": "record_001",
            "modality_data": {
                "hr": (("0", 60.0), ("15", 62.0), ("30", 64.0), ("45", 63.0)),
                "accel_x": (("0", 0.1), ("15", 0.2), ("30", 0.1), ("45", 0.2)),
                "accel_y": (("0", 0.2), ("15", 0.3), ("30", 0.2), ("45", 0.2)),
                "accel_z": (("0", 0.1), ("15", 0.2), ("30", 0.2), ("45", 0.1)),
            },
            "stage_labels": ("WAKE", "N2"),
            "timestamps": ("0", "30"),
        },
        epoch_seconds=30,
    )

    assert len(features) == 2
    assert features[0]["reference_stage"] == "WAKE"
    assert "hr_mean" in features[0]["wearable_features"]
