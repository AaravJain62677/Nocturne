from __future__ import annotations

import pytest
from src.config import canonical_sleep_stage
from src.preprocessing import (
    align_recording_to_epochs,
    build_wearable_feature_table,
    canonicalize_stage_labels,
    compute_hrv_features,
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
