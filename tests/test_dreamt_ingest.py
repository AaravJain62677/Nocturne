from __future__ import annotations

import pytest
from src.ingest import load_dataset
from src.ingest.load_dreamt import (
    discover_dreamt_records,
    load_dreamt_record,
    validate_dreamt_layout,
)


def _write_text(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_dreamt_record(tmp_path):
    record_dir = tmp_path / "subject_01" / "record_001"
    _write_text(
        record_dir / "accel_x.csv",
        "timestamp,accel_x\n0,0.10\n1,0.20\n",
    )
    _write_text(
        record_dir / "accel_z.csv",
        "timestamp,accel_z\n0,0.30\n1,0.40\n",
    )
    _write_text(
        record_dir / "hr.csv",
        "timestamp,hr\n0,60\n1,62\n",
    )
    _write_text(
        record_dir / "eeg_fpz_cz.csv",
        "timestamp,eeg_fpz_cz\n0,1.0\n1,1.1\n",
    )
    _write_text(
        record_dir / "stage_labels.csv",
        "timestamp,stage\n0,W\n1,N1\n2,REM\n",
    )
    return record_dir


def test_dreamt_discovery_finds_subject_records(tmp_path) -> None:
    _make_dreamt_record(tmp_path)

    records = discover_dreamt_records(tmp_path)

    assert len(records) == 1
    assert records[0].name == "record_001"


def test_dreamt_layout_validation_requires_real_record_dirs(tmp_path) -> None:
    _make_dreamt_record(tmp_path)

    validated = validate_dreamt_layout(tmp_path)

    assert len(validated) == 1
    assert validated[0].name == "record_001"


def test_dreamt_record_parses_subject_and_timestamps(tmp_path) -> None:
    record_dir = _make_dreamt_record(tmp_path)

    record = load_dreamt_record(record_dir)

    assert record.subject_id == "subject_01"
    assert record.recording_id == "record_001"
    assert record.start_timestamp == "0"
    assert record.end_timestamp == "2"
    assert "accel_x" in record.available_modalities
    assert "hr" in record.available_modalities


def test_dreamt_missing_wearable_modalities_are_explicit(tmp_path) -> None:
    record_dir = tmp_path / "subject_01" / "record_001"
    _write_text(
        record_dir / "hr.csv",
        "timestamp,hr\n0,60\n1,62\n",
    )
    _write_text(
        record_dir / "eeg_fpz_cz.csv",
        "timestamp,eeg_fpz_cz\n0,1.0\n1,1.1\n",
    )
    _write_text(
        record_dir / "stage_labels.csv",
        "timestamp,stage\n0,W\n1,N1\n",
    )

    with pytest.raises(ValueError, match="wearable|missing"):
        load_dreamt_record(record_dir, sensor_set="wearable")


def test_dreamt_malformed_stage_annotations_raise_value_error(tmp_path) -> None:
    record_dir = tmp_path / "subject_01" / "record_001"
    _write_text(
        record_dir / "accel_x.csv",
        "timestamp,accel_x\n0,0.10\n1,0.20\n",
    )
    _write_text(
        record_dir / "eeg_fpz_cz.csv",
        "timestamp,eeg_fpz_cz\n0,1.0\n1,1.1\n",
    )
    _write_text(
        record_dir / "stage_labels.csv",
        "timestamp,label\n0,banana\n1,UNKNOWN\n",
    )

    with pytest.raises(ValueError, match="stage|annotation|label"):
        load_dreamt_record(record_dir)


def test_dreamt_unknown_stage_labels_are_preserved(tmp_path) -> None:
    record_dir = tmp_path / "subject_01" / "record_001"
    _write_text(
        record_dir / "accel_x.csv",
        "timestamp,accel_x\n0,0.10\n1,0.20\n",
    )
    _write_text(
        record_dir / "eeg_fpz_cz.csv",
        "timestamp,eeg_fpz_cz\n0,1.0\n1,1.1\n",
    )
    _write_text(
        record_dir / "stage_labels.csv",
        "timestamp,stage\n0,UNKNOWN\n1,UNSCORED\n2,REM\n",
    )

    record = load_dreamt_record(record_dir)

    assert "UNKNOWN" in record.stage_labels
    assert "UNSCORED" in record.stage_labels


def test_load_dataset_accepts_dreamt_fixture_directory(tmp_path) -> None:
    record_dir = _make_dreamt_record(tmp_path)

    loaded = load_dataset("dreamt", data_dir=tmp_path)

    assert len(loaded) == 1
    assert loaded[0].recording_id == record_dir.name
