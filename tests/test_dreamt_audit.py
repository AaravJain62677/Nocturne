from __future__ import annotations

from src.inspect_data import audit_dreamt_dataset


def _write_text(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_dreamt_record(tmp_path):
    record_dir = tmp_path / "subject_01" / "record_001"
    _write_text(record_dir / "accel_x.csv", "timestamp,accel_x\n0,0.10\n1,0.20\n")
    _write_text(record_dir / "accel_y.csv", "timestamp,accel_y\n0,0.20\n1,0.25\n")
    _write_text(record_dir / "accel_z.csv", "timestamp,accel_z\n0,0.30\n1,0.40\n")
    _write_text(record_dir / "bvp.csv", "timestamp,bvp\n0,1.10\n1,1.20\n")
    _write_text(record_dir / "eda.csv", "timestamp,eda\n0,1.50\n1,1.60\n")
    _write_text(record_dir / "temp.csv", "timestamp,temp\n0,33.0\n1,33.1\n")
    _write_text(record_dir / "hr.csv", "timestamp,hr\n0,60\n1,62\n")
    _write_text(record_dir / "ibi.csv", "timestamp,ibi\n0,1000\n1,980\n")
    _write_text(record_dir / "eeg_fpz_cz.csv", "timestamp,eeg_fpz_cz\n0,1.0\n1,1.1\n")
    _write_text(record_dir / "eeg_pz_oz.csv", "timestamp,eeg_pz_oz\n0,0.9\n1,1.0\n")
    _write_text(record_dir / "emg.csv", "timestamp,emg\n0,2.0\n1,2.2\n")
    _write_text(record_dir / "eog.csv", "timestamp,eog\n0,0.5\n1,0.6\n")
    _write_text(record_dir / "stage_labels.csv", "timestamp,stage\n0,W\n1,N1\n2,REM\n")
    return record_dir


def test_dreamt_audit_reports_subjects_recordings_and_modalities(tmp_path) -> None:
    _make_dreamt_record(tmp_path)

    summary = audit_dreamt_dataset(tmp_path)

    assert summary["subject_count"] == 1
    assert summary["record_count"] == 1
    assert summary["file_count"] >= 5
    assert summary["modalities"]
    assert "accel_x" in summary["modalities"]
    assert "hr" in summary["modalities"]
    assert summary["label_files"]
    assert summary["missing_files"] == []
    assert summary["corrupt_files"] == []


def test_dreamt_audit_flags_missing_and_corrupt_files(tmp_path) -> None:
    record_dir = tmp_path / "subject_01" / "record_001"
    _write_text(record_dir / "accel_x.csv", "timestamp,accel_x\n0,0.10\n")
    _write_text(record_dir / "accel_y.csv", "timestamp,accel_y\n0,0.20\n")
    _write_text(record_dir / "accel_z.csv", "timestamp,accel_z\n0,0.30\n")
    _write_text(record_dir / "bvp.csv", "timestamp,bvp\n0,1.10\n")
    _write_text(record_dir / "temp.csv", "timestamp,temp\n0,33.0\n")
    _write_text(record_dir / "hr.csv", "timestamp,hr\n0,60\n")
    _write_text(record_dir / "ibi.csv", "timestamp,ibi\n0,1000\n")
    _write_text(record_dir / "stage_labels.csv", "timestamp,stage\n0,W\n")
    _write_text(record_dir / "corrupt.csv", "")

    summary = audit_dreamt_dataset(tmp_path)

    assert summary["record_count"] == 1
    assert summary["missing_files"]
    assert summary["corrupt_files"]
