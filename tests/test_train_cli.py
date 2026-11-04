import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from src.models import RandomForestSleepStageModel
from src.pipeline import PipelineConfig, run_pipeline
from src.preprocess import (
    generate_synthetic_demo_records,
    generate_wearable_dataset_artifact,
    preprocess_dataset,
)
from src.train import main


def _build_multi_subject_fixture() -> list[dict]:
    records: list[dict] = []
    subject_order = ["s1", "s2", "s3", "s4", "s5", "s6"]
    for subject_index, subject_id in enumerate(subject_order):
        base = 60.0 + subject_index
        for recording_index in range(2):
            epochs = []
            for epoch_index in range(3):
                phase = 30 * epoch_index
                epochs.append({
                    "epoch_index": epoch_index,
                    "epoch_start": float(phase),
                    "epoch_end": float(phase + 30),
                    "reference_stage": "WAKE" if subject_id in {"s1", "s2"} else "N2",
                    "excluded": False,
                    "exclusion_reason": "",
                    "wearable_features": {
                        "hr_mean": base + epoch_index,
                        "hrv_rmssd": 20.0 + epoch_index,
                        "accel_magnitude_mean": 0.15 + epoch_index * 0.04,
                        "eda_mean": 1.2 + epoch_index * 0.15,
                        "eda_std": 0.2,
                        "movement_intensity": 0.2 + epoch_index * 0.03,
                    },
                    "modality_availability": {
                        "hr": True,
                        "ibi": True,
                        "accel_x": True,
                        "accel_y": True,
                        "accel_z": True,
                        "eda": True,
                    },
                    "missingness": {"hr": 0.0, "accel": 0.0, "eda": 0.0},
                })
            records.append({
                "subject_id": subject_id,
                "recording_id": f"{subject_id}_rec_{recording_index}",
                "epochs": epochs,
            })
    return records


def test_cli_help_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--help"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_validation_mode_returns_zero() -> None:
    exit_code = main([
        "--dataset",
        "dreamt",
        "--model",
        "baseline",
        "--mode",
        "validate",
        "--taxonomy",
        "standard",
    ])
    assert exit_code == 0


def test_unimplemented_training_mode_exits_cleanly() -> None:
    with pytest.raises(SystemExit, match="unavailable|not implemented|unsupported"):
        main([
            "--dataset",
            "dreamt",
            "--model",
            "baseline",
            "--mode",
            "train",
        ])


def test_synthetic_end_to_end_pipeline_runs_without_subject_leakage(
    tmp_path: Path,
) -> None:
    records = _build_multi_subject_fixture()
    artifact_dir = tmp_path / "artifact"
    artifact = generate_wearable_dataset_artifact(
        records,
        output_dir=artifact_dir,
        dataset_id="synthetic_dreamt",
        epoch_seconds=30,
        dataset_config={"sensor_set": "wearable", "taxonomy": "standard"},
    )

    output_dir = tmp_path / "pipeline_outputs"
    summary = run_pipeline(PipelineConfig(
        dataset="dreamt",
        model="random_forest",
        taxonomy="standard",
        sensor_set="wearable",
        seed=7,
        artifact=artifact["data_path"],
        output_dir=output_dir,
        feature_group="all_wearable",
        split="test",
    ))

    assert summary["checkpoint_path"].endswith("random_forest_checkpoint.json")
    assert output_dir.joinpath("pipeline_summary.json").exists()
    assert output_dir.joinpath("metrics.json").exists()

    split_map = summary["split_map"]
    assert set(split_map["train"]) & set(split_map["val"]) == set()
    assert set(split_map["train"]) & set(split_map["test"]) == set()
    assert set(split_map["val"]) & set(split_map["test"]) == set()

    feature_columns = summary["feature_columns"]
    assert all("eeg" not in str(col).lower() for col in feature_columns)
    assert all("psg" not in str(col).lower() for col in feature_columns)

    checkpoint = RandomForestSleepStageModel.load_checkpoint(summary["checkpoint_path"])
    assert checkpoint.model_name == "random_forest"
    assert checkpoint.feature_columns == feature_columns

    loaded = pd.read_parquet(artifact["data_path"])
    assert len(summary["metrics"]["labels"]) >= 1
    selected_rows = loaded[loaded["subject_id"].isin(summary["selected_subjects"])]
    assert summary["prediction_count"] == len(selected_rows)


def test_preprocess_dataset_rejects_empty_data_dir(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="real DREAMT|--synthetic-demo|No valid recordings"):
        preprocess_dataset("dreamt", data_dir=empty_dir, output_dir=tmp_path / "empty_output")


def test_pipeline_fails_without_synthetic_demo_when_dreamt_missing(tmp_path: Path) -> None:
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()

    with pytest.raises(
        (FileNotFoundError, ValueError),
        match="--synthetic-demo|real DREAMT|No valid",
    ):
        run_pipeline(PipelineConfig(
            dataset="dreamt",
            model="random_forest",
            taxonomy="standard",
            sensor_set="wearable",
            seed=42,
            data_dir=empty_data_dir,
            output_dir=tmp_path / "expected_failure",
            split_output_dir=tmp_path / "expected_failure" / "splits",
            feature_group="all_wearable",
            split="test",
            synthetic_demo=False,
        ))


def test_generate_synthetic_demo_records_are_seeded_and_label_covered() -> None:
    records_a = generate_synthetic_demo_records(seed=42, subject_count=6)
    records_b = generate_synthetic_demo_records(seed=42, subject_count=6)
    records_c = generate_synthetic_demo_records(seed=99, subject_count=6)

    assert len(records_a) >= 6
    labels = {epoch["reference_stage"] for record in records_a for epoch in record["epochs"]}
    assert labels == {"WAKE", "N1", "N2", "N3", "REM"}
    assert records_a == records_b
    assert records_a != records_c


def test_pipeline_runs_on_seeded_synthetic_demo_data(tmp_path: Path) -> None:
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()
    output_dir = tmp_path / "demo_pipeline"

    summary = run_pipeline(PipelineConfig(
        dataset="dreamt",
        model="random_forest",
        taxonomy="standard",
        sensor_set="wearable",
        seed=42,
        data_dir=empty_data_dir,
        output_dir=output_dir,
        split_output_dir=output_dir / "splits",
        feature_group="all_wearable",
        split="test",
        synthetic_demo=True,
    ))

    assert summary["selected_split"] == "test"
    assert summary["metrics"]["accuracy"] >= 0.0
    assert summary["metrics"]["macro_f1"] >= 0.0
    assert output_dir.joinpath("metrics.json").exists()
    assert output_dir.joinpath("pipeline_summary.json").exists()
    assert "SYNTHETIC DEMONSTRATION" in json.dumps(
        summary,
        sort_keys=True,
    ).upper()
    assert summary["synthetic_data_disclaimer"] == (
        "SYNTHETIC DEMONSTRATION — NOT REAL DREAMT RESULTS"
    )
