from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.eval import (
    bootstrap_subject_metrics,
    compute_subject_level_metrics,
    evaluate_predictions,
    get_feature_group_sequence,
    run_feature_group_ablation,
)
from src.models import build_subject_splits, train_model


def _build_eval_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in ["s1", "s2", "s3", "s4", "s5", "s6"]:
        for epoch_index in range(4):
            stage = "WAKE" if subject in {"s1", "s2"} else "N2"
            rows.append(
                {
                    "subject_id": subject,
                    "recording_id": f"rec-{subject}",
                    "epoch_index": epoch_index,
                    "epoch_start": float(epoch_index * 30),
                    "epoch_end": float((epoch_index + 1) * 30),
                    "reference_stage": stage,
                    "excluded": False,
                    "feature_hr_mean": 60.0 + epoch_index,
                    "feature_hrv_rmssd": 40.0 + epoch_index,
                    "feature_accel_magnitude_mean": 0.2 + epoch_index * 0.05,
                    "feature_eda_mean": 1.5 + epoch_index * 0.1,
                    "feature_eda_std": 0.3,
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_predictions_reports_known_metrics() -> None:
    metrics = evaluate_predictions(
        ["WAKE", "N1", "N2", "REM"],
        ["WAKE", "N1", "N2", "REM"],
        taxonomy="standard",
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["cohens_kappa"] == pytest.approx(1.0)


def test_bootstrap_subject_metrics_are_reproducible() -> None:
    subjects = ["s1", "s2", "s3", "s4"]
    scores = {
        "s1": [0.8, 0.8],
        "s2": [0.6, 0.6],
        "s3": [0.5, 0.5],
        "s4": [0.9, 0.9],
    }
    first = bootstrap_subject_metrics(subjects, scores, n_resamples=50, seed=7)
    second = bootstrap_subject_metrics(subjects, scores, n_resamples=50, seed=7)
    assert first == second


def test_feature_group_ablation_runs_with_same_subject_splits() -> None:
    frame = _build_eval_frame()
    split_map = build_subject_splits(frame, seed=3)
    summary = run_feature_group_ablation(
        frame,
        split_map=split_map,
        model_name="dummy",
        feature_groups=["hr", "hrv", "accel", "hr+hrv", "hr+accel"],
        label_column="reference_stage",
        taxonomy="standard",
        seed=17,
    )
    assert len(summary["rows"]) == 5
    assert set(summary["rows"][0].keys()) >= {"feature_group", "macro_f1", "accuracy"}
    assert all(row["test_subjects"] == sorted(split_map["test"]) for row in summary["rows"])


def test_secondary_taxonomy_and_missing_classes_are_handled() -> None:
    metrics = evaluate_predictions(
        ["WAKE", "N1", "N2", "REM", "WAKE"],
        ["WAKE", "N1", "REM", "REM", "WAKE"],
        taxonomy="secondary",
    )
    assert metrics["labels"] == ["WAKE", "LIGHT", "DEEP", "REM"]
    assert metrics["per_class"]["DEEP"]["support"] == 0
    assert metrics["confusion_matrix"][0][0] == 2
    assert metrics["normalized_confusion_matrix"][1][2] == pytest.approx(0.0)


def test_subject_level_metrics_aggregate_without_leakage() -> None:
    frame = _build_eval_frame()
    predictions = ["WAKE" if subject.startswith("s1") else "N2" for subject in frame["subject_id"]]
    subject_metrics = compute_subject_level_metrics(frame, predictions, taxonomy="standard")
    assert set(subject_metrics.columns) >= {"subject_id", "n_epochs", "macro_f1", "weighted_f1"}
    assert subject_metrics["n_epochs"].sum() == len(frame)
    assert subject_metrics["subject_id"].nunique() == frame["subject_id"].nunique()


def test_feature_group_sequence_matches_repository_contract() -> None:
    assert get_feature_group_sequence()[:3] == ["hr", "hrv", "accel"]
    assert "all_wearable" in get_feature_group_sequence()
    assert len(get_feature_group_sequence()) == 7


def test_cli_renders_evaluation_artifacts(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.csv"
    frame = _build_eval_frame()
    artifact_path.write_text(frame.to_csv(index=False), encoding="utf-8")
    train_frame = frame[
        frame["subject_id"].isin(build_subject_splits(frame, seed=9)["train"])
    ].copy()
    model = train_model(train_frame, model_name="random_forest", feature_group="hr+accel", seed=9)
    checkpoint_path = tmp_path / "rf_checkpoint.json"
    model.save_checkpoint(checkpoint_path)
    output_dir = tmp_path / "eval_out"

    from src.train import main

    exit_code = main([
        "--dataset",
        "dreamt",
        "--model",
        "random_forest",
        "--mode",
        "eval",
        "--artifact",
        str(artifact_path),
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        "test",
        "--output-dir",
        str(output_dir),
        "--bootstrap-repetitions",
        "10",
        "--bootstrap-seed",
        "7",
    ])
    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "per_subject_metrics.csv").exists()
    assert (output_dir / "confusion_matrix.csv").exists()
    assert (output_dir / "normalized_confusion_matrix.csv").exists()
    assert (output_dir / "evaluation_metadata.json").exists()
