from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from src.models import (
    DummySleepStageModel,
    GRUSleepStageModel,
    HistGradientSleepStageModel,
    LSTMSleepStageModel,
    RandomForestSleepStageModel,
    build_subject_splits,
    compute_partition_summary,
    prepare_sequence_batches,
    resolve_feature_group,
    save_split_assignments,
)


def _build_synthetic_artifact_frame() -> pd.DataFrame:
    rows = []
    for subject in ["s1", "s2", "s3", "s4", "s5", "s6"]:
        for epoch_index in range(3):
            rows.append(
                {
                    "subject_id": subject,
                    "recording_id": f"rec-{subject}",
                    "epoch_index": epoch_index,
                    "epoch_start": float(epoch_index * 30),
                    "epoch_end": float((epoch_index + 1) * 30),
                    "reference_stage": "WAKE" if subject in {"s1", "s2"} else "N2",
                    "excluded": False,
                    "exclusion_reason": "",
                    "feature_hr_mean": 60.0 + epoch_index,
                    "feature_hrv_rmssd": 40.0 + epoch_index,
                    "feature_accel_magnitude_mean": 0.2 + epoch_index * 0.05,
                    "feature_eda_mean": 1.5 + epoch_index * 0.1,
                    "feature_eda_std": 0.3,
                    "modality_available_hr": True,
                    "modality_available_eda": True,
                    "modality_available_accel_x": True,
                    "modality_available_accel_y": True,
                    "modality_available_accel_z": True,
                }
            )
    return pd.DataFrame(rows)


def test_subject_split_is_subject_independent_and_deterministic() -> None:
    frame = _build_synthetic_artifact_frame()
    first = build_subject_splits(frame, seed=7)
    second = build_subject_splits(frame, seed=7)

    assert set(first.keys()) == {"train", "val", "test"}
    assert first["train"] != first["val"]
    assert first["train"] != first["test"]
    assert first["val"] != first["test"]
    assert first == second

    all_subjects = set(frame["subject_id"].unique())
    partition_subjects = set().union(*first.values())
    assert partition_subjects == all_subjects
    assert len(set(first["train"]) & set(first["val"])) == 0
    assert len(set(first["train"]) & set(first["test"])) == 0
    assert len(set(first["val"]) & set(first["test"])) == 0


def test_small_subject_splits_keep_all_partitions_non_empty() -> None:
    frame = _build_synthetic_artifact_frame().query("subject_id in ['s1', 's2', 's3', 's4']")
    split_map = build_subject_splits(frame, seed=5)

    assert split_map["train"]
    assert split_map["val"]
    assert split_map["test"]
    assert len(set(split_map["train"]) & set(split_map["val"])) == 0
    assert len(set(split_map["train"]) & set(split_map["test"])) == 0
    assert len(set(split_map["val"]) & set(split_map["test"])) == 0


def test_excluded_epochs_are_removed_before_split_summary() -> None:
    frame = _build_synthetic_artifact_frame()
    frame.loc[0, "excluded"] = True
    frame.loc[0, "exclusion_reason"] = "unknown_label"
    summary = compute_partition_summary(frame, build_subject_splits(frame, seed=3))
    assert summary["train"]["epochs"] >= 0
    assert summary["val"]["epochs"] >= 0
    assert summary["test"]["epochs"] >= 0
    assert "class_counts" in summary["train"]


def test_feature_group_selection_uses_real_preprocessing_columns() -> None:
    frame = _build_synthetic_artifact_frame()
    group = resolve_feature_group(frame, "hr+accel")
    assert "feature_hr_mean" in group
    assert "feature_accel_magnitude_mean" in group
    assert "feature_eda_mean" not in group


def test_random_forest_model_initializes_and_saves_checkpoint(tmp_path: Path) -> None:
    frame = _build_synthetic_artifact_frame().head(12)
    model = RandomForestSleepStageModel(
        label_column="reference_stage",
        feature_columns=["feature_hr_mean", "feature_hrv_rmssd", "feature_accel_magnitude_mean"],
        seed=3,
    )
    model.fit(frame)
    checkpoint_path = tmp_path / "rf_model.json"
    payload = model.save_checkpoint(checkpoint_path)
    assert checkpoint_path.exists()
    assert payload["model_name"] == "random_forest"
    assert payload["feature_columns"]
    loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert loaded["model_name"] == "random_forest"


def test_sequence_batches_preserve_subject_and_recording_boundaries() -> None:
    frame = _build_synthetic_artifact_frame().copy()
    batches = prepare_sequence_batches(frame, sequence_length=2, label_column="reference_stage")
    assert len(batches) >= 1
    for batch in batches:
        assert batch["subject_ids"]
        assert batch["recording_ids"]
        assert len(batch["labels"]) == len(batch["features"]) > 0


def test_split_assignments_round_trip_without_changing_membership(tmp_path: Path) -> None:
    frame = _build_synthetic_artifact_frame()
    split_map = build_subject_splits(frame, seed=11)
    output_dir = tmp_path / "splits"
    saved = save_split_assignments(split_map, output_dir)
    assignments_path = saved["assignments"]

    assert assignments_path.exists()
    payload = json.loads(assignments_path.read_text(encoding="utf-8"))
    loaded = {"train": [], "val": [], "test": []}
    for row in payload:
        loaded[row["split"]].append(row["subject_id"])
    assert {label: set(subjects) for label, subjects in loaded.items()} == {
        split_name: set(subjects) for split_name, subjects in split_map.items()
    }


def test_feature_group_resolution_requires_required_modalities() -> None:
    frame = _build_synthetic_artifact_frame().drop(
        columns=["feature_accel_magnitude_mean"]
    )
    with pytest.raises(
        ValueError,
        match="Feature group|Unsupported feature group|requires unavailable|Available groups",
    ):
        resolve_feature_group(frame, "hr+accel")

    with pytest.raises(
        ValueError,
        match="Feature group|Unsupported feature group|requires unavailable|Available groups",
    ):
        resolve_feature_group(frame, "hr+hrv+accel")


def test_classical_models_train_and_predict_on_holdout_data() -> None:
    frame = _build_synthetic_artifact_frame()
    split_map = build_subject_splits(frame, seed=3)
    train_frame = frame[frame["subject_id"].isin(split_map["train"])].copy()
    val_frame = frame[frame["subject_id"].isin(split_map["val"])].copy()

    for model_cls in (
        DummySleepStageModel,
        RandomForestSleepStageModel,
        HistGradientSleepStageModel,
    ):
        model = model_cls(label_column="reference_stage", seed=3)
        model.fit(train_frame)
        predictions = model.predict(val_frame)
        assert len(predictions) == len(val_frame)
        assert set(predictions).issubset(set(frame["reference_stage"].unique()))
        assert all(
            col not in model.feature_columns
            for col in {"subject_id", "recording_id", "reference_stage"}
        )
        assert all("psg" not in col.lower() for col in model.feature_columns)


def test_sequence_models_train_for_minimal_epoch_and_save_checkpoint(tmp_path: Path) -> None:
    frame = _build_synthetic_artifact_frame()
    for model_cls in (GRUSleepStageModel, LSTMSleepStageModel):
        model = model_cls(
            label_column="reference_stage",
            feature_columns=[
                "feature_hr_mean",
                "feature_hrv_rmssd",
                "feature_accel_magnitude_mean",
            ],
            seed=5,
            hidden_dim=8,
            sequence_length=2,
            epochs=2,
            patience=1,
        )
        model.fit(frame)
        predictions = model.predict(frame)
        expected_count = len(
            prepare_sequence_batches(
                frame,
                sequence_length=2,
                label_column="reference_stage",
            )
        )
        assert len(predictions) == expected_count
        checkpoint_path = tmp_path / f"{model.model_name}_checkpoint.json"
        payload = model.save_checkpoint(checkpoint_path)
        assert payload["model_name"] == model.model_name
        assert checkpoint_path.exists()

        reloaded = type(model).load_checkpoint(checkpoint_path)
        assert reloaded.model_name == model.model_name
        assert reloaded.label_to_index == model.label_to_index
