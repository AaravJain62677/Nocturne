"""Evaluation utilities for synthetic sleep-stage model assessment."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import DEFAULT_SEED, canonical_sleep_stage, to_secondary_stage
from src.models import build_subject_splits, resolve_feature_group, train_model

STANDARD_LABELS: list[str] = ["WAKE", "N1", "N2", "N3", "REM"]
SECONDARY_LABELS: list[str] = ["WAKE", "LIGHT", "DEEP", "REM"]


def get_feature_group_sequence() -> list[str]:
    """Return the repository's canonical seven wearable feature-group ablation order."""
    return ["hr", "hrv", "accel", "hr+hrv", "hr+accel", "hr+hrv+accel", "all_wearable"]


def _ordered_labels_for_taxonomy(taxonomy: str) -> list[str]:
    if taxonomy == "standard":
        return STANDARD_LABELS.copy()
    if taxonomy == "secondary":
        return SECONDARY_LABELS.copy()
    raise ValueError(f"Unsupported taxonomy '{taxonomy}'.")


def _normalize_stage_label(value: object, taxonomy: str) -> str:
    if value is None:
        raise ValueError("Stage labels cannot be None.")
    text = str(value).strip()
    if not text:
        raise ValueError("Stage labels cannot be empty.")
    canonical = canonical_sleep_stage(text)
    if taxonomy == "standard":
        if canonical in STANDARD_LABELS:
            return canonical
        if canonical in {"UNKNOWN", "MOVEMENT", "UNSCORED"}:
            return "UNKNOWN"
        raise ValueError(f"Unsupported stage label for evaluation: {value!r}")
    mapped = to_secondary_stage(canonical)
    if mapped in SECONDARY_LABELS:
        return mapped
    if mapped in {"MOVEMENT", "UNKNOWN", "UNSCORED"}:
        return "UNKNOWN"
    raise ValueError(f"Unsupported stage label for evaluation: {value!r}")


def _prepare_prediction_vectors(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    taxonomy: str,
) -> tuple[list[str], list[str], dict[str, int]]:
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Label sequences must have the same length before metric computation. "
            f"Received {len(y_true)} true labels and {len(y_pred)} predicted labels."
        )

    true_tokens: list[str] = []
    pred_tokens: list[str] = []
    removed_counts: dict[str, int] = {}
    allowed = _ordered_labels_for_taxonomy(taxonomy)

    for true_value, pred_value in zip(y_true, y_pred, strict=True):
        try:
            true_token = _normalize_stage_label(true_value, taxonomy)
            pred_token = _normalize_stage_label(pred_value, taxonomy)
        except ValueError:
            removed_counts["unsupported"] = removed_counts.get("unsupported", 0) + 1
            continue

        if true_token not in allowed:
            removed_counts[true_token] = removed_counts.get(true_token, 0) + 1
            continue
        if pred_token not in allowed:
            removed_counts[pred_token] = removed_counts.get(pred_token, 0) + 1
            continue

        true_tokens.append(true_token)
        pred_tokens.append(pred_token)

    if not true_tokens:
        raise ValueError(
            "No valid labels remained after applying the taxonomy mapping; "
            f"taxonomy={taxonomy}."
        )

    return true_tokens, pred_tokens, removed_counts


def compute_confusion_matrix(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    labels: Sequence[str] | None = None,
    taxonomy: str = "standard",
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Compute the raw confusion matrix and return the matched label order."""
    allowed = list(labels) if labels is not None else _ordered_labels_for_taxonomy(taxonomy)
    true_tokens, pred_tokens, removed = _prepare_prediction_vectors(y_true, y_pred, taxonomy)
    matrix = confusion_matrix(true_tokens, pred_tokens, labels=allowed)
    return matrix, allowed, removed


def evaluate_predictions(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    labels: Sequence[str] | None = None,
    taxonomy: str = "standard",
) -> dict[str, Any]:
    """Compute overall classification metrics and confidence matrices."""
    if taxonomy not in {"standard", "secondary"}:
        raise ValueError(f"Unsupported taxonomy '{taxonomy}'.")

    allowed = list(labels) if labels is not None else _ordered_labels_for_taxonomy(taxonomy)
    true_tokens, pred_tokens, removed = _prepare_prediction_vectors(y_true, y_pred, taxonomy)

    present_labels = [
        label for label in allowed if label in set(true_tokens) or label in set(pred_tokens)
    ]
    if not present_labels:
        present_labels = allowed[:1]
    accuracy = float(accuracy_score(true_tokens, pred_tokens))
    balanced = float(balanced_accuracy_score(true_tokens, pred_tokens))
    macro_f1 = float(
        f1_score(
            true_tokens,
            pred_tokens,
            labels=present_labels,
            average="macro",
            zero_division=0,
        )
    )
    weighted_f1 = float(
        f1_score(
            true_tokens,
            pred_tokens,
            labels=present_labels,
            average="weighted",
            zero_division=0,
        )
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        true_tokens,
        pred_tokens,
        labels=allowed,
        zero_division=0,
        average=None,
    )

    try:
        kappa = float(cohen_kappa_score(true_tokens, pred_tokens))
    except ValueError:
        kappa = None

    raw_matrix = confusion_matrix(true_tokens, pred_tokens, labels=allowed)
    row_sums = raw_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        raw_matrix.astype(float),
        row_sums,
        out=np.zeros_like(raw_matrix, dtype=float),
        where=row_sums != 0,
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for label, p, r, f_score, count in zip(allowed, precision, recall, f1, support, strict=True):
        per_class[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f_score),
            "support": int(count),
        }

    metrics: dict[str, Any] = {
        "taxonomy": taxonomy,
        "labels": allowed,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohens_kappa": kappa,
        "per_class": per_class,
        "support": {label: int(count) for label, count in zip(allowed, support, strict=True)},
        "confusion_matrix": raw_matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "removed_labels": removed,
    }
    return metrics


def compute_subject_level_metrics(
    frame: pd.DataFrame,
    predictions: Sequence[object],
    label_column: str = "reference_stage",
    taxonomy: str = "standard",
) -> pd.DataFrame:
    """Aggregate epoch-level results to subject-level macro metrics."""
    if len(frame) != len(predictions):
        raise ValueError(
            "The prediction vector must have the same length as the frame being evaluated. "
            f"Received {len(frame)} rows and {len(predictions)} predictions."
        )

    rows: list[dict[str, Any]] = []
    prediction_array = np.asarray(list(predictions), dtype=object)
    for subject_id, subject_frame in frame.groupby("subject_id", sort=True):
        subject_mask = frame["subject_id"].astype(str) == str(subject_id)
        subject_indices = np.nonzero(subject_mask.to_numpy())[0]
        subject_true = subject_frame[label_column].tolist()
        subject_pred = prediction_array[subject_indices].tolist()
        metrics = evaluate_predictions(subject_true, subject_pred, taxonomy=taxonomy)
        rows.append(
            {
                "subject_id": str(subject_id),
                "n_epochs": int(len(subject_true)),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "cohens_kappa": metrics["cohens_kappa"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_subject_metrics(
    subject_ids: Sequence[str],
    score_map: dict[str, float | Sequence[float]],
    n_resamples: int = 200,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Bootstrap subject-level scores while preserving the subject unit of resampling."""
    ids = [str(subject_id) for subject_id in subject_ids]
    if not ids:
        raise ValueError("No subjects are available for bootstrap resampling.")

    values: dict[str, float] = {}
    for subject_id in ids:
        score_value = score_map.get(subject_id)
        if score_value is None:
            raise KeyError(f"Missing subject score for '{subject_id}'.")
        if isinstance(score_value, (list, tuple, np.ndarray)):
            values[subject_id] = float(np.mean(score_value))
        else:
            values[subject_id] = float(score_value)

    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    for _ in range(int(n_resamples)):
        sampled = [values[ids[idx]] for idx in rng.choice(len(ids), size=len(ids), replace=True)]
        draws.append(float(np.mean(sampled)))

    lower = float(np.quantile(draws, 0.05))
    upper = float(np.quantile(draws, 0.95))
    return {
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "subject_count": len(ids),
        "bootstrap_mean": float(np.mean(draws)),
        "ci_lower_05": lower,
        "ci_upper_95": upper,
        "bootstrap_samples": draws,
    }


def _feature_group_coverage(
    frame: pd.DataFrame,
    feature_group: str,
) -> tuple[list[str], list[str]]:
    try:
        selected = resolve_feature_group(frame, feature_group)
    except ValueError as exc:
        raise ValueError(
            f"Feature group '{feature_group}' is unavailable in the artifact: {exc}"
        ) from exc

    available = [col for col in frame.columns if str(col).startswith("feature_")]
    missing = [col for col in selected if col not in available]
    return selected, missing


def run_feature_group_ablation(
    frame: pd.DataFrame,
    split_map: dict[str, list[str]] | None = None,
    model_name: str = "dummy",
    feature_groups: Sequence[str] | None = None,
    label_column: str = "reference_stage",
    taxonomy: str = "standard",
    seed: int = DEFAULT_SEED,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train a model on each feature group using the same subject splits.

    The function summarizes performance for each selected feature group using the
    same subject-based partition so the comparison remains aligned.
    """
    if split_map is None:
        split_map = build_subject_splits(frame, seed=seed)

    if feature_groups is None:
        feature_groups = [
            "hr",
            "hrv",
            "accel",
            "hr+hrv",
            "hr+accel",
            "hr+hrv+accel",
            "all",
        ]

    train_subjects = sorted({str(subject) for subject in split_map.get("train", [])})
    test_subjects = sorted({str(subject) for subject in split_map.get("test", [])})
    if not train_subjects or not test_subjects:
        raise ValueError("A valid train/test split is required for feature-group ablations.")

    train_frame = frame[frame["subject_id"].astype(str).isin(train_subjects)].copy()
    test_frame = frame[frame["subject_id"].astype(str).isin(test_subjects)].copy()

    rows: list[dict[str, Any]] = []
    for feature_group in feature_groups:
        feature_columns, missing = _feature_group_coverage(train_frame, feature_group)
        model = train_model(
            train_frame,
            model_name=model_name,
            feature_group=feature_group,
            label_column=label_column,
            seed=seed,
        )
        predictions = model.predict(test_frame)
        metrics = evaluate_predictions(
            test_frame[label_column].astype(str).tolist(),
            predictions,
            taxonomy=taxonomy,
        )
        rows.append(
            {
                "feature_group": feature_group,
                "model_name": model_name,
                "proxy_test_subjects": test_subjects,
                "test_subjects": test_subjects,
                "feature_columns": feature_columns,
                "missing_columns": missing,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "cohens_kappa": metrics["cohens_kappa"],
            }
        )

    summary = {"rows": rows, "test_subjects": test_subjects, "train_subjects": train_subjects}
    if output_dir is not None:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path / "ablation_summary.csv", index=False)
    return summary


def _subject_index_map(frame: pd.DataFrame) -> dict[str, list[int]]:
    index_map: dict[str, list[int]] = {}
    for subject_id, subject_frame in frame.groupby("subject_id", sort=True):
        index_map[str(subject_id)] = subject_frame.index.tolist()
    return index_map


def write_evaluation_outputs(
    output_dir: str | Path,
    *,
    metrics: dict[str, Any],
    per_subject: pd.DataFrame,
    confusion: np.ndarray,
    normalized_confusion: np.ndarray,
    labels: Sequence[str],
    metadata: dict[str, Any],
    ablation_summary: pd.DataFrame | None = None,
    bootstrap_summary: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Persist evaluation results as JSON and CSV artifacts."""
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = target_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    metrics_df = pd.DataFrame([
        {
            "metric": name,
            "value": value,
        }
        for name, value in {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "cohens_kappa": metrics["cohens_kappa"],
        }.items()
    ])
    metrics_df.to_csv(target_dir / "metrics.csv", index=False)

    per_subject.to_csv(target_dir / "per_subject_metrics.csv", index=False)

    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    confusion_df.index.name = "true_label"
    confusion_df.columns.name = "predicted_label"
    confusion_df.to_csv(target_dir / "confusion_matrix.csv")

    normalized_df = pd.DataFrame(normalized_confusion, index=labels, columns=labels)
    normalized_df.index.name = "true_label"
    normalized_df.columns.name = "predicted_label"
    normalized_df.to_csv(target_dir / "normalized_confusion_matrix.csv")

    if ablation_summary is not None:
        ablation_summary.to_csv(target_dir / "ablation_summary.csv", index=False)
    if bootstrap_summary is not None:
        bootstrap_summary.to_csv(target_dir / "bootstrap_results.csv", index=False)

    metadata_path = target_dir / "evaluation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "metrics": metrics_path,
        "metrics_csv": target_dir / "metrics.csv",
        "per_subject": target_dir / "per_subject_metrics.csv",
        "confusion": target_dir / "confusion_matrix.csv",
        "normalized_confusion": target_dir / "normalized_confusion_matrix.csv",
        "ablation": target_dir / "ablation_summary.csv",
        "bootstrap": target_dir / "bootstrap_results.csv",
        "metadata": metadata_path,
    }


def write_optional_visuals(
    output_dir: str | Path,
    confusion: np.ndarray,
    labels: Sequence[str],
    ablation_summary: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Create basic plots if matplotlib is available."""
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return outputs

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.set_title("Confusion matrix")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, int(confusion[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    path = target_dir / "confusion_matrix_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs["heatmap"] = path

    if (
        ablation_summary is not None
        and not ablation_summary.empty
        and ablation_summary["feature_group"].nunique() > 1
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(ablation_summary["feature_group"], ablation_summary["macro_f1"], color="steelblue")
        ax.set_ylabel("Macro-F1")
        ax.set_xlabel("Feature group")
        ax.set_title("Feature-group macro-F1 comparison")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = target_dir / "feature_group_macro_f1.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["macro_f1"] = path

    return outputs


__all__ = [
    "STANDARD_LABELS",
    "SECONDARY_LABELS",
    "bootstrap_subject_metrics",
    "compute_confusion_matrix",
    "compute_subject_level_metrics",
    "evaluate_predictions",
    "get_feature_group_sequence",
    "run_feature_group_ablation",
    "write_evaluation_outputs",
    "write_optional_visuals",
]
