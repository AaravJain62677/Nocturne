"""Public evaluation API for the sleep-staging project."""

from __future__ import annotations

from src.config import SUPPORTED_TAXONOMIES
from src.eval.metrics import (
    SECONDARY_LABELS,
    STANDARD_LABELS,
    bootstrap_subject_metrics,
    compute_confusion_matrix,
    compute_subject_level_metrics,
    evaluate_predictions,
    get_feature_group_sequence,
    run_feature_group_ablation,
    write_evaluation_outputs,
    write_optional_visuals,
)


def compute_stage_metrics(y_true, y_pred, taxonomy: str):
    """Compatibility wrapper for the historical evaluation entry point."""
    if taxonomy not in SUPPORTED_TAXONOMIES:
        raise ValueError(
            f"Unsupported taxonomy '{taxonomy}'. "
            f"Supported values: {', '.join(SUPPORTED_TAXONOMIES)}."
        )
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Label sequences must have the same length before metric computation. "
            f"Received {len(y_true)} true labels and {len(y_pred)} predicted labels."
        )
    if not y_true and not y_pred:
        raise NotImplementedError(
            "Stage metric computation is unavailable in this phase; "
            "evaluation requires a real artifact and model checkpoint."
        )
    return evaluate_predictions(y_true, y_pred, taxonomy=taxonomy)


__all__ = [
    "STANDARD_LABELS",
    "SECONDARY_LABELS",
    "bootstrap_subject_metrics",
    "compute_confusion_matrix",
    "compute_stage_metrics",
    "compute_subject_level_metrics",
    "evaluate_predictions",
    "get_feature_group_sequence",
    "run_feature_group_ablation",
    "write_evaluation_outputs",
    "write_optional_visuals",
]
