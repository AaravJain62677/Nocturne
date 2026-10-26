"""Evaluation utilities and placeholders for sleep-staging metrics."""

from __future__ import annotations

from collections.abc import Sequence

from src.config import SUPPORTED_TAXONOMIES


def compute_stage_metrics(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    taxonomy: str,
) -> float:
    """Compute stage metrics for a model run.

    This Phase 1 placeholder raises a clear error to keep the project contract
    explicit and avoid accidental false success on unavailable evaluation code.
    """
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

    _ = y_true, y_pred
    raise NotImplementedError(
        "Stage metric computation is not implemented yet; evaluation is unavailable in this phase."
    )


__all__ = ["compute_stage_metrics"]
