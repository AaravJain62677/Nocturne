"""Preprocessing helpers and taxonomy definitions for sleep staging."""

from __future__ import annotations

from src.config import SECONDARY_SLEEP_STAGE_MAP, canonical_sleep_stage
from src.preprocessing.eeg_features import compute_eeg_band_stats
from src.preprocessing.epoch_segmentation import (
    align_recording_to_epochs,
    canonicalize_stage_labels,
)
from src.preprocessing.watch_features import build_wearable_feature_table, compute_hrv_features

STANDARD_STAGE_ORDER: tuple[str, ...] = ("WAKE", "N1", "N2", "N3", "REM")
SECONDARY_STAGE_ORDER: tuple[str, ...] = ("WAKE", "LIGHT", "DEEP", "REM")


def resolve_stage_taxonomy(taxonomy: str) -> tuple[str, ...]:
    """Resolve the requested label taxonomy.

    The function validates arguments and raises a meaningful error rather than
    silently accepting unsupported values or converting unknown labels.
    """
    if taxonomy == "standard":
        return STANDARD_STAGE_ORDER
    if taxonomy == "secondary":
        return SECONDARY_STAGE_ORDER
    raise ValueError(
        f"Unsupported taxonomy '{taxonomy}'. Supported values: standard, secondary."
    )


def apply_stage_taxonomy(labels: list[str], taxonomy: str) -> list[str]:
    """Apply a taxonomy mapping to a sequence of stage labels.

    Unknown and unscored labels remain explicit and are never silently converted
    to WAKE. This function only transforms known labels to the chosen taxonomy.
    """
    if taxonomy == "standard":
        return [canonical_sleep_stage(label) for label in labels]

    if taxonomy == "secondary":
        mapped: list[str] = []
        for label in labels:
            canonical = canonical_sleep_stage(label)
            mapped.append(SECONDARY_SLEEP_STAGE_MAP.get(canonical, canonical))
        return mapped

    raise ValueError(
        f"Unsupported taxonomy '{taxonomy}'. Supported values: standard, secondary."
    )


__all__ = [
    "SECONDARY_STAGE_ORDER",
    "STANDARD_STAGE_ORDER",
    "align_recording_to_epochs",
    "apply_stage_taxonomy",
    "build_wearable_feature_table",
    "canonicalize_stage_labels",
    "compute_eeg_band_stats",
    "compute_hrv_features",
    "resolve_stage_taxonomy",
]
