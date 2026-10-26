"""Centralized project configuration and sleep-stage canonicalization."""

from __future__ import annotations

from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = REPO_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSING_DIR: Final[Path] = DATA_DIR / "processed"
SPLIT_DIR: Final[Path] = DATA_DIR / "splits"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"
TABLES_DIR: Final[Path] = RESULTS_DIR / "tables"

DEFAULT_SEED: Final[int] = 42
EPOCH_SECONDS: Final[int] = 30

SUPPORTED_DATASETS: Final[tuple[str, ...]] = ("dreamt", "sleep_edf")
SUPPORTED_MODELS: Final[tuple[str, ...]] = ("baseline", "hybrid")
SUPPORTED_TAXONOMIES: Final[tuple[str, ...]] = ("standard", "secondary")
SUPPORTED_SENSOR_SETS: Final[tuple[str, ...]] = ("wearable", "psg", "combined")
SUPPORTED_MODES: Final[tuple[str, ...]] = ("validate", "train", "eval")

STAGE_ALIASES: Final[dict[str, str]] = {
    "W": "WAKE",
    "WAKE": "WAKE",
    "AWAKE": "WAKE",
    "N1": "N1",
    "S1": "N1",
    "STAGE1": "N1",
    "STAGE 1": "N1",
    "SLEEP STAGE 1": "N1",
    "N2": "N2",
    "S2": "N2",
    "STAGE2": "N2",
    "STAGE 2": "N2",
    "SLEEP STAGE 2": "N2",
    "N3": "N3",
    "S3": "N3",
    "STAGE3": "N3",
    "STAGE 3": "N3",
    "SLEEP STAGE 3": "N3",
    "REM": "REM",
    "R": "REM",
    "REMSLEEP": "REM",
    "STAGE R": "REM",
    "SLEEP STAGE R": "REM",
    "M": "MOVEMENT",
    "MOVEMENT": "MOVEMENT",
    "MOVEMENT TIME": "MOVEMENT",
    "MT": "MOVEMENT",
    "UNSCORED": "UNSCORED",
    "UNSCORED/NOT SCORED": "UNSCORED",
    "UNKNOWN": "UNKNOWN",
    "?": "UNKNOWN",
    "NA": "UNKNOWN",
    "N/A": "UNKNOWN",
}

SECONDARY_SLEEP_STAGE_MAP: Final[dict[str, str]] = {
    "WAKE": "WAKE",
    "N1": "LIGHT",
    "N2": "LIGHT",
    "N3": "DEEP",
    "REM": "REM",
    "MOVEMENT": "MOVEMENT",
    "UNSCORED": "UNSCORED",
    "UNKNOWN": "UNKNOWN",
}


def resolve_repo_root() -> Path:
    """Return the repository root independent of the current working directory."""
    return REPO_ROOT


def canonical_sleep_stage(label: str | None) -> str:
    """Normalize a raw sleep-stage label into a canonical internal representation.

    Unknown labels are preserved as ``UNKNOWN`` instead of being silently coerced into
    ``WAKE``. Empty or blank labels raise ``ValueError`` to force callers to handle
    missing or malformed inputs explicitly.
    """
    if label is None:
        raise ValueError("Sleep-stage label is missing; received None.")

    normalized = str(label).strip()
    if not normalized:
        raise ValueError("Sleep-stage label is empty or blank.")

    key = normalized.upper()
    canonical = STAGE_ALIASES.get(key)
    if canonical is not None:
        return canonical

    if key.startswith("STAGE ") or key.startswith("SLEEP STAGE "):
        return "UNKNOWN"

    raise ValueError(
        f"Unsupported sleep-stage label {label!r}; "
        "unknown labels are preserved as UNKNOWN."
    )


def to_secondary_stage(label: str | None) -> str:
    """Map a canonical stage label into the secondary taxonomy."""
    canonical = canonical_sleep_stage(label)
    return SECONDARY_SLEEP_STAGE_MAP.get(canonical, "UNKNOWN")


__all__ = [
    "DATA_DIR",
    "DEFAULT_SEED",
    "EPOCH_SECONDS",
    "FIGURES_DIR",
    "PROCESSING_DIR",
    "RAW_DATA_DIR",
    "REPO_ROOT",
    "RESULTS_DIR",
    "SECONDARY_SLEEP_STAGE_MAP",
    "SPLIT_DIR",
    "SUPPORTED_DATASETS",
    "SUPPORTED_MODES",
    "SUPPORTED_MODELS",
    "SUPPORTED_SENSOR_SETS",
    "SUPPORTED_TAXONOMIES",
    "TABLES_DIR",
    "canonical_sleep_stage",
    "resolve_repo_root",
    "to_secondary_stage",
]
