
import pytest
from src.config import (
    DATA_DIR,
    PROCESSING_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    SECONDARY_SLEEP_STAGE_MAP,
    canonical_sleep_stage,
    resolve_repo_root,
)


def test_repo_root_and_paths_are_absolute() -> None:
    root = resolve_repo_root()
    assert root.is_absolute()
    assert DATA_DIR.is_absolute()
    assert RAW_DATA_DIR == root / "data" / "raw"
    assert PROCESSING_DIR == root / "data" / "processed"
    assert RESULTS_DIR == root / "results"


def test_canonical_sleep_stage_handles_known_and_unknown_labels() -> None:
    assert canonical_sleep_stage("W") == "WAKE"
    assert canonical_sleep_stage("Wake") == "WAKE"
    assert canonical_sleep_stage("N2") == "N2"
    assert canonical_sleep_stage("Sleep stage 3") == "N3"
    assert canonical_sleep_stage("R") == "REM"
    assert canonical_sleep_stage("M") == "MOVEMENT"
    assert canonical_sleep_stage("Unscored") == "UNSCORED"
    assert canonical_sleep_stage("Unknown") == "UNKNOWN"
    assert SECONDARY_SLEEP_STAGE_MAP["N2"] == "LIGHT"
    assert SECONDARY_SLEEP_STAGE_MAP["N3"] == "DEEP"


def test_canonical_sleep_stage_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="empty|blank"):
        canonical_sleep_stage(" ")

    with pytest.raises(ValueError, match="Unsupported|unknown"):
        canonical_sleep_stage("not-a-real-stage")
