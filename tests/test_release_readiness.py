import pandas as pd
import pytest
from src import __version__
from src.config import canonical_sleep_stage
from src.models import resolve_feature_group
from src.train import validate_configuration


def test_version_metadata_matches_release() -> None:
    assert __version__ == "0.1.0"


def test_configuration_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset"):
        validate_configuration("invalid_dataset", "random_forest", "standard", "validate")

    with pytest.raises(ValueError, match="Unsupported model"):
        validate_configuration("dreamt", "invalid_model", "standard", "validate")

    with pytest.raises(ValueError, match="Unsupported taxonomy"):
        validate_configuration("dreamt", "random_forest", "invalid_taxonomy", "validate")


def test_feature_group_evaluation_rejects_invalid_groups() -> None:
    frame = pd.DataFrame(
        [
            {
                "subject_id": "s1",
                "recording_id": "r1",
                "epoch_index": 0,
                "reference_stage": "WAKE",
                "feature_hr_mean": 60.0,
                "feature_hrv_rmssd": 40.0,
                "feature_accel_magnitude_mean": 0.15,
                "feature_eda_mean": 1.2,
            }
        ]
    )

    with pytest.raises(ValueError, match="Unsupported feature group|requires unavailable"):
        resolve_feature_group(frame, "invalid_group")


def test_canonical_sleep_stage_contract_is_explicit() -> None:
    assert canonical_sleep_stage("W") == "WAKE"
    assert canonical_sleep_stage("Unknown") == "UNKNOWN"
    with pytest.raises(ValueError, match="empty|blank|Unsupported"):
        canonical_sleep_stage("")
