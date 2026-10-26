import pytest
from src.config import SECONDARY_SLEEP_STAGE_MAP, canonical_sleep_stage
from src.eval import compute_stage_metrics
from src.experiment import build_experiment_specs
from src.ingest import (
    DATASET_REGISTRY,
    get_dataset_spec,
    load_dataset,
    validate_dataset_layout,
    validate_sensor_set,
)
from src.models import BaselineSleepStageModel
from src.preprocessing import (
    SECONDARY_STAGE_ORDER,
    STANDARD_STAGE_ORDER,
    apply_stage_taxonomy,
    resolve_stage_taxonomy,
)


def test_registry_contains_supported_datasets() -> None:
    assert set(DATASET_REGISTRY) == {"dreamt", "sleep_edf"}
    assert get_dataset_spec("dreamt").name == "dreamt"
    assert get_dataset_spec("sleep_edf").source == "public"
    assert "hr" in get_dataset_spec("dreamt").wearable_signals
    assert "eeg_fpz_cz" in get_dataset_spec("sleep_edf").psg_signals


def test_stage_taxonomies_are_explicit_and_ordered() -> None:
    assert STANDARD_STAGE_ORDER == ("WAKE", "N1", "N2", "N3", "REM")
    assert SECONDARY_STAGE_ORDER == ("WAKE", "LIGHT", "DEEP", "REM")
    taxonomy = resolve_stage_taxonomy("secondary")
    assert taxonomy == SECONDARY_STAGE_ORDER
    assert SECONDARY_SLEEP_STAGE_MAP["N2"] == "LIGHT"


def test_stage_taxonomy_application_is_explicit() -> None:
    labels = ["WAKE", "N1", "N2", "N3", "REM", "UNKNOWN", "UNSCORED"]
    transformed = apply_stage_taxonomy(labels, "secondary")
    assert transformed == [
        "WAKE",
        "LIGHT",
        "LIGHT",
        "DEEP",
        "REM",
        "UNKNOWN",
        "UNSCORED",
    ]


def test_dataset_layout_validation_requires_existing_dirs() -> None:
    layout = validate_dataset_layout("dreamt")
    assert layout["dataset"] == "dreamt"
    assert "raw" in layout
    assert "processed" in layout


def test_sensor_set_validation_for_dataset() -> None:
    signals = get_dataset_spec("dreamt").wearable_signals
    assert "accel_z" in signals
    assert "hr" in signals
    assert validate_sensor_set("dreamt", "wearable") == signals

    with pytest.raises(ValueError, match="sensor set|unsupported"):
        validate_sensor_set("dreamt", "not_a_real_sensor_set")


def test_experiment_specs_are_structured() -> None:
    specs = build_experiment_specs(
        dataset="dreamt",
        sensor_set="wearable",
        taxonomy="secondary",
        model_name="baseline",
    )
    assert len(specs) == 1
    assert specs[0].dataset == "dreamt"
    assert specs[0].sensor_set == "wearable"
    assert specs[0].taxonomy == "secondary"
    assert specs[0].model_name == "baseline"


def test_unimplemented_dataset_and_model_calls_fail_cleanly() -> None:
    with pytest.raises(ValueError, match="sensor set|unsupported"):
        load_dataset("dreamt", sensor_set="not_a_real_sensor_set")

    with pytest.raises(NotImplementedError, match="not implemented|unavailable"):
        load_dataset("dreamt", sensor_set="wearable")

    model = BaselineSleepStageModel(num_classes=5, signal_channels=5, taxonomy="standard")
    with pytest.raises(NotImplementedError, match="not implemented|unavailable"):
        model.fit([], [])


def test_evaluation_placeholder_is_explicitly_unavailable() -> None:
    with pytest.raises(ValueError, match="Unsupported taxonomy|unsupported"):
        compute_stage_metrics([], [], "invalid")

    with pytest.raises(NotImplementedError, match="not implemented|unavailable"):
        compute_stage_metrics([], [], "standard")


def test_unknown_label_is_not_coerced_to_wake() -> None:
    assert canonical_sleep_stage("Unscored") == "UNSCORED"
    assert canonical_sleep_stage("Unknown") == "UNKNOWN"
