"""Experiment specification registry for the sleep-staging repository."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
    SUPPORTED_SENSOR_SETS,
    SUPPORTED_TAXONOMIES,
)
from src.ingest import get_dataset_spec, validate_sensor_set


@dataclass(frozen=True)
class ExperimentSpec:
    """Describe one concrete experiment configuration."""

    dataset: str
    sensor_set: str
    taxonomy: str
    model_name: str

    def __post_init__(self) -> None:
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{self.dataset}'. "
                f"Supported values: {', '.join(SUPPORTED_DATASETS)}."
            )
        if self.sensor_set not in SUPPORTED_SENSOR_SETS:
            raise ValueError(
                f"Unsupported sensor set '{self.sensor_set}'. "
                f"Supported values: {', '.join(SUPPORTED_SENSOR_SETS)}."
            )
        if self.taxonomy not in SUPPORTED_TAXONOMIES:
            raise ValueError(
                f"Unsupported taxonomy '{self.taxonomy}'. "
                f"Supported values: {', '.join(SUPPORTED_TAXONOMIES)}."
            )
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{self.model_name}'. "
                f"Supported values: {', '.join(SUPPORTED_MODELS)}."
            )


def validate_experiment_config(
    dataset: str,
    sensor_set: str,
    taxonomy: str,
    model_name: str,
) -> tuple[str, ...]:
    """Validate the dataset, sensor set, taxonomy, and model combination.

    Returns the canonical signal names associated with the requested sensor set,
    ensuring the experiment specification remains consistent before any training
    or evaluation is attempted.
    """
    get_dataset_spec(dataset)
    if taxonomy not in SUPPORTED_TAXONOMIES:
        raise ValueError(
            f"Unsupported taxonomy '{taxonomy}'. "
            f"Supported values: {', '.join(SUPPORTED_TAXONOMIES)}."
        )
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported values: {', '.join(SUPPORTED_MODELS)}."
        )
    return validate_sensor_set(dataset, sensor_set)


def build_experiment_specs(
    dataset: str,
    sensor_set: str,
    taxonomy: str,
    model_name: str,
) -> list[ExperimentSpec]:
    """Build the experiment configuration objects for a concrete study setup.

    The current implementation intentionally keeps the registry small and explicit
    while leaving the actual training workload for a later phase.
    """
    validate_experiment_config(dataset, sensor_set, taxonomy, model_name)
    return [
        ExperimentSpec(
            dataset=dataset,
            sensor_set=sensor_set,
            taxonomy=taxonomy,
            model_name=model_name,
        )
    ]


__all__ = ["ExperimentSpec", "build_experiment_specs", "validate_experiment_config"]
