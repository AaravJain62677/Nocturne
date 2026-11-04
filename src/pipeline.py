"""Unified synthetic pipeline orchestration for Nocturne.

This module intentionally builds on the existing training, preprocessing, and
inspection utilities without replacing the individual CLI commands already in the
repository. It provides a single, explicit interface for running the full
synthetic workflow: inspection, preprocessing, splitting, training, checkpoint
persistence, evaluation, and artifact export.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_SEED
from src.eval import (
    compute_subject_level_metrics,
    evaluate_predictions,
    write_evaluation_outputs,
    write_optional_visuals,
)
from src.models import (
    build_subject_splits,
    compute_partition_summary,
    load_preprocessed_artifact,
    resolve_feature_group,
    save_split_assignments,
    train_model,
)
from src.preprocess import (
    SYNTHETIC_DEMONSTRATION_LABEL,
    _synthetic_demo_payload,
    preprocess_dataset,
)
from src.train import validate_configuration


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Explicit configuration for a synthetic end-to-end pipeline run.

    Synthetic demo generation is opt-in and never automatic. Missing real DREAMT
    data must fail unless the user explicitly requests the demo fallback.
    """

    dataset: str = "dreamt"
    model: str = "random_forest"
    taxonomy: str = "standard"
    sensor_set: str = "wearable"
    seed: int = DEFAULT_SEED
    artifact: str | Path | None = None
    data_dir: str | Path | None = None
    output_dir: str | Path = "results/pipeline"
    split_output_dir: str | Path | None = None
    feature_group: str = "all_wearable"
    split: str = "test"
    checkpoint: str | Path | None = None
    synthetic_demo: bool = False


def _resolve_artifact_path(config: PipelineConfig) -> Path:
    """Resolve a real artifact or require an explicit synthetic demo opt-in."""
    if config.artifact is not None:
        artifact_path = Path(config.artifact).expanduser().resolve()
        if artifact_path.exists():
            return artifact_path
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    if config.data_dir is not None:
        dataset_dir = Path(config.data_dir).expanduser().resolve()
        real_pattern_files = [
            path for path in dataset_dir.rglob("*")
            if path.is_file() and path.name not in {".gitkeep", ".DS_Store"}
        ]
        if not real_pattern_files:
            if not config.synthetic_demo:
                raise FileNotFoundError(
                    f"No valid DREAMT recordings were found in '{dataset_dir}'. "
                    "Supply a real DREAMT dataset or rerun with --synthetic-demo to generate "
                    "a deterministic synthetic demonstration fixture."
                )
            demo_output_dir = Path(config.output_dir).expanduser().resolve() / "processed"
            payload = _synthetic_demo_payload(
                data_dir=dataset_dir,
                output_dir=demo_output_dir,
                seed=int(config.seed),
                subject_count=6,
                recordings_per_subject=2,
                epochs_per_record=12,
            )
            artifact_path = Path(payload["artifact"]["data_path"]).expanduser().resolve()
            if artifact_path.exists():
                return artifact_path
            raise FileNotFoundError(f"Generated synthetic artifact not found: {artifact_path}")

        try:
            payload = preprocess_dataset(
                dataset=config.dataset,
                data_dir=dataset_dir,
                output_dir=Path(config.output_dir).expanduser().resolve() / "processed",
                synthetic_demo=config.synthetic_demo,
            )
        except ValueError as exc:
            if not config.synthetic_demo:
                raise ValueError(
                    f"No valid DREAMT recordings were discovered under '{dataset_dir}'. "
                    "Provide a real DREAMT dataset or rerun with --synthetic-demo to generate "
                    "an explicit synthetic demonstration fixture."
                ) from exc
            demo_output_dir = Path(config.output_dir).expanduser().resolve() / "processed"
            payload = _synthetic_demo_payload(
                data_dir=dataset_dir,
                output_dir=demo_output_dir,
                seed=int(config.seed),
                subject_count=6,
                recordings_per_subject=2,
                epochs_per_record=12,
            )
        artifact_path = Path(payload["artifact"]["data_path"]).expanduser().resolve()
        if artifact_path.exists():
            return artifact_path
        raise FileNotFoundError(f"Generated artifact not found: {artifact_path}")

    fallback = Path("data/processed") / f"{config.dataset}_wearable_epochs.parquet"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(
        "No artifact path was supplied and no processed data was found. "
        "Provide --artifact, --data-dir with real DREAMT data, or rerun with --synthetic-demo "
        "to generate an explicit synthetic demonstration artifact."
    )


def run_pipeline(config: PipelineConfig | dict[str, Any] | argparse.Namespace) -> dict[str, Any]:
    """Execute the synthetic end-to-end pipeline and return a summary payload."""
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    if isinstance(config, dict):
        config = PipelineConfig(**config)

    validate_configuration(
        config.dataset,
        config.model,
        config.taxonomy,
        "train",
        sensor_set=config.sensor_set,
    )

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = _resolve_artifact_path(config)
    artifact = load_preprocessed_artifact(artifact_path)

    split_map = build_subject_splits(artifact, seed=int(config.seed))
    split_dir = (
        Path(config.split_output_dir).expanduser().resolve()
        if config.split_output_dir is not None
        else output_dir / "splits"
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    save_split_assignments(split_map, split_dir)
    summary = compute_partition_summary(artifact, split_map)

    train_subjects = sorted({str(subject_id) for subject_id in split_map.get("train", [])})
    if not train_subjects:
        raise ValueError("Train split is empty; the pipeline cannot train without subjects.")
    train_frame = artifact[artifact["subject_id"].astype(str).isin(train_subjects)].copy()

    model = train_model(
        train_frame,
        config.model,
        feature_group=config.feature_group,
        label_column="reference_stage",
        seed=int(config.seed),
    )

    checkpoint_path = (
        Path(config.checkpoint).expanduser().resolve()
        if config.checkpoint is not None
        else output_dir / f"{config.model}_checkpoint.json"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(checkpoint_path)

    selected_subjects = sorted({str(subject_id) for subject_id in split_map.get(config.split, [])})
    if not selected_subjects:
        raise ValueError(f"Split '{config.split}' is empty; no subjects available for evaluation.")

    eval_frame = artifact[artifact["subject_id"].astype(str).isin(selected_subjects)].copy()
    feature_columns = resolve_feature_group(eval_frame, config.feature_group)
    feature_frame = eval_frame[feature_columns].copy()
    predictions = np.asarray(model.predict(feature_frame), dtype=object)
    metrics = evaluate_predictions(
        eval_frame["reference_stage"].tolist(),
        predictions,
        taxonomy=config.taxonomy,
    )
    per_subject = compute_subject_level_metrics(
        eval_frame,
        predictions,
        taxonomy=config.taxonomy,
    )

    metadata = {
        "dataset": config.dataset,
        "model": config.model,
        "taxonomy": config.taxonomy,
        "sensor_set": config.sensor_set,
        "feature_group": config.feature_group,
        "seed": int(config.seed),
        "artifact_path": str(artifact_path),
        "split_map": split_map,
        "selected_split": config.split,
        "selected_subjects": selected_subjects,
        "checkpoint_path": str(checkpoint_path),
        "synthetic_data_disclaimer": SYNTHETIC_DEMONSTRATION_LABEL,
        "synthetic_data_generation_method": (
            "deterministic synthetic demonstration fixture; "
            "not a real DREAMT dataset"
        ),
    }
    ablation_summary = pd.DataFrame([
        {
            "feature_group": config.feature_group,
            "model_name": config.model,
            "split": config.split,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "cohens_kappa": metrics["cohens_kappa"],
            "n_subjects": len(selected_subjects),
            "n_epochs": int(len(eval_frame)),
        }
    ])
    evaluation_paths = write_evaluation_outputs(
        output_dir,
        metrics=metrics,
        per_subject=per_subject,
        confusion=np.asarray(metrics["confusion_matrix"], dtype=int),
        normalized_confusion=np.asarray(metrics["normalized_confusion_matrix"], dtype=float),
        labels=metrics["labels"],
        metadata=metadata,
        ablation_summary=ablation_summary,
    )
    write_optional_visuals(
        output_dir,
        np.asarray(metrics["confusion_matrix"], dtype=int),
        metrics["labels"],
        ablation_summary,
    )

    pipeline_summary = {
        "dataset": config.dataset,
        "model": config.model,
        "taxonomy": config.taxonomy,
        "sensor_set": config.sensor_set,
        "feature_group": config.feature_group,
        "seed": int(config.seed),
        "artifact_path": str(artifact_path),
        "checkpoint_path": str(checkpoint_path),
        "split_map": split_map,
        "partition_summary": summary,
        "selected_split": config.split,
        "selected_subjects": selected_subjects,
        "feature_columns": feature_columns,
        "prediction_count": int(len(predictions)),
        "metrics": metrics,
        "evaluation_paths": {key: str(path) for key, path in evaluation_paths.items()},
        "synthetic_data_disclaimer": SYNTHETIC_DEMONSTRATION_LABEL,
        "synthetic_data_generation_method": (
            "deterministic synthetic demonstration fixture; "
            "not a real DREAMT dataset"
        ),
    }
    pipeline_path = output_dir / "pipeline_summary.json"
    pipeline_path.write_text(
        json.dumps(pipeline_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return pipeline_summary


def build_parser() -> argparse.ArgumentParser:
    """Construct the unified Nocturne pipeline CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the synthetic Nocturne workflow end-to-end: preprocess, split, train, "
            "checkpoint, and evaluate. Synthetic demo generation is explicit and must be "
            "requested with --synthetic-demo; it is never substituted automatically for "
            "missing real DREAMT recordings."
        )
    )
    parser.add_argument(
        "--dataset",
        default="dreamt",
        help="Dataset identifier to validate and run.",
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        help="Model family to train.",
    )
    parser.add_argument(
        "--taxonomy",
        default="standard",
        help="Sleep-stage taxonomy.",
    )
    parser.add_argument(
        "--sensor-set",
        default="wearable",
        help="Sensor set to validate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible runs.",
    )
    parser.add_argument(
        "--artifact",
        default=None,
        help="Path to an existing preprocessed artifact.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Directory containing DREAMT recordings to preprocess. If no real data is found, "
            "the run fails unless --synthetic-demo is set explicitly."
        ),
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help=(
            "Generate a deterministic synthetic demonstration artifact for DREAMT-style "
            "validation. This flag is required for synthetic fallback and is never automatic."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/pipeline",
        help="Directory for model checkpoints and evaluation outputs.",
    )
    parser.add_argument(
        "--split-output-dir",
        default=None,
        help="Directory for saved subject split assignments.",
    )
    parser.add_argument(
        "--feature-group",
        default="all_wearable",
        help="Feature group to train and evaluate on.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Subject split to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint output path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the end-to-end synthetic pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_pipeline(PipelineConfig(
            dataset=args.dataset,
            model=args.model,
            taxonomy=args.taxonomy,
            sensor_set=args.sensor_set,
            seed=args.seed,
            artifact=args.artifact,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            split_output_dir=args.split_output_dir,
            feature_group=args.feature_group,
            split=args.split,
            checkpoint=args.checkpoint,
            synthetic_demo=args.synthetic_demo,
        ))
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        parser.exit(status=2, message=f"Pipeline error: {exc}\n")

    print(json.dumps({
        "dataset": summary["dataset"],
        "model": summary["model"],
        "feature_group": summary["feature_group"],
        "checkpoint_path": summary["checkpoint_path"],
        "selected_split": summary["selected_split"],
        "accuracy": summary["metrics"]["accuracy"],
        "macro_f1": summary["metrics"]["macro_f1"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PipelineConfig", "build_parser", "main", "run_pipeline"]
