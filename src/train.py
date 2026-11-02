"""CLI entry point for the sleep-staging research project."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_SEED,
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
    SUPPORTED_MODES,
    SUPPORTED_SENSOR_SETS,
    SUPPORTED_TAXONOMIES,
    resolve_repo_root,
)
from src.eval import (
    bootstrap_subject_metrics,
    compute_subject_level_metrics,
    evaluate_predictions,
    write_evaluation_outputs,
    write_optional_visuals,
)
from src.experiment import validate_experiment_config
from src.models import (
    SleepStageModel,
    build_subject_splits,
    compute_partition_summary,
    load_preprocessed_artifact,
    save_split_assignments,
    train_model,
)
from src.utils import configure_logging, get_logger, set_deterministic_seed

logger = get_logger("train")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the repository."""
    parser = argparse.ArgumentParser(
        description=(
            "Sleep-stage research pipeline. Phase 1 validates configuration and "
            "marks unsupported workflows as unavailable."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Dataset identifier for the sleep-staging experiments.",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        required=True,
        help="Model family to use for the experiment.",
    )
    parser.add_argument(
        "--taxonomy",
        choices=SUPPORTED_TAXONOMIES,
        default="standard",
        help="Sleep-stage taxonomy to apply: standard or secondary.",
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="validate",
        help=(
            "Validation checks the configuration; training and evaluation "
            "are unavailable in Phase 1."
        ),
    )
    parser.add_argument(
        "--sensor-set",
        choices=SUPPORTED_SENSOR_SETS,
        default="wearable",
        help="Input sensor set for the experiment: wearable, PSG, or combined.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic random seed for Python, NumPy, and PyTorch.",
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Path to a preprocessed parquet or CSV artifact to train on.",
    )
    parser.add_argument(
        "--feature-group",
        type=str,
        default="all",
        help="Feature group to train on, resolved from the actual artifact schema.",
    )
    parser.add_argument(
        "--split-output-dir",
        type=str,
        default=None,
        help="Optional directory to write subject split assignments and metadata.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved model checkpoint for evaluation mode.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Subject split to evaluate when running checkpoint-based evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write evaluation artifacts.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=200,
        help="Number of subject-level bootstrap resamples for the evaluation summary.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for subject-level bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the canonical seven wearable feature-group ablation analysis.",
    )
    parser.add_argument(
        "--split-map",
        type=str,
        default=None,
        help="Optional JSON file containing subject split assignments to reuse.",
    )
    return parser


def validate_configuration(
    dataset: str,
    model: str,
    taxonomy: str,
    mode: str,
    sensor_set: str = "wearable",
) -> dict[str, object]:
    """Validate the requested dataset, model, taxonomy, mode, and sensor set."""
    repo_root = resolve_repo_root()
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root not found: {repo_root}")

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported values: {', '.join(SUPPORTED_DATASETS)}."
        )

    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model}'. Supported values: {', '.join(SUPPORTED_MODELS)}."
        )

    if taxonomy not in SUPPORTED_TAXONOMIES:
        raise ValueError(
            f"Unsupported taxonomy '{taxonomy}'. "
            f"Supported values: {', '.join(SUPPORTED_TAXONOMIES)}."
        )

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported mode '{mode}'. Supported values: {', '.join(SUPPORTED_MODES)}."
        )

    if sensor_set not in SUPPORTED_SENSOR_SETS:
        raise ValueError(
            f"Unsupported sensor set '{sensor_set}'. "
            f"Supported values: {', '.join(SUPPORTED_SENSOR_SETS)}."
        )

    signals = validate_experiment_config(dataset, sensor_set, taxonomy, model)
    config = {
        "dataset": dataset,
        "model": model,
        "taxonomy": taxonomy,
        "mode": mode,
        "sensor_set": sensor_set,
        "repo_root": repo_root,
        "signals": signals,
    }

    logger.info(
        "Validated configuration: dataset=%s model=%s taxonomy=%s "
        "mode=%s sensor_set=%s repo_root=%s signal_count=%s",
        dataset,
        model,
        taxonomy,
        mode,
        sensor_set,
        repo_root,
        len(signals),
    )
    return config


def save_experiment_metadata(
    output_dir: str | Path,
    *,
    dataset: str,
    model: str,
    taxonomy: str,
    sensor_set: str,
    feature_group: str,
    seed: int,
    artifact: str,
    split_map: dict[str, list[str]],
    summary: dict[str, dict[str, object]],
    checkpoint_path: str | Path,
) -> Path:
    """Persist experiment metadata alongside split assignments."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "model": model,
        "taxonomy": taxonomy,
        "sensor_set": sensor_set,
        "feature_group": feature_group,
        "seed": seed,
        "artifact": artifact,
        "split_map": split_map,
        "summary": summary,
        "checkpoint_path": str(checkpoint_path),
    }
    metadata_path = output_path / "experiment_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def _load_split_map(artifact: pd.DataFrame, args: argparse.Namespace) -> dict[str, list[str]]:
    """Load a split map from a file if provided, otherwise derive a reproducible split."""
    if args.split_map is not None:
        split_path = Path(args.split_map).expanduser().resolve()
        if not split_path.exists():
            raise FileNotFoundError(f"Split map not found: {split_path}")
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
            split_map = {"train": [], "val": [], "test": []}
            for row in rows:
                split_name = str(row.get("split", "")).lower()
                subject_id = str(row.get("subject_id", ""))
                if split_name in split_map and subject_id:
                    split_map[split_name].append(subject_id)
            return {key: sorted(value) for key, value in split_map.items()}
        if isinstance(payload, dict):
            return {
                str(key): sorted(str(value) for value in values)
                for key, values in payload.items()
            }
        raise ValueError(
            f"Unsupported split-map schema in {split_path}: {type(payload)!r}"
        )
    return build_subject_splits(artifact, seed=args.seed)


def _run_single_checkpoint_eval(args: argparse.Namespace, artifact: pd.DataFrame) -> int:
    """Evaluate a saved checkpoint on a selected subject split."""
    if args.checkpoint is None:
        raise ValueError("Checkpoint-based evaluation requires --checkpoint.")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    split_map = _load_split_map(artifact, args)
    selected_subjects = sorted({str(subject_id) for subject_id in split_map.get(args.split, [])})
    if not selected_subjects:
        raise ValueError(f"Split '{args.split}' is empty; no subjects available for evaluation.")

    split_frame = artifact[artifact["subject_id"].astype(str).isin(selected_subjects)].copy()
    model = SleepStageModel.load_checkpoint(checkpoint_path)
    feature_columns = list(getattr(model, "feature_columns", []) or [])
    if not feature_columns:
        feature_columns = [
            column for column in split_frame.columns if str(column).startswith("feature_")
        ]
    if not feature_columns:
        raise ValueError(
            "The checkpoint does not record usable feature columns for evaluation."
        )
    if args.feature_group is not None:
        selected_columns = [col for col in feature_columns if col in split_frame.columns]
        if not selected_columns:
            raise ValueError(
                "Checkpoint feature columns "
                f"{feature_columns} are not present in the loaded artifact."
            )
        split_frame = split_frame.loc[
            :,
            list(dict.fromkeys(selected_columns + ["subject_id", "reference_stage"])),
        ]
    feature_frame = (
        split_frame[feature_columns].copy() if feature_columns else split_frame.copy()
    )
    predictions = np.asarray(model.predict(feature_frame), dtype=object)
    metrics = evaluate_predictions(
        split_frame["reference_stage"].tolist(),
        predictions,
        taxonomy=args.taxonomy,
    )
    per_subject = compute_subject_level_metrics(
        split_frame,
        predictions,
        taxonomy=args.taxonomy,
    )
    bootstrap_rows = []
    for metric_name in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        score_map = {
            str(subject_id): float(row[metric_name])
            for subject_id, row in per_subject.set_index("subject_id").iterrows()
        }
        bootstrap_result = bootstrap_subject_metrics(
            list(score_map.keys()),
            score_map,
            n_resamples=max(1, int(args.bootstrap_repetitions)),
            seed=int(args.bootstrap_seed),
        )
        bootstrap_rows.append(
            {
                "metric": metric_name,
                "n_resamples": bootstrap_result["n_resamples"],
                "seed": bootstrap_result["seed"],
                "subject_count": bootstrap_result["subject_count"],
                "bootstrap_mean": bootstrap_result["bootstrap_mean"],
                "ci_lower_05": bootstrap_result["ci_lower_05"],
                "ci_upper_95": bootstrap_result["ci_upper_95"],
            }
        )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("results") / "experiments" / (
            f"{args.dataset}_{args.model}_{args.taxonomy}_{args.split}"
        )
    )
    ablation_summary = pd.DataFrame([
        {
            "feature_group": args.feature_group or "checkpoint",
            "model_name": args.model,
            "split": args.split,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "cohens_kappa": metrics["cohens_kappa"],
            "n_subjects": len(selected_subjects),
            "n_epochs": int(len(split_frame)),
        }
    ])

    metadata = {
        "dataset": args.dataset,
        "model_name": args.model,
        "taxonomy": args.taxonomy,
        "sensor_set": args.sensor_set,
        "feature_group": args.feature_group or "checkpoint",
        "split": args.split,
        "split_subjects": selected_subjects,
        "seed": args.seed,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "bootstrap_seed": int(args.bootstrap_seed),
        "checkpoint_path": str(checkpoint_path),
        "artifact_path": str(Path(args.artifact).expanduser().resolve()),
        "subject_count": len(selected_subjects),
        "epoch_count": int(len(split_frame)),
    }
    write_evaluation_outputs(
        output_dir,
        metrics=metrics,
        per_subject=per_subject,
        confusion=np.asarray(metrics["confusion_matrix"], dtype=int),
        normalized_confusion=np.asarray(metrics["normalized_confusion_matrix"], dtype=float),
        labels=metrics["labels"],
        metadata=metadata,
        ablation_summary=ablation_summary,
        bootstrap_summary=bootstrap_frame,
    )
    write_optional_visuals(
        output_dir,
        np.asarray(metrics["confusion_matrix"], dtype=int),
        metrics["labels"],
        ablation_summary,
    )
    logger.info(
        "Evaluation complete for model=%s split=%s feature_group=%s output_dir=%s",
        args.model,
        args.split,
        args.feature_group or "checkpoint",
        output_dir,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI. Returns 0 on successful validation or training."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    set_deterministic_seed(args.seed)

    try:
        validate_configuration(
            args.dataset,
            args.model,
            args.taxonomy,
            args.mode,
            sensor_set=args.sensor_set,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(status=2, message=f"Configuration error: {exc}\n")

    if args.mode == "validate":
        logger.info("Configuration validation succeeded; no training run was executed.")
        return 0

    if args.mode in {"train", "eval"}:
        if args.artifact is None:
            raise SystemExit(
                "Training and evaluation workflows are unavailable without an explicit "
                "--artifact path; use --mode validate to check configuration only."
            )

        artifact = load_preprocessed_artifact(args.artifact)
        if args.mode == "eval":
            return _run_single_checkpoint_eval(args, artifact)

        split_map = _load_split_map(artifact, args)
        if args.split_output_dir is not None:
            save_split_assignments(split_map, args.split_output_dir)
        summary = compute_partition_summary(artifact, split_map)
        model = train_model(
            artifact,
            args.model,
            feature_group=args.feature_group,
            label_column="reference_stage",
            seed=args.seed,
        )
        checkpoint_path = (
            Path(args.artifact).resolve().parent / f"{args.model}_checkpoint.json"
        )
        model.save_checkpoint(checkpoint_path)
        if args.split_output_dir is not None:
            save_experiment_metadata(
                args.split_output_dir,
                dataset=args.dataset,
                model=args.model,
                taxonomy=args.taxonomy,
                sensor_set=args.sensor_set,
                feature_group=args.feature_group,
                seed=args.seed,
                artifact=str(Path(args.artifact).resolve()),
                split_map=split_map,
                summary=summary,
                checkpoint_path=checkpoint_path,
            )
        logger.info(
            "Training complete for model=%s feature_group=%s summary=%s checkpoint=%s",
            args.model,
            args.feature_group,
            summary,
            checkpoint_path,
        )
        return 0

    parser.error(f"Unsupported mode requested: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
