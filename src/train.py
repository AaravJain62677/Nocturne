"""CLI entry point for the sleep-staging research project."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.config import (
    DEFAULT_SEED,
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
    SUPPORTED_MODES,
    SUPPORTED_SENSOR_SETS,
    SUPPORTED_TAXONOMIES,
    resolve_repo_root,
)
from src.experiment import validate_experiment_config
from src.models import (
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
        split_map = build_subject_splits(artifact, seed=args.seed)
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
