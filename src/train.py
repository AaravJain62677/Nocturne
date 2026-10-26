"""CLI entry point for the sleep-staging research project.

This Phase 1 foundation validates configuration and exposes unsupported workflows
with explicit, user-facing errors instead of silently constructing a training run.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI.

    Returns 0 when validation mode succeeds. Unsupported workflows raise
    ``SystemExit`` with a clear explanation so the CLI fails gracefully rather than
    performing hidden or invalid work.
    """
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
        raise SystemExit(
            "Training and evaluation workflows are unavailable in Phase 1; "
            "use --mode validate to check configuration only."
        )

    parser.error(f"Unsupported mode requested: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
