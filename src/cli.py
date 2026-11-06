from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_section(title: str) -> None:
    print()
    print(title)
    print("─" * len(title))


def render_results(results_dir: Path) -> None:
    summary_path = results_dir / "pipeline_summary.json"

    if not summary_path.exists():
        raise SystemExit(
            f"Missing pipeline_summary.json in results directory: {results_dir}"
        )

    summary = load_json(summary_path)
    metrics = summary.get("metrics", {})

    print("NOCTURNE")
    print("Reproducible wearable sleep-stage evaluation")

    print_section("Experiment")
    print(f"  Dataset:        {summary.get('dataset', 'N/A').upper()}")
    print(f"  Model:          {summary.get('model', 'N/A')}")
    print(f"  Taxonomy:       {summary.get('taxonomy', 'N/A')}")
    print(f"  Sensor set:     {summary.get('sensor_set', 'N/A')}")
    print(f"  Feature group:  {summary.get('feature_group', 'N/A')}")
    print(f"  Seed:           {summary.get('seed', 'N/A')}")
    print(f"  Split:          {summary.get('selected_split', 'N/A')}")

    print_section("Metrics")

    metric_values = [
        ("Accuracy", metrics.get("accuracy")),
        ("Balanced Accuracy", metrics.get("balanced_accuracy")),
        ("Macro F1", metrics.get("macro_f1")),
        ("Weighted F1", metrics.get("weighted_f1")),
        ("Cohen's Kappa", metrics.get("cohens_kappa")),
    ]

    for label, value in metric_values:
        if value is None:
            print(f"  {label:<20} N/A")
        else:
            print(f"  {label:<20} {value:.4f}")

    print_section("Evaluation")

    partition_summary = summary.get("partition_summary", {})
    test_partition = partition_summary.get("test", {})

    print(
        f"  Test subjects:       "
        f"{test_partition.get('subjects', 'N/A')}"
    )
    print(
        f"  Test epochs:         "
        f"{test_partition.get('epochs', 'N/A')}"
    )
    print(
        f"  Test recordings:     "
        f"{test_partition.get('recordings', 'N/A')}"
    )

    train_subjects = set(summary.get("split_map", {}).get("train", []))
    val_subjects = set(summary.get("split_map", {}).get("val", []))
    test_subjects = set(summary.get("split_map", {}).get("test", []))

    leakage = (
        not (train_subjects & test_subjects)
        and not (val_subjects & test_subjects)
        and not (train_subjects & val_subjects)
    )

    print(
        f"  Subject independence: "
        f"{'PASS' if leakage else 'FAIL'}"
    )

    labels = metrics.get("labels", [])
    support = metrics.get("support", {})
    covered = all(label in support for label in labels)

    print(
        f"  Class coverage:       "
        f"{'PASS' if covered else 'FAIL'}"
    )

    print_section("Artifacts")

    artifact_files = [
        ("Metrics", "metrics.json"),
        ("Confusion matrix", "confusion_matrix.csv"),
        ("Normalized confusion", "normalized_confusion_matrix.csv"),
        ("Per-subject metrics", "per_subject_metrics.csv"),
        ("Ablation summary", "ablation_summary.csv"),
        ("Evaluation metadata", "evaluation_metadata.json"),
    ]

    for label, filename in artifact_files:
        status = "FOUND" if (results_dir / filename).exists() else "MISSING"
        print(f"  {label:<25} {status}")

    disclaimer = summary.get("synthetic_data_disclaimer")

    if disclaimer:
        print()
        print("!" * 64)
        print(f"  {disclaimer}")
        print("  These metrics are pipeline-validation results only.")
        print("  They are NOT real DREAMT benchmark results.")
        print("!" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Render Nocturne evaluation results.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    results_parser = subparsers.add_parser(
        "results",
        help="Display an evaluation result directory.",
    )
    results_parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the results directory.",
    )

    args = parser.parse_args()

    if args.command == "results":
        render_results(args.results_dir)


if __name__ == "__main__":
    main()