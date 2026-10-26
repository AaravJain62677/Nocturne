import subprocess
import sys
from pathlib import Path

import pytest
from src.train import main


def test_cli_help_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--help"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_validation_mode_returns_zero() -> None:
    exit_code = main([
        "--dataset",
        "dreamt",
        "--model",
        "baseline",
        "--mode",
        "validate",
        "--taxonomy",
        "standard",
    ])
    assert exit_code == 0


def test_unimplemented_training_mode_exits_cleanly() -> None:
    with pytest.raises(SystemExit, match="unavailable|not implemented|unsupported"):
        main([
            "--dataset",
            "dreamt",
            "--model",
            "baseline",
            "--mode",
            "train",
        ])
