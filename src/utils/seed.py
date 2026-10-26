"""Deterministic seeding helpers for Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
from typing import Final

import numpy as np
import torch

DEFAULT_SEED: Final[int] = 42


def set_deterministic_seed(seed: int = DEFAULT_SEED) -> int:
    """Set Python, NumPy, and PyTorch seeds for reproducible execution.

    The function does not guarantee identical results across all hardware or
    library configurations, but it provides the best available deterministic
    baseline for local reproducible runs.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed!r}.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import random

        random.seed(seed)
    except Exception:
        pass

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    except Exception:
        pass

    return seed
