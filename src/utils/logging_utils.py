"""Logging configuration for the sleep-staging project."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a module-level logger with consistent formatting."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("sleep_degradation")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger configured for package-level use."""
    logger_name = "sleep_degradation" if name is None else f"sleep_degradation.{name}"
    return logging.getLogger(logger_name)
