"""Logging configuration for the sleep-staging project."""

from __future__ import annotations

import logging

LOGGER_NAMESPACE = "nocturne"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a module-level logger with consistent formatting."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(LOGGER_NAMESPACE)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger configured for package-level use."""
    logger_name = LOGGER_NAMESPACE if name is None else f"{LOGGER_NAMESPACE}.{name}"
    return logging.getLogger(logger_name)
