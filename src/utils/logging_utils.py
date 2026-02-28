"""Logging helpers for CLI and service entrypoints."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a consistent key-value style log format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s event=%(message)s",
    )
