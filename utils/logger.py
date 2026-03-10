"""
Klassify – Structured logging utility.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a consistently configured logger.

    Args:
        name:  Logger name – typically ``__name__`` of the calling module.
        level: Override log level (INFO/DEBUG/WARNING/ERROR).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    from configs.settings import settings  # lazy to avoid circular imports

    effective_level = level or settings.LOG_LEVEL

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(effective_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(effective_level.upper())

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
