"""Structured logging setup for the sentiment dashboard."""

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    return logger
