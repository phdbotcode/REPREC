"""Lightweight logging setup for the project."""

import logging
import sys
from typing import Optional


_CONFIGURED = False


def get_logger(
    name: str = "llm4rec",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a logger with console (and optional file) handlers.

    Calling this multiple times with the same *name* returns the same
    logger instance; handlers are only attached once.

    Parameters
    ----------
    name : logger name (dot-separated hierarchy is respected).
    level : logging level (default ``INFO``).
    log_file : if given, also write logs to this file path.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Already set up – just return
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (optional)
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    _CONFIGURED = True
    return logger
