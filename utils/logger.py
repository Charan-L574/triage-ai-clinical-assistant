"""
utils/logger.py — Structured rotating logger for TriageAI.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import config


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured for TriageAI."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Rotating file handler ─────────────────────────────────────────────────
    try:
        log_path = Path("triageai.log")
        fh = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # If we can't write logs to disk (e.g. read-only FS), console is enough
        pass

    logger.propagate = False
    return logger
