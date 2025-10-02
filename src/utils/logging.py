# -*- coding: utf-8 -*-
"""
Simple logging setup that logs to both console and an optional file.
"""

from __future__ import annotations
import logging
import os
from logging import Logger
from logging.handlers import RotatingFileHandler
from typing import Optional


_DEF_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None, *, overwrite: bool = False) -> None:
    logging.basicConfig(level=level, format=_DEF_FORMAT, datefmt=_DATEFMT)
    logger = logging.getLogger()
    logger.handlers.clear()  # reset basicConfig handlers

    # console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(_DEF_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(ch)

    # file
    if log_file:
        if overwrite and os.path.exists(log_file):
            try:
                os.remove(log_file)
            except Exception:
                pass
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_DEF_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(fh)


def get_logger(name: str | None = None) -> Logger:
    return logging.getLogger(name if name else __name__)
