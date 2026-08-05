"""Logging setup — console + rotating file, structured for grep-ability."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    logger.remove()  # drop loguru's default handler before adding ours

    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        log_dir / "llm_extract_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="50 MB",
        retention="30 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    )
