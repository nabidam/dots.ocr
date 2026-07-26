"""Rich console and rotating file logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from app.config import LoggingConfig


def configure_logging(level: str, config: LoggingConfig) -> None:
    """Configure detailed rotating file logs and readable Rich console logs."""

    log_path = Path(config.file)
    if not log_path.is_absolute():
        log_path = Path(__file__).resolve().parent.parent / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    root_logger.handlers.clear()

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )

    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=False,
        show_path=False,
    )
    console_handler.setLevel(level.upper())
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("uvicorn.access").setLevel(level.upper())
    logging.getLogger("uvicorn.error").setLevel(level.upper())


__all__ = ["configure_logging"]
