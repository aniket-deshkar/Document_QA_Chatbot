"""File-based logging helpers for application-wide and per-user logs."""

from __future__ import annotations

import logging
import os
import re
from threading import Lock

_LOG_LOCK = Lock()


def _safe_user_filename(user_id: str) -> str:
    """Convert user id into a filesystem-safe log filename."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (user_id or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown_user"


def _has_file_handler(logger: logging.Logger, file_path: str) -> bool:
    """Check whether logger already has a file handler attached for the given path."""
    expected = os.path.abspath(file_path)
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        base_name = os.path.abspath(getattr(handler, "baseFilename", ""))
        if base_name == expected:
            return True
    return False


def _build_formatter() -> logging.Formatter:
    """Create the standard log line formatter used by app and user log files."""
    return logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def configure_app_file_logging(log_dir: str, logger_name: str = "main") -> logging.Logger:
    """Configure append-only application log file and return the logger."""
    os.makedirs(log_dir, exist_ok=True)
    app_log_path = os.path.join(log_dir, "application.log")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    with _LOG_LOCK:
        if not _has_file_handler(logger, app_log_path):
            file_handler = logging.FileHandler(app_log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(_build_formatter())
            logger.addHandler(file_handler)

    return logger


def get_user_logger(user_id: str, log_dir: str) -> logging.Logger:
    """Create or reuse a dedicated append-only logger for a specific user."""
    user_logs_dir = os.path.join(log_dir, "users")
    os.makedirs(user_logs_dir, exist_ok=True)

    safe_user = _safe_user_filename(user_id)
    user_log_path = os.path.join(user_logs_dir, f"{safe_user}.log")
    logger = logging.getLogger(f"user.{safe_user}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    with _LOG_LOCK:
        if not _has_file_handler(logger, user_log_path):
            file_handler = logging.FileHandler(user_log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(_build_formatter())
            logger.addHandler(file_handler)

    return logger


def log_user_event(
    user_id: str,
    log_dir: str,
    event: str,
    level: int = logging.INFO,
    **fields,
):
    """Write a structured user-scoped event line into the user's log file."""
    logger = get_user_logger(user_id=user_id, log_dir=log_dir)
    cleaned_fields = {k: v for k, v in fields.items() if v is not None}
    if cleaned_fields:
        meta = " ".join(f"{k}={cleaned_fields[k]}" for k in sorted(cleaned_fields))
        message = f"{event} | {meta}"
    else:
        message = event
    logger.log(level, message)
