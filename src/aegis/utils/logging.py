"""Structured logging for AEGIS."""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

_console = Console()
_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure rich structured logging."""
    global _configured
    if _configured:
        return
    handler = RichHandler(console=_console, show_time=True, show_path=False, markup=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("aegis")
    root.setLevel(getattr(logging, level.upper()))
    root.addHandler(handler)
    _configured = True


class StructuredLogger:
    """Logger with structured key-value output."""

    def __init__(self, name: str) -> None:
        setup_logging()
        self._logger = logging.getLogger(f"aegis.{name}")

    def _fmt(self, msg: str, **kw: Any) -> str:
        if kw:
            pairs = " ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
            return f"{msg} | {pairs}"
        return msg

    def info(self, msg: str, **kw: Any) -> None:
        self._logger.info(self._fmt(msg, **kw))

    def warning(self, msg: str, **kw: Any) -> None:
        self._logger.warning(self._fmt(msg, **kw))

    def error(self, msg: str, **kw: Any) -> None:
        self._logger.error(self._fmt(msg, **kw))

    def debug(self, msg: str, **kw: Any) -> None:
        self._logger.debug(self._fmt(msg, **kw))


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)
