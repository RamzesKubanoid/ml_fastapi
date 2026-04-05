"""
log_control.py — centralised logging configuration for the churn service.

All modules obtain their logger via get_logger(__name__). This gives each
module its own named logger while sharing the same format, level, and
handlers configured here.

Log format is structured text that is both human-readable locally and
easily parsed by log aggregators in production. Each line contains:

    LEVEL     | ISO-8601 timestamp | logger name | message
"""
import logging
import sys


# ── Format ───────────────────────────────────────────────────────────────────

_FORMAT = "%(levelname)-8s | %(asctime)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


# ── Root logger setup (runs once on first import) ────────────────────────────

def _configure_root_logger() -> None:
    """
    Configures the root logger exactly once.

    - Streams to stdout so Docker/container runtimes capture logs without
      any extra configuration (stdout is the standard log sink for containers).
    - Uses WARNING as the root level so noisy third-party libraries
      (sklearn, uvicorn internals, etc.) stay quiet by default.
    - The churn service's own loggers are set to INFO below.
    """
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. pytest captures logging itself)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    root.setLevel(logging.WARNING)   # suppress noisy third-party libs
    root.addHandler(handler)


_configure_root_logger()


# ── Public API ───────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a named logger for the given module.

    Sets the logger's own level to INFO (or whatever is passed) so that
    churn-service messages are always captured, independently of the root
    logger's WARNING threshold.

    Args:
        name:  Typically __name__ of the calling module.
        level: Log level for this logger (default INFO).

    Returns:
        Configured logging.Logger instance.

    Example:
        log = get_logger(__name__)
        log.info("Dataset loaded: %d rows", len(df))
        log.warning("No model found on disk — service starts untrained")
        log.error("Prediction failed: %s", exc)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
