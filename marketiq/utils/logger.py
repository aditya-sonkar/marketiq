"""Centralized logging framework for MarketIQ pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = "marketiq.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "marketiq",
    log_dir: Path = DEFAULT_LOG_DIR,
    log_file: str = DEFAULT_LOG_FILE,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a thread-safe logger with console and file handlers.

    Args:
        name (str): Name of the logger namespace.
        log_dir (Path): Directory where log files are stored.
        log_file (str): Log filename.
        level (int): Logging severity level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize file logger at {log_dir / log_file}: {e}")

    logger.propagate = False
    return logger


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """Retrieves a logger instance under the 'marketiq' parent namespace.

    Args:
        module_name (Optional[str]): Optional sub-module identifier (e.g., 'scraper').

    Returns:
        logging.Logger: Logger instance under marketiq.<module_name>.
    """
    full_name = f"marketiq.{module_name}" if module_name else "marketiq"
    logger = logging.getLogger(full_name)
    if not logger.handlers and not logger.parent.handlers:
        setup_logger(name="marketiq")
    return logger
