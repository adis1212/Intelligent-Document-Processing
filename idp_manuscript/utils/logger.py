"""
Logging Utility
Sets up structured logging for the IDP pipeline.
"""

import os
import logging
from utils.config_loader import load_config


def setup_logger(name="idp_manuscript"):
    """Setup and return a configured logger."""
    try:
        config = load_config()
        log_level = config.get("logging", {}).get("level", "INFO")
        log_file = config.get("logging", {}).get("log_file", "idp_manuscript.log")
    except Exception:
        log_level = "INFO"
        log_file = "idp_manuscript.log"

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level, logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")

    return logger
