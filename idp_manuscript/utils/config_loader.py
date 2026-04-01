"""
Configuration Loader Utility
Loads and validates config.yaml for the IDP pipeline.
"""

import os
import yaml
import logging

logger = logging.getLogger(__name__)

_config_cache = None


def load_config(config_path=None):
    """Load configuration from YAML file with caching."""
    global _config_cache

    if _config_cache is not None and config_path is None:
        return _config_cache

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml"
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        _config_cache = config
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise


def get_config_value(section, key, default=None):
    """Get a specific config value with optional default."""
    config = load_config()
    try:
        return config[section][key]
    except KeyError:
        logger.warning(f"Config key '{section}.{key}' not found, using default: {default}")
        return default
