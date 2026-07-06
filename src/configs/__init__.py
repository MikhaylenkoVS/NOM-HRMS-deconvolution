"""Single import point for project-wide configuration.

Typical usage::

    from src.configs import CHEM, PIPELINE, PATHS

    delta = CHEM.derivatization_shifts["delta_cd3"]
    ppm = PIPELINE.run_pipeline_defaults["ppm_tol"]

Use :func:`load_config` for dynamic access by name.
"""
from src.configs.loader import (
    CHEM,
    CONFIG_DIR,
    ConfigError,
    ConfigNamespace,
    PATHS,
    PIPELINE,
    load_config,
)

__all__ = [
    "CHEM",
    "PIPELINE",
    "PATHS",
    "load_config",
    "ConfigNamespace",
    "ConfigError",
    "CONFIG_DIR",
]
