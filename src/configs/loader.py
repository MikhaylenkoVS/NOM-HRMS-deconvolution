"""Central configuration loader for the NOM-HRMS-deconvolution project.

All shared scientific constants and default settings live in the JSON files
next to this module (``chemistry.json``, ``pipeline.json``, ``paths.json``).
This module loads them once (cached) and exposes them through lightweight,
attribute-accessible namespaces so that every other module reads its
constants from a single source of truth.

Notes
-----
JSON files are resolved relative to this module's directory
(``Path(__file__).parent``), never relative to the current working
directory, so configuration loads identically regardless of where the
program is launched from.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent

#: Config name -> file name mapping. Keep in sync with the JSON files here.
_CONFIG_FILES = {
    "chemistry": "chemistry.json",
    "pipeline": "pipeline.json",
    "paths": "paths.json",
}


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or malformed."""


class ConfigNamespace:
    """Read-only, attribute-accessible view over a parsed JSON mapping.

    Top-level keys are exposed both as attributes (``cfg.delta_cd3``) and via
    item access (``cfg["delta_cd3"]``). Nested containers are returned as
    plain ``dict``/``list`` objects, preserving their original types so that,
    e.g., mass lookups keep working as ``cfg.monoisotopic_masses["C"]``.

    Parameters
    ----------
    name : str
        Logical config name (for error messages).
    data : dict
        Parsed JSON mapping.
    """

    __slots__ = ("_name", "_data")

    def __init__(self, name: str, data: dict[str, Any]):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_data", data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(
                f"Ключ {key!r} отсутствует в конфиге {self._name!r}. "
                f"Доступные ключи: {sorted(self._data)}"
            ) from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the underlying mapping."""
        return dict(self._data)

    def __repr__(self) -> str:
        return f"ConfigNamespace({self._name!r}, keys={sorted(self._data)})"


@lru_cache(maxsize=None)
def load_config(name: str) -> ConfigNamespace:
    """Load and cache one configuration file by logical name.

    Parameters
    ----------
    name : {"chemistry", "pipeline", "paths"}
        Logical name of the configuration to load.

    Returns
    -------
    ConfigNamespace
        Parsed configuration with attribute and item access.

    Raises
    ------
    ConfigError
        If ``name`` is unknown, the file is missing, or its contents are not
        a valid JSON object.
    """
    if name not in _CONFIG_FILES:
        raise ConfigError(
            f"Неизвестный конфиг {name!r}. Доступные: {sorted(_CONFIG_FILES)}"
        )

    path = CONFIG_DIR / _CONFIG_FILES[name]
    if not path.exists():
        raise ConfigError(f"Файл конфигурации не найден: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Некорректный JSON в {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Ожидался JSON-объект в {path}, получено {type(data).__name__}"
        )

    return ConfigNamespace(name, data)


#: Convenience singletons — the common import points used across the project.
CHEM = load_config("chemistry")
PIPELINE = load_config("pipeline")
PATHS = load_config("paths")
