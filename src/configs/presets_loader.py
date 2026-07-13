"""Загрузчик пресетов параметров (soil, water, peat, coal)."""

from __future__ import annotations

import json
import os
from typing import Any

_PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")


def list_presets() -> list[dict[str, Any]]:
    """Вернуть список доступных пресетов как list of dict."""
    presets: list[dict[str, Any]] = []
    if not os.path.isdir(_PRESETS_DIR):
        return presets
    for fname in sorted(os.listdir(_PRESETS_DIR)):
        if fname.endswith(".json"):
            path = os.path.join(_PRESETS_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    preset = json.load(f)
                preset["id"] = fname.replace(".json", "")
                presets.append(preset)
            except Exception:
                pass
    return presets


def load_preset(preset_id: str) -> dict[str, Any] | None:
    """Загрузить один пресет по идентификатору (имени файла без .json)."""
    path = os.path.join(_PRESETS_DIR, f"{preset_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
