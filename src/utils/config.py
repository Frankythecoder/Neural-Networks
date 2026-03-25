from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(
    path: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML config and apply dot-notation overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            _set_nested(config, key.split("."), value)

    return config


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dict using a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
