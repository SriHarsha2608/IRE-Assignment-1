from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return _REPO_ROOT


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or _REPO_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def raw_dir(config: dict[str, Any]) -> Path:
    return repo_root() / config["paths"]["raw_dir"]


def processed_dir(config: dict[str, Any]) -> Path:
    return repo_root() / config["paths"]["processed_dir"]


def temp_dir(config: dict[str, Any]) -> Path:
    return repo_root() / config["paths"]["temp_dir"]