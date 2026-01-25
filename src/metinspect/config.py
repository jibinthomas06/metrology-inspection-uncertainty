from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @property
    def mvtec_dir(self) -> Path:
        return Path(self.raw["paths"]["mvtec_dir"])


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return Config(raw=raw)
