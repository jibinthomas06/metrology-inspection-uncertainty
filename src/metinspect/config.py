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

    @property
    def reports_dir(self) -> Path:
        return Path(self.raw["paths"]["reports_dir"])

    @property
    def category(self) -> str:
        return str(self.raw["mvtec"]["category"])

    @property
    def image_size(self) -> int:
        return int(self.raw["mvtec"]["image_size"])

    @property
    def device(self) -> str:
        return str(self.raw["runtime"]["device"])

    @property
    def seed(self) -> int:
        return int(self.raw["runtime"]["seed"])


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return Config(raw=raw)
