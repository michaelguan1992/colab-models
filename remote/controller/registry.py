from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .policy import Spec


@dataclass
class Registry:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return self.raw["models"]

    def model(self, model_id: str) -> dict[str, Any]:
        if model_id not in self.models:
            raise KeyError(f"未知模型 {model_id!r}，可选: {', '.join(self.models)}")
        return self.models[model_id]

    @property
    def specs(self) -> dict[str, Spec]:
        return {k: Spec(k, v["kind"], float(v["est_vram_gb"])) for k, v in self.models.items()}

    @property
    def models_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("models_dir", "/content/models"))

    @property
    def outputs_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("outputs_dir", "/content/outputs"))

    @property
    def logs_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("logs_dir", "/content/lm/logs"))

    @property
    def strata(self) -> dict[str, Any]:
        return self.raw.get("strata", {})

    @property
    def idle_unload_minutes(self) -> float:
        return float(self.raw.get("idle_unload_minutes", 45))
