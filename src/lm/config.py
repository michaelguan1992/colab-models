from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("LM_CONFIG", "~/.config/lm/config.toml")).expanduser()
STATE_DIR = Path(os.environ.get("LM_STATE_DIR", "~/.local/state/lm")).expanduser()
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    session: str = "localmodels"
    gpu: str = "G4"
    high_mem: bool = True
    default_model: str = "inkling"
    local_port: int = 9000
    remote_port: int = 9000
    remote_dir: str = "/content/lm/app"
    ssh_user: str = "root"
    ssh_extra: list[str] = field(default_factory=list)
    colab_bin: str = "colab"
    remote_src: str = str(REPO_ROOT / "remote")
    dsh_home: str = os.environ.get("DSH_HOME", "~/.dsh")
    dsh_profile: str = "web"
    outputs_dir: str = "outputs"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"


def load() -> Config:
    cfg = Config()
    if CONFIG_PATH.exists():
        data = tomllib.loads(CONFIG_PATH.read_text())
        known = {f.name for f in fields(Config)}
        for k, v in data.items():
            if k in known:
                setattr(cfg, k, v)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return cfg
