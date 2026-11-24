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
    # Tried in order on every `lm up`. G4 has no high-mem shape (HTTP 400; its standard shape
    # already has ~185GB RAM). A100 is only 80GB with high_mem. H100 is left out on purpose:
    # community reports (2026-11) show it returning 503 almost always.
    accelerators: list[dict] = field(default_factory=lambda: [
        {"gpu": "G4"},
        {"gpu": "A100", "high_mem": True},
    ])
    min_vram_gb: float = 75.0          # assigned GPU smaller than this (e.g. A100 40GB) -> stop it, try next
    assign_retry_minutes: float = 12.0  # wait between rounds with `lm up --wait` (+/-20% jitter)
    assign_daily_limit: int = 12        # max assignment requests per day across all `lm up` runs
    idle_stop_minutes: float = 60.0     # watchdog runs `lm down` after this long idle; 0 = off
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
    claude_bin: str = "claude"
    outputs_dir: str = "outputs"
    hf_token_env: str = "HF_TOKEN"     # Mac env var whose value deploy copies to the runtime's HF token file

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
        if "gpu" in data and "accelerators" not in data:  # old single-GPU config
            cfg.accelerators = [{"gpu": data["gpu"], "high_mem": bool(data.get("high_mem", False))}]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return cfg
