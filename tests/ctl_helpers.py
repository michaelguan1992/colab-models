import tempfile
import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from controller.app import create_app
from controller.backends.base import Backend
from controller.gpu import GpuSnapshot
from controller.manager import Manager
from controller.registry import Registry

REGISTRY_YAML = Path(__file__).parents[1] / "remote" / "models.yaml"


class FakeBackend(Backend):
    vram = {"inkling": 17.5, "qwen": 48.0, "pika": 68.0}
    instances: list = []

    def __init__(self, model_id, cfg, registry):
        super().__init__(model_id, cfg, registry)
        self.kind = cfg["kind"]
        self.upstream_model = f"upstream-{model_id}"
        self.healthy = True
        self.release = True
        self.checks_until_free = 0
        FakeBackend.instances.append(self)

    async def start(self):
        self.started_at = self.ready_at = time.time()
        self.state = "ready"

    async def stop(self):
        self.state = "stopped"

    def released(self):
        if self.checks_until_free > 0:
            self.checks_until_free -= 1
            return False
        return self.release

    async def check_health(self, client):
        return self.healthy

    @property
    def base_url(self):
        return "http://upstream.invalid"

    def vram_gb(self, per_pid_gb):
        return self.vram[self.id]


class FakeReaper:
    def __init__(self):
        self.calls = []

    async def stop_strata(self, port):
        self.calls.append(("strata", port))

    async def kill_pids(self, pids):
        self.calls.append(("pids", tuple(sorted(pids))))


def tmp_registry(raw=None):
    """models.yaml with every path under a fresh temp dir, so tests never touch /content."""
    raw = raw or yaml.safe_load(REGISTRY_YAML.read_text())
    root = Path(tempfile.mkdtemp(prefix="lm-test-"))
    raw["paths"] = {"models_dir": str(root / "models"), "outputs_dir": str(root / "out"),
                    "logs_dir": str(root / "logs"), "state_file": str(root / "state.json")}
    return Registry(raw)


def make_manager(total_gb=96, free_bytes=lambda p: 10**15, registry=None, compute_cap="", hf_check=None,
                 factory=FakeBackend, **kw):
    reg = registry or tmp_registry()
    return Manager(reg, factory=factory,
                   probe=lambda: GpuSnapshot("RTX PRO 6000", 0, total_gb, {}, True, compute_cap),
                   free_bytes=free_bytes, hf_check=hf_check or (lambda repo, rev: None),
                   reaper=FakeReaper(), **kw)


def make_client(total_gb=96, free_bytes=lambda p: 10**15, registry=None, autoload=None, **kw):
    mgr = make_manager(total_gb, free_bytes, registry, **kw)
    return TestClient(create_app(mgr, idle_poll_s=3600, autoload=autoload)), mgr


def wait_ready(client, model_id):
    for _ in range(50):
        s = client.get("/admin/status").json()
        if model_id in s["online"]:
            return s
        time.sleep(0.02)
    raise AssertionError(f"{model_id} never came online")
