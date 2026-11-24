import tempfile
import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from controller.app import create_app
from controller.backends.base import Backend
from controller.gpu import GpuSnapshot
from controller.install import Installer
from controller.manager import Manager
from controller.registry import Registry

REGISTRY_YAML = Path(__file__).parents[1] / "remote" / "models.yaml"


async def forbidden_fetch(url, dest):
    raise AssertionError(f"test tried to download {url}")


def offline_installer(reg):
    return Installer(reg, fetch=forbidden_fetch)


class FakeBackend(Backend):
    vram = {"inkling": 17.5, "qwen": 48.0, "qwen-tiny": 1.5, "pika": 68.0}
    instances: list = []

    def __init__(self, model, route, registry, installer):
        super().__init__(model, route, registry, installer)
        self.upstream_model = f"upstream-{model.id}"
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
        return f"http://{self.id}.invalid"

    def vram_gb(self, per_pid_gb):
        return self.vram.get(self.id, self.route.vram_gb)


class FakeReaper:
    def __init__(self):
        self.calls = []

    async def stop_engine(self, engine, port):
        self.calls.append((engine.name, port))

    async def kill_pids(self, pids):
        self.calls.append(("pids", tuple(sorted(pids))))


def tmp_registry(raw=None):
    """models.yaml with every path under a fresh temp dir, so tests never touch /content."""
    raw = raw or yaml.safe_load(REGISTRY_YAML.read_text())
    root = Path(tempfile.mkdtemp(prefix="lm-test-"))
    raw["paths"] = {"models_dir": str(root / "models"), "outputs_dir": str(root / "out"),
                    "logs_dir": str(root / "logs"), "state_file": str(root / "state.json"),
                    "engines_dir": str(root / "engines"), "venvs_dir": str(root / "venvs"),
                    "holds_dir": str(root / "holds")}
    return Registry(raw)


def make_manager(total_gb=96, free_bytes=lambda p: 10**15, registry=None, compute_cap="", hf_check=None,
                 factory=FakeBackend, **kw):
    reg = registry or tmp_registry()
    kw.setdefault("installer", offline_installer(reg))
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


def build(cls, model_id="inkling", registry=None, installer=None, compute_cap="", route=None, **kw):
    """The route the controller would pick on this GPU (default: unknown GPU, so no sm-gated route)."""
    reg = registry or tmp_registry()
    model = reg.model(model_id)
    return cls(model, reg.choose(model_id, compute_cap, route).route, reg, installer or offline_installer(reg), **kw)
