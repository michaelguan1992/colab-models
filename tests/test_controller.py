import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from controller.app import create_app
from controller.backends.base import Backend
from controller.gpu import GpuSnapshot
from controller.manager import Manager
from controller.registry import Registry

REGISTRY_YAML = Path(__file__).parents[1] / "remote" / "models.yaml"


class FakeBackend(Backend):
    vram = {"inkling": 17.5, "qwen": 48.0, "pika": 68.0}

    def __init__(self, model_id, cfg, registry):
        super().__init__(model_id, cfg, registry)
        self.kind = cfg["kind"]
        self.upstream_model = f"upstream-{model_id}"

    async def start(self):
        self.state = "ready"
        self.ready_at = time.time()

    async def stop(self):
        self.state = "stopped"

    @property
    def base_url(self):
        return "http://upstream.invalid"

    def vram_gb(self, per_pid_gb):
        return self.vram[self.id]


def make_client():
    reg = Registry.load(REGISTRY_YAML)
    mgr = Manager(reg, factory=FakeBackend,
                  probe=lambda: GpuSnapshot("RTX PRO 6000", 0, 96, {}, True))
    return TestClient(create_app(mgr, idle_poll_s=3600)), mgr


def wait_ready(client, model_id):
    for _ in range(50):
        s = client.get("/admin/status").json()
        if model_id in s["online"]:
            return s
        time.sleep(0.02)
    raise AssertionError(f"{model_id} never came online")


def test_models_empty_and_proxy_503():
    client, _ = make_client()
    with client:
        assert client.get("/v1/models").json()["data"] == []
        r = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
        assert r.status_code == 503
        assert "没有在线" in r.json()["error"]["message"]


def test_load_lists_only_online_llm_and_swaps():
    client, _ = make_client()
    with client:
        assert client.post("/admin/load", json={"id": "inkling"}).json()["ok"]
        wait_ready(client, "inkling")
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["inkling"]

        r = client.post("/admin/load", json={"id": "qwen"}).json()
        assert r["ok"] and r["unloaded"] == ["inkling"]
        s = wait_ready(client, "qwen")
        assert s["online"] == ["qwen"]


def test_pika_gate():
    client, _ = make_client()
    with client:
        client.post("/admin/load", json={"id": "qwen"})
        wait_ready(client, "qwen")
        r = client.post("/admin/load", json={"id": "pika"})
        assert r.status_code == 409 and "qwen" in r.json()["reason"]

        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        assert client.post("/admin/load", json={"id": "pika"}).json()["ok"]
        s = wait_ready(client, "pika")
        assert sorted(s["online"]) == ["inkling", "pika"]
        assert "[+pika]" in client.get("/v1/models").json()["data"][0]["name"]

        # Qwen must be refused while Pika is online.
        assert client.post("/admin/load", json={"id": "qwen"}).status_code == 409


def test_unknown_model_404_and_unload_all():
    client, _ = make_client()
    with client:
        assert client.post("/admin/load", json={"id": "nope"}).status_code == 404
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        assert client.post("/admin/unload", json={}).json()["unloaded"] == ["inkling"]
        assert client.get("/admin/status").json()["online"] == []


def test_video_requires_pika():
    client, _ = make_client()
    with client:
        assert client.post("/video/generate", json={"prompt": "a cat"}).status_code == 503


def test_idle_unload():
    reg = Registry.load(REGISTRY_YAML)
    mgr = Manager(reg, factory=FakeBackend, probe=lambda: GpuSnapshot())

    async def run():
        await mgr.load("inkling")
        await asyncio.sleep(0)
        mgr.last_activity -= reg.idle_unload_minutes * 60 + 1
        await mgr.idle_check()

    asyncio.run(run())
    assert mgr.backends == {} and mgr.idle_unloaded


def test_files_rejects_traversal():
    client, _ = make_client()
    with client:
        assert client.get("/files/..%2Fetc%2Fpasswd").status_code in (400, 404)
