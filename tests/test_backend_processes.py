import asyncio
import os
import socket
import time

import pytest

from controller.backends import base as base_mod
from controller.backends.base import Backend
from controller.backends.strata import served_model_matches
from controller.procs import pid_alive
from ctl_helpers import tmp_registry


class SleepBackend(Backend):
    """download() runs `sh` with a background `sleep` child, like `hf download` with its workers."""

    def __init__(self, model_id, cfg, registry, pidfile):
        super().__init__(model_id, cfg, registry)
        self.pidfile = pidfile

    async def download(self):
        await self._run_logged(["sh", "-c", f"sleep 30 & echo $! > {self.pidfile}; wait"])

    async def _launch(self):
        await self._spawn(["sh", "-c", "sleep 30"])

    async def _ready(self, client):
        return False


def make(tmp_path, port=18999):
    reg = tmp_registry()
    return SleepBackend("inkling", {"port": port}, reg, tmp_path / "child.pid")


async def child_pid(b):
    for _ in range(100):
        if b.pidfile.exists() and b.pidfile.read_text().strip():
            return int(b.pidfile.read_text())
        await asyncio.sleep(0.05)
    raise AssertionError("download never started")


async def gone(pid, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


def test_stop_kills_the_downloads_whole_process_group(tmp_path):
    async def run():
        b = make(tmp_path)
        task = asyncio.create_task(b.start())
        pid = await child_pid(b)
        assert pid_alive(pid)
        await b.stop()
        assert await gone(pid), "hf-style child survived stop()"
        await task
    asyncio.run(run())


def test_cancelling_the_start_task_kills_the_download_child(tmp_path):
    async def run():
        b = make(tmp_path)
        task = asyncio.create_task(b.start())
        pid = await child_pid(b)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await gone(pid), "download child survived cancellation"
        assert b.state == "stopped" and not b._download_pgids
    asyncio.run(run())


def test_spawn_closes_the_log_file_handle(tmp_path, monkeypatch):
    opened = []
    real_open = open

    def spy(*a, **kw):
        f = real_open(*a, **kw)
        opened.append(f)
        return f

    monkeypatch.setattr(base_mod, "open", spy, raising=False)

    async def run():
        b = make(tmp_path)
        await b._spawn(["true"])
        await b.proc.wait()
    asyncio.run(run())
    assert opened and all(f.closed for f in opened)


def test_wait_released_sees_port_and_pids(tmp_path):
    async def run():
        b = make(tmp_path)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            b.cfg["port"] = s.getsockname()[1]
            assert not b.released()
            assert not await b.wait_released(0.3, poll=0.05)
        assert b.released()
        b.recorded_pids = {os.getpid()}
        assert not b.released()
        b.recorded_pids = set()
        assert await b.wait_released(1, poll=0.05)
    asyncio.run(run())


OTHERS = {"Inkling-2-Mini-Uncensored-AD-Q4_K.gguf", "IQ3_XXS"}


def test_strata_identity_rejects_a_leftover_server_for_another_model():
    old = [{"id": "Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"}]
    assert served_model_matches(old, "IQ3_XXS", OTHERS) == (False, None)
    assert served_model_matches(
        [{"id": "x", "path": "/content/models/inkling/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"}],
        "IQ3_XXS", OTHERS) == (False, None)


def test_strata_identity_accepts_the_expected_model_and_opaque_ids():
    ok = [{"id": "Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"}]
    assert served_model_matches(ok, "Inkling-2-Mini-Uncensored-AD-Q4_K.gguf", OTHERS) == (True, ok[0]["id"])
    assert served_model_matches([{"id": "default"}], "IQ3_XXS", OTHERS) == (True, "default")
    assert served_model_matches([], "IQ3_XXS", OTHERS) == (True, None)
