from __future__ import annotations

import asyncio
import contextlib
import os
import time
from pathlib import Path
from typing import Callable

import httpx

from ..gpu import process_tree
from ..install import Installed, Installer
from ..procs import kill_group, pid_alive, port_listening
from ..registry import Registry
from ..spec import Engine, ModelSpec, Route, install_needs_model
from ..templating import engine_values, expand


class Backend:
    """A model process on the runtime. Subclasses implement ``_ready``; the engine config drives the rest."""

    ready_poll_s = 3.0

    def __init__(self, model: ModelSpec, route: Route, registry: Registry, installer: Installer) -> None:
        self.id = model.id
        self.model = model
        self.route = route
        self.engine: Engine = registry.engines[route.engine]
        self.registry = registry
        self.installer = installer
        self.installed: Installed = installer.installed(self.engine) or Installed()
        self.state = "pending"  # pending|installing|downloading|starting|ready|stopping|stopped|error
        self.error: str | None = None
        self.proc: asyncio.subprocess.Process | None = None
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self.local_dir: Path = registry.models_dir / model.id
        self.lora_dir: Path = registry.models_dir / f"{model.id}-lora"
        self.log_path: Path = registry.logs_dir / f"{model.id}.log"
        self.upstream_model: str | None = None
        self.health_failures = 0
        self.queue_depth = 0  # running + queued jobs, for backends that serve jobs
        self.recorded_pids: set[int] = set()
        self.on_launch: Callable[[Backend], None] | None = None
        self._job_pgids: set[int] = set()

    @property
    def port(self) -> int:
        return self.route.port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def values(self) -> dict[str, str]:
        r = self.route
        v = engine_values(self.registry.models_dir, self.registry.outputs_dir, self.port, self.installed.values())
        v.update(r.params)
        v["model_dir"] = str(self.local_dir)
        if r.model_path:
            v["model_path"] = str(self.local_dir / r.model_path)
        if r.lora:
            v.update(lora_dir=str(self.lora_dir), lora_file=r.lora.file)
        return v

    # ---- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        self.started_at = time.time()
        try:
            install = self.engine.install
            install_first = install is not None and not install_needs_model(install)
            if install_first:
                await self._install()
            if self.route.repo:
                self.state = "downloading"
                await self.download()
            if install is not None and not install_first:
                await self._install()
            self.state = "starting"
            await self._launch()
            if self.on_launch:
                self.on_launch(self)
            await self._wait_ready(timeout=self.route.ready_timeout_s)
            self.state = "ready"
            self.ready_at = time.time()
        except asyncio.CancelledError:
            await self.stop()
            raise
        except Exception as e:  # surfaced via /admin/status
            self.state = "error"
            self.error = f"{type(e).__name__}: {e}"
            await self._kill()

    async def stop(self) -> None:
        self.recorded_pids |= self.pids()
        self.state = "stopping"
        await self._kill()
        self.state = "stopped"

    async def _kill(self) -> None:
        groups = set(self._job_pgids)
        if self.proc and self.proc.returncode is None:
            groups.add(self.proc.pid)
        await asyncio.gather(*(kill_group(g, grace=30) for g in groups))
        if self.proc:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.proc.wait(), timeout=5)

    def released(self) -> bool:
        """The engine's port is closed and none of its recorded processes remain."""
        return not port_listening(self.port) and not any(pid_alive(p) for p in self.recorded_pids)

    async def wait_released(self, timeout: float, poll: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if self.released():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(poll)

    def busy(self) -> bool:
        return self.queue_depth > 0

    # ---- download --------------------------------------------------------
    async def _hf_download(self, repo: str, include: list[str] | None, local_dir: Path,
                           revision: str | None = None) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["hf", "download", repo, "--local-dir", str(local_dir)]
        if revision:
            cmd += ["--revision", revision]
        for pat in include or []:
            cmd += ["--include", pat]
        env = {**os.environ, "HF_XET_HIGH_PERFORMANCE": "1"}  # hf_transfer is gone in huggingface_hub 1.x
        await self._run_logged(cmd, env=env)

    async def download(self) -> None:
        r = self.route
        await self._hf_download(r.repo, list(r.include) or None, self.local_dir, r.revision)
        if r.lora:
            await self._hf_download(r.lora.repo, [r.lora.file], self.lora_dir, r.lora.revision)

    def download_progress(self) -> float | None:
        est = self.route.est_download_gb
        if not est or not self.local_dir.exists():
            return None
        size = sum(f.stat().st_size for f in self.local_dir.rglob("*") if f.is_file())
        return min(1.0, size / (est * 1024**3))

    async def _install(self) -> None:
        self.state = "installing"
        self.installed = await self.installer.ensure(self.engine, self.local_dir, self._run_logged)

    async def _run_logged(self, cmd: list[str], env: dict | None = None) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as log:
            log.write(f"\n$ {' '.join(cmd)}\n".encode())
            log.flush()
            p = await asyncio.create_subprocess_exec(
                *cmd, stdout=log, stderr=log, env=env, start_new_session=True)
            self._job_pgids.add(p.pid)
            try:
                rc = await p.wait()
            finally:
                if p.returncode is None:  # cancelled mid-download: take the whole group down
                    await asyncio.shield(kill_group(p.pid, grace=10))
                self._job_pgids.discard(p.pid)
        if rc != 0:
            raise RuntimeError(f"{cmd[0]} 退出码 {rc}，见 {self.log_path}")

    async def _spawn(self, cmd: list[str], env: dict | None = None) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        try:
            log.write(f"\n$ {' '.join(cmd)}\n".encode())
            log.flush()
            self.proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=log, stderr=log, env=env, start_new_session=True)
        finally:
            log.close()

    # ---- readiness -------------------------------------------------------
    async def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        async with httpx.AsyncClient(timeout=5) as client:
            while time.time() < deadline:
                if self.proc and self.proc.returncode not in (None, 0):
                    raise RuntimeError(f"进程退出 (code {self.proc.returncode})，见 {self.log_path}")
                try:
                    if await self._ready(client):
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(self.ready_poll_s)
        raise TimeoutError(f"{timeout:.0f}s 内未就绪")

    async def check_health(self, client: httpx.AsyncClient) -> bool:
        try:
            return await self._ready(client)
        except (httpx.HTTPError, RuntimeError, ValueError):
            return False

    async def _launch(self) -> None:
        argv = expand(self.engine.launch, self.values(), self.route.extra)
        await self._spawn(argv, env=self.installed.env(os.environ))

    async def _ready(self, client: httpx.AsyncClient) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- introspection ---------------------------------------------------
    def pids(self) -> set[int]:
        if not self.proc or self.proc.returncode is not None:
            return set()
        return process_tree(self.proc.pid)

    def vram_gb(self, per_pid_gb: dict[int, float]) -> float | None:
        pids = self.pids()
        if not pids or not per_pid_gb:
            return None
        used = sum(per_pid_gb.get(p, 0.0) for p in pids)
        return used if used > 0 else None
