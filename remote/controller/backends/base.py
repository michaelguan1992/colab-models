from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import Any

import httpx

from ..gpu import process_tree


class Backend:
    """A model process on the runtime. Subclasses implement ``_launch`` and ``_ready``."""

    kind = "llm"

    def __init__(self, model_id: str, cfg: dict[str, Any], registry) -> None:
        self.id = model_id
        self.cfg = cfg
        self.registry = registry
        self.state = "pending"  # pending|downloading|starting|ready|stopping|stopped|error
        self.error: str | None = None
        self.proc: asyncio.subprocess.Process | None = None
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self.local_dir: Path = registry.models_dir / model_id
        self.log_path: Path = registry.logs_dir / f"{model_id}.log"
        self.upstream_model: str | None = None

    # ---- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        self.started_at = time.time()
        try:
            if not self.cfg.get("skip_download"):
                self.state = "downloading"
                await self.download()
            self.state = "starting"
            await self._launch()
            await self._wait_ready(timeout=float(self.cfg.get("ready_timeout_s", 1800)))
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
        self.state = "stopping"
        await self._kill()
        self.state = "stopped"

    async def _kill(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=30)
            except asyncio.TimeoutError:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    # ---- download --------------------------------------------------------
    async def _hf_download(self, repo: str, include: list[str] | None, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["hf", "download", repo, "--local-dir", str(local_dir)]
        for pat in include or []:
            cmd += ["--include", pat]
        env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
        await self._run_logged(cmd, env=env)

    async def download(self) -> None:
        await self._hf_download(self.cfg["repo"], self.cfg.get("include"), self.local_dir)

    def download_progress(self) -> float | None:
        est = self.cfg.get("est_download_gb")
        if not est or not self.local_dir.exists():
            return None
        size = sum(f.stat().st_size for f in self.local_dir.rglob("*") if f.is_file())
        return min(1.0, size / (float(est) * 1024**3))

    async def _run_logged(self, cmd: list[str], env: dict | None = None) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as log:
            log.write(f"\n$ {' '.join(cmd)}\n".encode())
            p = await asyncio.create_subprocess_exec(*cmd, stdout=log, stderr=log, env=env)
            rc = await p.wait()
        if rc != 0:
            raise RuntimeError(f"{cmd[0]} 退出码 {rc}，见 {self.log_path}")

    async def _spawn(self, cmd: list[str], env: dict | None = None) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        log.write(f"\n$ {' '.join(cmd)}\n".encode())
        log.flush()
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=log, stderr=log, env=env, start_new_session=True)

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
                await asyncio.sleep(3)
        raise TimeoutError(f"{timeout:.0f}s 内未就绪")

    async def _launch(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    async def _ready(self, client: httpx.AsyncClient) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- introspection ---------------------------------------------------
    @property
    def base_url(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

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
