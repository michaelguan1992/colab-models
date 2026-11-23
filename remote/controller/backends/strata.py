from __future__ import annotations

import asyncio
import os

import httpx

from ..gpu import process_tree
from .base import Backend


def _runtime_root(pid: int) -> int:
    """Walk up from the port listener to the topmost strata process (the daemonised runtime)."""
    try:
        import psutil
        proc = psutil.Process(pid)
        for parent in proc.parents():
            if "strata" not in " ".join(parent.cmdline()):
                break
            proc = parent
        return proc.pid
    except Exception:
        return pid


def listener_pid(port: int) -> int | None:
    try:
        import psutil
        for c in psutil.net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
                return c.pid
    except Exception:
        pass
    return None


class StrataBackend(Backend):
    """One LLM served by Strata on a local port (OpenAI + Anthropic compatible)."""

    kind = "llm"

    @property
    def port(self) -> int:
        return int(self.cfg.get("port") or self.registry.strata.get("port", 8081))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _fmt(self, args: list[str]) -> list[str]:
        model_path = str(self.local_dir / self.cfg["model_path"])
        return [a.format(model_path=model_path, port=self.port) for a in args]

    async def _launch(self) -> None:
        s = self.registry.strata
        cmd = [s.get("bin", "strata"), *self._fmt(s["launch"]), *self.cfg.get("extra", [])]
        await self._spawn(cmd, env={**os.environ})

    async def _ready(self, client: httpx.AsyncClient) -> bool:
        r = await client.get(f"{self.base_url}/v1/models")
        if r.status_code != 200:
            return False
        data = r.json().get("data") or []
        if data:
            self.upstream_model = data[0].get("id")
        return True

    def pids(self) -> set[int]:
        # `strata launch` daemonises (runtime -> serve.server -> engine, re-parented to init) and
        # exits, so our own child pid says nothing about VRAM. Find the server by its port.
        if self.state in ("stopped", "stopping", "error", "pending", "downloading"):
            return set()
        lp = listener_pid(self.port)
        if lp is None:
            return super().pids()
        return process_tree(_runtime_root(lp))

    async def _kill(self) -> None:
        # Strata may daemonise its server; ask it to stop on our port first.
        s = self.registry.strata
        if s.get("stop"):
            try:
                p = await asyncio.create_subprocess_exec(
                    s.get("bin", "strata"), *self._fmt(s["stop"]),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await asyncio.wait_for(p.wait(), timeout=60)
            except (OSError, asyncio.TimeoutError):
                pass
        await super()._kill()
