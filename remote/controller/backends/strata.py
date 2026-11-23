from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

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


IDENTITY_KEYS = ("id", "root", "path", "model_path", "model", "name")


def _stem(path: str) -> str:
    name = Path(path).name.lower()
    return name[:-5] if name.endswith(".gguf") else name


def served_model_matches(entries: list[dict[str, Any]], expected: str, others: set[str]) -> tuple[bool, str | None]:
    """Does /v1/models describe ``expected`` (a model_path)? Returns (ok, upstream id).

    Strata's id format is unverified, so only a clear mismatch (the entry names a file or another
    registry model, not ours) rejects; an opaque id passes. The port-free precondition
    (wait_released) is the primary guard against a leftover server.
    """
    if not entries:
        return True, None
    want = _stem(expected)
    other_stems = {_stem(o) for o in others} - {want}
    opaque: dict[str, Any] | None = None
    for e in entries:
        tokens = [str(e[k]).lower() for k in IDENTITY_KEYS if isinstance(e.get(k), str)]
        if any(want in t for t in tokens):
            return True, e.get("id")
        conflicting = any(o in t for t in tokens for o in other_stems) or any(
            t.endswith(".gguf") or "/" in t for t in tokens)
        if not conflicting and opaque is None:
            opaque = e
    return (True, opaque.get("id")) if opaque is not None else (False, None)


class StrataBackend(Backend):
    """One LLM served by Strata on a local port (OpenAI + Anthropic compatible)."""

    kind = "llm"

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
        others = {rt.cfg["model_path"] for _, rt in self.registry.all_routes() if rt.engine == "strata"}
        ok, upstream = served_model_matches(data, self.cfg["model_path"], others)
        if ok and upstream:
            self.upstream_model = upstream
        return ok

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
