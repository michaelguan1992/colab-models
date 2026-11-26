from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from ..gpu import process_tree
from ..procs import run_quiet
from ..templating import expand
from .base import Backend


def _runtime_root(pid: int, marker: str) -> int:
    """Walk up from the port listener to the topmost ancestor whose command line mentions ``marker``."""
    try:
        import psutil
        proc = psutil.Process(pid)
        for parent in proc.parents():
            if marker not in " ".join(parent.cmdline()):
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

    The id format is engine-specific and unverified, so only a clear mismatch (the entry names a file
    or another registry model, not ours) rejects; an opaque id passes. The port-free precondition
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


def dotted(obj: Any, path: str) -> Any:
    for key in path.split("."):
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


class ServerBackend(Backend):
    """An engine that serves HTTP on the model's port; the controller proxies to it."""

    async def _ready(self, client: httpx.AsyncClient) -> bool:
        r = await client.get(f"{self.base_url}{self.engine.ready.path}")
        if r.status_code != 200:
            return False
        ready = self.engine.ready
        try:
            data = r.json().get("data") or []
        except (ValueError, AttributeError):
            data = []
        if ready.identity == "model_path":
            others = {rt.model_path for _, rt in self.registry.all_routes()
                      if rt.engine == self.route.engine and rt.model_path}
            ok, upstream = served_model_matches(data, self.route.model_path, others)
            if not ok:
                return False
        else:
            entry = data[0] if data and isinstance(data[0], dict) else {}
            upstream = entry.get("id")
        if ready.field is not None:
            entry = next((e for e in data if isinstance(e, dict) and e.get("id") == upstream), {})
            if str(dotted(entry, ready.field)) != ready.equals:
                return False
        if upstream:
            self.upstream_model = upstream
        if ready.busy is not None:
            self.queue_depth = await self._queued(client, ready.busy)
        return True

    async def _queued(self, client: httpx.AsyncClient, path: str) -> int:
        try:
            body = (await client.get(f"{self.base_url}{path}")).json()
        except (httpx.HTTPError, ValueError):
            return 0
        return sum(len(v) for v in body.values() if isinstance(v, list)) if isinstance(body, dict) else 0

    def pids(self) -> set[int]:
        # A daemonizing launcher exits and re-parents the server to init, so our own child pid says
        # nothing about VRAM. Find the server by its port.
        if not self.engine.daemonizes:
            return super().pids()
        if self.state not in ("starting", "ready"):
            return set()
        lp = listener_pid(self.port)
        if lp is None:
            return super().pids()
        return process_tree(_runtime_root(lp, Path(expand(self.engine.launch[:1], self.values())[0]).name))

    async def _kill(self) -> None:
        if self.engine.stop:
            try:
                argv = expand(self.engine.stop, self.values())
            except KeyError:  # the engine was never installed, so nothing of it can be running
                argv = None
            if argv:
                await run_quiet(argv)
        await super()._kill()
