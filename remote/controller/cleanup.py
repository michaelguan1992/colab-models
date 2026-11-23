"""Startup cleanup: stop engines a previous controller left running (e.g. after `lm up` redeploys)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Iterable, Protocol

from .procs import kill_group
from .registry import Registry
from .state import StateFile

log = logging.getLogger("lm.controller")


class Reaper(Protocol):
    async def stop_strata(self, port: int) -> None: ...
    async def kill_pids(self, pids: Iterable[int]) -> None: ...


class SystemReaper:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    async def stop_strata(self, port: int) -> None:
        s = self.registry.strata
        if not s.get("stop"):
            return
        try:
            p = await asyncio.create_subprocess_exec(
                s.get("bin", "strata"), *[a.format(port=port) for a in s["stop"]],
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await asyncio.wait_for(p.wait(), timeout=60)
        except (OSError, asyncio.TimeoutError):
            pass

    async def kill_pids(self, pids: Iterable[int]) -> None:
        groups: set[int] = set()
        for pid in pids:
            try:
                groups.add(os.getpgid(pid))
            except (ProcessLookupError, PermissionError):
                continue
        await asyncio.gather(*(kill_group(g) for g in groups))


async def reap_stale(registry: Registry, state: StateFile, reaper: Reaper) -> None:
    """Idempotent: a missing or corrupt state file just means there is nothing recorded."""
    recorded = state.read()
    strata_ports: set[int] = {r.port for _, r in registry.all_routes() if r.engine == "strata"}
    for model_id, entry in recorded.items():
        try:
            if entry.get("engine") == "strata":
                if isinstance(entry.get("port"), int):
                    strata_ports.add(entry["port"])
            else:
                pids = [p for p in entry.get("pids", []) if isinstance(p, int)]
                if pids:
                    log.info("清理上次遗留的 %s (pids %s)", model_id, pids)
                    await reaper.kill_pids(pids)
        except Exception:
            log.exception("清理 %s 失败", model_id)
    for port in sorted(strata_ports):
        try:
            await reaper.stop_strata(port)
        except Exception:
            log.exception("停止端口 %s 上的 Strata 失败", port)
    state.clear()
