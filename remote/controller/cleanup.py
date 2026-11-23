"""Startup cleanup: stop engines a previous controller left running (e.g. after `lm up` redeploys)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Iterable, Protocol

from .install import Installer
from .procs import kill_group, run_quiet
from .registry import Registry
from .spec import Engine
from .state import StateFile
from .templating import engine_values, expand

log = logging.getLogger("lm.controller")


class Reaper(Protocol):
    async def stop_engine(self, engine: Engine, port: int) -> None: ...
    async def kill_pids(self, pids: Iterable[int]) -> None: ...


class SystemReaper:
    def __init__(self, registry: Registry, installer: Installer) -> None:
        self.registry = registry
        self.installer = installer

    async def stop_engine(self, engine: Engine, port: int) -> None:
        installed = self.installer.installed(engine)
        if not engine.stop or installed is None:  # never installed here, so nothing of ours is running
            return
        values = engine_values(self.registry.models_dir, self.registry.outputs_dir, port, installed.values())
        await run_quiet(expand(engine.stop, values))

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
    stop_ports: dict[str, set[int]] = {
        name: {r.port for _, r in registry.all_routes() if r.engine == name}
        for name, e in registry.engines.items() if e.stop}
    for model_id, entry in state.read().items():
        try:
            engine = registry.engines.get(entry.get("engine"))
            if engine and engine.stop:
                if isinstance(entry.get("port"), int):
                    stop_ports[engine.name].add(entry["port"])
            else:
                pids = [p for p in entry.get("pids", []) if isinstance(p, int)]
                if pids:
                    log.info("清理上次遗留的 %s (pids %s)", model_id, pids)
                    await reaper.kill_pids(pids)
        except Exception:
            log.exception("清理 %s 失败", model_id)
    for name, ports in stop_ports.items():
        for port in sorted(ports):
            try:
                await reaper.stop_engine(registry.engines[name], port)
            except Exception:
                log.exception("停止端口 %s 上的 %s 失败", port, name)
    state.clear()
