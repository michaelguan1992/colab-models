"""Owns all model backends and enforces the scheduling policy."""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from . import gpu as gpu_mod
from .backends.base import Backend
from .policy import Decision, Online, plan_load, violations
from .registry import Registry

BackendFactory = Callable[[str, dict, Registry], Backend]
GiB = 1024**3


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def disk_free(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def default_factory(model_id: str, cfg: dict, registry: Registry) -> Backend:
    from .backends.pika import PikaBackend
    from .backends.strata import StrataBackend

    return {"llm": StrataBackend, "video": PikaBackend}[cfg["kind"]](model_id, cfg, registry)


class Manager:
    def __init__(self, registry: Registry, factory: BackendFactory = default_factory,
                 probe: Callable[[], gpu_mod.GpuSnapshot] = gpu_mod.probe,
                 free_bytes: Callable[[Path], int] = disk_free) -> None:
        self.registry = registry
        self.factory = factory
        self.probe = probe
        self.free_bytes = free_bytes
        self.backends: dict[str, Backend] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()
        self.last_activity = time.time()
        self.idle_unloaded = False
        self.events: list[str] = []

    # ---- state -----------------------------------------------------------
    def touch(self) -> None:
        self.last_activity = time.time()
        self.idle_unloaded = False

    def _log(self, msg: str) -> None:
        self.events = (self.events + [f"{time.strftime('%H:%M:%S')} {msg}"])[-50:]

    def _live(self) -> list[Backend]:
        return [b for b in self.backends.values() if b.state not in ("stopped", "error")]

    def online(self, snap: gpu_mod.GpuSnapshot | None = None) -> list[Online]:
        per_pid = (snap or self.probe()).per_pid_gb
        return [Online(b.id, b.kind, b.vram_gb(per_pid)) for b in self._live()]

    def active_llm(self) -> Backend | None:
        for b in self.backends.values():
            if b.kind == "llm" and b.state == "ready":
                return b
        return None

    def active_video(self) -> Backend | None:
        for b in self.backends.values():
            if b.kind == "video" and b.state == "ready":
                return b
        return None

    # ---- actions ---------------------------------------------------------
    async def load(self, model_id: str, swap: bool = True) -> Decision:
        self.touch()
        cfg = self.registry.model(model_id)
        async with self.lock:
            spec = self.registry.specs[model_id]
            snap = self.probe()
            decision = plan_load(spec, self.online(snap), self.registry.specs, swap=swap,
                                 total_gb=snap.total_gb if snap.available else None,
                                 margin_gb=self.registry.vram_margin_gb)
            if not decision.ok:
                self._log(f"拒绝加载 {model_id}: {decision.reason}")
                return decision
            if model_id in self.backends and self.backends[model_id].state not in ("stopped", "error"):
                return decision
            for other in decision.unload:
                await self._unload_locked(other)
            backend = self.factory(model_id, cfg, self.registry)
            if not cfg.get("skip_download"):
                ok, why = self._ensure_disk(backend)
                if not ok:
                    self._log(f"拒绝加载 {model_id}: {why}")
                    return Decision(False, unload=decision.unload, reason=why)
            self.backends[model_id] = backend
            self.tasks[model_id] = asyncio.create_task(self._start(backend))
            self._log(f"开始加载 {model_id}" + (f"（已卸载 {', '.join(decision.unload)}）" if decision.unload else ""))
            return decision

    # ---- disk ------------------------------------------------------------
    def _weight_dirs(self, model_id: str) -> list[Path]:
        base = self.registry.models_dir
        return [base / model_id, base / f"{model_id}-lora"]

    def _ensure_disk(self, backend: Backend) -> tuple[bool, str]:
        """Make room for ``backend``'s download by deleting weights of offline models.

        Weights are re-downloaded from HF on the next load anyway, so evicting is cheap
        compared with failing a 150GB download halfway (G4 disk is ~236GB).
        """
        est = float(backend.cfg.get("est_download_gb") or 0) * GiB
        have = sum(dir_size(d) for d in self._weight_dirs(backend.id))
        need = max(0.0, est - have) + self.registry.disk_reserve_gb * GiB
        free = self.free_bytes(self.registry.models_dir)
        if free >= need:
            return True, ""
        live = {b.id for b in self._live()} | {backend.id}
        candidates = [(sum(dir_size(d) for d in self._weight_dirs(m)), m)
                      for m in self.registry.models if m not in live]
        evicted = []
        for size, mid in sorted(candidates, reverse=True):
            if free >= need:
                break
            if size == 0:
                continue
            for d in self._weight_dirs(mid):
                shutil.rmtree(d, ignore_errors=True)
            evicted.append(mid)
            free = self.free_bytes(self.registry.models_dir)
        if evicted:
            self._log(f"磁盘不够，已删除离线模型的权重：{', '.join(evicted)}")
        if free >= need:
            return True, ""
        return False, (f"磁盘不够：{backend.id} 还需约 {need / GiB:.0f}GB（含 {self.registry.disk_reserve_gb:.0f}GB 余量），"
                       f"只剩 {free / GiB:.0f}GB；请先卸载其他模型")

    async def _start(self, backend: Backend) -> None:
        await backend.start()
        if backend.state != "ready":
            self._log(f"{backend.id} 加载失败: {backend.error}")
            return
        self._log(f"{backend.id} 已上线")
        # Post-load check with measured VRAM (rule 2 uses real numbers).
        bad = violations(self.online(), self.registry.specs)
        if bad:
            async with self.lock:
                await self._unload_locked(backend.id)
            backend.state = "error"
            backend.error = "；".join(bad) + "，已自动卸载"
            self._log(f"{backend.id} 违反调度规则: {backend.error}")

    async def unload(self, model_id: str) -> bool:
        self.touch()
        async with self.lock:
            return await self._unload_locked(model_id)

    async def _unload_locked(self, model_id: str) -> bool:
        b = self.backends.get(model_id)
        if not b:
            return False
        task = self.tasks.pop(model_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await b.stop()
        self.backends.pop(model_id, None)
        self._log(f"{model_id} 已下线")
        return True

    async def unload_all(self) -> list[str]:
        ids = list(self.backends)
        async with self.lock:
            for i in ids:
                await self._unload_locked(i)
        return ids

    async def idle_check(self) -> None:
        limit = self.registry.idle_unload_minutes * 60
        if limit > 0 and self.backends and time.time() - self.last_activity > limit:
            ids = await self.unload_all()
            self.idle_unloaded = True
            self._log(f"空闲超过 {self.registry.idle_unload_minutes:.0f} 分钟，已卸载 {', '.join(ids)}")

    # ---- reporting -------------------------------------------------------
    def status(self) -> dict[str, Any]:
        snap = self.probe()
        models = []
        for mid, cfg in self.registry.models.items():
            b = self.backends.get(mid)
            entry: dict[str, Any] = {
                "id": mid, "kind": cfg["kind"], "display": cfg.get("display", mid),
                "state": b.state if b else "offline",
                "est_vram_gb": cfg["est_vram_gb"],
            }
            if b:
                entry["vram_gb"] = b.vram_gb(snap.per_pid_gb)
                entry["error"] = b.error
                if b.state == "downloading":
                    entry["download_progress"] = b.download_progress()
                if b.ready_at:
                    entry["online_since"] = b.ready_at
            models.append(entry)
        return {
            "gpu": {"name": snap.name, "used_gb": round(snap.used_gb, 1),
                    "total_gb": round(snap.total_gb, 1), "available": snap.available},
            "models": models,
            "online": [m["id"] for m in models if m["state"] == "ready"],
            "idle_seconds": round(time.time() - self.last_activity),
            "idle_unload_minutes": self.registry.idle_unload_minutes,
            "idle_unloaded": self.idle_unloaded,
            "events": self.events[-10:],
        }
