"""Owns all model backends and enforces the scheduling policy."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from . import gpu as gpu_mod
from .backends.base import Backend
from .backends.jobs import JobsBackend
from .backends.server import ServerBackend
from .cleanup import Reaper, SystemReaper, reap_stale
from .hf import hf_preflight
from .install import Installer
from .policy import GiB, DiskView, Online, Plan, Spec, plan_load, plan_rules, violations
from .registry import Registry
from .spec import ModelSpec, Route
from .state import StateFile

log = logging.getLogger("lm.controller")

BackendFactory = Callable[[ModelSpec, Route, Registry, Installer], Backend]
HfCheck = Callable[[str, "str | None"], "str | None"]


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def disk_free(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def default_factory(model: ModelSpec, route: Route, registry: Registry, installer: Installer) -> Backend:
    cls = {"server": ServerBackend, "jobs": JobsBackend}[registry.engines[route.engine].shape]
    return cls(model, route, registry, installer)


LOADING_STATES = ("installing", "downloading", "starting")


class RequestHandle:
    """One proxied request in flight. ``finish`` is idempotent (stream end and background task both call it)."""

    def __init__(self, manager: "Manager", activity: bool) -> None:
        self._manager = manager
        self._activity = activity
        self._done = False

    def finish(self) -> None:
        if self._done:
            return
        self._done = True
        self._manager.requests_inflight -= 1
        if self._activity:
            self._manager.touch()


class Manager:
    release_timeout_s = 90.0
    release_poll_s = 1.0
    health_failures_limit = 2

    def __init__(self, registry: Registry, factory: BackendFactory = default_factory,
                 probe: Callable[[], gpu_mod.GpuSnapshot] = gpu_mod.probe,
                 free_bytes: Callable[[Path], int] = disk_free,
                 hf_check: HfCheck = hf_preflight,
                 reaper: Reaper | None = None,
                 state: StateFile | None = None,
                 installer: Installer | None = None) -> None:
        self.registry = registry
        self.factory = factory
        self.probe = probe
        self.free_bytes = free_bytes
        self.hf_check = hf_check
        self.installer = installer or Installer(registry)
        self.reaper = reaper or SystemReaper(registry, self.installer)
        self.state = state or StateFile(registry.state_file)
        self.backends: dict[str, Backend] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()
        self.last_activity = time.time()
        self.idle_unloaded = False
        self.requests_inflight = 0
        self.events: list[str] = []

    # ---- state -----------------------------------------------------------
    def touch(self) -> None:
        self.last_activity = time.time()
        self.idle_unloaded = False

    def begin_request(self, activity: bool) -> RequestHandle:
        self.requests_inflight += 1
        return RequestHandle(self, activity)

    def inflight(self) -> dict[str, int]:
        return {
            "loading": sum(1 for b in self.backends.values() if b.state in LOADING_STATES),
            "requests": self.requests_inflight,
            "jobs": sum(b.queue_depth for b in self.backends.values() if b.state == "ready"),
        }

    def busy(self) -> bool:
        return any(self.inflight().values())

    def _log(self, msg: str) -> None:
        self.events = (self.events + [f"{time.strftime('%H:%M:%S')} {msg}"])[-50:]

    def _live(self) -> list[Backend]:
        return [b for b in self.backends.values() if b.state not in ("stopped", "error")]

    def online(self, snap: gpu_mod.GpuSnapshot | None = None) -> list[Online]:
        per_pid = (snap or self.probe()).per_pid_gb
        return [Online(b.id, b.model.group, b.vram_gb(per_pid)) for b in self._live()]

    def _specs(self) -> dict[str, Spec]:
        """Registry estimates, overridden by the route each live backend actually runs."""
        specs = dict(self.registry.specs)
        for b in self.backends.values():
            specs[b.id] = Spec(b.id, b.model.group, b.route.vram_gb)
        return specs

    def serving(self, capability: str) -> list[Backend]:
        """Ready backends whose model serves ``capability``, in registry order."""
        return [b for mid in self.registry.models
                if (b := self.backends.get(mid)) and b.state == "ready" and capability in b.model.serves]

    def loading(self, capability: str) -> list[Backend]:
        return [b for mid in self.registry.models
                if (b := self.backends.get(mid)) and b.state in LOADING_STATES and capability in b.model.serves]

    # ---- startup ---------------------------------------------------------
    async def startup_cleanup(self) -> None:
        try:
            await reap_stale(self.registry, self.state, self.reaper)
        except Exception:
            log.exception("启动清理失败")

    # ---- actions ---------------------------------------------------------
    async def load(self, model_id: str, swap: bool = True, route: str | None = None) -> Plan:
        self.touch()
        self.registry.model(model_id)
        async with self.lock:
            existing = self.backends.get(model_id)
            online = existing is not None and existing.state not in ("stopped", "error")
            if online and route in (None, existing.route.name):
                return Plan(True, reason=f"{model_id} 已在线")
            snap = self.probe()
            choice = self.registry.choose(model_id, snap.compute_cap, route)
            if choice.route is None:
                return self._refuse(model_id, choice.reason)
            if online:  # another route of the same model: take this one down before planning the new one
                b = await self._stop_entry(model_id)
                if b is not None and not await self._released(b):
                    return self._refuse(model_id, f"{model_id} 的 {existing.route.name} 路线已停止但资源未释放：{b.error}")
                existing = None
                snap = self.probe()
            model, route = self.registry.model(model_id), choice.route

            why = await self._hf_refusal(route)
            if why:
                return self._refuse(model_id, why)

            plan = self._plan(model, route, snap, swap)
            if not plan.ok:
                return self._refuse(model_id, plan.reason)

            if existing is not None:
                self.backends.pop(model_id)
                if not await self._released(existing):
                    return self._refuse(model_id, f"上次的 {model_id} 还没完全退出：{existing.error}")

            for other in plan.unload:
                b = await self._stop_entry(other)
                if b is not None and not await self._released(b):
                    return self._refuse(model_id, f"{other} 已停止但资源未释放：{b.error}；未加载 {model_id}")
            for victim in plan.evict:
                for d in self._weight_dirs(victim):
                    shutil.rmtree(d, ignore_errors=True)
            if plan.evict:
                self._log(f"磁盘不够，已删除离线模型的权重：{', '.join(plan.evict)}")

            backend = self.factory(model, route, self.registry, self.installer)
            backend.on_launch = self._record
            self.backends[model_id] = backend
            self.tasks[model_id] = asyncio.create_task(self._start(backend))
            self._log(f"开始加载 {model_id}（路线 {route.name}）"
                      + (f"（已卸载 {', '.join(plan.unload)}）" if plan.unload else ""))
            return plan

    def _refuse(self, model_id: str, reason: str) -> Plan:
        self._log(f"拒绝加载 {model_id}: {reason}")
        return Plan(False, reason=reason)

    async def _hf_refusal(self, route: Route) -> str | None:
        for repo, revision in ((s.repo, s.revision) for s in route.sources()):
            try:
                reason = await asyncio.wait_for(asyncio.to_thread(self.hf_check, repo, revision), timeout=30)
            except Exception as e:
                log.warning("HF 访问检查 %s 失败（继续加载）: %s", repo, e)
                continue
            if reason:
                return reason
        return None

    # ---- disk ------------------------------------------------------------
    def _weight_dirs(self, model_id: str) -> list[Path]:
        base = self.registry.models_dir
        return [base / model_id]

    def _plan(self, model: ModelSpec, route: Route, snap: gpu_mod.GpuSnapshot, swap: bool) -> Plan:
        target = Spec(model.id, model.group, route.vram_gb)
        specs = {**self._specs(), model.id: target}
        online = self.online(snap)
        rules = self.registry.rules
        unloading = set(plan_rules(target, online, specs, rules, swap).unload)
        disk = self._disk_view(model.id, route, unloading)
        return plan_load(target, online, specs, rules, swap=swap, disk=disk,
                         total_gb=snap.total_gb if snap.available else None,
                         margin_gb=self.registry.vram_margin_gb if route.vram_margin_gb is None
                         else route.vram_margin_gb)

    def _disk_view(self, model_id: str, route: Route, unloading: set[str]) -> DiskView:
        sizes = {m: sum(dir_size(d) for d in self._weight_dirs(m)) for m in self.registry.models}
        downloads = route.repo is not None
        need = max(0, int(route.est_download_gb * GiB) - sizes[model_id]) if downloads else 0
        reserve = int(self.registry.disk_reserve_gb * GiB) if downloads else 0
        inflight = sum(max(0, int(b.route.est_download_gb * GiB) - sizes.get(b.id, 0))
                       for b in self.backends.values()
                       if b.state == "downloading" and b.id != model_id and b.id not in unloading)
        return DiskView(free_bytes=self.free_bytes(self.registry.models_dir), weights_bytes=sizes,
                        need_bytes=need, reserve_bytes=reserve, inflight_bytes=inflight)

    # ---- lifecycle -------------------------------------------------------
    def _record(self, backend: Backend) -> None:
        pids = backend.pids()
        backend.recorded_pids |= pids
        self.state.set(backend.id, {
            "route": backend.route.name, "engine": backend.route.engine,
            "port": backend.port, "pids": sorted(pids), "started_at": backend.started_at,
        })

    async def _start(self, backend: Backend) -> None:
        await backend.start()
        if backend.state != "ready":
            self.state.remove(backend.id)
            self._log(f"{backend.id} 加载失败: {backend.error}")
            return
        self._record(backend)
        self._log(f"{backend.id} 已上线")
        # Post-load check with measured VRAM (rule 2 uses real numbers).
        bad = violations(self.online(), self._specs(), self.registry.rules)
        if bad:
            async with self.lock:
                if self.backends.get(backend.id) is not backend:
                    return
                await backend.stop()
                self.state.remove(backend.id)
                backend.state = "error"
                backend.error = "；".join(bad) + "，已自动卸载"
                self._log(f"{backend.id} 违反调度规则: {backend.error}")

    async def _stop_entry(self, model_id: str) -> Backend | None:
        b = self.backends.get(model_id)
        if not b:
            return None
        task = self.tasks.pop(model_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await b.stop()
        self.state.remove(model_id)
        self.backends.pop(model_id, None)
        self._log(f"{model_id} 已下线")
        return b

    async def _released(self, b: Backend) -> bool:
        """Wait for the stopped engine to free its port and processes; otherwise keep it visible as an error."""
        if await b.wait_released(self.release_timeout_s, self.release_poll_s):
            return True
        b.state = "error"
        b.error = f"停止后 {self.release_timeout_s:.0f}s 内端口 {b.port} 仍被占用或进程未退出"
        self.backends[b.id] = b
        self._log(f"{b.id} {b.error}")
        return False

    async def unload(self, model_id: str) -> bool:
        self.touch()
        async with self.lock:
            b = await self._stop_entry(model_id)
            if b is None:
                return False
            await self._released(b)
            return True

    async def unload_all(self) -> list[str]:
        ids = list(self.backends)
        async with self.lock:
            for i in ids:
                await self._stop_entry(i)
        return ids

    async def health_check(self) -> None:
        ready = [b for b in self.backends.values() if b.state == "ready"]
        if not ready:
            return
        async with httpx.AsyncClient(timeout=5) as client:
            for b in ready:
                if await b.check_health(client):
                    b.health_failures = 0
                    continue
                b.health_failures += 1
                if b.health_failures < self.health_failures_limit:
                    continue
                async with self.lock:
                    if self.backends.get(b.id) is b and b.state == "ready":
                        await b.stop()
                        self.state.remove(b.id)
                        b.state = "error"
                        b.error = "引擎无响应/已退出"
                        self._log(f"{b.id} {b.error}")

    def external_work(self) -> list[str]:
        """Work started outside the controller that must keep the runtime alive.

        A benchmark or conversion run over ssh sends no requests, so without this the idle
        timers (here and in the Mac watchdog) stop a runtime that is busy. Two signals: a GPU
        process no backend owns, and any file in holds_dir (for CPU or network work).
        """
        ours: set[int] = set()
        for b in self.backends.values():
            ours |= b.pids()
        reasons = [f"GPU 进程 {pid}" for pid in sorted(set(self.probe().per_pid_gb) - ours)]
        if self.registry.holds_dir.is_dir():
            reasons += [f"hold {p.name}" for p in sorted(self.registry.holds_dir.iterdir())]
        return reasons

    async def idle_check(self) -> None:
        # Queued jobs count as activity while they run, like external work: a web UI the user drives directly
        # (ComfyUI) sends the controller no requests, so otherwise the end of a long job finds the idle clock
        # already past both limits and the runtime is stopped the moment its queue empties.
        if self.external_work() or self.inflight()["jobs"]:
            self.touch()
            return
        limit = self.registry.idle_unload_minutes * 60
        if limit <= 0 or not self.backends or self.busy() or time.time() - self.last_activity <= limit:
            return
        ids = await self.unload_all()
        self.idle_unloaded = True
        self._log(f"空闲超过 {self.registry.idle_unload_minutes:.0f} 分钟，已卸载 {', '.join(ids)}")

    # ---- reporting -------------------------------------------------------
    def status(self) -> dict[str, Any]:
        snap = self.probe()
        models = []
        for mid, model in self.registry.models.items():
            b = self.backends.get(mid)
            entry: dict[str, Any] = {
                "id": mid, "group": model.group, "serves": list(model.serves), "display": model.display,
                "state": b.state if b else "offline",
                "routes": [r.name for r in model.routes],
            }
            if b:
                entry.update(route=b.route.name, engine=b.route.engine, port=b.route.port, caps=dict(b.route.caps),
                             params=dict(b.route.params),
                             est_vram_gb=b.route.vram_gb, vram_gb=b.vram_gb(snap.per_pid_gb), error=b.error)
                if b.state == "downloading":
                    entry["download_progress"] = b.download_progress()
                if b.ready_at:
                    entry["online_since"] = b.ready_at
            else:
                choice = self.registry.choose(mid, snap.compute_cap)
                chosen = choice.route or model.routes[0]
                entry.update(route=choice.route.name if choice.route else None, engine=chosen.engine, port=chosen.port,
                             caps=dict(chosen.caps), est_vram_gb=chosen.vram_gb)
                if choice.route is None:
                    entry["route_reason"] = choice.reason
            models.append(entry)
        return {
            "gpu": {"name": snap.name, "used_gb": round(snap.used_gb, 1),
                    "total_gb": round(snap.total_gb, 1), "available": snap.available,
                    "compute_cap": snap.compute_cap},
            "models": models,
            "online": [m["id"] for m in models if m["state"] == "ready"],
            "idle_seconds": round(time.time() - self.last_activity),
            "idle_unload_minutes": self.registry.idle_unload_minutes,
            "idle_unloaded": self.idle_unloaded,
            "busy": self.busy(),
            "external_work": self.external_work(),
            "inflight": self.inflight(),
            "events": self.events[-10:],
        }
