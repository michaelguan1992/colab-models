from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .policy import Spec

ENGINE_KINDS = {"strata": "llm", "pika": "video"}
KINDS = ("llm", "video")
AUTO_PORT_BASE = 8081


class RegistryError(ValueError):
    """models.yaml is malformed; raised at startup so the error names the field."""


@dataclass(frozen=True)
class Route:
    """One way to run a model: engine + weights + hardware requirement. ``cfg`` is the flat dict backends get."""
    name: str
    engine: str
    vram_gb: float
    est_download_gb: float
    port: int
    requires_sm: tuple[int, ...]
    cfg: Mapping[str, Any] = field(compare=False, repr=False)

    def fits(self, sm: int | None) -> bool:
        return not self.requires_sm or (sm is not None and sm in self.requires_sm)

    def requirement(self) -> str:
        if not self.requires_sm:
            return "无特殊要求"
        return "GPU 计算能力 " + " / ".join(f"sm_{s}" for s in self.requires_sm)


@dataclass(frozen=True)
class RouteChoice:
    route: Route | None
    reason: str = ""


def parse_sm(compute_cap: str) -> int | None:
    """nvidia-smi reports "12.0"; routes are written as sm: [120]."""
    try:
        major, minor = compute_cap.strip().split(".")
        return int(major) * 10 + int(minor)
    except ValueError:
        return None


def choose_route(model_id: str, routes: tuple[Route, ...], compute_cap: str, wanted: str | None = None) -> RouteChoice:
    """First route whose requirements the GPU meets; ``wanted`` restricts the search to one named route."""
    candidates = routes
    if wanted:
        candidates = tuple(r for r in routes if r.name == wanted)
        if not candidates:
            raise KeyError(f"{model_id} 没有路线 {wanted!r}，可选: {', '.join(r.name for r in routes)}")
    sm = parse_sm(compute_cap)
    for r in candidates:
        if r.fits(sm):
            return RouteChoice(r)
    reqs = "；".join(f"{r.name}: {r.requirement()}" for r in candidates)
    gpu = f"sm_{sm}" if sm is not None else "未知"
    return RouteChoice(None, f"没有可用路线：{model_id} 的路线要求 {reqs}；当前 GPU 计算能力为 {gpu}")


def _num(where: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (positive and value <= 0):
        raise RegistryError(f"{where} 必须是{'正' if positive else ''}数字，得到 {value!r}")
    return float(value)


def _parse_requires(where: str, requires: Any) -> tuple[int, ...]:
    if requires is None:
        return ()
    if not isinstance(requires, dict) or set(requires) - {"sm"}:
        raise RegistryError(f"{where}.requires 只支持 sm，得到 {requires!r}")
    sm = requires.get("sm", [])
    if not isinstance(sm, list) or not all(isinstance(s, int) and not isinstance(s, bool) for s in sm):
        raise RegistryError(f"{where}.requires.sm 必须是整数列表，得到 {sm!r}")
    return tuple(sm)


def _build_routes(model_id: str, model: dict[str, Any], ports: dict[int, str]) -> list[tuple[dict, str, int | None]]:
    where = f"models.{model_id}"
    if not isinstance(model, dict):
        raise RegistryError(f"{where} 必须是映射")
    if model.get("kind") not in KINDS:
        raise RegistryError(f"{where}.kind 必须是 {' 或 '.join(KINDS)}，得到 {model.get('kind')!r}")
    routes = model.get("routes")
    if not isinstance(routes, list) or not routes:
        raise RegistryError(f"{where}.routes 必须是非空列表（每条路线写 engine、vram_gb 和权重来源）")
    out: list[tuple[dict, str, int | None]] = []
    names: set[str] = set()
    for i, route in enumerate(routes):
        rwhere = f"{where}.routes[{i}]"
        if not isinstance(route, dict):
            raise RegistryError(f"{rwhere} 必须是映射")
        engine = route.get("engine")
        if engine not in ENGINE_KINDS:
            raise RegistryError(f"{rwhere}.engine 未知: {engine!r}，可选: {', '.join(ENGINE_KINDS)}")
        if ENGINE_KINDS[engine] != model["kind"]:
            raise RegistryError(f"{rwhere}.engine {engine} 只能跑 {ENGINE_KINDS[engine]}，但 {where}.kind 是 {model['kind']}")
        name = route.get("name", engine)
        if name in names:
            raise RegistryError(f"{where} 的路线名重复: {name!r}（同引擎多条路线请加 name）")
        names.add(name)
        _num(f"{rwhere}.vram_gb", route.get("vram_gb"), positive=True)
        if "est_download_gb" in route:
            _num(f"{rwhere}.est_download_gb", route["est_download_gb"])
        _parse_requires(rwhere, route.get("requires"))
        port = route.get("port")
        if port is not None:
            if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                raise RegistryError(f"{rwhere}.port 必须是 1-65535 的整数，得到 {port!r}")
            owner = ports.setdefault(port, model_id)
            if owner != model_id:
                raise RegistryError(f"端口 {port} 被 {owner} 和 {model_id} 同时使用，每个模型需要独立端口")
        out.append((route, name, port))
    return out


class Registry:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        models = raw.get("models") if isinstance(raw, dict) else None
        if not isinstance(models, dict) or not models:
            raise RegistryError("models 必须是非空映射")
        explicit: dict[int, str] = {}
        parsed = {mid: _build_routes(mid, m, explicit) for mid, m in models.items()}

        taken = set(explicit)
        self._routes: dict[str, tuple[Route, ...]] = {}
        for mid, items in parsed.items():
            auto: int | None = None
            routes: list[Route] = []
            for route, name, port in items:
                if port is None:
                    if auto is None:
                        auto = AUTO_PORT_BASE
                        while auto in taken:
                            auto += 1
                        taken.add(auto)
                    port = auto
                routes.append(self._route(mid, models[mid], route, name, port))
            self._routes[mid] = tuple(routes)

    @staticmethod
    def _route(model_id: str, model: dict[str, Any], route: dict[str, Any], name: str, port: int) -> Route:
        cfg = {k: v for k, v in model.items() if k != "routes"}
        cfg.update({k: v for k, v in route.items() if k not in ("name", "requires", "vram_gb", "caps")})
        if "caps" in model or "caps" in route:
            cfg["caps"] = {**model.get("caps", {}), **route.get("caps", {})}
        vram = float(route["vram_gb"])
        est_dl = float(route.get("est_download_gb") or 0)
        cfg.update(route=name, engine=route["engine"], port=port, est_vram_gb=vram, est_download_gb=est_dl)
        return Route(name, route["engine"], vram, est_dl, port, _parse_requires("", route.get("requires")), cfg)

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return self.raw["models"]

    def model(self, model_id: str) -> dict[str, Any]:
        if model_id not in self.models:
            raise KeyError(f"未知模型 {model_id!r}，可选: {', '.join(self.models)}")
        return self.models[model_id]

    def routes(self, model_id: str) -> tuple[Route, ...]:
        self.model(model_id)
        return self._routes[model_id]

    def all_routes(self) -> list[tuple[str, Route]]:
        return [(mid, r) for mid, routes in self._routes.items() for r in routes]

    def choose(self, model_id: str, compute_cap: str, route: str | None = None) -> RouteChoice:
        return choose_route(model_id, self.routes(model_id), compute_cap, route)

    def ids_of_kind(self, kind: str) -> list[str]:
        return [mid for mid, m in self.models.items() if m["kind"] == kind]

    @property
    def specs(self) -> dict[str, Spec]:
        """Estimates from each model's first route; the manager overrides these with the chosen route."""
        return {mid: Spec(mid, self.models[mid]["kind"], routes[0].vram_gb) for mid, routes in self._routes.items()}

    @property
    def models_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("models_dir", "/content/models"))

    @property
    def outputs_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("outputs_dir", "/content/outputs"))

    @property
    def logs_dir(self) -> Path:
        return Path(self.raw.get("paths", {}).get("logs_dir", "/content/lm/logs"))

    @property
    def state_file(self) -> Path:
        return Path(self.raw.get("paths", {}).get("state_file", "/content/lm/state.json"))

    @property
    def strata(self) -> dict[str, Any]:
        return self.raw.get("strata", {})

    @property
    def vram_margin_gb(self) -> float:
        return float(self.raw.get("vram_margin_gb", 4))

    @property
    def disk_reserve_gb(self) -> float:
        return float(self.raw.get("disk_reserve_gb", 10))

    @property
    def idle_unload_minutes(self) -> float:
        return float(self.raw.get("idle_unload_minutes", 45))
