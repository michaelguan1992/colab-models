"""models.yaml -> typed registry. Every field is validated here, so errors name the field at startup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .policy import MaxOnline, RequiresBelow, Rule, Spec
from .spec import (Archive, ArchiveInstall, Engine, Install, Lora, ModelSpec, Ready, Route, ScriptInstall,
                   VenvInstall)
from .templating import EXTRA, INSTALL_PLACEHOLDERS, engine_provides, placeholders, route_provides

AUTO_PORT_BASE = 8081
SHAPES = ("server", "jobs")
APIS = ("openai", "anthropic")
SHA256 = re.compile(r"[0-9a-fA-F]{64}")

TOP_KEYS = {"paths", "idle_unload_minutes", "vram_margin_gb", "disk_reserve_gb", "engines", "models", "rules"}
PATH_KEYS = {"models_dir", "outputs_dir", "logs_dir", "state_file", "engines_dir", "venvs_dir", "holds_dir"}
ENGINE_KEYS = {"shape", "apis", "install", "launch", "stop", "daemonizes", "ready", "requires"}
MODEL_KEYS = {"group", "serves", "display", "caps", "routes"}
ROUTE_KEYS = {"name", "engine", "repo", "include", "revision", "model_path", "lora", "vram_gb",
              "est_download_gb", "port", "requires", "caps", "extra", "params", "ready_timeout_s"}


class RegistryError(ValueError):
    """models.yaml is malformed; raised at startup so the error names the field."""


# ---- primitives -------------------------------------------------------------------------------

def _keys(where: str, raw: Any, allowed: Iterable[str], required: Iterable[str] = ()) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RegistryError(f"{where} 必须是映射")
    allowed = set(allowed)
    unknown = set(raw) - allowed
    if unknown:
        raise RegistryError(f"{where} 有未知字段 {', '.join(sorted(map(str, unknown)))}，可用: {', '.join(sorted(allowed))}")
    for k in required:
        if k not in raw:
            raise RegistryError(f"{where}.{k} 必须填写")
    return raw


def _str(where: str, v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise RegistryError(f"{where} 必须是非空字符串，得到 {v!r}")
    return v


def _strs(where: str, v: Any, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(v, list) or not all(isinstance(s, str) and s for s in v) or (nonempty and not v):
        raise RegistryError(f"{where} 必须是{'非空' if nonempty else ''}字符串列表，得到 {v!r}")
    return tuple(v)


def _num(where: str, v: Any, *, positive: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (positive and v <= 0):
        raise RegistryError(f"{where} 必须是{'正' if positive else ''}数字，得到 {v!r}")
    return float(v)


def _int(where: str, v: Any, lo: int, hi: int | None = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
        raise RegistryError(f"{where} 必须是 {lo}{'-' + str(hi) if hi else ' 以上'} 的整数，得到 {v!r}")
    return v


def _bool(where: str, v: Any) -> bool:
    if not isinstance(v, bool):
        raise RegistryError(f"{where} 必须是 true 或 false，得到 {v!r}")
    return v


def _sha256(where: str, v: Any) -> str:
    if not isinstance(v, str) or not SHA256.fullmatch(v):
        raise RegistryError(f"{where} 必须是 64 位十六进制 sha256，得到 {v!r}")
    return v.lower()


def _argv(where: str, v: Any) -> tuple[str, ...]:
    argv = _strs(where, v, nonempty=True)
    for a in argv:
        if EXTRA in a and a != EXTRA:
            raise RegistryError(f"{where} 里 {EXTRA} 必须单独占一个元素，得到 {a!r}")
    return argv


def _names(missing: Iterable[str]) -> str:
    return ", ".join("{" + n + "}" for n in sorted(missing))


def _sm(where: str, requires: Any) -> tuple[int, ...]:
    if requires is None:
        return ()
    if not isinstance(requires, dict) or set(requires) - {"sm"}:
        raise RegistryError(f"{where}.requires 只支持 sm，得到 {requires!r}")
    sm = requires.get("sm", [])
    if not isinstance(sm, list) or not all(isinstance(s, int) and not isinstance(s, bool) for s in sm):
        raise RegistryError(f"{where}.requires.sm 必须是整数列表，得到 {sm!r}")
    return tuple(sm)


# ---- engines ----------------------------------------------------------------------------------

def _parse_install(where: str, raw: Any) -> Install:
    kind = raw.get("kind") if isinstance(raw, dict) else None
    if kind == "script":
        d = _keys(where, raw, {"kind", "url", "sums_url", "sha256", "env", "bin"}, {"url", "bin"})
        if "sha256" not in d and "sums_url" not in d:
            raise RegistryError(f"{where} 必须给 sha256 或 sums_url，才能校验安装脚本")
        env = d.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise RegistryError(f"{where}.env 必须是字符串到字符串的映射，得到 {env!r}")
        return ScriptInstall(
            _str(f"{where}.url", d["url"]), _str(f"{where}.bin", d["bin"]),
            _sha256(f"{where}.sha256", d["sha256"]) if "sha256" in d else None,
            _str(f"{where}.sums_url", d["sums_url"]) if "sums_url" in d else None, dict(env))
    if kind == "archive":
        d = _keys(where, raw, {"kind", "url", "sha256", "extra_archives", "bin", "requirements"}, {"url", "sha256", "bin"})
        archives = [Archive(_str(f"{where}.url", d["url"]), _sha256(f"{where}.sha256", d["sha256"]))]
        extras = d.get("extra_archives", [])
        if not isinstance(extras, list):
            raise RegistryError(f"{where}.extra_archives 必须是列表，得到 {extras!r}")
        for i, a in enumerate(extras):
            aw = f"{where}.extra_archives[{i}]"
            a = _keys(aw, a, {"url", "sha256"}, {"url", "sha256"})
            archives.append(Archive(_str(f"{aw}.url", a["url"]), _sha256(f"{aw}.sha256", a["sha256"])))
        req = _str(f"{where}.requirements", d["requirements"]) if "requirements" in d else None
        return ArchiveInstall(tuple(archives), _str(f"{where}.bin", d["bin"]), req)
    if kind == "venv":
        d = _keys(where, raw, {"kind", "pip", "pip_files"})
        pip_files = _strs(f"{where}.pip_files", d.get("pip_files", []))
        for f in pip_files:
            extra = placeholders([f]) - INSTALL_PLACEHOLDERS
            if extra:
                raise RegistryError(f"{where}.pip_files 只能使用 {{model_dir}} {{models_dir}}，得到 {sorted(extra)}")
        return VenvInstall(_strs(f"{where}.pip", d.get("pip", [])), pip_files)
    raise RegistryError(f"{where}.kind 必须是 script、archive 或 venv，得到 {kind!r}")


def _parse_engine(name: str, raw: Any) -> Engine:
    where = f"engines.{name}"
    d = _keys(where, raw, ENGINE_KEYS, {"shape", "launch", "ready"})
    shape = d["shape"]
    if shape not in SHAPES:
        raise RegistryError(f"{where}.shape 必须是 {' 或 '.join(SHAPES)}，得到 {shape!r}")
    apis = _strs(f"{where}.apis", d.get("apis", []))
    if set(apis) - set(APIS):
        raise RegistryError(f"{where}.apis 只支持 {', '.join(APIS)}，得到 {list(apis)!r}")
    daemonizes = _bool(f"{where}.daemonizes", d.get("daemonizes", False))
    stop = _argv(f"{where}.stop", d["stop"]) if "stop" in d else None
    if shape == "jobs" and (daemonizes or stop):
        raise RegistryError(f"{where} 是 jobs 引擎，不支持 daemonizes / stop")
    r = _keys(f"{where}.ready", d["ready"], {"path", "identity", "field", "equals"}, {"path"})
    path = _str(f"{where}.ready.path", r["path"])
    if not path.startswith("/"):
        raise RegistryError(f"{where}.ready.path 必须以 / 开头，得到 {path!r}")
    identity = r.get("identity")
    if identity not in (None, "model_path") or (identity and shape != "server"):
        raise RegistryError(f"{where}.ready.identity 只能是 model_path（且仅 server 引擎），得到 {identity!r}")
    if ("field" in r) != ("equals" in r) or (("field" in r) and shape != "server"):
        raise RegistryError(f"{where}.ready.field 和 ready.equals 要一起给（且仅 server 引擎）")
    field = _str(f"{where}.ready.field", r["field"]) if "field" in r else None
    equals = _str(f"{where}.ready.equals", r["equals"]) if "equals" in r else None
    engine = Engine(name, shape, apis, _parse_install(f"{where}.install", d["install"]) if "install" in d else None,
                    _argv(f"{where}.launch", d["launch"]), stop, daemonizes, Ready(path, identity, field, equals),
                    _sm(where, d.get("requires")))
    if engine.stop:
        missing = placeholders(engine.stop) - engine_provides(engine)
        if missing:
            raise RegistryError(f"{where}.stop 使用了没有来源的 {_names(missing)}")
    return engine


# ---- models -----------------------------------------------------------------------------------

def _parse_caps(where: str, raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RegistryError(f"{where} 必须是映射")
    for k in ("tools", "vision"):
        if k in raw:
            _bool(f"{where}.{k}", raw[k])
    for k in ("context", "max_output"):
        if k in raw:
            _int(f"{where}.{k}", raw[k], 1)
    if "reasoning" in raw:
        _strs(f"{where}.reasoning", raw["reasoning"])
    return dict(raw)


def _extra(where: str, v: Any) -> tuple[str, ...]:
    if not isinstance(v, list) or not all(isinstance(e, (str, int, float)) and not isinstance(e, bool) for e in v):
        raise RegistryError(f"{where} 必须是字符串列表，得到 {v!r}")
    return tuple(str(e) for e in v)


def _route_port(where: str, route: Any) -> int | None:
    if not isinstance(route, dict):
        raise RegistryError(f"{where} 必须是映射")
    return _int(f"{where}.port", route["port"], 1, 65535) if route.get("port") is not None else None


def _parse_route(where: str, raw: dict[str, Any], model_caps: Mapping[str, Any], engines: dict[str, Engine],
                 port: int) -> Route:
    _keys(where, raw, ROUTE_KEYS, {"engine", "vram_gb"})
    ename = raw["engine"]
    if ename not in engines:
        raise RegistryError(f"{where}.engine 未知: {ename!r}，可选: {', '.join(engines)}")
    engine = engines[ename]
    route_sm = _sm(where, raw.get("requires"))
    if route_sm and engine.requires_sm:
        sm = tuple(s for s in route_sm if s in engine.requires_sm)
        if not sm:
            raise RegistryError(f"{where}.requires.sm {list(route_sm)} 与引擎 {ename} 的要求 {list(engine.requires_sm)} 没有交集")
    else:
        sm = route_sm or engine.requires_sm

    lora = None
    if "lora" in raw:
        ld = _keys(f"{where}.lora", raw["lora"], {"repo", "file", "revision"}, {"repo", "file"})
        lora = Lora(_str(f"{where}.lora.repo", ld["repo"]), _str(f"{where}.lora.file", ld["file"]),
                    _str(f"{where}.lora.revision", ld["revision"]) if ld.get("revision") else None)
    params = raw.get("params", {})
    if not isinstance(params, dict) or not all(
            isinstance(k, str) and isinstance(v, (str, int, float)) for k, v in params.items()):
        raise RegistryError(f"{where}.params 必须是名字到标量的映射，得到 {params!r}")
    extra = _extra(f"{where}.extra", raw.get("extra", []))
    include = _strs(f"{where}.include", raw.get("include", []))
    repo = _str(f"{where}.repo", raw["repo"]) if "repo" in raw else None
    if (include or lora) and not repo:
        raise RegistryError(f"{where} 写了 include / lora 但没有 repo")
    est = _num(f"{where}.est_download_gb", raw["est_download_gb"]) if "est_download_gb" in raw else 0.0
    route = Route(
        name=_str(f"{where}.name", raw["name"]) if "name" in raw else ename, engine=ename,
        vram_gb=_num(f"{where}.vram_gb", raw["vram_gb"], positive=True), est_download_gb=est, port=port,
        requires_sm=sm, repo=repo, include=include,
        revision=_str(f"{where}.revision", raw["revision"]) if raw.get("revision") else None,
        model_path=_str(f"{where}.model_path", raw["model_path"]) if "model_path" in raw else None,
        lora=lora, extra=extra, params={k: str(v) for k, v in params.items()},
        caps={**model_caps, **_parse_caps(f"{where}.caps", raw.get("caps"))},
        ready_timeout_s=_num(f"{where}.ready_timeout_s", raw.get("ready_timeout_s", 1800), positive=True))

    clash = set(route.params) & (engine_provides(engine) | {"model_dir", "extra", "model_path", "lora_dir", "lora_file"})
    if clash:
        raise RegistryError(f"{where}.params 的名字与内置占位符冲突: {', '.join(sorted(clash))}")
    missing = placeholders(engine.launch) - route_provides(engine, route)
    if missing:
        raise RegistryError(f"{where} 的 engines.{ename}.launch 使用了没有来源的 {_names(missing)}"
                            "（可由 params、model_path、lora 或引擎安装提供）")
    if route.extra and EXTRA not in engine.launch:
        raise RegistryError(f"{where}.extra 非空，但 engines.{ename}.launch 没有 {EXTRA}")
    if engine.ready.identity == "model_path" and not route.model_path:
        raise RegistryError(f"{where}.model_path 必须填写（engines.{ename}.ready.identity 是 model_path）")
    return route


def _parse_rules(raw: Any, groups: set[str]) -> tuple[Rule, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RegistryError("rules 必须是列表")
    out: list[Rule] = []
    limited: set[str] = set()
    for i, r in enumerate(raw):
        where = f"rules[{i}]"
        d = _keys(where, r, {"group", "max_online", "on_conflict", "requires_online_below"}, {"group"})
        group = d["group"]
        if group not in groups:
            raise RegistryError(f"{where}.group 未知: {group!r}，现有: {', '.join(sorted(groups))}")
        if "on_conflict" in d and "max_online" not in d:
            raise RegistryError(f"{where}.on_conflict 只能和 max_online 一起用")
        if "max_online" not in d and "requires_online_below" not in d:
            raise RegistryError(f"{where} 需要 max_online 或 requires_online_below")
        if "max_online" in d:
            if group in limited:
                raise RegistryError(f"{where}: {group} 组重复定义了 max_online")
            limited.add(group)
            conflict = d.get("on_conflict", "refuse")
            if conflict not in ("swap", "refuse"):
                raise RegistryError(f"{where}.on_conflict 必须是 swap 或 refuse，得到 {conflict!r}")
            out.append(MaxOnline(group, _int(f"{where}.max_online", d["max_online"], 1), conflict == "swap"))
        if "requires_online_below" in d:
            b = _keys(f"{where}.requires_online_below", d["requires_online_below"], {"group", "vram_gb"},
                      {"group", "vram_gb"})
            if b["group"] not in groups:
                raise RegistryError(f"{where}.requires_online_below.group 未知: {b['group']!r}，现有: {', '.join(sorted(groups))}")
            if b["group"] == group:
                raise RegistryError(f"{where}.requires_online_below.group 不能和 group 相同")
            out.append(RequiresBelow(group, b["group"], _num(f"{where}.requires_online_below.vram_gb",
                                                            b["vram_gb"], positive=True)))
    return tuple(out)


class Registry:
    def __init__(self, raw: Any) -> None:
        top = _keys("models.yaml", raw, TOP_KEYS, {"engines", "models"})
        paths = _keys("paths", top.get("paths", {}), PATH_KEYS)
        self.models_dir = Path(paths.get("models_dir", "/content/models"))
        self.outputs_dir = Path(paths.get("outputs_dir", "/content/outputs"))
        self.logs_dir = Path(paths.get("logs_dir", "/content/lm/logs"))
        self.state_file = Path(paths.get("state_file", "/content/lm/state.json"))
        self.engines_dir = Path(paths.get("engines_dir", "/content/lm/engines"))
        self.venvs_dir = Path(paths.get("venvs_dir", "/content/lm/venvs"))
        self.holds_dir = Path(paths.get("holds_dir", "/content/lm/holds"))
        self.vram_margin_gb = _num("vram_margin_gb", top.get("vram_margin_gb", 4))
        self.disk_reserve_gb = _num("disk_reserve_gb", top.get("disk_reserve_gb", 10))
        self.idle_unload_minutes = _num("idle_unload_minutes", top.get("idle_unload_minutes", 45))

        raw_engines = top["engines"]
        if not isinstance(raw_engines, dict) or not raw_engines:
            raise RegistryError("engines 必须是非空映射")
        self.engines: dict[str, Engine] = {n: _parse_engine(n, e) for n, e in raw_engines.items()}

        raw_models = top["models"]
        if not isinstance(raw_models, dict) or not raw_models:
            raise RegistryError("models 必须是非空映射")
        explicit: dict[int, str] = {}
        route_ports: dict[str, list[int | None]] = {}
        for mid, m in raw_models.items():
            where = f"models.{mid}"
            _keys(where, m, MODEL_KEYS, {"group", "serves", "routes"})
            if not isinstance(m["routes"], list) or not m["routes"]:
                raise RegistryError(f"{where}.routes 必须是非空列表（每条路线写 engine、vram_gb 和权重来源）")
            route_ports[mid] = []
            for i, r in enumerate(m["routes"]):
                port = _route_port(f"{where}.routes[{i}]", r)
                if port is not None:
                    owner = explicit.setdefault(port, mid)
                    if owner != mid:
                        raise RegistryError(f"端口 {port} 被 {owner} 和 {mid} 同时使用，每个模型需要独立端口")
                route_ports[mid].append(port)

        taken = set(explicit)
        models: dict[str, ModelSpec] = {}
        for mid, m in raw_models.items():
            where = f"models.{mid}"
            group = _str(f"{where}.group", m["group"])
            serves = _strs(f"{where}.serves", m["serves"], nonempty=True)
            caps = _parse_caps(f"{where}.caps", m.get("caps"))
            auto: int | None = None
            routes: list[Route] = []
            for i, (r, port) in enumerate(zip(m["routes"], route_ports[mid])):
                if port is None:
                    if auto is None:
                        auto = AUTO_PORT_BASE
                        while auto in taken:
                            auto += 1
                        taken.add(auto)
                    port = auto
                routes.append(_parse_route(f"{where}.routes[{i}]", r, caps, self.engines, port))
            names = [r.name for r in routes]
            dup = next((n for n in names if names.count(n) > 1), None)
            if dup:
                raise RegistryError(f"{where} 的路线名重复: {dup!r}（同引擎多条路线请加 name）")
            models[mid] = ModelSpec(mid, group, serves, _str(f"{where}.display", m["display"]) if "display" in m else mid,
                                    caps, tuple(routes))
        self.models = models
        self.rules = _parse_rules(top.get("rules"), {m.group for m in models.values()})

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def model(self, model_id: str) -> ModelSpec:
        if model_id not in self.models:
            raise KeyError(f"未知模型 {model_id!r}，可选: {', '.join(self.models)}")
        return self.models[model_id]

    def routes(self, model_id: str) -> tuple[Route, ...]:
        return self.model(model_id).routes

    def all_routes(self) -> list[tuple[str, Route]]:
        return [(mid, r) for mid, m in self.models.items() for r in m.routes]

    def choose(self, model_id: str, compute_cap: str, route: str | None = None) -> "RouteChoice":
        return choose_route(model_id, self.routes(model_id), compute_cap, route)

    def ids_serving(self, capability: str) -> list[str]:
        return [mid for mid, m in self.models.items() if capability in m.serves]

    @property
    def specs(self) -> dict[str, Spec]:
        """Estimates from each model's first route; the manager overrides these with the chosen route."""
        return {mid: Spec(mid, m.group, m.routes[0].vram_gb) for mid, m in self.models.items()}


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
