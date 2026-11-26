"""Typed registry data. Built once by ``registry.py``; everything else trusts these shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

Shape = Literal["server", "jobs"]


@dataclass(frozen=True)
class Archive:
    url: str
    sha256: str


@dataclass(frozen=True)
class ScriptInstall:
    url: str
    bin: str
    sha256: str | None
    sums_url: str | None
    env: Mapping[str, str]


@dataclass(frozen=True)
class ArchiveInstall:
    archives: tuple[Archive, ...]  # the first is the engine itself, the rest are extracted next to it
    bin: str                       # file name searched for inside the extracted tree
    requirements: str | None = None  # pip -r this file (relative to the engine's top folder) into <top>/.venv


@dataclass(frozen=True)
class VenvInstall:
    pip: tuple[str, ...]
    pip_files: tuple[str, ...]     # may use {model_dir}, so they install after the weights download


Install = ScriptInstall | ArchiveInstall | VenvInstall


def install_needs_model(install: Install) -> bool:
    return isinstance(install, VenvInstall) and bool(install.pip_files)


@dataclass(frozen=True)
class Ready:
    path: str
    identity: Literal["model_path"] | None
    # Engines that answer /v1/models while still loading (Basalt): the served entry's dotted field must
    # equal this value before the model counts as ready.
    field: str | None = None
    equals: str | None = None
    # A web UI the user drives directly sends the controller no requests. This path answers JSON whose
    # top-level lists are its running and queued work; their total length keeps the idle timers off.
    busy: str | None = None


@dataclass(frozen=True)
class Engine:
    name: str
    shape: Shape
    apis: tuple[str, ...]
    install: Install | None
    launch: tuple[str, ...]
    stop: tuple[str, ...] | None
    daemonizes: bool
    ready: Ready
    requires_sm: tuple[int, ...]


@dataclass(frozen=True)
class Source:
    """Files from one HuggingFace repo, placed under ``into`` (relative to the model's weights folder)."""
    repo: str
    include: tuple[str, ...]
    revision: str | None = None
    into: str = ""


@dataclass(frozen=True)
class Route:
    """One way to run a model. ``requires_sm`` and ``caps`` are already merged with the engine / model."""
    name: str
    engine: str
    vram_gb: float
    est_download_gb: float
    port: int
    requires_sm: tuple[int, ...]
    repo: str | None
    include: tuple[str, ...]
    revision: str | None
    model_path: str | None
    extra: tuple[str, ...]
    params: Mapping[str, str]
    caps: Mapping[str, Any]
    ready_timeout_s: float
    vram_margin_gb: float | None = None   # overrides the registry-wide margin when this route is the load target
    also: tuple[Source, ...] = ()         # more repos whose files join the same weights folder

    def sources(self) -> tuple[Source, ...]:
        main = (Source(self.repo, self.include, self.revision),) if self.repo else ()
        return main + self.also

    def fits(self, sm: int | None) -> bool:
        return not self.requires_sm or (sm is not None and sm in self.requires_sm)

    def requirement(self) -> str:
        if not self.requires_sm:
            return "无特殊要求"
        return "GPU 计算能力 " + " / ".join(f"sm_{s}" for s in self.requires_sm)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    group: str
    serves: tuple[str, ...]
    display: str
    caps: Mapping[str, Any]
    routes: tuple[Route, ...]
