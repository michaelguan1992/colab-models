"""Command templates: ``{name}`` placeholders in launch / stop / pip_files, plus ``{extra}`` as a list."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from .spec import ArchiveInstall, Engine, Route, ScriptInstall, VenvInstall

PLACEHOLDER = re.compile(r"\{(\w+)\}")
EXTRA = "{extra}"
INSTALL_PLACEHOLDERS = frozenset({"model_dir", "models_dir"})


def placeholders(argv: Sequence[str]) -> set[str]:
    return {name for a in argv for name in PLACEHOLDER.findall(a)}


def engine_provides(engine: Engine) -> set[str]:
    """Names available wherever the engine runs, with or without a model (stop command)."""
    names = {"port", "models_dir", "outputs_dir"}
    if isinstance(engine.install, (ScriptInstall, ArchiveInstall)):
        names.add("bin")
    if isinstance(engine.install, VenvInstall) or (
            isinstance(engine.install, ArchiveInstall) and engine.install.requirements):
        names.add("venv_python")
    return names


def route_provides(engine: Engine, route: Route) -> set[str]:
    names = engine_provides(engine) | {"model_dir", "extra", *route.params}
    if route.model_path:
        names.add("model_path")
    return names


def engine_values(models_dir, outputs_dir, port: int, installed: Mapping[str, str]) -> dict[str, str]:
    return {"port": str(port), "models_dir": str(models_dir), "outputs_dir": str(outputs_dir), **installed}


def expand(argv: Sequence[str], values: Mapping[str, str], extra: Sequence[str] = ()) -> list[str]:
    out: list[str] = []
    for a in argv:
        if a == EXTRA:
            out.extend(extra)
        else:
            out.append(PLACEHOLDER.sub(lambda m: values[m.group(1)], a))
    return out
