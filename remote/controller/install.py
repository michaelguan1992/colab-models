"""Lazy, idempotent engine installs under /content/lm/engines/<name>/, keyed by a hash of the install spec."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Mapping

import httpx

from .registry import Registry
from .spec import ArchiveInstall, Engine, Install, ScriptInstall, VenvInstall
from .templating import expand

CODE_DIR = Path(__file__).resolve().parents[1]  # holds the `controller` package, for `-m controller.x` inside a venv
MARKER = "installed.json"

Fetch = Callable[[str, Path], Awaitable[None]]
Run = Callable[[list[str], "dict | None"], Awaitable[None]]


class InstallError(RuntimeError):
    pass


@dataclass(frozen=True)
class Installed:
    bin: Path | None = None
    lib_path: str | None = None
    venv_python: Path | None = None

    def values(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.bin:
            out["bin"] = str(self.bin)
        if self.venv_python:
            out["venv_python"] = str(self.venv_python)
        return out

    def env(self, base: Mapping[str, str]) -> dict[str, str]:
        env = dict(base)
        if self.lib_path:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [self.lib_path, env.get("LD_LIBRARY_PATH")]))
        if self.venv_python:
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(CODE_DIR), env.get("PYTHONPATH")]))
        return env

    def exists(self) -> bool:
        return all(p is None or p.exists() for p in (self.bin, self.venv_python))


async def http_fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(None, connect=15)) as client:
        async with client.stream("GET", url) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                async for chunk in r.aiter_bytes(1 << 20):
                    f.write(chunk)
    os.replace(part, dest)


def spec_hash(spec: Install) -> str:
    # "layout" bumps when what an install records changes, so old markers reinstall.
    blob = json.dumps({"kind": type(spec).__name__, "layout": 2, **dataclasses.asdict(spec)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")


class Installer:
    def __init__(self, registry: Registry, fetch: Fetch = http_fetch) -> None:
        self.registry = registry
        self.fetch = fetch
        self._locks: dict[str, asyncio.Lock] = {}

    def _root(self, engine: Engine) -> Path:
        return self.registry.engines_dir / engine.name

    def installed(self, engine: Engine) -> Installed | None:
        """The recorded install if its marker matches the current spec and its files are still there."""
        if engine.install is None:
            return Installed()
        try:
            marker = json.loads((self._root(engine) / MARKER).read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(marker, dict) or marker.get("hash") != spec_hash(engine.install):
            return None
        found = Installed(bin=Path(marker["bin"]) if marker.get("bin") else None,
                          lib_path=marker.get("lib_path") or None,
                          venv_python=Path(marker["venv_python"]) if marker.get("venv_python") else None)
        return found if found.exists() else None

    async def ensure(self, engine: Engine, model_dir: Path, run: Run) -> Installed:
        spec = engine.install
        if spec is None:
            return Installed()
        async with self._locks.setdefault(engine.name, asyncio.Lock()):
            have = self.installed(engine)
            if have is not None:
                return have
            root = self._root(engine)
            root.mkdir(parents=True, exist_ok=True)
            (root / MARKER).unlink(missing_ok=True)
            if isinstance(spec, ScriptInstall):
                result = await self._script(engine, spec, root, run)
            elif isinstance(spec, ArchiveInstall):
                result = await self._archive(engine, spec, root, run)
            else:
                result = await self._venv(engine, spec, model_dir, run)
            marker = {"hash": spec_hash(spec), **{k: str(v) if v else None for k, v in dataclasses.asdict(result).items()}}
            (root / MARKER).write_text(json.dumps(marker))
            return result

    async def _verified(self, engine: Engine, url: str, dest: Path, want: str) -> None:
        await self.fetch(url, dest)
        got = await asyncio.to_thread(_sha256_file, dest)
        if got != want.lower():
            dest.unlink(missing_ok=True)
            raise InstallError(f"{engine.name} 的 {dest.name} 校验失败：期望 sha256 {want[:12]}…，实际 {got[:12]}…，已拒绝安装")

    async def _script(self, engine: Engine, spec: ScriptInstall, root: Path, run: Run) -> Installed:
        script = root / Path(spec.url).name
        want = spec.sha256 or await self._listed_sum(engine, spec.sums_url, script.name, root)
        await self._verified(engine, spec.url, script, want)
        await run(["sh", str(script)], {**os.environ, **spec.env})
        bin_path = Path(spec.bin).expanduser()
        if not bin_path.exists():
            raise InstallError(f"{engine.name} 安装脚本跑完了，但没有找到 {bin_path}")
        return Installed(bin=bin_path)

    async def _listed_sum(self, engine: Engine, sums_url: str | None, name: str, root: Path) -> str:
        sums = root / "SHA256SUMS"
        await self.fetch(sums_url, sums)
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == name:
                return parts[0]
        raise InstallError(f"{sums_url} 里没有 {name} 的校验和")

    async def _archive(self, engine: Engine, spec: ArchiveInstall, root: Path, run: Run) -> Installed:
        downloads = root / "download"
        shutil.rmtree(downloads, ignore_errors=True)
        files = []
        for a in spec.archives:
            dest = downloads / Path(a.url).name
            await self._verified(engine, a.url, dest, a.sha256)
            files.append(dest)
        tree = root / "tree"
        shutil.rmtree(tree, ignore_errors=True)
        for f in files:
            await asyncio.to_thread(_extract, f, tree)
        shutil.rmtree(downloads, ignore_errors=True)
        found = next((p for p in sorted(tree.rglob(spec.bin)) if p.is_file()), None)
        if found is None:
            raise InstallError(f"{engine.name} 的压缩包里没有找到 {spec.bin}")
        found.chmod(found.stat().st_mode | 0o111)
        # Companion archives (llama.cpp's cudart bundle) extract beside the binary's folder, not into it.
        lib_dirs = sorted({found.parent, *(so.parent for so in tree.rglob("*.so*") if so.is_file())})
        lib_path = os.pathsep.join(str(d) for d in lib_dirs)
        if spec.requirements is None:
            return Installed(bin=found, lib_path=lib_path)
        top = tree / found.relative_to(tree).parts[0]
        req = top / spec.requirements
        if not req.is_file():
            raise InstallError(f"{engine.name} 的压缩包里没有找到 {spec.requirements}")
        # Engines like Basalt pick up <top>/.venv on their own; same borrowed-pip trick as the venv kind.
        python = top / ".venv" / "bin" / "python"
        await run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(top / ".venv")], None)
        await run([str(python), "-m", "pip", "install", "-q", "-r", str(req)], None)
        return Installed(bin=found, lib_path=lib_path, venv_python=python)

    async def _venv(self, engine: Engine, spec: VenvInstall, model_dir: Path, run: Run) -> Installed:
        venv = self.registry.venvs_dir / engine.name
        shutil.rmtree(venv, ignore_errors=True)
        python = venv / "bin" / "python"
        # Colab's python has no ensurepip; the venv borrows the system pip, which installs into the venv.
        await run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)], None)
        pip = [str(python), "-m", "pip", "install", "-q"]
        if spec.pip:
            await run([*pip, *spec.pip], None)
        values = {"model_dir": str(model_dir), "models_dir": str(self.registry.models_dir)}
        for f in spec.pip_files:
            await run([*pip, "-r", *expand([f], values)], None)
        return Installed(venv_python=python)
