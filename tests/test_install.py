import asyncio
import dataclasses
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import pytest
import yaml

from controller.backends.server import ServerBackend
from controller.install import CODE_DIR, MARKER, InstallError, Installed, Installer
from controller.procs import pid_alive
from ctl_helpers import REGISTRY_YAML, build, tmp_registry


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Net:
    """Offline stand-in for the network: url -> bytes, and a log of what was fetched."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.fetched: list[str] = []

    async def __call__(self, url, dest):
        self.fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[url])


class Runner:
    def __init__(self, effect=None) -> None:
        self.calls: list[tuple[list[str], dict | None]] = []
        self.effect = effect

    async def __call__(self, cmd, env):
        self.calls.append((cmd, env))
        if self.effect:
            self.effect(cmd)


def registry_with(engine_name, mutate):
    raw = yaml.safe_load(REGISTRY_YAML.read_text())
    mutate(raw["engines"][engine_name]["install"])
    return tmp_registry(raw)


def marker(reg, name):
    return reg.engines_dir / name / MARKER


SCRIPT = b"#!/bin/sh\necho install\n"
SUMS = b"deadbeef  other.txt\n" + f"{sha(SCRIPT)}  install.sh\n".encode()
BASE = "https://github.com/Niko1221/Strata/releases/download/v0.2.20"


def script_setup(tmp_path, **spec):
    bin_path = tmp_path / "bin" / "strata"
    reg = registry_with("strata", lambda i: i.update(bin=str(bin_path), **spec))
    net = Net({f"{BASE}/install.sh": SCRIPT, f"{BASE}/SHA256SUMS.txt": SUMS})

    def create(cmd):
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        bin_path.write_text("#!/bin/sh\n")
    return reg, net, Runner(create), bin_path


def run(coro):
    return asyncio.run(coro)


def test_script_install_verifies_against_sums_runs_with_env_and_records_a_marker(tmp_path):
    reg, net, runner, bin_path = script_setup(tmp_path)
    installer = Installer(reg, fetch=net)
    got = run(installer.ensure(reg.engines["strata"], tmp_path, runner))
    assert got.bin == bin_path
    (cmd, env), = runner.calls
    assert cmd[0] == "sh" and cmd[1].endswith("install.sh")
    assert env["STRATA_FEED"] == BASE
    assert json.loads(marker(reg, "strata").read_text())["bin"] == str(bin_path)
    assert installer.installed(reg.engines["strata"]) == got


def test_second_ensure_is_idempotent_and_does_not_touch_the_network(tmp_path):
    reg, net, runner, _ = script_setup(tmp_path)
    installer = Installer(reg, fetch=net)
    first = run(installer.ensure(reg.engines["strata"], tmp_path, runner))
    net.fetched.clear()
    runner.calls.clear()
    assert run(Installer(reg, fetch=net).ensure(reg.engines["strata"], tmp_path, runner)) == first
    assert net.fetched == [] and runner.calls == []


def test_changed_install_spec_reinstalls(tmp_path):
    reg, net, runner, _ = script_setup(tmp_path)
    installer = Installer(reg, fetch=net)
    engine = reg.engines["strata"]
    run(installer.ensure(engine, tmp_path, runner))
    changed = dataclasses.replace(engine, install=dataclasses.replace(engine.install, env={"STRATA_FEED": "x"}))
    assert installer.installed(changed) is None
    run(installer.ensure(changed, tmp_path, runner))
    assert len(runner.calls) == 2 and installer.installed(changed) is not None


def test_missing_binary_means_not_installed(tmp_path):
    reg, net, runner, bin_path = script_setup(tmp_path)
    installer = Installer(reg, fetch=net)
    run(installer.ensure(reg.engines["strata"], tmp_path, runner))
    bin_path.unlink()
    assert installer.installed(reg.engines["strata"]) is None


def test_sha256_mismatch_refuses_runs_nothing_and_leaves_no_marker(tmp_path):
    reg, net, runner, _ = script_setup(tmp_path)
    net.files[f"{BASE}/install.sh"] = b"#!/bin/sh\nrm -rf /\n"
    with pytest.raises(InstallError, match="校验失败"):
        run(Installer(reg, fetch=net).ensure(reg.engines["strata"], tmp_path, runner))
    assert runner.calls == [] and not marker(reg, "strata").exists()
    assert not (reg.engines_dir / "strata" / "install.sh").exists()


def test_explicit_sha256_is_checked_without_a_sums_file(tmp_path):
    reg, net, runner, _ = script_setup(tmp_path, sha256="0" * 64)
    with pytest.raises(InstallError, match="校验失败"):
        run(Installer(reg, fetch=net).ensure(reg.engines["strata"], tmp_path, runner))
    assert f"{BASE}/SHA256SUMS.txt" not in net.fetched and not marker(reg, "strata").exists()


def test_script_missing_from_sums_is_refused(tmp_path):
    reg, net, runner, _ = script_setup(tmp_path)
    net.files[f"{BASE}/SHA256SUMS.txt"] = b"deadbeef  other.txt\n"
    with pytest.raises(InstallError, match="没有 install.sh 的校验和"):
        run(Installer(reg, fetch=net).ensure(reg.engines["strata"], tmp_path, runner))
    assert runner.calls == []


def tar_with(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def archive_setup():
    main = tar_with({"llama-b1/llama-server": b"#!/bin/sh\n", "llama-b1/libggml.so": b"ggml"})
    cudart = tar_with({"cudart-b1/libcudart.so.12": b"cudart"})
    reg = registry_with("llama-cpp", lambda i: None)
    inst = reg.engines["llama-cpp"].install
    files = {inst.archives[0].url: main, inst.archives[1].url: cudart}
    reg = registry_with("llama-cpp", lambda i: (i.update(sha256=sha(main)),
                                                i["extra_archives"][0].update(sha256=sha(cudart))))
    return reg, Net(files)


def test_archive_install_extracts_all_finds_bin_and_sets_library_path(tmp_path):
    reg, net = archive_setup()
    got = run(Installer(reg, fetch=net).ensure(reg.engines["llama-cpp"], tmp_path, Runner()))
    assert got.bin.name == "llama-server" and got.bin.exists() and os.access(got.bin, os.X_OK)
    cudart_dir = got.bin.parent.parent / "cudart-b1"
    assert (cudart_dir / "libcudart.so.12").exists()
    env = got.env({"LD_LIBRARY_PATH": "/usr/lib"})
    paths = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert {str(got.bin.parent), str(cudart_dir)} <= set(paths), "companion archive libs must be loadable"
    assert paths[-1] == "/usr/lib" and "PYTHONPATH" not in env
    assert not (reg.engines_dir / "llama-cpp" / "download").exists()


def test_archive_checksum_mismatch_extracts_nothing_and_leaves_no_marker(tmp_path):
    reg, net = archive_setup()
    extra_url = reg.engines["llama-cpp"].install.archives[1].url
    net.files[extra_url] = tar_with({"evil.so": b"x"})
    with pytest.raises(InstallError, match="校验失败"):
        run(Installer(reg, fetch=net).ensure(reg.engines["llama-cpp"], tmp_path, Runner()))
    assert not marker(reg, "llama-cpp").exists()
    assert not (reg.engines_dir / "llama-cpp" / "tree").exists()


def test_archive_without_the_binary_is_an_error(tmp_path):
    reg, net = archive_setup()
    engine = reg.engines["llama-cpp"]
    engine = dataclasses.replace(engine, install=dataclasses.replace(engine.install, bin="missing-server"))
    with pytest.raises(InstallError, match="missing-server"):
        run(Installer(reg, fetch=net).ensure(engine, tmp_path, Runner()))
    assert not marker(reg, "llama-cpp").exists()


def test_venv_install_creates_system_site_venv_and_pips_after_expanding_model_dir(tmp_path):
    reg = tmp_registry()
    runner = Runner()
    model_dir = tmp_path / "models" / "pika"
    got = run(Installer(reg).ensure(reg.engines["diffusers"], model_dir, runner))
    venv = reg.venvs_dir / "diffusers"
    assert got.venv_python == venv / "bin" / "python" and got.bin is None
    cmds = [c for c, _ in runner.calls]
    assert cmds[0] == [sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)]
    assert cmds[1][:5] == [str(got.venv_python), "-m", "pip", "install", "-q"]
    assert cmds[1][5:] == list(reg.engines["diffusers"].install.pip)
    assert cmds[2][-2:] == ["-r", f"{model_dir}/examples/requirements-diffusers.txt"]
    assert str(CODE_DIR) in got.env({})["PYTHONPATH"].split(os.pathsep)
    assert marker(reg, "diffusers").exists()


def test_failed_install_command_leaves_no_marker(tmp_path):
    reg = tmp_registry()

    def boom(cmd):
        raise RuntimeError("pip 退出码 1")
    with pytest.raises(RuntimeError):
        run(Installer(reg).ensure(reg.engines["diffusers"], tmp_path, Runner(boom)))
    assert not marker(reg, "diffusers").exists()


def test_engine_without_install_needs_nothing(tmp_path):
    raw = yaml.safe_load(REGISTRY_YAML.read_text())
    raw["engines"]["strata"].pop("install")
    raw["engines"]["strata"]["launch"] = [a for a in raw["engines"]["strata"]["launch"] if a != "{bin}"]
    raw["engines"]["strata"]["stop"] = ["strata", "launch", "--stop"]
    reg = tmp_registry(raw)
    assert run(Installer(reg).ensure(reg.engines["strata"], tmp_path, Runner())) == Installed()


def test_concurrent_loads_of_one_engine_install_once(tmp_path):
    reg, net, runner, _ = script_setup(tmp_path)
    installer = Installer(reg, fetch=net)

    async def both():
        return await asyncio.gather(*(installer.ensure(reg.engines["strata"], tmp_path, runner) for _ in range(2)))
    a, b = run(both())
    assert a == b and len(runner.calls) == 1


# ---- through the backend: state, cancellation, ordering ------------------------------------------

def test_backend_installs_first_for_script_engines_and_after_the_download_for_pip_files():
    events = []

    class Spy(Installer):
        def installed(self, engine):
            return None

        async def ensure(self, engine, model_dir, run):
            events.append(f"install:{engine.name}")
            return Installed(bin=Path("/x/bin"), venv_python=Path("/x/python"))

    class Recording(ServerBackend):
        async def download(self):
            events.append("download")

        async def _launch(self):
            events.append("launch")

        async def _wait_ready(self, timeout):
            events.append("ready")

    from controller.backends.jobs import JobsBackend

    class RecordingJobs(JobsBackend):
        download = Recording.download
        _launch = Recording._launch
        _wait_ready = Recording._wait_ready

    reg = tmp_registry()
    run(build(Recording, "inkling", reg, Spy(reg)).start())
    assert events == ["install:strata", "download", "launch", "ready"]
    events.clear()
    run(build(RecordingJobs, "pika", reg, Spy(reg)).start())
    assert events == ["download", "install:diffusers", "launch", "ready"]


def test_install_state_is_visible_and_cancel_kills_the_install_process_group(tmp_path):
    script = f"#!/bin/sh\nsleep 30 & echo $! > {tmp_path / 'child.pid'}\nwait\n".encode()
    bin_path = tmp_path / "bin" / "strata"
    reg = registry_with("strata", lambda i: (i.update(bin=str(bin_path), sha256=sha(script)), i.pop("sums_url")))
    installer = Installer(reg, fetch=Net({reg.engines["strata"].install.url: script}))

    async def go():
        b = build(ServerBackend, "inkling", reg, installer)
        task = asyncio.create_task(b.start())
        pidfile = tmp_path / "child.pid"
        for _ in range(200):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            await asyncio.sleep(0.05)
        assert b.state == "installing"
        child = int(pidfile.read_text())
        assert pid_alive(child)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        deadline = time.monotonic() + 5
        while pid_alive(child) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert not pid_alive(child), "install child survived cancellation"
        assert b.state == "stopped" and not b._job_pgids
    run(go())
    assert not marker(reg, "strata").exists()


def test_failed_install_surfaces_as_error_state_without_downloading_weights(tmp_path):
    reg = tmp_registry()
    net = Net({})

    class NoDownload(ServerBackend):
        async def download(self):
            raise AssertionError("weights must not download after a failed install")

    async def go():
        b = build(NoDownload, "inkling", reg, Installer(reg, fetch=net))
        await b.start()
        return b
    b = run(go())
    assert b.state == "error" and "SHA256SUMS" in b.error


def test_archive_with_requirements_builds_the_engines_own_venv(tmp_path):
    engine_tar = tar_with({"eng-1/bin/eng": b"#!/bin/sh\n", "eng-1/requirements.txt": b"jinja2\n"})
    reg = registry_with("llama-cpp", lambda i: (i.update(url="https://x/eng-1.tar.gz", sha256=sha(engine_tar),
                                                         bin="eng", requirements="requirements.txt"),
                                                i.pop("extra_archives")))
    runner = Runner()
    got = run(Installer(reg, fetch=Net({"https://x/eng-1.tar.gz": engine_tar})).ensure(
        reg.engines["llama-cpp"], tmp_path, runner))
    top = got.bin.parent.parent
    assert got.venv_python == top / ".venv" / "bin" / "python", "engine finds <top>/.venv by itself"
    cmds = [c for c, _ in runner.calls]
    assert cmds[0] == [sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(top / ".venv")]
    assert cmds[1] == [str(got.venv_python), "-m", "pip", "install", "-q", "-r", str(top / "requirements.txt")]


def test_archive_requirements_missing_from_the_archive_is_an_install_error(tmp_path):
    engine_tar = tar_with({"eng-1/bin/eng": b"#!/bin/sh\n"})
    reg = registry_with("llama-cpp", lambda i: (i.update(url="https://x/eng-1.tar.gz", sha256=sha(engine_tar),
                                                         bin="eng", requirements="requirements.txt"),
                                                i.pop("extra_archives")))
    with pytest.raises(InstallError, match="requirements.txt"):
        run(Installer(reg, fetch=Net({"https://x/eng-1.tar.gz": engine_tar})).ensure(
            reg.engines["llama-cpp"], tmp_path, Runner()))
    assert not marker(reg, "llama-cpp").exists()
