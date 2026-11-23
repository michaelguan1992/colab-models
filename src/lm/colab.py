"""Thin wrapper over the Google Colab CLI (`uv tool install google-colab-cli`)."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from .config import Config


class ColabError(RuntimeError):
    pass


class Colab:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if not shutil.which(cfg.colab_bin):
            raise ColabError("找不到 colab CLI。先运行: uv tool install google-colab-cli")

    def _run(self, *args: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.cfg.colab_bin, *args]
        r = subprocess.run(cmd, text=True, capture_output=capture)
        if check and r.returncode != 0:
            detail = (r.stderr or r.stdout or "").strip() if capture else ""
            raise ColabError(f"{shlex.join(cmd)} 失败 (code {r.returncode}) {detail}")
        return r

    # ---- sessions --------------------------------------------------------
    def session_active(self) -> bool:
        r = self._run("status", "-s", self.cfg.session, capture=True, check=False)
        return r.returncode == 0

    def new(self) -> None:
        args = ["new", "-s", self.cfg.session, "--gpu", self.cfg.gpu]
        if self.cfg.high_mem:
            args.append("--high-mem")
        self._run(*args)

    def stop(self) -> None:
        self._run("stop", "-s", self.cfg.session)

    def usage(self) -> str:
        r = self._run("usage", capture=True, check=False)
        return (r.stdout or r.stderr or "").strip()

    # ---- remote execution ------------------------------------------------
    def ssh(self, command: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
        return self._run("ssh", "-s", self.cfg.session, "--", "bash", "-lc", command,
                         capture=capture, check=check)

    def upload(self, local: Path, remote: str) -> None:
        self._run("upload", "-s", self.cfg.session, str(local), remote)

    def download(self, remote: str, local: Path) -> None:
        self._run("download", "-s", self.cfg.session, remote, str(local))

    def deploy(self, load: str | None) -> None:
        """Ship remote/ to the runtime and (re)start the controller there."""
        src = Path(self.cfg.remote_src)
        if not (src / "bootstrap.sh").exists():
            raise ColabError(f"找不到远端代码目录 {src}")
        with tempfile.TemporaryDirectory() as tmp:
            tgz = Path(tmp) / "lm-remote.tgz"
            with tarfile.open(tgz, "w:gz") as tar:
                tar.add(src, arcname=".", filter=lambda ti: None if "__pycache__" in ti.name else ti)
            self.upload(tgz, "/content/lm-remote.tgz")
        rd = shlex.quote(self.cfg.remote_dir)
        env = f"LOAD={shlex.quote(load)} " if load else ""
        self.ssh(f"mkdir -p {rd} && tar xzf /content/lm-remote.tgz -C {rd} && {env}bash {rd}/bootstrap.sh")

    def proxy_command(self) -> str:
        return shlex.join([self.cfg.colab_bin, "ssh", "-s", self.cfg.session, "--proxy-mode"])


def eprint(*a) -> None:
    print(*a, file=sys.stderr)
