"""Thin wrapper over the Google Colab CLI (`uv tool install google-colab-cli`)."""

from __future__ import annotations

import os
import re
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


def classify_assign_error(output: str) -> str:
    """Map `colab new` failure output to capacity | rejected | quota | error."""
    o = output.lower()
    if "service unavailable" in o or re.search(r"\b50[234]\b", o) or "timed out" in o or "readtimeout" in o:
        return "capacity"
    if "toomanyassignments" in o or re.search(r"\b412\b", o):
        return "quota"
    if "bad request" in o or re.search(r"\b400\b", o) or "not eligible" in o or "entitle" in o:
        return "rejected"
    return "error"


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
        # `colab status` exits 0 even when the session is missing, so parse `colab sessions`.
        r = self._run("sessions", capture=True, check=False)
        return any(line.strip().startswith(f"[{self.cfg.session}]")
                   for line in (r.stdout or "").splitlines())

    def new(self, gpu: str, high_mem: bool = False) -> tuple[bool, str, str]:
        """One assignment request, no CLI-side retries (lm owns retry policy).

        Returns (ok, kind, detail); kind is "" on success, else classify_assign_error().
        """
        args = [self.cfg.colab_bin, "new", "-s", self.cfg.session, "--gpu", gpu]
        if high_mem:
            args.append("--high-mem")
        env = {**os.environ, "COLAB_CLI_ASSIGN_RETRIES": "0"}
        r = subprocess.run(args, text=True, capture_output=True, env=env)
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0 and "error" not in out.lower():
            return True, "", ""
        # A timed-out POST can still leave a (billed) runtime behind: adopt it if it exists.
        if self.session_active():
            return True, "", "请求报错，但会话已存在，直接使用"
        lines = [ln.strip() for ln in out.splitlines() if ln.strip() and "│" not in ln and "╭" not in ln and "╰" not in ln]
        return False, classify_assign_error(out), (lines[-1] if lines else f"exit {r.returncode}")

    def gpu_info(self) -> tuple[str, float]:
        """(name, total VRAM in GB) of the assigned runtime."""
        r = self.ssh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
                     capture=True, check=False)
        for line in (r.stdout or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                try:
                    return parts[0], float(parts[1]) / 1024
                except ValueError:
                    pass
        return "unknown", 0.0

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
