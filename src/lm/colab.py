"""Thin wrapper over the Google Colab CLI (`uv tool install google-colab-cli`)."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from . import tunnel
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


def remote_files(src: Path) -> list[Path]:
    """Everything `deploy` uploads from remote/, as sorted paths relative to src."""
    return sorted(p.relative_to(src) for p in src.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts)


def code_version(src: Path) -> str:
    h = hashlib.sha256()
    for rel in remote_files(src):
        h.update(rel.as_posix().encode() + b"\0" + (src / rel).read_bytes() + b"\0")
    return h.hexdigest()[:12]


class Colab:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if not shutil.which(cfg.colab_bin):
            raise ColabError("找不到 colab CLI。先运行: uv tool install google-colab-cli")

    def _run(self, *args: str, capture: bool = False, check: bool = True,
             input: str | None = None) -> subprocess.CompletedProcess:
        cmd = [self.cfg.colab_bin, *args]
        r = subprocess.run(cmd, text=True, capture_output=capture, input=input)
        if check and r.returncode != 0:
            detail = (r.stderr or r.stdout or "").strip() if capture else ""
            raise ColabError(f"{shlex.join(cmd)} 失败 (code {r.returncode}) {detail}")
        return r

    # ---- sessions --------------------------------------------------------
    def session_active(self) -> bool | None:
        """True / False when `colab sessions` answered; None when the CLI itself failed (auth, network)."""
        # `colab status` exits 0 even when the session is missing, so parse `colab sessions`.
        r = self._run("sessions", capture=True, check=False)
        if r.returncode != 0:
            return None
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
        if self.session_active() is True:
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
    def ssh(self, command: str, capture: bool = False, check: bool = True,
            input: str | None = None) -> subprocess.CompletedProcess:
        r = tunnel.run(self.cfg, command, capture=capture, input=input)  # reuse the tunnel's connection if up
        if r is not None:
            if check and r.returncode != 0:
                detail = (r.stderr or r.stdout or "").strip() if capture else ""
                raise ColabError(f"远端命令失败 (code {r.returncode}) {detail}")
            return r
        return self._run("ssh", "-s", self.cfg.session, "--", "bash", "-lc", command,
                         capture=capture, check=check, input=input)

    def upload(self, local: Path, remote: str) -> None:
        self._run("upload", "-s", self.cfg.session, str(local), remote)

    def download(self, remote: str, local: Path) -> None:
        self._run("download", "-s", self.cfg.session, remote, str(local))

    @property
    def version(self) -> str:
        return code_version(Path(self.cfg.remote_src))

    def push_hf_token(self) -> bool:
        """Copy the Mac's HF token to the runtime over ssh stdin, so it never appears in an argv or log."""
        token = os.environ.get(self.cfg.hf_token_env, "").strip()
        if not token:
            return False
        # colab ssh may not forward stdin; never replace a good token with an empty file.
        self.ssh("umask 077 && mkdir -p ~/.cache/huggingface && t=~/.cache/huggingface/token.new "
                 "&& cat > $t && if [ -s $t ]; then chmod 600 $t && mv $t ~/.cache/huggingface/token; "
                 "else rm -f $t; echo 'HF token 未送达（stdin 未转发）' >&2; fi", input=token)
        return True

    def deploy(self, load: str | None) -> None:
        """Ship remote/ to the runtime and (re)start the controller there."""
        src = Path(self.cfg.remote_src)
        if not (src / "bootstrap.sh").exists():
            raise ColabError(f"找不到远端代码目录 {src}")
        version = self.version
        with tempfile.TemporaryDirectory() as tmp:
            tgz = Path(tmp) / "lm-remote.tgz"
            with tarfile.open(tgz, "w:gz") as tar:
                for rel in remote_files(src):
                    tar.add(src / rel, arcname=rel.as_posix())
            self.upload(tgz, "/content/lm-remote.tgz")
        if self.push_hf_token():
            print("已同步 HF token 到运行时")
        else:
            print(f"注意：Mac 上没有 ${self.cfg.hf_token_env}，私有仓库（如 Basalt 包）会下载失败", file=sys.stderr)
        rd = shlex.quote(self.cfg.remote_dir)
        env = {"LM_VERSION": version, "LM_PORT": str(self.cfg.remote_port)}
        if load:
            env["LOAD"] = load
        assigns = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        self.ssh(f"mkdir -p {rd} && tar xzf /content/lm-remote.tgz -C {rd} && {assigns} bash {rd}/bootstrap.sh")

    def proxy_command(self) -> str:
        return shlex.join([self.cfg.colab_bin, "ssh", "-s", self.cfg.session, "--proxy-mode"])


def eprint(*a) -> None:
    print(*a, file=sys.stderr)
