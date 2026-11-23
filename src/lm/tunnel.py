"""ssh -L port forward to the controller, using `colab ssh --proxy-mode` as ProxyCommand."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time

import httpx

from .config import STATE_DIR, Config

PID_FILE = STATE_DIR / "tunnel.pid"
# Colab allows one SSH connection per runtime (a second `colab ssh` gets HTTP 429), so the
# tunnel is an OpenSSH ControlMaster and remote commands are multiplexed over it.
CONTROL_PATH = STATE_DIR / "cm.sock"
LOG_FILE = STATE_DIR / "tunnel.log"


def _pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def healthy(cfg: Config, timeout: float = 3) -> bool:
    try:
        return httpx.get(f"{cfg.base_url}/health", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def _host(cfg: Config) -> str:
    return f"{cfg.ssh_user}@colab-{cfg.session}"


def master_alive(cfg: Config) -> bool:
    if not CONTROL_PATH.exists():
        return False
    r = subprocess.run(["ssh", "-S", str(CONTROL_PATH), "-O", "check", _host(cfg)],
                       capture_output=True, text=True)
    return r.returncode == 0


def run(cfg: Config, command: str, capture: bool = False) -> subprocess.CompletedProcess | None:
    """Run a remote shell command over the tunnel's master connection; None if no master."""
    if not master_alive(cfg):
        return None
    return subprocess.run(["ssh", "-S", str(CONTROL_PATH), "-o", "LogLevel=ERROR", _host(cfg),
                           shlex.join(["bash", "-lc", command])], text=True, capture_output=capture)


def start(cfg: Config, proxy_command: str, wait_s: float = 60) -> None:
    stop()
    CONTROL_PATH.unlink(missing_ok=True)
    cmd = [
        "ssh", "-N",
        "-L", f"127.0.0.1:{cfg.local_port}:127.0.0.1:{cfg.remote_port}",
        "-o", f"ProxyCommand={proxy_command}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=4",
        "-o", "LogLevel=ERROR",
        "-o", "ControlMaster=yes",
        "-o", f"ControlPath={CONTROL_PATH}",
        *cfg.ssh_extra,
        _host(cfg),
    ]
    log = open(LOG_FILE, "ab")
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    PID_FILE.write_text(str(p.pid))
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"ssh 隧道退出 (code {p.returncode})，见 {LOG_FILE}")
        if healthy(cfg):
            return
        time.sleep(1)
    raise RuntimeError(f"{wait_s:.0f}s 内隧道未连通，见 {LOG_FILE}")


def stop() -> bool:
    pid = _pid()
    PID_FILE.unlink(missing_ok=True)
    if pid is None:
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return True


def is_running() -> bool:
    return _pid() is not None
