"""ssh -L port forward to the controller, using `colab ssh --proxy-mode` as ProxyCommand."""

from __future__ import annotations

import os
import signal
import subprocess
import time

import httpx

from .config import STATE_DIR, Config

PID_FILE = STATE_DIR / "tunnel.pid"
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


def start(cfg: Config, proxy_command: str, wait_s: float = 60) -> None:
    stop()
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
        *cfg.ssh_extra,
        f"{cfg.ssh_user}@colab-{cfg.session}",
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
