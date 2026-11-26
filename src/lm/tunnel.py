"""ssh -L port forward to the controller, using `colab ssh --proxy-mode` as ProxyCommand."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Literal

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


SERVICE = "lm-controller"


class PortConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class Probe:
    state: Literal["down", "ours", "foreign"]
    version: str = ""


def probe(cfg: Config, timeout: float = 3) -> Probe:
    try:
        r = httpx.get(f"{cfg.base_url}/health", timeout=timeout)
    except httpx.HTTPError:
        return Probe("down")
    if r.status_code != 200:
        return Probe("down")
    try:
        body = r.json()
    except ValueError:
        return Probe("foreign")
    if isinstance(body, dict) and body.get("service") == SERVICE:
        return Probe("ours", str(body.get("version", "")))
    # Controllers before /health carried an identity answered exactly {"ok": true}; an empty
    # version makes lm up redeploy over them instead of refusing the port.
    if body == {"ok": True}:
        return Probe("ours", "")
    return Probe("foreign")


def healthy(cfg: Config, timeout: float = 3) -> bool:
    return probe(cfg, timeout).state == "ours"


def conflict_message(cfg: Config) -> str:
    return f"本地端口 {cfg.local_port} 被其他程序占用。在 config.toml 里改 local_port 换一个端口。"


def _host(cfg: Config) -> str:
    return f"{cfg.ssh_user}@colab-{cfg.session}"


def master_alive(cfg: Config) -> bool:
    return _master_alive_host(_host(cfg))


def _master_alive_host(host: str) -> bool:
    if not CONTROL_PATH.exists():
        return False
    r = subprocess.run(["ssh", "-S", str(CONTROL_PATH), "-O", "check", host], capture_output=True, text=True)
    return r.returncode == 0


def run(cfg: Config, command: str, capture: bool = False,
        input: str | None = None) -> subprocess.CompletedProcess | None:
    """Run a remote shell command over the tunnel's master connection; None if no master."""
    if not master_alive(cfg):
        return None
    return subprocess.run(["ssh", "-S", str(CONTROL_PATH), "-o", "LogLevel=ERROR", _host(cfg),
                           shlex.join(["bash", "-lc", command])], text=True, capture_output=capture,
                          input=input)


def forward(cfg: Config, port: int) -> bool:
    """Forward one more local port to the same port on the runtime, over the master connection.

    Asking again for a forward the master already has succeeds. A reconnect drops it, so callers ask each time.
    """
    if not master_alive(cfg):
        return False
    spec = f"127.0.0.1:{port}:127.0.0.1:{port}"
    r = subprocess.run(["ssh", "-S", str(CONTROL_PATH), "-O", "forward", "-L", spec, _host(cfg)],
                       capture_output=True, text=True)
    return r.returncode == 0


def start(cfg: Config, proxy_command: str, wait_s: float = 60, require_health: bool = True) -> None:
    """Open the tunnel. require_health=False only waits for the ssh master, for use before the
    controller exists (deploy runs over the master).

    The pid on file is a keeper that reopens ssh whenever it exits after having connected once
    (network drop, Mac waking from sleep), so harnesses see a gap of seconds, not minutes.
    """
    stop()
    if probe(cfg).state == "foreign":
        raise PortConflict(conflict_message(cfg))
    CONTROL_PATH.unlink(missing_ok=True)
    cmd = [
        "ssh", "-N",
        "-L", f"127.0.0.1:{cfg.local_port}:127.0.0.1:{cfg.remote_port}",
        "-o", f"ProxyCommand={proxy_command}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ExitOnForwardFailure=yes",
        # A dead link (sleep, Wi-Fi change) is noticed after ~30s instead of ~2min.
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=2",
        "-o", "LogLevel=ERROR",
        "-o", "ControlMaster=yes",
        "-o", f"ControlPath={CONTROL_PATH}",
        *cfg.ssh_extra,
        _host(cfg),
    ]
    log = open(LOG_FILE, "ab")
    p = subprocess.Popen([sys.executable, "-m", "lm.tunnel", _host(cfg), json.dumps(cmd)],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    PID_FILE.write_text(str(p.pid))
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"ssh 隧道退出 (code {p.returncode})，见 {LOG_FILE}")
        if (healthy(cfg) if require_health else master_alive(cfg)):
            return
        time.sleep(1)
    raise RuntimeError(f"{wait_s:.0f}s 内隧道未连通，见 {LOG_FILE}")


def keep(host: str, cmd: list[str], poll_s: float = 1, max_backoff_s: float = 30,
         alive=None, sleep=time.sleep) -> int:
    """Run ssh, and run it again each time it exits, once it has connected at least once.

    A first attempt that never connects returns ssh's exit code, so `start` still fails fast
    on a wrong session or a taken port. Retries back off from 1s to max_backoff_s and reset
    once a connection is up again.
    """
    alive = alive or (lambda: _master_alive_host(host))
    ever_up, backoff = False, 1.0
    while True:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
        up = False
        while p.poll() is None:
            if not up and alive():
                up = ever_up = True
                backoff = 1.0
            sleep(poll_s)
        if not ever_up:
            return p.returncode
        print(f"{time.strftime('%H:%M:%S')} ssh 退出 (code {p.returncode})，{backoff:.0f}s 后重连", flush=True)
        sleep(backoff)
        backoff = min(backoff * 2, max_backoff_s)


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


if __name__ == "__main__":  # the keeper process spawned by start()
    sys.exit(keep(sys.argv[1], json.loads(sys.argv[2])))
