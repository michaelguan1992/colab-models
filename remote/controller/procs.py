"""Process and port helpers shared by backends, the manager and startup cleanup."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import time


def pid_alive(pid: int) -> bool:
    try:
        import psutil
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
    except ImportError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def port_listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def kill_group(pgid: int, grace: float = 10.0) -> None:
    """SIGTERM the whole group, SIGKILL whatever survives ``grace`` seconds."""
    if pgid <= 1 or pgid == os.getpgrp():
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            return
        await asyncio.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    for _ in range(50):
        if not _group_alive(pgid):
            return
        await asyncio.sleep(0.1)


async def run_quiet(argv: list[str], timeout: float = 60.0) -> None:
    """Run a short helper command (an engine's stop); a missing binary or a hang is not an error."""
    try:
        p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.DEVNULL,
                                                 stderr=asyncio.subprocess.DEVNULL)
        await asyncio.wait_for(p.wait(), timeout=timeout)
    except (OSError, asyncio.TimeoutError):
        pass
