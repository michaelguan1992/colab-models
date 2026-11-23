"""Mac-side exclusive lock so two `lm` commands never mutate state at the same time."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager

from .config import STATE_DIR

LOCK_PATH = STATE_DIR / "lock"


@contextmanager
def state_lock(wait: bool = True) -> Iterator[bool]:
    """Hold the lock for the block. Yields False only when wait=False and someone else holds it."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                yield False
                return
            print("另一个 lm 命令正在运行，等待…", flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
