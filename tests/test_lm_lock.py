import threading
import time

import pytest

from lm import cli, lock
from lm.config import Config


@pytest.fixture(autouse=True)
def lock_path(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "LOCK_PATH", tmp_path / "lock")


def test_nonblocking_returns_false_while_held():
    with lock.state_lock():
        with lock.state_lock(wait=False) as got:
            assert got is False
    with lock.state_lock(wait=False) as got:
        assert got is True


def test_concurrent_acquisitions_serialize(capsys):
    order = []
    holding = threading.Event()
    release = threading.Event()

    def first():
        with lock.state_lock():
            order.append("a-in")
            holding.set()
            release.wait(5)
            order.append("a-out")

    def second():
        holding.wait(5)
        with lock.state_lock():
            order.append("b-in")

    ta, tb = threading.Thread(target=first), threading.Thread(target=second)
    ta.start()
    tb.start()
    holding.wait(5)
    time.sleep(0.2)
    assert order == ["a-in"]
    release.set()
    ta.join(5)
    tb.join(5)
    assert order == ["a-in", "a-out", "b-in"]
    assert "另一个 lm 命令正在运行，等待…" in capsys.readouterr().out


@pytest.mark.parametrize("cmd,locked", [("unload", True), ("down", True), ("load", True), ("up", True),
                                        ("tunnel", True), ("video", True), ("connect", True),
                                        ("status", False), ("logs", False)])
def test_main_locks_state_changing_commands(monkeypatch, cmd, locked):
    seen = []

    def probe(cfg, a):
        with lock.state_lock(wait=False) as got:
            seen.append(got)
        return 0

    monkeypatch.setattr(cli, "load", lambda: Config())
    for name in ("up", "down", "load", "unload", "tunnel", "video", "connect", "status", "logs"):
        monkeypatch.setattr(cli, f"cmd_{name}", probe)
    argv = {"load": ["load", "m"], "video": ["video", "p"], "connect": ["connect", "dsh"]}.get(cmd, [cmd])
    assert cli.main(argv) == 0
    assert seen == [not locked]
