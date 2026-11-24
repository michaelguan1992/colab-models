import httpx
import pytest

from lm import cli, lock, tunnel
from lm.colab import ColabError
from lm.config import Config

CFG = Config(idle_stop_minutes=1)  # limit = 60s


class FakeColab:
    def __init__(self, active=True):
        self.active = active
        self.stopped = 0
        self.fail_stop = False

    def session_active(self):
        return self.active

    def stop(self):
        if self.fail_stop:
            raise ColabError("stop failed")
        self.stopped += 1

    def proxy_command(self):
        return "proxy"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(cli, "SESSION_STATE", tmp_path / "session.json")
    monkeypatch.setattr(tunnel, "healthy", lambda cfg, timeout=3: False)
    monkeypatch.setattr(tunnel, "stop", lambda: False)
    monkeypatch.setattr(tunnel, "start", lambda cfg, proxy, **kw: None)


def idle(seconds, busy=False):
    return {"idle_seconds": seconds, "busy": busy}


def test_confirmed_absent_session_exits():
    c = FakeColab(active=False)
    assert cli.watchdog_tick(CFG, c, cli.WatchState(), 0, None) is True
    assert c.stopped == 0


def test_failed_query_keeps_guarding():
    c = FakeColab(active=None)
    st = cli.WatchState()
    assert cli.watchdog_tick(CFG, c, st, 0, None) is False
    assert cli.watchdog_tick(CFG, c, st, 60, None) is False
    assert st.query_failing


def test_busy_never_stops():
    c = FakeColab()
    assert cli.watchdog_tick(CFG, c, cli.WatchState(), 0, idle(10_000, busy=True)) is False
    assert c.stopped == 0


def test_idle_and_not_busy_stops():
    c = FakeColab()
    assert cli.watchdog_tick(CFG, c, cli.WatchState(), 0, idle(100)) is True
    assert c.stopped == 1


def test_below_limit_does_not_stop():
    c = FakeColab()
    assert cli.watchdog_tick(CFG, c, cli.WatchState(), 0, idle(30)) is False


def test_wake_from_sleep_requires_a_full_idle_window_after_wake():
    c = FakeColab()
    st = cli.WatchState()
    assert cli.watchdog_tick(CFG, c, st, 0, idle(0)) is False
    assert cli.watchdog_tick(CFG, c, st, 10_000, idle(10_000)) is False
    assert cli.watchdog_tick(CFG, c, st, 10_030, idle(10_030)) is False
    assert c.stopped == 0
    assert cli.watchdog_tick(CFG, c, st, 10_060, idle(10_060)) is True
    assert c.stopped == 1


def test_activity_after_wake_prevents_stop():
    c = FakeColab()
    st = cli.WatchState()
    cli.watchdog_tick(CFG, c, st, 0, idle(0))
    cli.watchdog_tick(CFG, c, st, 10_000, idle(10_000))
    assert cli.watchdog_tick(CFG, c, st, 10_060, idle(5)) is False
    assert c.stopped == 0


def test_wake_resets_unreachable_clock():
    c = FakeColab()
    st = cli.WatchState()
    cli.watchdog_tick(CFG, c, st, 0, None)
    assert cli.watchdog_tick(CFG, c, st, 10_000, None) is False
    assert c.stopped == 0
    assert cli.watchdog_tick(CFG, c, st, 10_060, None) is True
    assert c.stopped == 1


def test_skips_tick_while_another_command_holds_the_lock():
    c = FakeColab()
    with lock.state_lock():
        assert cli.watchdog_tick(CFG, c, cli.WatchState(), 0, idle(100)) is False
    assert c.stopped == 0
    assert cli.watchdog_tick(CFG, c, cli.WatchState(), 0, idle(100)) is True


def test_stop_path_survives_unreachable_controller(monkeypatch):
    def refuse(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(tunnel, "healthy", lambda cfg, timeout=3: True)
    monkeypatch.setattr(cli.httpx, "request", refuse)
    c = FakeColab()
    assert cli.watchdog_tick(CFG, c, cli.WatchState(), 0, idle(100)) is True
    assert c.stopped == 1


def test_failed_stop_keeps_guarding():
    c = FakeColab()
    c.fail_stop = True
    assert cli.watchdog_tick(CFG, c, cli.WatchState(), 0, idle(100)) is False


@pytest.mark.parametrize("keeper_running,restarts", [(True, 0), (False, 1)])
def test_watchdog_leaves_reconnecting_to_a_live_keeper(monkeypatch, keeper_running, restarts):
    started = []
    monkeypatch.setattr(tunnel, "start", lambda cfg, proxy, **kw: started.append(proxy))
    monkeypatch.setattr(tunnel, "is_running", lambda: keeper_running)
    cfg, st = Config(idle_stop_minutes=10), cli.WatchState()
    cli.watchdog_tick(cfg, FakeColab(), st, 1000, None)
    cli.watchdog_tick(cfg, FakeColab(), st, 1150, None)
    assert len(started) == restarts
