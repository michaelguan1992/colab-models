import pytest

from lm import cli, tunnel
from lm.colab import ColabError
from lm.config import Config


class FakeColab:
    version = "v-local"

    def __init__(self, cfg=None, active=True):
        self.active = active
        self.deploys = []
        self.stopped = 0

    def session_active(self):
        return self.active

    def deploy(self, load):
        self.deploys.append(load)

    def proxy_command(self):
        return "proxy"

    def stop(self):
        self.stopped += 1


class Unloaded:
    def json(self):
        return {"unloaded": []}

    def raise_for_status(self):
        pass


@pytest.fixture
def env(monkeypatch, tmp_path):
    fake = FakeColab()
    state = {"probe": tunnel.Probe("ours", "v-local"), "healthy": True, "master": True, "starts": [], "loads": [],
             "api": []}
    monkeypatch.setattr(cli, "Colab", lambda cfg: fake)
    monkeypatch.setattr(cli, "SESSION_STATE", tmp_path / "session.json")
    monkeypatch.setattr(cli, "start_watchdog", lambda cfg: None)
    monkeypatch.setattr(cli, "wait_model", lambda cfg, m: True)
    monkeypatch.setattr(cli, "request_load", lambda cfg, m, swap, route=None: state["loads"].append(m) or True)
    monkeypatch.setattr(tunnel, "probe", lambda cfg, timeout=3: state["probe"])
    monkeypatch.setattr(tunnel, "healthy", lambda cfg, timeout=3: state["healthy"])
    monkeypatch.setattr(tunnel, "master_alive", lambda cfg: state["master"])
    monkeypatch.setattr(tunnel, "start", lambda cfg, proxy, **kw: state["starts"].append(kw))
    monkeypatch.setattr(cli, "api", lambda cfg, method, path, **kw: state["api"].append((method, path)) or Unloaded())
    return fake, state


def args(**kw):
    ns = dict(model="inkling", no_load=False, wait=False, open=False)
    return type("A", (), {**ns, **kw})()


def test_same_version_skips_deploy_and_keeps_healthy_tunnel(env, capsys):
    fake, state = env
    assert cli.cmd_up(Config(), args()) == 0
    assert fake.deploys == []
    assert state["starts"] == []
    assert state["loads"] == ["inkling"]
    assert "跳过部署" in capsys.readouterr().out


def test_different_version_deploys(env):
    fake, state = env
    state["probe"] = tunnel.Probe("ours", "v-old")
    assert cli.cmd_up(Config(), args()) == 0
    assert fake.deploys == ["inkling"]
    assert state["starts"] == []


def test_foreign_service_is_not_ours_and_blocks_deploy(env):
    fake, state = env
    state["probe"] = tunnel.Probe("foreign")
    with pytest.raises(tunnel.PortConflict, match="本地端口 9000 被其他程序占用"):
        cli.cmd_up(Config(), args())
    assert fake.deploys == []


def test_dead_controller_reuses_ssh_master_then_deploys(env):
    fake, state = env
    state["probe"] = tunnel.Probe("down")
    state["healthy"] = False
    fake.deploy = lambda load: (fake.deploys.append(load), state.update(healthy=True))
    assert cli.cmd_up(Config(), args()) == 0
    assert fake.deploys == ["inkling"]
    assert state["starts"] == []


def test_no_master_starts_tunnel_without_waiting_for_controller(env):
    fake, state = env
    state["probe"] = tunnel.Probe("down")
    state["healthy"] = False
    state["master"] = False
    cli.cmd_up(Config(), args())
    assert state["starts"][0] == {"require_health": False}
    assert fake.deploys == ["inkling"]


def test_unknown_session_state_does_not_acquire(env, monkeypatch):
    fake, _ = env
    fake.active = None
    monkeypatch.setattr(cli, "acquire", lambda *a, **k: pytest.fail("must not request a new runtime"))
    assert cli.cmd_up(Config(), args()) == 1
    assert fake.deploys == []


def test_down_on_unknown_state_still_stops_and_reports(env, monkeypatch, capsys):
    fake, state = env
    fake.active = None
    monkeypatch.setattr(tunnel, "stop", lambda: False)
    monkeypatch.setattr(cli, "stop_watchdog", lambda: None)
    assert cli.cmd_down(Config(), type("A", (), {"keep_session": False})()) == 0
    assert fake.stopped == 1
    assert ("POST", "/admin/unload") in state["api"]
    assert "无法确认会话状态" in capsys.readouterr().out


def test_down_on_unknown_state_reports_failed_stop(env, monkeypatch, capsys):
    fake, _ = env
    fake.active = None

    def boom():
        raise ColabError("auth")
    fake.stop = boom
    monkeypatch.setattr(tunnel, "stop", lambda: False)
    monkeypatch.setattr(cli, "stop_watchdog", lambda: None)
    assert cli.cmd_down(Config(), type("A", (), {"keep_session": False})()) == 1
    assert "可能仍在计费" in capsys.readouterr().out
