import yaml

from lm import cli
from lm.config import Config


class A:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_load_passes_route(monkeypatch):
    sent = []

    class R:
        status_code = 200

        def json(self):
            return {"ok": True}
    monkeypatch.setattr(cli, "ensure_tunnel", lambda cfg: None)
    monkeypatch.setattr(cli, "api", lambda cfg, m, p, **kw: sent.append(kw["json"]) or R())
    assert cli.cmd_load(Config(), A(model="qwen", no_swap=False, no_wait=True, route="basalt")) == 0
    assert cli.cmd_load(Config(), A(model="qwen", no_swap=False, no_wait=True, route=None)) == 0
    assert sent[0] == {"id": "qwen", "swap": True, "route": "basalt"}
    assert "route" not in sent[1]


def test_logs_follow_remote_dir(monkeypatch):
    cmds = []

    class C:
        def __init__(self, cfg): pass
        def ssh(self, command): cmds.append(command)
    monkeypatch.setattr(cli, "Colab", C)
    cli.cmd_logs(Config(remote_dir="/data/lm/app"), A(name="inkling", lines=5))
    assert cmds == ["tail -n 5 /data/lm/logs/inkling.log"]


def test_dsh_launch_carries_the_placeholder_key_and_the_plugin(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, env: seen.update(cmd=cmd, env=env) or 0)
    cli.cmd_dsh(Config(), A())
    assert seen["env"]["LM_COLAB_API_KEY"] and seen["cmd"][-1].endswith("plugin-colab/cordis.yml")
