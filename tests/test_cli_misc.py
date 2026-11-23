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


def test_setup_dsh_keeps_only_latest_backup(tmp_path):
    cfg = Config(dsh_home=str(tmp_path))
    patch = tmp_path / "profiles" / "web" / "cordis.patch.yml"
    patch.parent.mkdir(parents=True)
    patch.write_text(yaml.safe_dump([{"id": "llm-pi-ai", "config": {"providers": {}}}]))
    for ts in ("20200101000000", "20200102000000"):
        (patch.parent / f"cordis.patch.yml.bak-{ts}").write_text("old")
    assert cli.cmd_setup_dsh(cfg, A(profile=None, force=False)) == 0
    backups = list(patch.parent.glob("cordis.patch.yml.bak-*"))
    assert len(backups) == 1 and "2020" not in backups[0].name


def test_dsh_gets_configured_port(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, env: seen.update(env=env) or 0)
    cli.cmd_dsh(Config(local_port=9123), A())
    assert seen["env"]["LM_BASE_URL"] == "http://127.0.0.1:9123"
