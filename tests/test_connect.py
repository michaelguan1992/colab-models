import json

import pytest
import yaml

from lm import cli, connect
from lm.config import Config

STATUS = {"models": [
    {"id": "inkling", "serves": ["chat"], "display": "Inkling-2-Mini", "route": "basalt",
     "caps": {"tools": True, "vision": True, "context": 1048576, "reasoning": ["none", "low", "medium", "high"]}},
    {"id": "qwen", "serves": ["chat"], "display": "Qwen3.8", "route": "strata",
     "caps": {"context": 32768, "reasoning": []}},
    {"id": "big", "serves": ["chat"], "display": "Needs sm_120", "route": None, "caps": {}},
    {"id": "pika", "serves": ["video"], "display": "Pika", "route": "diffusers", "caps": {}},
]}


def test_dsh_models_come_from_caps_and_skip_non_chat_and_unrunnable_models():
    p = connect.dsh_provider(STATUS, "http://127.0.0.1:9000")
    assert p["baseURL"] == "http://127.0.0.1:9000/v1"
    assert p["models"] == [
        {"id": "inkling", "name": "Inkling-2-Mini", "contextWindow": 1048576, "input": ["text", "image"],
         "reasoningEfforts": {"off": "none", "low": "low", "medium": "medium", "high": "high"}},
        {"id": "qwen", "name": "Qwen3.8", "contextWindow": 32768, "input": ["text"], "reasoningEfforts": False},
    ]


def test_merge_keeps_other_rows_providers_and_mcp_servers():
    rows = [{"id": "ui", "config": {"x": 1}},
            {"id": "llm-pi-ai", "config": {"providers": {"other": {"a": 1}, "colab": {"old": True}}}},
            {"id": "mcp-manager", "config": {"servers": {"github": {"transport": "stdio"}}}}]
    out = connect.merge_dsh_patch(rows, {"new": True}, {"command": "/bin/lm"})
    assert out[0] == {"id": "ui", "config": {"x": 1}}
    assert out[1]["config"]["providers"] == {"other": {"a": 1}, "colab": {"new": True}}
    assert out[2]["config"]["servers"] == {"github": {"transport": "stdio"}, "lm": {"command": "/bin/lm"}}
    assert [r["id"] for r in connect.merge_dsh_patch([{"id": "llm-pi-ai"}], {}, {})] == ["llm-pi-ai", "mcp-manager"]


def test_claude_env_points_every_alias_at_the_model_and_registers_mcp():
    env, mcp = connect.claude_env(STATUS, "http://127.0.0.1:9000", "inkling", "/bin/lm")
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000" and env["ANTHROPIC_MODEL"] == "inkling"
    assert {env[k] for k in env if k.startswith("ANTHROPIC_DEFAULT_")} == {"inkling"}
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1048576"
    assert json.loads(mcp) == {"mcpServers": {"lm": {"command": "/bin/lm", "args": ["mcp"]}}}
    with pytest.raises(ValueError, match="inkling, qwen"):
        connect.claude_env(STATUS, "u", "pika", "/bin/lm")


def stub_controller(monkeypatch):
    class R:
        def json(self):
            return STATUS
    monkeypatch.setattr(cli, "ensure_tunnel", lambda cfg: None)
    monkeypatch.setattr(cli, "api", lambda cfg, method, path, **kw: R())
    monkeypatch.setattr(cli, "lm_bin", lambda: "/bin/lm")


class A:
    def __init__(self, **kw):
        self.__dict__.update(dict(harness="dsh", model=None, profile=None, force=False) | kw)


def test_connect_dsh_writes_provider_and_mcp_and_keeps_one_backup(monkeypatch, tmp_path):
    stub_controller(monkeypatch)
    patch = tmp_path / "profiles" / "web" / "cordis.patch.yml"
    patch.parent.mkdir(parents=True)
    patch.write_text(yaml.safe_dump([{"id": "llm-pi-ai", "config": {"providers": {}}}]))
    for ts in ("20200101000000", "20200102000000"):
        (patch.parent / f"cordis.patch.yml.bak-{ts}").write_text("old")
    assert cli.cmd_connect(Config(dsh_home=str(tmp_path)), A()) == 0
    rows = yaml.safe_load(patch.read_text())
    assert [m["id"] for m in rows[0]["config"]["providers"]["colab"]["models"]] == ["inkling", "qwen"]
    assert rows[1]["config"]["servers"]["lm"]["command"] == "/bin/lm"
    backups = list(patch.parent.glob("cordis.patch.yml.bak-*"))
    assert len(backups) == 1 and "2020" not in backups[0].name


def test_connect_dsh_refuses_to_create_llm_row_without_force(monkeypatch, tmp_path):
    stub_controller(monkeypatch)
    assert cli.cmd_connect(Config(dsh_home=str(tmp_path)), A()) == 1
    assert not (tmp_path / "profiles").exists()
    assert cli.cmd_connect(Config(dsh_home=str(tmp_path)), A(force=True)) == 0


def test_lm_claude_runs_claude_with_the_env_and_mcp_config(monkeypatch):
    stub_controller(monkeypatch)
    seen = {}
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, env: seen.update(cmd=cmd, env=env) or 0)
    assert cli.cmd_claude(Config(claude_bin="/x/claude"), A(args=["-p", "hi"])) == 0
    assert seen["cmd"][0] == "/x/claude" and seen["cmd"][1] == "--mcp-config" and seen["cmd"][3:] == ["-p", "hi"]
    assert seen["env"]["ANTHROPIC_MODEL"] == "inkling"


def test_lm_claude_passes_unknown_flags_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "load", lambda: Config())
    monkeypatch.setattr(cli, "cmd_claude", lambda cfg, a: seen.update(args=a.args, model=a.model) or 0)
    assert cli.main(["claude", "--model", "qwen", "-p", "hi", "--output-format", "json"]) == 0
    assert seen == {"args": ["-p", "hi", "--output-format", "json"], "model": "qwen"}
