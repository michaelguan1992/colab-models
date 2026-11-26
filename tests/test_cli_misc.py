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


def test_image_and_video_jobs_ask_for_their_capability(monkeypatch):
    sent = []

    class R:
        status_code = 400

        def json(self):
            return {"detail": "x"}
    monkeypatch.setattr(cli, "ensure_tunnel", lambda cfg: None)
    monkeypatch.setattr(cli, "api", lambda cfg, m, p, **kw: sent.append(kw["json"]) or R())
    cli.cmd_image(Config(), A(prompt="a fox", model=None, height=None, width=768, seed=3, negative_prompt=None, cfg=None))
    cli.cmd_video(Config(), A(prompt="a fox", model=None, seconds=5, image=None, last_image=None, height=None,
                              width=None, seed=None))
    cli.cmd_image(Config(), A(prompt="a fox", model="moss", height=None, width=None, seed=None,
                              negative_prompt="blurry", cfg=None))
    assert sent == [{"model": None, "serves": "image", "input": {"prompt": "a fox", "width": 768, "seed": 3}},
                    {"model": None, "serves": "video", "input": {"prompt": "a fox", "seconds": 5}},
                    {"model": "moss", "serves": None, "input": {"prompt": "a fox", "negative_prompt": "blurry"}}]


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


def test_comfy_loads_the_model_forwards_its_port_and_opens_it(monkeypatch):
    states = iter(["offline", "ready", "ready", "offline"])
    calls = []

    class R:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def api(cfg, method, path, **kw):
        calls.append((method, path))
        if path == "/admin/load":
            return R({"ok": True})
        return R({"models": [{"id": "moss-comfy", "serves": ["comfyui"], "state": next(states), "port": 8188,
                              "params": {"home": "image_moss_image_t2i"}},
                             {"id": "inkling", "serves": ["chat"], "state": "offline", "port": 8081}]})
    monkeypatch.setattr(cli, "ensure_tunnel", lambda cfg: None)
    monkeypatch.setattr(cli, "api", api)
    monkeypatch.setattr(cli.tunnel, "forward", lambda cfg, port: calls.append(("forward", port)) or True)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: calls.append(("open", url)))
    assert cli.cmd_comfy(Config(), A(model=None, no_open=False)) == 0
    assert calls == [("GET", "/admin/status"), ("POST", "/admin/load"), ("GET", "/admin/status"),
                     ("GET", "/admin/status"), ("forward", 8188), ("open", "http://127.0.0.1:8188/?template=image_moss_image_t2i")]
    assert cli.cmd_comfy(Config(), A(model="inkling", no_open=True)) == 1
