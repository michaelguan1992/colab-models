import types

import httpx
import pytest

from lm import cli, tunnel


def _refuse(*args, **kwargs):
    raise AssertionError(
        "a test reached the network through lm's httpx; mock lm.cli.api / lm.tunnel.probe instead "
        "(the default base_url is the live tunnel to the Colab controller)")


@pytest.fixture(autouse=True)
def no_live_controller(monkeypatch):
    """lm's default base_url is 127.0.0.1:9000, the live tunnel. An unmocked call once unloaded real models."""
    stub = types.SimpleNamespace(
        **{name: getattr(httpx, name) for name in dir(httpx) if not name.startswith("_")})
    for name in ("request", "get", "post", "put", "delete", "stream"):
        setattr(stub, name, _refuse)
    monkeypatch.setattr(cli, "httpx", stub)
    monkeypatch.setattr(tunnel, "httpx", stub)
