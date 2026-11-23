import httpx
import pytest

from lm import tunnel
from lm.config import Config


class Resp:
    def __init__(self, status=200, body=None, bad_json=False):
        self.status_code, self._body, self._bad = status, body, bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


def fake_get(resp):
    def get(url, timeout):
        if isinstance(resp, Exception):
            raise resp
        return resp
    return get


@pytest.mark.parametrize("resp,state,version", [
    (Resp(200, {"ok": True, "service": "lm-controller", "version": "abc"}), "ours", "abc"),
    (Resp(200, {"ok": True, "service": "something-else"}), "foreign", ""),
    (Resp(200, {"ok": True}), "ours", ""),
    (Resp(200, bad_json=True), "foreign", ""),
    (Resp(502, {}), "down", ""),
    (httpx.ConnectError("refused"), "down", ""),
])
def test_probe_identifies_controller(monkeypatch, resp, state, version):
    monkeypatch.setattr(tunnel.httpx, "get", fake_get(resp))
    p = tunnel.probe(Config())
    assert (p.state, p.version) == (state, version)
    assert tunnel.healthy(Config()) is (state == "ours")


def test_start_refuses_foreign_port_owner(monkeypatch):
    cfg = Config(local_port=9123)
    monkeypatch.setattr(tunnel.httpx, "get", fake_get(Resp(200, {"service": "nginx"})))
    monkeypatch.setattr(tunnel, "stop", lambda: False)
    with pytest.raises(tunnel.PortConflict, match="本地端口 9123 被其他程序占用.*local_port"):
        tunnel.start(cfg, "proxy")
