import subprocess
import tarfile

import pytest

from lm import colab as colab_mod
from lm.colab import Colab, code_version, remote_files
from lm.config import Config


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "remote"
    (d / "controller").mkdir(parents=True)
    (d / "bootstrap.sh").write_text("echo hi\n")
    (d / "controller" / "app.py").write_text("x = 1\n")
    (d / "controller" / "__pycache__").mkdir()
    (d / "controller" / "__pycache__" / "app.pyc").write_bytes(b"junk")
    return d


class Runner:
    def __init__(self, sessions_rc=0, sessions_out=""):
        self.calls = []
        self.uploaded = []
        self.sessions = (sessions_rc, sessions_out)

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        if cmd[1] == "upload":
            with tarfile.open(cmd[-2]) as t:
                self.uploaded = sorted(t.getnames())
        if cmd[1] == "sessions":
            return subprocess.CompletedProcess(cmd, self.sessions[0], self.sessions[1], "")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def make(monkeypatch, src):
    def build(**cfg_kw):
        runner = Runner()
        monkeypatch.setattr(colab_mod.subprocess, "run", runner)
        monkeypatch.setattr(colab_mod.tunnel, "master_alive", lambda cfg: False)
        monkeypatch.setattr(colab_mod.shutil, "which", lambda b: "/usr/bin/colab")
        return Colab(Config(remote_src=str(src), **cfg_kw)), runner
    return build


def ssh_command(runner):
    return next(c for c, _ in runner.calls if c[1] == "ssh")[-1]


def test_version_is_deterministic_and_short(src):
    v = code_version(src)
    assert v == code_version(src) and len(v) == 12


def test_version_changes_with_content_and_path_but_not_pycache(src):
    v = code_version(src)
    (src / "controller" / "__pycache__" / "app.pyc").write_bytes(b"other")
    assert code_version(src) == v
    (src / "controller" / "app.py").write_text("x = 2\n")
    v2 = code_version(src)
    assert v2 != v
    (src / "controller" / "app.py").rename(src / "controller" / "main.py")
    assert code_version(src) not in (v, v2)


def test_deploy_passes_version_port_and_load(make, src):
    c, runner = make(remote_port=9100)
    c.deploy(load="inkling")
    cmd = ssh_command(runner)
    assert f"LM_VERSION={code_version(src)}" in cmd
    assert "LM_PORT=9100" in cmd
    assert "LOAD=inkling" in cmd


def test_deploy_uploads_exactly_the_versioned_files(make, src):
    c, runner = make()
    c.deploy(load=None)
    assert runner.uploaded == sorted(p.as_posix() for p in remote_files(src))
    assert "LOAD=" not in ssh_command(runner)


def test_hf_token_goes_via_stdin_never_argv(make, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret")
    c, runner = make()
    c.deploy(load=None)
    assert not any("hf_supersecret" in part for cmd, _ in runner.calls for part in cmd)
    token_call = next((cmd, kw) for cmd, kw in runner.calls if kw.get("input"))
    assert token_call[1]["input"] == "hf_supersecret"
    assert "chmod 600" in token_call[0][-1] and ".cache/huggingface/token" in token_call[0][-1]


def test_hf_token_env_name_is_configurable(make, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("MY_HF", "hf_other")
    c, runner = make(hf_token_env="MY_HF")
    c.deploy(load=None)
    assert [kw["input"] for _, kw in runner.calls if kw.get("input")] == ["hf_other"]


def test_no_token_means_no_token_call(make, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    c, runner = make()
    c.deploy(load=None)
    assert not any(kw.get("input") for _, kw in runner.calls)


@pytest.mark.parametrize("rc,out,expected", [
    (0, "[localmodels] G4 running\n", True),
    (0, "", False),
    (0, "[other] G4\n", False),
    (1, "", None),
    (2, "auth expired", None),
])
def test_session_active_is_tri_state(make, rc, out, expected):
    c, runner = make()
    runner.sessions = (rc, out)
    assert c.session_active() is expected
