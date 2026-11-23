import math
import pytest

from lm import cli
from lm.colab import classify_assign_error
from lm.config import Config


class FakeColab:
    def __init__(self, results, vram=95.6):
        self.results = list(results)  # per new() call: (ok, kind, detail)
        self.calls = []
        self.vram = vram
        self.stopped = 0

    def new(self, gpu, high_mem=False):
        self.calls.append((gpu, high_mem))
        return self.results.pop(0)

    def gpu_info(self):
        return "GPU", self.vram

    def stop(self):
        self.stopped += 1


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ASSIGN_STATE", tmp_path / "assign.json")
    monkeypatch.setattr(cli, "SESSION_STATE", tmp_path / "session.json")
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)


CAP = (False, "capacity", "503")


def test_falls_back_to_a100_high_mem():
    c = FakeColab([CAP, (True, "", "")], vram=80)
    assert cli.acquire(Config(), c, wait=None)
    assert c.calls == [("G4", False), ("A100", True)]


def test_one_round_without_wait():
    c = FakeColab([CAP, CAP])
    assert not cli.acquire(Config(), c, wait=None)
    assert len(c.calls) == 2


def test_wait_retries_rounds_until_success():
    c = FakeColab([CAP, CAP, (True, "", "")])
    assert cli.acquire(Config(), c, wait=math.inf)
    assert c.calls[-1] == ("G4", False)


def test_daily_limit_stops_retrying():
    cfg = Config(assign_daily_limit=3)
    c = FakeColab([CAP] * 10)
    assert not cli.acquire(cfg, c, wait=math.inf)
    assert len(c.calls) == 3
    # Limit is shared across `lm up` runs on the same day.
    assert not cli.acquire(cfg, FakeColab([]), wait=None)


def test_rejected_accelerator_is_dropped():
    c = FakeColab([(False, "rejected", "400"), CAP, CAP])
    assert not cli.acquire(Config(assign_daily_limit=3), c, wait=math.inf)
    assert c.calls == [("G4", False), ("A100", True), ("A100", True)]


def test_small_gpu_is_stopped_and_next_tried():
    c = FakeColab([(True, "", ""), CAP])
    c.vram = 40
    assert not cli.acquire(Config(), c, wait=None)
    assert c.stopped == 1 and len(c.calls) == 2


def test_classify():
    assert classify_assign_error("ColabRequestError: ... accelerator=G4: Service Unavailable") == "capacity"
    assert classify_assign_error("Failed to issue request GET ...: Bad Request") == "rejected"
    assert classify_assign_error("TooManyAssignmentsError") == "quota"
    assert classify_assign_error("something odd") == "error"


def test_wait_minutes_gives_up_at_the_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cli.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    c = FakeColab([(False, "capacity", "")] * 100)
    assert not cli.acquire(Config(assign_retry_minutes=12, assign_daily_limit=100), c, wait=30)
    assert clock[0] <= 30 * 60, "never sleeps past the wait limit"
    assert 2 <= len(c.calls) <= 3 * len(Config().accelerators), "about 30/12 rounds, not until the daily cap"


def test_wait_flag_parses_optional_minutes():
    p = cli.build_parser()
    assert p.parse_args(["up"]).wait is None
    assert p.parse_args(["up", "--wait"]).wait == math.inf
    assert p.parse_args(["up", "--wait", "90"]).wait == 90
