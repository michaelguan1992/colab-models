import asyncio
import json
import sys
import time
import types

import httpx
import yaml

from controller.app import tracked_stream
from controller.backends.jobs import JobsBackend
from controller.backends.server import ServerBackend
from controller.cleanup import SystemReaper, reap_stale
from controller.hf import classify_hf_error, hf_preflight
from controller.gpu import GpuSnapshot
from controller.state import StateFile
from ctl_helpers import (REGISTRY_YAML, FakeBackend, FakeReaper, build, make_client, make_manager,
                         offline_installer, tmp_registry, wait_ready)


def wait_state(client, model_id, state):
    for _ in range(100):
        m = next(m for m in client.get("/admin/status").json()["models"] if m["id"] == model_id)
        if m["state"] == state:
            return m
        time.sleep(0.02)
    raise AssertionError(f"{model_id} never reached {state}")


class FailingBackend(FakeBackend):
    async def start(self):
        self.state = "error"
        self.error = "RuntimeError: hf 退出码 1"


class StuckDownload(FakeBackend):
    async def start(self):
        self.state = "downloading"
        await asyncio.sleep(3600)


class StuckInstall(FakeBackend):
    async def start(self):
        self.state = "installing"
        await asyncio.sleep(3600)


# ---- error state ---------------------------------------------------------------------------

def test_failed_load_stays_visible_as_error_until_next_action():
    client, mgr = make_client(factory=FailingBackend)
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        m = wait_state(client, "inkling", "error")
        assert "退出码 1" in m["error"]
        s = client.get("/admin/status").json()
        assert s["online"] == []
        r = client.post("/v1/chat/completions", json={"messages": []})
        assert r.status_code == 503 and "没有在线" in r.json()["error"]["message"]
        assert client.post("/admin/unload", json={"id": "inkling"}).json()["ok"]
        assert wait_state(client, "inkling", "offline")


def test_rule_violation_after_load_keeps_error_entry_with_reason(monkeypatch):
    monkeypatch.setitem(FakeBackend.vram, "inkling", 22.0)  # measured above the 20GB video limit
    client, mgr = make_client()
    with client:
        client.post("/admin/load", json={"id": "pika"})
        wait_ready(client, "pika")
        assert client.post("/admin/load", json={"id": "inkling"}).json()["ok"]
        m = wait_state(client, "inkling", "error")
        assert "已自动卸载" in m["error"]
        assert client.get("/admin/status").json()["online"] == ["pika"]


def test_error_entries_do_not_count_as_online_for_policy():
    client, mgr = make_client(factory=FailingBackend)
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_state(client, "inkling", "error")
        mgr.factory = FakeBackend
        r = client.post("/admin/load", json={"id": "qwen"}).json()
        assert r["ok"] and r["unloaded"] == []


def test_loading_message_ignores_errored_backends():
    client, _ = make_client(factory=FailingBackend)
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_state(client, "inkling", "error")
        msg = client.post("/v1/chat/completions", json={}).json()["error"]["message"]
        assert "正在加载" not in msg


# ---- 503 text, /health, startup ----------------------------------------------------------------

def test_503_texts_list_registry_ids():
    raw = yaml.safe_load(REGISTRY_YAML.read_text())
    raw["models"]["alpha"] = raw["models"].pop("inkling")
    raw["models"].pop("qwen")
    raw["models"].pop("qwen-tiny")
    raw["models"]["clip"] = raw["models"].pop("pika")
    client, _ = make_client(registry=tmp_registry(raw))
    with client:
        llm = client.post("/v1/chat/completions", json={}).json()["error"]["message"]
        assert "/load alpha" in llm and "inkling" not in llm and "qwen" not in llm
        video = client.post("/jobs", json={"input": {"prompt": "x"}})
        assert video.status_code == 503 and "/load clip" in video.json()["detail"]


def test_health_contract(monkeypatch):
    client, _ = make_client()
    with client:
        monkeypatch.delenv("LM_VERSION", raising=False)
        assert client.get("/health").json() == {"ok": True, "service": "lm-controller", "version": "dev"}
        monkeypatch.setenv("LM_VERSION", "abc123")
        assert client.get("/health").json()["version"] == "abc123"


def test_bad_autoload_id_does_not_crash_startup():
    client, mgr = make_client(autoload="nope")
    with client:
        assert client.get("/health").json()["ok"]
        assert mgr.backends == {}


def test_refused_autoload_does_not_crash_startup():
    client, mgr = make_client(autoload="inkling", hf_check=lambda repo, rev: "无法访问")
    with client:
        assert client.get("/health").json()["ok"]
        assert mgr.backends == {}


# ---- state file and startup cleanup ----------------------------------------------------------

def test_state_file_records_ready_backends_and_forgets_them_on_stop():
    client, mgr = make_client()
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        entry = mgr.state.read()["inkling"]
        assert entry["route"] == "strata" and entry["engine"] == "strata" and entry["port"] == 8081
        assert isinstance(entry["pids"], list) and entry["started_at"]
        client.post("/admin/unload", json={"id": "inkling"})
        assert mgr.state.read() == {}


def test_state_file_write_is_atomic_and_tolerates_corruption(tmp_path):
    sf = StateFile(tmp_path / "sub" / "state.json")
    assert sf.read() == {}
    sf.set("a", {"port": 1})
    assert sf.read() == {"a": {"port": 1}}
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["state.json"]
    sf.path.write_text("{not json")
    assert sf.read() == {}
    sf.path.write_text("[1, 2]")
    assert sf.read() == {}


def test_startup_cleanup_reaps_recorded_and_registry_ports_before_autoload():
    reg = tmp_registry()
    sf = StateFile(reg.state_file)
    sf.set("pika", {"engine": "diffusers", "port": 8090, "pids": [4242, 4243]})
    sf.set("inkling", {"engine": "strata", "port": 8081, "pids": [5000]})
    client, mgr = make_client(registry=reg, autoload="inkling")
    with client:
        wait_ready(client, "inkling")
        calls = mgr.reaper.calls
        assert ("pids", (4242, 4243)) in calls
        assert {("strata", 8081), ("strata", 8082)} <= set(calls)
        assert not [c for c in calls if c[0] in ("llama-cpp", "diffusers")]  # engines without `stop` are not swept
        assert ("pids", (5000,)) not in calls
        assert list(mgr.state.read()) == ["inkling"]  # stale entries cleared before the new load recorded itself


def test_startup_cleanup_is_safe_with_missing_or_corrupt_state(tmp_path):
    reg = tmp_registry()
    reaper = FakeReaper()
    asyncio.run(reap_stale(reg, StateFile(reg.state_file), reaper))
    assert [c for c in reaper.calls if c[0] == "pids"] == []
    reg.state_file.parent.mkdir(parents=True, exist_ok=True)
    reg.state_file.write_text("\x00garbage")
    asyncio.run(reap_stale(reg, StateFile(reg.state_file), reaper))
    assert json.loads(reg.state_file.read_text()) == {}


def test_system_reaper_kills_recorded_process_groups():
    reg = tmp_registry()
    sf = StateFile(reg.state_file)

    async def run():
        child = await asyncio.create_subprocess_exec("sh", "-c", "sleep 60 & echo $! > /dev/null; wait",
                                                     start_new_session=True)
        sf.set("pika", {"engine": "diffusers", "port": 8090, "pids": [child.pid]})
        started = time.monotonic()
        await reap_stale(reg, sf, SystemReaper(reg, offline_installer(reg)))
        await asyncio.wait_for(child.wait(), 5)
        assert time.monotonic() - started < 5
    asyncio.run(run())
    assert sf.read() == {}


# ---- release wait, health watch ----------------------------------------------------------------

def test_next_backend_starts_only_after_previous_one_released():
    client, mgr = make_client()
    mgr.release_poll_s = 0.01
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        old = mgr.backends["inkling"]
        old.checks_until_free = 5
        assert client.post("/admin/load", json={"id": "qwen"}).json()["ok"]
        assert old.checks_until_free == 0
        wait_ready(client, "qwen")


def test_load_refused_and_error_state_when_old_engine_never_releases():
    client, mgr = make_client()
    mgr.release_poll_s, mgr.release_timeout_s = 0.01, 0.1
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        mgr.backends["inkling"].release = False
        r = client.post("/admin/load", json={"id": "qwen"})
        assert r.status_code == 409 and "inkling" in r.json()["reason"]
        m = wait_state(client, "inkling", "error")
        assert "仍被占用" in m["error"]
        assert "qwen" not in mgr.backends


def test_two_failed_health_checks_mark_backend_error_and_unroute_it():
    client, mgr = make_client()
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        b = mgr.backends["inkling"]
        b.healthy = False
        client.portal.call(mgr.health_check)
        assert b.state == "ready"
        b.healthy = True
        client.portal.call(mgr.health_check)
        assert b.health_failures == 0
        b.healthy = False
        client.portal.call(mgr.health_check)
        client.portal.call(mgr.health_check)
        m = wait_state(client, "inkling", "error")
        assert m["error"] == "引擎无响应/已退出"
        assert client.get("/admin/status").json()["online"] == []
        assert client.post("/v1/chat/completions", json={}).status_code == 503
        assert mgr.state.read() == {}


# ---- idle and in-flight work -------------------------------------------------------------------

def make_idle_manager(factory=FakeBackend):
    mgr = make_manager(factory=factory)
    return mgr


def expire(mgr):
    mgr.last_activity -= mgr.registry.idle_unload_minutes * 60 + 1


def test_idle_unload_waits_for_downloads():
    mgr = make_idle_manager(StuckDownload)

    async def run():
        await mgr.load("inkling")
        await asyncio.sleep(0.01)
        expire(mgr)
        assert mgr.busy() and mgr.inflight()["loading"] == 1
        await mgr.idle_check()
        assert "inkling" in mgr.backends and not mgr.idle_unloaded
        mgr.backends["inkling"].state = "ready"
        await mgr.idle_check()
        assert mgr.backends == {} and mgr.idle_unloaded
    asyncio.run(run())


def test_install_counts_as_in_flight_work_and_shows_in_status():
    client, mgr = make_client(factory=StuckInstall)
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_state(client, "inkling", "installing")
        assert mgr.busy() and mgr.inflight()["loading"] == 1
        expire(mgr)
        client.portal.call(mgr.idle_check)
        assert "inkling" in mgr.backends and not mgr.idle_unloaded
        assert client.post("/admin/unload", json={"id": "inkling"}).json()["ok"]


def test_idle_unload_waits_for_in_flight_requests_then_counts_their_end_as_activity():
    mgr = make_idle_manager()

    async def run():
        await mgr.load("inkling")
        await asyncio.sleep(0)
        h = mgr.begin_request(activity=True)
        expire(mgr)
        await mgr.idle_check()
        assert "inkling" in mgr.backends and mgr.inflight()["requests"] == 1
        h.finish()
        h.finish()
        assert mgr.inflight()["requests"] == 0
        await mgr.idle_check()  # finishing refreshed last_activity
        assert "inkling" in mgr.backends
        expire(mgr)
        await mgr.idle_check()
        assert mgr.backends == {}
    asyncio.run(run())


def test_idle_unload_waits_for_video_jobs():
    mgr = make_idle_manager()

    async def run():
        await mgr.load("pika")
        await asyncio.sleep(0)
        mgr.backends["pika"].queue_depth = 2
        expire(mgr)
        assert mgr.busy() and mgr.inflight()["jobs"] == 2
        await mgr.idle_check()
        assert "pika" in mgr.backends
        mgr.backends["pika"].queue_depth = 0
        await mgr.idle_check()
        assert mgr.backends == {}
    asyncio.run(run())


def test_external_gpu_process_or_hold_file_counts_as_activity():
    snap = {"per_pid": {}}
    mgr = make_manager(factory=FakeBackend)
    mgr.probe = lambda: GpuSnapshot("RTX PRO 6000", 0, 96, dict(snap["per_pid"]), True, "")

    async def run():
        await mgr.load("inkling")
        await asyncio.sleep(0)
        snap["per_pid"] = {4242: 30.0}
        expire(mgr)
        await mgr.idle_check()
        assert "inkling" in mgr.backends and mgr.status()["idle_seconds"] < 5
        assert mgr.status()["external_work"] == ["GPU 进程 4242"]

        snap["per_pid"] = {}
        mgr.registry.holds_dir.mkdir(parents=True)
        (mgr.registry.holds_dir / "qwen-convert").touch()
        expire(mgr)
        await mgr.idle_check()
        assert "inkling" in mgr.backends and mgr.status()["external_work"] == ["hold qwen-convert"]

        (mgr.registry.holds_dir / "qwen-convert").unlink()
        expire(mgr)
        await mgr.idle_check()
        assert mgr.backends == {}
    asyncio.run(run())


def test_status_reports_busy_and_inflight():
    client, mgr = make_client()
    with client:
        s = client.get("/admin/status").json()
        assert s["busy"] is False and s["inflight"] == {"loading": 0, "requests": 0, "jobs": 0}
        h = mgr.begin_request(activity=False)
        s = client.get("/admin/status").json()
        assert s["busy"] is True and s["inflight"]["requests"] == 1
        h.finish()


def test_tracked_stream_counts_until_the_stream_ends():
    mgr = make_manager()
    seen = []

    async def upstream():
        for chunk in (b"a", b"b"):
            seen.append(mgr.inflight()["requests"])
            yield chunk

    async def run():
        h = mgr.begin_request(activity=True)
        out = [c async for c in tracked_stream(h, upstream())]
        assert out == [b"a", b"b"] and seen == [1, 1] and mgr.inflight()["requests"] == 0
        h2 = mgr.begin_request(activity=True)
        gen = tracked_stream(h2, upstream())
        await gen.__anext__()
        assert mgr.busy()
        await gen.aclose()  # client went away mid-stream
        assert not mgr.busy()
    asyncio.run(run())


def proxy_client():
    client, mgr = make_client()

    def upstream(request):
        async def body():
            yield json.dumps({"path": request.url.path}).encode()
        return httpx.Response(200, content=body())

    return client, mgr, upstream


def test_only_inference_posts_count_as_activity_through_the_proxy():
    client, mgr, upstream = proxy_client()
    with client:
        client.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        mgr.last_activity -= 1000
        assert client.get("/v1/models").status_code == 200
        assert client.get("/v1/something").json() == {"path": "/v1/something"}
        assert client.get("/admin/status").json()["idle_seconds"] >= 999
        assert client.post("/v1/chat/completions", json={"model": "inkling"}).status_code == 200
        s = client.get("/admin/status").json()
        assert s["idle_seconds"] < 5 and s["inflight"]["requests"] == 0


def test_job_calls_count_as_activity():
    client, mgr, upstream = proxy_client()
    with client:
        client.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        client.post("/admin/load", json={"id": "pika"})
        wait_ready(client, "pika")
        mgr.last_activity -= 1000
        client.get("/jobs/abc")
        assert client.get("/admin/status").json()["idle_seconds"] < 5


def test_pika_health_reports_queue_depth_and_busy():
    reg = tmp_registry()
    b = build(JobsBackend, "pika", reg)
    state = {"depth": 3}

    def handler(request):
        return httpx.Response(200, json={"ready": True, "error": None, "queue_depth": state["depth"]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await b.check_health(client)
            assert b.queue_depth == 3 and b.busy()
            state["depth"] = 0
            await b.check_health(client)
            assert not b.busy()
    asyncio.run(run())


def test_strata_ready_requires_the_expected_model_on_the_port():
    reg = tmp_registry()
    b = build(ServerBackend, "qwen", reg)
    served = {"data": [{"id": "Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"}]}

    async def run():
        handler = lambda request: httpx.Response(200, json=served)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert not await b._ready(client)
            assert b.upstream_model is None
            served["data"] = [{"id": "IQ3_XXS"}]
            assert await b._ready(client) and b.upstream_model == "IQ3_XXS"
    asyncio.run(run())


# ---- disk accounting for concurrent downloads --------------------------------------------------

def test_other_downloads_in_progress_reduce_available_disk():
    raw = yaml.safe_load(REGISTRY_YAML.read_text())
    raw["disk_reserve_gb"] = 0
    reg = tmp_registry(raw)
    gib = 1024**3
    factory = lambda model, route, r, inst: (StuckDownload if model.id == "pika" else FakeBackend)(model, route, r, inst)
    mgr = make_manager(registry=reg, free_bytes=lambda p: 160 * gib, factory=factory)

    async def run():
        assert (await mgr.load("pika")).ok
        await asyncio.sleep(0.01)
        plan = await mgr.load("inkling")
        assert not plan.ok and "磁盘不够" in plan.reason
        assert "inkling" not in mgr.backends
        await mgr.unload_all()
    asyncio.run(run())


# ---- HF preflight ------------------------------------------------------------------------------

class GatedRepoError(Exception):
    pass


def test_gated_repo_refuses_before_touching_anything():
    reg = tmp_registry()
    kept = reg.models_dir / "qwen"
    kept.mkdir(parents=True)
    (kept / "w").write_bytes(b"x" * 100)
    calls = []

    def check(repo, rev):
        calls.append(repo)
        return f"无法访问 HuggingFace 仓库 {repo}；请设置 HF_TOKEN" if "Inkling" in repo else None

    client, mgr = make_client(registry=reg, hf_check=check, free_bytes=lambda p: 0)
    with client:
        r = client.post("/admin/load", json={"id": "inkling"})
        assert r.status_code == 409
        assert "AtomicChat/Inkling-2-Mini-Uncensored-GGUF" in r.json()["reason"] and "HF_TOKEN" in r.json()["reason"]
        assert kept.exists() and mgr.backends == {}
        assert calls == ["AtomicChat/Inkling-2-Mini-Uncensored-GGUF"]


def test_pika_checks_both_the_model_and_lora_repos():
    calls = []
    client, _ = make_client(hf_check=lambda repo, rev: calls.append(repo))
    with client:
        assert client.post("/admin/load", json={"id": "pika"}).json()["ok"]
    assert calls == ["Pika-Labs/Pika-H3-Flex", "Pika-Labs/Pika-H3-Flex-Turbo-LoRA"]


def test_hf_check_crash_proceeds_with_the_load():
    def boom(repo, rev):
        raise ConnectionError("dns")

    client, _ = make_client(hf_check=boom)
    with client:
        assert client.post("/admin/load", json={"id": "inkling"}).json()["ok"]


def fake_hub(monkeypatch, exc):
    class HfApi:
        def model_info(self, repo, revision=None):
            raise exc

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=HfApi))


def test_hf_preflight_classifies_auth_errors(monkeypatch):
    fake_hub(monkeypatch, GatedRepoError("401"))
    reason = hf_preflight("org/private", "main")
    assert "org/private" in reason and "HF_TOKEN" in reason

    resp = types.SimpleNamespace(status_code=404)
    fake_hub(monkeypatch, type("HfHubHTTPError", (Exception,), {"response": resp})())
    assert "org/private" in hf_preflight("org/private")


def test_hf_preflight_lets_network_errors_through(monkeypatch, caplog):
    fake_hub(monkeypatch, ConnectionError("unreachable"))
    assert hf_preflight("org/model") is None
    assert "org/model" in caplog.text
    resp = types.SimpleNamespace(status_code=503)
    assert classify_hf_error("o/m", type("HfHubHTTPError", (Exception,), {"response": resp})()) is None


def test_loading_another_route_of_an_online_model_switches_routes():
    from test_registry_routes import two_route_raw
    mgr = make_manager(registry=tmp_registry(two_route_raw()), compute_cap="12.0")

    async def run():
        await mgr.load("inkling")
        await asyncio.sleep(0)
        first = mgr.backends["inkling"]
        assert first.route.name == "fast"
        same = await mgr.load("inkling", route="fast")
        assert same.ok and "已在线" in same.reason and mgr.backends["inkling"] is first
        switched = await mgr.load("inkling", route="strata")
        await asyncio.sleep(0)
        assert switched.ok and mgr.backends["inkling"].route.name == "strata" and first.state == "stopped"
    asyncio.run(run())
