import asyncio
import copy
import json

import httpx
import yaml

from ctl_helpers import (REGISTRY_YAML, make_client, make_manager,
                         tmp_registry, wait_ready)


def test_models_empty_and_proxy_503():
    client, _ = make_client()
    with client:
        assert client.get("/v1/models").json()["data"] == []
        r = client.post("/v1/chat/completions", json={"messages": []})
        assert r.status_code == 503
        assert "没有在线" in r.json()["error"]["message"]


def test_load_lists_only_online_llm_and_swaps():
    client, _ = make_client()
    with client:
        assert client.post("/admin/load", json={"id": "inkling"}).json()["ok"]
        wait_ready(client, "inkling")
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["inkling"]

        r = client.post("/admin/load", json={"id": "qwen"}).json()
        assert r["ok"] and r["unloaded"] == ["inkling"]
        s = wait_ready(client, "qwen")
        assert s["online"] == ["qwen"]


def test_pika_gate():
    client, _ = make_client()
    with client:
        client.post("/admin/load", json={"id": "qwen"})
        wait_ready(client, "qwen")
        r = client.post("/admin/load", json={"id": "pika"})
        assert r.status_code == 409 and "qwen" in r.json()["reason"]

        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        assert client.post("/admin/load", json={"id": "pika"}).json()["ok"]
        s = wait_ready(client, "pika")
        assert sorted(s["online"]) == ["inkling", "pika"]
        assert "[+pika]" in client.get("/v1/models").json()["data"][0]["name"]

        # Qwen must be refused while Pika is online.
        assert client.post("/admin/load", json={"id": "qwen"}).status_code == 409


def test_unknown_model_404_and_unload_all():
    client, _ = make_client()
    with client:
        assert client.post("/admin/load", json={"id": "nope"}).status_code == 404
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        assert client.post("/admin/unload", json={}).json()["unloaded"] == ["inkling"]
        assert client.get("/admin/status").json()["online"] == []


def test_jobs_require_a_jobs_model_online():
    client, _ = make_client()
    with client:
        assert client.post("/jobs", json={"input": {"prompt": "a cat"}}).status_code == 503
        assert client.post("/jobs", json={"model": "pika", "input": {}}).status_code == 503
        assert client.post("/jobs", json={"model": "nope", "input": {}}).status_code == 404
        assert client.get("/jobs/unknown").status_code == 404


def test_idle_unload():
    mgr = make_manager()
    reg = mgr.registry

    async def run():
        await mgr.load("inkling")
        await asyncio.sleep(0)
        mgr.last_activity -= reg.idle_unload_minutes * 60 + 1
        await mgr.idle_check()

    asyncio.run(run())
    assert mgr.backends == {} and mgr.idle_unloaded


def test_files_rejects_traversal():
    client, _ = make_client()
    with client:
        assert client.get("/files/..%2Fetc%2Fpasswd").status_code in (400, 404)


def test_a100_80_rejects_pika_next_to_inkling():
    client, _ = make_client(total_gb=80)
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        r = client.post("/admin/load", json={"id": "pika"})
        assert r.status_code == 409 and "显存不够" in r.json()["reason"]


def _disk_registry(**est_gb):
    raw = yaml.safe_load(REGISTRY_YAML.read_text())
    raw["disk_reserve_gb"] = 0
    for mid, gb in est_gb.items():
        for route in raw["models"][mid]["routes"]:
            route["est_download_gb"] = gb
    return tmp_registry(raw)


def _weights(reg, model_id, nbytes):
    d = reg.models_dir / model_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "w.gguf").write_bytes(b"x" * nbytes)
    return d


def test_disk_evicts_offline_weights_then_loads():
    reg = _disk_registry(inkling=0.000001)  # ~1KB download
    qwen = _weights(reg, "qwen", 2000)
    free = lambda p: 10**15 if not qwen.exists() else 0
    client, _ = make_client(free_bytes=free, registry=reg)
    with client:
        r = client.post("/admin/load", json={"id": "inkling"})
        assert r.status_code == 200, r.json()
        assert r.json()["evicted"] == ["qwen"]
        assert not qwen.exists()


def test_disk_rejects_when_eviction_is_not_enough():
    client, _ = make_client(free_bytes=lambda p: 0, registry=_disk_registry())
    with client:
        r = client.post("/admin/load", json={"id": "pika"})
        assert r.status_code == 409 and "磁盘不够" in r.json()["reason"]


def test_disk_full_swap_refusal_keeps_online_model():
    reg = _disk_registry()
    disk = {"free": 10**15}
    client, _ = make_client(free_bytes=lambda p: disk["free"], registry=reg)
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        _weights(reg, "inkling", 1000)
        disk["free"] = 0
        r = client.post("/admin/load", json={"id": "qwen"})
        assert r.status_code == 409 and "磁盘不够" in r.json()["reason"]
        assert client.get("/admin/status").json()["online"] == ["inkling"]
        assert (reg.models_dir / "inkling").exists()


def test_refused_load_keeps_cached_weights_of_offline_models():
    reg = _disk_registry()
    qwen = _weights(reg, "qwen", 1000)
    client, _ = make_client(free_bytes=lambda p: 0, registry=reg)
    with client:
        r = client.post("/admin/load", json={"id": "pika"})
        assert r.status_code == 409
        assert qwen.exists()


def test_swap_succeeds_when_deleting_the_outgoing_models_weights_makes_room():
    reg = _disk_registry(qwen=0.000002)
    disk = {"tight": False}
    inkling = reg.models_dir / "inkling"
    free = lambda p: 0 if disk["tight"] and inkling.exists() else 10**15
    client, _ = make_client(free_bytes=free, registry=reg)
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        _weights(reg, "inkling", 5000)
        disk["tight"] = True
        r = client.post("/admin/load", json={"id": "qwen"})
        assert r.status_code == 200, r.json()
        assert r.json()["unloaded"] == ["inkling"] and r.json()["evicted"] == ["inkling"]
        assert not inkling.exists()
        assert wait_ready(client, "qwen")["online"] == ["qwen"]


def test_model_list_polling_does_not_reset_idle():
    client, mgr = make_client()
    with client:
        mgr.last_activity -= 1000
        client.get("/v1/models")
        assert client.get("/admin/status").json()["idle_seconds"] >= 999


# ---- API surface follows `serves`, not a model kind ----------------------------------------------

def streamed(obj):
    async def body():
        yield json.dumps(obj).encode()
    return httpx.Response(200, content=body())


def raw_registry():
    return copy.deepcopy(yaml.safe_load(REGISTRY_YAML.read_text()))


def test_models_endpoint_lists_online_chat_models_with_context_length():
    client, _ = make_client()
    with client:
        for mid in ("inkling", "pika"):
            client.post("/admin/load", json={"id": mid})
            wait_ready(client, mid)
        data = client.get("/v1/models").json()["data"]
        assert [m["id"] for m in data] == ["inkling"]
        assert data[0]["context_length"] == 32768


def test_models_endpoint_omits_context_length_when_caps_lack_it():
    raw = raw_registry()
    del raw["models"]["inkling"]["caps"]["context"]
    client, _ = make_client(registry=tmp_registry(raw))
    with client:
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        assert "context_length" not in client.get("/v1/models").json()["data"][0]


def test_status_reports_group_serves_caps_and_engine_per_model():
    raw = raw_registry()
    next(r for r in raw["models"]["inkling"]["routes"] if r.get("name", r["engine"]) == "strata")["caps"] = {"vision": True}
    client, _ = make_client(registry=tmp_registry(raw))
    with client:
        models = {m["id"]: m for m in client.get("/admin/status").json()["models"]}
        assert "kind" not in models["inkling"]
        assert (models["inkling"]["group"], models["inkling"]["serves"], models["inkling"]["engine"]) == \
            ("llm", ["chat"], "strata")
        assert models["inkling"]["caps"]["vision"] is True and models["qwen"]["caps"]["reasoning"] == \
            ["none", "low", "medium", "high"]
        assert (models["pika"]["group"], models["pika"]["serves"], models["pika"]["engine"]) == \
            ("video", ["video"], "diffusers")
        assert models["qwen-tiny"]["engine"] == "llama-cpp" and models["qwen-tiny"]["caps"] == {"context": 8192}
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        online = next(m for m in client.get("/admin/status").json()["models"] if m["id"] == "inkling")
        assert online["engine"] == "strata" and online["caps"]["vision"] is True


def test_group_names_mean_nothing_to_the_code():
    raw = raw_registry()
    for m in raw["models"].values():
        m["group"] = {"llm": "text", "video": "clips", "image": "pictures"}[m["group"]]
    raw["rules"] = [{"group": "text", "max_online": 1, "on_conflict": "swap"},
                    {"group": "clips", "requires_online_below": {"group": "text", "vram_gb": 20}}]
    client, _ = make_client(registry=tmp_registry(raw))
    with client:
        client.post("/admin/load", json={"id": "qwen"})
        wait_ready(client, "qwen")
        r = client.post("/admin/load", json={"id": "pika"})
        assert r.status_code == 409 and "qwen" in r.json()["reason"]
        assert client.post("/admin/load", json={"id": "inkling"}).json()["unloaded"] == ["qwen"]


def test_a_chat_model_in_any_group_is_listed_and_proxied():
    raw = raw_registry()
    raw["models"]["inkling"]["group"] = "assistants"
    raw["rules"] = [{"group": "video", "requires_online_below": {"group": "assistants", "vram_gb": 20}}]
    client, _ = make_client(registry=tmp_registry(raw))
    with client:
        client.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda r: streamed({"host": r.url.host})))
        client.post("/admin/load", json={"id": "inkling"})
        wait_ready(client, "inkling")
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["inkling"]
        assert client.post("/v1/chat/completions", json={"model": "inkling"}).json() == {"host": "inkling.invalid"}


def test_proxy_routes_by_the_requested_model_and_never_substitutes():
    raw = raw_registry()
    raw["rules"][0]["max_online"] = 2
    client, mgr = make_client(registry=tmp_registry(raw))
    with client:
        seen = []

        def upstream(request):
            seen.append((request.url.host, json.loads(request.content)["model"]))
            return streamed({})
        client.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        for mid in ("qwen-tiny", "inkling"):
            client.post("/admin/load", json={"id": mid})
            wait_ready(client, mid)
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["inkling", "qwen-tiny"]
        client.post("/v1/chat/completions", json={"model": "qwen-tiny"})
        client.post("/v1/chat/completions", json={"model": "inkling"})
        assert seen == [("qwen-tiny.invalid", "upstream-qwen-tiny"), ("inkling.invalid", "upstream-inkling")]
        r = client.post("/v1/chat/completions", json={"model": "anything"})
        assert r.status_code == 404 and "inkling" in r.json()["error"]["message"]
        r = client.post("/v1/chat/completions", json={"model": "qwen"})
        assert r.status_code == 404 and "/load qwen" in r.json()["error"]["message"]
        r = client.post("/v1/chat/completions", json={"messages": []})
        assert r.status_code == 400 and "qwen-tiny" in r.json()["error"]["message"]
        assert len(seen) == 2
        client.post("/admin/unload", json={"id": "qwen-tiny"})
        client.post("/v1/chat/completions", json={"messages": []})
        assert seen[-1] == ("inkling.invalid", "upstream-inkling"), "a lone online model fills in the missing id"


def test_jobs_route_to_the_online_jobs_model_and_remember_the_owner():
    raw = raw_registry()
    raw["models"]["clip"] = raw["models"].pop("pika")
    raw["models"]["clip"]["routes"][0]["port"] = 8095
    client, mgr = make_client(registry=tmp_registry(raw))
    with client:
        hosts = []

        def upstream(request):
            hosts.append((request.url.host, request.url.path, request.content))
            return httpx.Response(200, json={"id": "j1", "state": "queued"})
        client.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        client.post("/admin/load", json={"id": "clip"})
        wait_ready(client, "clip")
        job = client.post("/jobs", json={"input": {"prompt": "x", "seconds": 6}}).json()
        assert job == {"id": "j1", "state": "queued", "model": "clip"}
        assert client.get("/jobs/j1").json()["model"] == "clip"
        assert hosts == [("clip.invalid", "/generate", b'{"prompt": "x", "seconds": 6}'),
                         ("clip.invalid", "/jobs/j1", b"")]
        assert mgr.backends["clip"].queue_depth == 1


def test_jobs_pick_the_online_model_that_serves_the_requested_capability():
    client, _ = make_client()
    with client:
        hosts = []

        def upstream(request):
            hosts.append(request.url.host)
            return httpx.Response(200, json={"id": f"j{len(hosts)}", "state": "queued"})
        client.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        for mid in ("pika", "moss"):
            client.post("/admin/load", json={"id": mid})
            wait_ready(client, mid)
        assert client.post("/jobs", json={"input": {"prompt": "x"}}).status_code == 400
        assert client.post("/jobs", json={"serves": "image", "input": {"prompt": "x"}}).json()["model"] == "moss"
        assert client.post("/jobs", json={"serves": "video", "input": {"prompt": "x"}}).json()["model"] == "pika"
        assert hosts == ["moss.invalid", "pika.invalid"]


def test_503_names_the_capability_and_loading_models():
    client, mgr = make_client()
    with client:
        msg = client.post("/v1/chat/completions", json={}).json()["error"]["message"]
        assert "chat" in msg and "/load inkling" in msg and "/load qwen-tiny" in msg and "/load pika" not in msg
        detail = client.post("/jobs", json={"input": {"prompt": "x"}}).json()["detail"]
        assert "/load pika" in detail and "/load inkling" not in detail
