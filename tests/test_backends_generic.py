import asyncio
import socket
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from controller.backends import server as server_mod
from controller.backends.jobs import JobsBackend
from controller.backends.server import ServerBackend
from controller.install import Installed
from controller.templating import expand
from ctl_helpers import REGISTRY_YAML, build, tmp_registry


def argv_of(model_id, installed):
    b = build(ServerBackend if model_id != "pika" else JobsBackend, model_id)
    b.installed = installed
    return b, expand(b.engine.launch, b.values(), b.route.extra)


def test_strata_launch_is_the_old_hardcoded_command_rendered_from_the_template():
    b, argv = argv_of("inkling", Installed(bin=Path("/root/.local/bin/strata")))
    assert argv == ["/root/.local/bin/strata", "launch", "--model-path",
                    str(b.registry.models_dir / "inkling" / "Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"),
                    "--port", "8081", "--access", "local", "--no-api-key", "--yes", "--no-tui",
                    "--context", "32k", "--drafting", "auto"]


def test_llama_cpp_launch_comes_from_data_alone():
    b, argv = argv_of("qwen-tiny", Installed(bin=Path("/e/llama-server"), lib_path="/e"))
    assert argv == ["/e/llama-server", "-m", str(b.registry.models_dir / "qwen-tiny" / "Qwen3-0.6B-Q8_0.gguf"),
                    "--port", "8083", "--host", "127.0.0.1", "-ngl", "999", "-c", "8192"]


def test_jobs_launch_uses_the_venv_python_and_route_params():
    b, argv = argv_of("pika", Installed(venv_python=Path("/v/bin/python")))
    reg = b.registry
    assert argv[:3] == ["/v/bin/python", "-m", "controller.pika_worker"]
    assert argv[-1] == "--cpu-offload"
    pairs = dict(zip(argv[3:-1:2], argv[4:-1:2]))
    assert pairs == {"--model-dir": str(reg.models_dir / "pika"), "--adapter-file": "fl2va/adapter.safetensors",
                     "--workflow": "fl2va", "--fps": "24", "--width": "832", "--height": "480",
                     "--outputs-dir": str(reg.outputs_dir), "--port": "8090"}


def test_expand_puts_the_extra_list_in_place_and_leaves_other_braces_alone():
    assert expand(["a", "{extra}", "--j={\"k\": 1}", "{port}"], {"port": "9"}, ["x", "y"]) == \
        ["a", "x", "y", "--j={\"k\": 1}", "9"]
    assert expand(["{extra}", "z"], {}, []) == ["z"]


def test_downloads_follow_the_route(monkeypatch):
    seen = []

    async def record(self, cmd, env=None):
        seen.append(cmd)

    from controller.backends.base import Backend
    monkeypatch.setattr(Backend, "_run_logged", record)
    reg = tmp_registry()
    asyncio.run(build(JobsBackend, "pika", reg).download())
    [base] = seen
    assert base[:3] == ["hf", "download", "Pika-Labs/Pika-H3-Flex"] and "fl2va/*" in base
    seen.clear()
    asyncio.run(build(ServerBackend, "qwen-tiny", reg).download())
    assert seen == [["hf", "download", "Qwen/Qwen3-0.6B-GGUF", "--local-dir", str(reg.models_dir / "qwen-tiny"),
                     "--include", "Qwen3-0.6B-Q8_0.gguf"]]


def test_also_sources_download_into_their_own_subfolders(monkeypatch):
    seen = []

    async def record(self, cmd, env=None):
        seen.append(cmd)

    from controller.backends.base import Backend
    monkeypatch.setattr(Backend, "_run_logged", record)
    reg = tmp_registry()
    asyncio.run(build(ServerBackend, "sulphur-comfy", reg, compute_cap="8.0").download())
    root = reg.models_dir / "sulphur-comfy"
    assert [(c[2], c[4]) for c in seen] == [
        ("Comfy-Org/MiniMax-H3", str(root)), ("SulphurAI/Sulphur-3", str(root / "diffusion_models")),
        ("Pika-Labs/Pika-H3-Flex-Turbo-LoRA", str(root / "loras"))]
    assert seen[1][5:] == ["--revision", "988224b9913740d60ee54b5a53c7a892467edc76", "--include", "s3.safetensors"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mini_registry(engine_extra=None, route_extra=None):
    raw = {
        "paths": yaml.safe_load(REGISTRY_YAML.read_text())["paths"],
        "engines": {"web": {
            "shape": "server",
            "launch": [sys.executable, "-m", "http.server", "{port}", "--bind", "127.0.0.1", "{extra}"],
            "ready": {"path": "/"}, **(engine_extra or {})}},
        "models": {"site": {"group": "tools", "serves": ["chat"],
                            "routes": [{"engine": "web", "vram_gb": 1, "port": free_port(), **(route_extra or {})}]}},
    }
    return tmp_registry(raw)


def test_a_new_engine_and_model_run_from_yaml_alone(tmp_path):
    reg = mini_registry(route_extra={"extra": ["--directory", str(tmp_path)]})

    async def go():
        b = build(ServerBackend, "site", reg)
        b.ready_poll_s = 0.05
        await b.start()
        assert b.state == "ready", b.error
        r = httpx.get(f"{b.base_url}/")
        assert r.status_code == 200
        await b.stop()
        assert b.state == "stopped" and await b.wait_released(5, poll=0.05)
    asyncio.run(go())


def test_route_without_repo_never_downloads_or_enters_the_downloading_state():
    reg = mini_registry()
    states = []

    class Watching(ServerBackend):
        async def download(self):
            raise AssertionError("no repo, no download")

        async def _ready(self, client):
            states.append(self.state)
            return await super()._ready(client)

    async def go():
        b = build(Watching, "site", reg)
        b.ready_poll_s = 0.05
        await b.start()
        await b.stop()
    asyncio.run(go())
    assert set(states) == {"starting"}


def test_daemonizing_engine_is_stopped_with_its_stop_command_before_the_kill(tmp_path):
    marker = tmp_path / "stopped.txt"
    reg = mini_registry(engine_extra={"daemonizes": True,
                                      "stop": ["sh", "-c", f"echo port {{port}} > {marker}"]})

    async def go():
        b = build(ServerBackend, "site", reg)
        b.ready_poll_s = 0.05
        await b.start()
        assert b.state == "ready", b.error
        await b.stop()
        assert marker.read_text().split() == ["port", str(b.port)]
    asyncio.run(go())


def test_daemonizing_pids_come_from_the_port_listener_tree(monkeypatch):
    reg = mini_registry(engine_extra={"daemonizes": True})
    b = build(ServerBackend, "site", reg)
    roots = []
    monkeypatch.setattr(server_mod, "listener_pid", lambda port: 4000)
    monkeypatch.setattr(server_mod, "_runtime_root", lambda pid, marker: roots.append((pid, marker)) or 3000)
    monkeypatch.setattr(server_mod, "process_tree", lambda pid: {pid, 3001})
    assert b.pids() == set()
    b.state = "ready"
    assert b.pids() == {3000, 3001}
    assert roots == [(4000, Path(sys.executable).name)]
    b.state = "stopped"
    assert b.pids() == set()


def test_non_daemonizing_engine_ignores_the_port_listener(monkeypatch):
    b = build(ServerBackend, "site", mini_registry())
    b.state = "ready"
    monkeypatch.setattr(server_mod, "listener_pid", lambda port: pytest.fail("listener consulted"))
    assert b.pids() == set()


def served(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_readiness_polls_the_engines_ready_path_and_only_checks_identity_when_asked():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"data": [{"id": "Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"}]})

    async def go():
        async with served(handler) as client:
            tiny = build(ServerBackend, "qwen-tiny")
            assert await tiny._ready(client)  # llama-cpp: no identity check, any 200 is ready
            assert tiny.upstream_model == "Inkling-2-Mini-Uncensored-AD-Q4_K.gguf", "proxy must use the engine's own id"
            qwen = build(ServerBackend, "qwen")
            assert not await qwen._ready(client)  # strata: another model's server answers on our port
    asyncio.run(go())
    assert paths == ["/v1/models", "/v1/models"]


def test_non_200_is_not_ready():
    async def go():
        async with served(lambda r: httpx.Response(503)) as client:
            assert not await build(ServerBackend, "qwen-tiny")._ready(client)
            assert not await build(JobsBackend, "pika")._ready(client)
    asyncio.run(go())


def test_jobs_readiness_reads_health_for_queue_depth_busy_and_errors():
    state = {"body": {"ready": True, "error": None, "queue_depth": 3}}
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json=state["body"])

    async def go():
        b = build(JobsBackend, "pika")
        async with served(handler) as client:
            assert await b.check_health(client) and b.busy() and b.queue_depth == 3
            state["body"] = {"ready": False, "error": None, "queue_depth": 0}
            assert not await b.check_health(client) and not b.busy()
            state["body"] = {"ready": False, "error": "OOM", "queue_depth": 0}
            assert not await b.check_health(client)  # the worker's error reads as unhealthy
            with pytest.raises(RuntimeError, match="OOM"):
                await b._ready(client)
    asyncio.run(go())
    assert set(paths) == {"/health"}


def test_basalt_counts_as_ready_only_once_its_model_reports_loaded():
    state = {"value": "loading"}

    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "Inkling-OU", "status": {"value": state["value"]}}]})

    async def go():
        async with served(handler) as client:
            b = build(ServerBackend, "inkling", compute_cap="12.0")
            assert b.engine.name == "basalt"
            assert not await b._ready(client), "Basalt answers /v1/models while weights are still loading"
            state["value"] = "loaded"
            assert await b._ready(client)
            assert b.upstream_model == "Inkling-OU"
    asyncio.run(go())


def test_failed_download_retries_with_fewer_connections(monkeypatch):
    calls = []

    async def flaky(self, cmd, env=None):
        calls.append(env.get("HF_XET_HIGH_PERFORMANCE") or env.get("HF_XET_FIXED_DOWNLOAD_CONCURRENCY"))
        if len(calls) < 3:
            raise RuntimeError("hf 退出码 1")

    from controller.backends.base import Backend
    monkeypatch.setattr(Backend, "_run_logged", flaky)
    b = build(ServerBackend, "qwen-tiny", tmp_registry())
    b.download_retry_s = 0
    asyncio.run(b.download())
    assert calls == ["1", "4", "4"]  # high-performance first, then 4 fixed connections


def test_download_gives_up_after_the_last_attempt(monkeypatch):
    async def broken(self, cmd, env=None):
        raise RuntimeError("hf 退出码 1")

    from controller.backends.base import Backend
    monkeypatch.setattr(Backend, "_run_logged", broken)
    b = build(ServerBackend, "qwen-tiny", tmp_registry())
    b.download_retry_s = 0
    with pytest.raises(RuntimeError):
        asyncio.run(b.download())


def test_comfyui_launches_through_the_launcher_and_its_queue_counts_as_work():
    b, argv = argv_of("moss-comfy", Installed(bin=Path("/e/ComfyUI/main.py"), venv_python=Path("/e/ComfyUI/.venv/bin/python")))
    assert argv[:9] == ["/e/ComfyUI/.venv/bin/python", "-m", "controller.comfy_launch", "--comfy", "/e/ComfyUI/main.py",
                        "--model-dir", str(b.registry.models_dir / "moss-comfy"), "--outputs-dir", str(b.registry.outputs_dir)]
    assert argv[9:11] == ["--port", "8188"] and argv[11:13] == ["--workflow", "image_moss_image_t2i=MOSS 文生图"]

    def handler(request):
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [[1]], "queue_pending": [[2], [3]]})
        return httpx.Response(200, json={"system": {}, "devices": []})

    async def go():
        async with served(handler) as client:
            assert await b._ready(client)
    asyncio.run(go())
    assert b.queue_depth == 3 and b.busy()
