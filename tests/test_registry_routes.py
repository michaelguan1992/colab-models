import copy

import pytest
import yaml

from controller.policy import MaxOnline, RequiresBelow
from controller.registry import Registry, RegistryError, choose_route, parse_sm
from controller.spec import ArchiveInstall, ScriptInstall, VenvInstall
from ctl_helpers import REGISTRY_YAML, make_client, tmp_registry, wait_ready

RAW = yaml.safe_load(REGISTRY_YAML.read_text())


def two_route_raw():
    raw = copy.deepcopy(RAW)
    fast = {"name": "fast", "engine": "strata", "model_path": "fast.gguf", "vram_gb": 20,
            "est_download_gb": 15, "requires": {"sm": [120]}, "caps": {"vision": True}}
    raw["models"]["inkling"]["caps"] = {"tools": True}
    strata = next(r for r in raw["models"]["inkling"]["routes"] if r.get("name", r["engine"]) == "strata")
    raw["models"]["inkling"]["routes"] = [fast, strata]
    return raw


def test_shipped_registry_parses_into_typed_data():
    reg = Registry(copy.deepcopy(RAW))
    assert reg.rules == (MaxOnline("llm", 1, True), RequiresBelow("video", "llm", 22.0))
    assert [type(reg.engines[n].install) for n in ("strata", "llama-cpp", "diffusers")] == \
        [ScriptInstall, ArchiveInstall, VenvInstall]
    assert {m.id: m.group for m in reg.models.values()} == \
        {"inkling": "llm", "qwen": "llm", "qwen-tiny": "llm", "pika": "video", "moss": "image",
         "moss-comfy": "image", "pika-comfy": "video", "sulphur-comfy": "video"}
    assert reg.ids_serving("chat") == ["inkling", "qwen", "qwen-tiny"] and reg.ids_serving("video") == ["pika"]
    assert reg.engines["strata"].daemonizes and reg.engines["strata"].ready.identity == "model_path"
    assert reg.engines["diffusers"].shape == "jobs"
    llama = reg.engines["llama-cpp"].install
    assert [a.sha256[:6] for a in llama.archives] == ["cb0993", "485df8"] and llama.bin == "llama-server"


def test_shipped_registry_gives_every_model_one_unique_port():
    reg = Registry(copy.deepcopy(RAW))
    ports = {mid: {r.port for r in reg.routes(mid)} for mid in reg.models}
    assert all(len(p) == 1 for p in ports.values())
    assert len({next(iter(p)) for p in ports.values()}) == len(ports)
    assert ports["pika"] == {8090}


def test_route_carries_merged_caps_and_params():
    reg = Registry(copy.deepcopy(RAW))
    r = reg.choose("inkling", "7.5").route
    assert r.caps == {"tools": True, "vision": False, "context": 32768, "reasoning": []}
    assert r.vram_gb == 18 and r.est_download_gb == 17.2 and r.engine == "strata" and r.name == "strata"
    assert r.vram_margin_gb is None
    assert reg.routes("pika")[0].params["workflow"] == "fl2va"


def test_duplicate_port_is_rejected_at_load():
    raw = copy.deepcopy(RAW)
    raw["models"]["inkling"]["routes"][0]["port"] = 8090
    with pytest.raises(RegistryError, match="8090"):
        Registry(raw)


def edit_engine(name, **changes):
    def edit(raw):
        raw["engines"][name].update(changes)
    return edit


BAD = [
    ("unknown engine", lambda r: r["models"]["inkling"]["routes"][0].update(engine="vllm"), "engine"),
    ("missing vram", lambda r: r["models"]["inkling"]["routes"][0].pop("vram_gb"), "vram_gb"),
    ("no routes", lambda r: r["models"]["inkling"].pop("routes"), "routes"),
    ("empty routes", lambda r: r["models"]["inkling"].update(routes=[]), "routes"),
    ("kind is gone", lambda r: r["models"]["inkling"].update(kind="llm"), "kind"),
    ("no group", lambda r: r["models"]["inkling"].pop("group"), "group"),
    ("empty serves", lambda r: r["models"]["inkling"].update(serves=[]), "serves"),
    ("no serves", lambda r: r["models"]["inkling"].pop("serves"), "serves"),
    ("bad sm", lambda r: r["models"]["inkling"]["routes"][0].update(requires={"sm": "120"}), "sm"),
    ("bad requires key", lambda r: r["models"]["inkling"]["routes"][0].update(requires={"arch": 1}), "requires"),
    ("duplicate route name", lambda r: r["models"]["inkling"]["routes"].append(
        dict(r["models"]["inkling"]["routes"][0])), "重复"),
    ("unknown route field", lambda r: r["models"]["inkling"]["routes"][0].update(vramgb=1), "vramgb"),
    ("old top-level strata block", lambda r: r.update(strata={"bin": "strata"}), "strata"),
    ("unknown group in rules", lambda r: r["rules"].append({"group": "audio", "max_online": 1}), "audio"),
    ("unknown group in requires_online_below",
     lambda r: r["rules"][1]["requires_online_below"].update(group="audio"), "audio"),
    ("rule group requires itself", lambda r: r["rules"][1]["requires_online_below"].update(group="video"), "相同"),
    ("rule without a primitive", lambda r: r["rules"].append({"group": "llm"}), "max_online"),
    ("on_conflict without max_online", lambda r: r["rules"].append({"group": "llm", "on_conflict": "swap"}),
     "on_conflict"),
    ("bad on_conflict", lambda r: r["rules"][0].update(on_conflict="explode"), "on_conflict"),
    ("max_online zero", lambda r: r["rules"][0].update(max_online=0), "max_online"),
    ("max_online twice", lambda r: r["rules"].append({"group": "llm", "max_online": 2}), "重复"),
    ("bad shape", edit_engine("strata", shape="daemon"), "shape"),
    ("engine without launch", lambda r: r["engines"]["strata"].pop("launch"), "launch"),
    ("engine without ready", lambda r: r["engines"]["strata"].pop("ready"), "ready"),
    ("ready path without slash", lambda r: r["engines"]["strata"]["ready"].update(path="v1/models"), "ready.path"),
    ("unknown api", edit_engine("strata", apis=["grpc"]), "apis"),
    ("jobs engine cannot daemonize", edit_engine("diffusers", daemonizes=True), "jobs"),
    ("extra must stand alone", lambda r: r["engines"]["strata"].update(
        launch=["{bin}", "--x={extra}"]), "{extra}"),
    ("launch placeholder nobody provides", lambda r: r["engines"]["llama-cpp"]["launch"].append("{nope}"), "nope"),
    ("launch needs a param the route lacks", lambda r: r["models"]["pika"]["routes"][0]["params"].pop("adapter_file"),
     "adapter_file"),
    ("stop placeholder nobody provides", lambda r: r["engines"]["strata"]["stop"].append("{model_path}"),
     "model_path"),
    ("model_path placeholder without model_path", lambda r: r["models"]["qwen-tiny"]["routes"][0].pop("model_path"),
     "model_path"),
    ("extra with no {extra} in launch", lambda r: r["engines"]["llama-cpp"].update(
        launch=["{bin}", "-m", "{model_path}", "--port", "{port}"]), "extra"),
    ("param shadows a builtin", lambda r: r["models"]["pika"]["routes"][0]["params"].update(port=1), "port"),
    ("script install without a checksum source", lambda r: r["engines"]["strata"]["install"].pop("sums_url"),
     "sha256"),
    ("archive install without sha256", lambda r: r["engines"]["llama-cpp"]["install"].pop("sha256"), "sha256"),
    ("extra archive without sha256",
     lambda r: r["engines"]["llama-cpp"]["install"]["extra_archives"][0].pop("sha256"), "sha256"),
    ("malformed sha256", lambda r: r["engines"]["llama-cpp"]["install"].update(sha256="abc"), "sha256"),
    ("unknown install kind", lambda r: r["engines"]["strata"]["install"].update(kind="brew"), "kind"),
    ("venv pip_files placeholder", lambda r: r["engines"]["diffusers"]["install"].update(
        pip_files=["{bin}/x"]), "pip_files"),
    ("engine sm and route sm disjoint", lambda r: (
        r["engines"]["strata"].update(requires={"sm": [120]}),
        r["models"]["inkling"]["routes"][0].update(requires={"sm": [80]})), "没有交集"),
    ("caps context type", lambda r: r["models"]["inkling"]["caps"].update(context="big"), "context"),
]


@pytest.mark.parametrize("name,edit,match", BAD, ids=[b[0] for b in BAD])
def test_bad_registry_is_rejected_with_a_message_naming_the_field(name, edit, match):
    raw = copy.deepcopy(RAW)
    edit(raw)
    with pytest.raises(RegistryError, match=match):
        Registry(raw)


def test_engine_and_route_sm_requirements_merge():
    raw = copy.deepcopy(RAW)
    raw["engines"]["strata"]["requires"] = {"sm": [120, 100, 89]}
    raw["models"]["inkling"]["routes"][0]["requires"] = {"sm": [120, 80]}
    reg = Registry(raw)
    assert reg.routes("inkling")[0].requires_sm == (120,)
    assert reg.routes("qwen")[-1].requires_sm == (120, 100, 89)  # the strata route


def test_parse_sm():
    assert parse_sm("12.0") == 120 and parse_sm("8.9") == 89 and parse_sm("") is None


def test_first_matching_route_wins_and_falls_back():
    reg = Registry(two_route_raw())
    assert reg.choose("inkling", "12.0").route.name == "fast"
    assert reg.choose("inkling", "8.9").route.name == "strata"
    assert reg.choose("inkling", "").route.name == "strata"


def test_route_caps_override_model_caps():
    reg = Registry(two_route_raw())
    assert reg.routes("inkling")[0].caps == {"tools": True, "vision": True}
    assert reg.routes("inkling")[1].caps == {"tools": True}


def test_no_fitting_route_lists_every_requirement():
    raw = two_route_raw()
    raw["models"]["inkling"]["routes"].pop()
    choice = Registry(raw).choose("inkling", "8.9")
    assert choice.route is None
    assert "fast" in choice.reason and "sm_120" in choice.reason and "sm_89" in choice.reason


def test_unknown_route_name_is_a_key_error():
    with pytest.raises(KeyError, match="nope"):
        choose_route("inkling", Registry(copy.deepcopy(RAW)).routes("inkling"), "", "nope")


def test_load_picks_route_by_gpu_and_status_shows_it():
    client, mgr = make_client(registry=tmp_registry(two_route_raw()), compute_cap="12.0")
    with client:
        models = {m["id"]: m for m in client.get("/admin/status").json()["models"]}
        assert models["inkling"]["route"] == "fast" and models["pika"]["route"] == "fp8"
        assert client.post("/admin/load", json={"id": "inkling"}).status_code == 200
        wait_ready(client, "inkling")
        assert mgr.backends["inkling"].route.name == "fast"
        assert mgr.backends["inkling"].route.model_path == "fast.gguf"


def test_load_falls_back_on_older_gpu_and_explicit_route_must_fit():
    client, mgr = make_client(registry=tmp_registry(two_route_raw()), compute_cap="8.9")
    with client:
        r = client.post("/admin/load", json={"id": "inkling", "route": "fast"})
        assert r.status_code == 409 and "sm_120" in r.json()["reason"] and "sm_89" in r.json()["reason"]
        assert mgr.backends == {}
        assert client.post("/admin/load", json={"id": "inkling"}).status_code == 200
        wait_ready(client, "inkling")
        assert mgr.backends["inkling"].route.name == "strata"
        assert client.post("/admin/load", json={"id": "qwen", "route": "nope"}).status_code == 404


def test_status_reports_why_no_route_fits():
    raw = two_route_raw()
    raw["models"]["inkling"]["routes"].pop()
    client, _ = make_client(registry=tmp_registry(raw), compute_cap="8.9")
    with client:
        m = next(m for m in client.get("/admin/status").json()["models"] if m["id"] == "inkling")
        assert m["route"] is None and "sm_120" in m["route_reason"]


def test_shipped_registry_prefers_basalt_on_blackwell_and_strata_elsewhere():
    reg = Registry(copy.deepcopy(RAW))
    assert reg.choose("inkling", "12.0").route.engine == "basalt"
    assert reg.choose("inkling", "8.9").route.engine == "strata"
    assert reg.choose("inkling", "8.0").route.engine == "strata"


def test_inkling_gets_1m_on_big_gpus_256k_on_the_l4_and_32k_elsewhere():
    reg = Registry(copy.deepcopy(RAW))
    assert [reg.choose("inkling", cc).route.caps["context"] for cc in ("12.0", "9.0", "8.0", "8.9", "7.5")] == \
        [1048576, 1048576, 1048576, 262144, 32768]


def test_l4_route_fits_only_because_it_drops_the_vram_margin():
    reg = Registry(copy.deepcopy(RAW))
    r = reg.choose("inkling", "8.9").route
    assert r.name == "strata-256k" and r.vram_margin_gb == 0
    l4_gb = 23034 / 1024
    assert r.vram_gb <= l4_gb - r.vram_margin_gb and r.vram_gb > l4_gb - reg.vram_margin_gb


def test_qwen_reads_images_on_every_route_and_inkling_only_on_basalt():
    reg = Registry(copy.deepcopy(RAW))
    assert all(r.caps["vision"] for r in reg.routes("qwen"))
    assert [r.caps["vision"] for r in reg.routes("inkling")] == [True, False, False, False]  # Strata refuses Inkling images


def test_pika_runs_fp8_only_where_the_gpu_has_fp8():
    reg = Registry(copy.deepcopy(RAW))
    assert reg.choose("pika", "12.0").route.name == "fp8" and reg.choose("pika", "8.9").route.name == "fp8"
    assert reg.choose("pika", "8.0").route.name == "bf16"  # A100
    assert "--cpu-offload" in reg.choose("pika", "8.0").route.extra


def test_moss_stays_resident_only_where_it_fits():
    reg = Registry(copy.deepcopy(RAW))
    assert reg.choose("moss", "12.0").route.name == "resident" and reg.choose("moss", "8.0").route.name == "resident"
    l4 = reg.choose("moss", "8.9").route
    assert l4.name == "offload" and "--cpu-offload" in l4.extra


def test_also_needs_a_repo_and_a_path_inside_the_weights_folder():
    raw = copy.deepcopy(RAW)
    route = raw["models"]["sulphur-comfy"]["routes"][0]
    assert [s.repo for s in Registry(copy.deepcopy(raw)).routes("sulphur-comfy")[0].sources()] == \
        ["Comfy-Org/MiniMax-H3", "SulphurAI/Sulphur-3", "Pika-Labs/Pika-H3-Flex-Turbo-LoRA"]
    route["also"][0]["into"] = "../elsewhere"
    with pytest.raises(RegistryError, match="into"):
        Registry(copy.deepcopy(raw))
    route["also"][0]["into"] = "diffusion_models"
    del route["repo"], route["include"], route["revision"]
    with pytest.raises(RegistryError, match="also"):
        Registry(raw)
