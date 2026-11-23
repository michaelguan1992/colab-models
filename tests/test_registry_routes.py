import copy

import pytest
import yaml

from controller.registry import Registry, RegistryError, choose_route, parse_sm
from ctl_helpers import REGISTRY_YAML, make_client, tmp_registry, wait_ready

RAW = yaml.safe_load(REGISTRY_YAML.read_text())


def two_route_raw():
    raw = copy.deepcopy(RAW)
    fast = {"name": "fast", "engine": "strata", "model_path": "fast.gguf", "vram_gb": 20,
            "est_download_gb": 15, "requires": {"sm": [120]}, "caps": {"vision": True}}
    raw["models"]["inkling"]["caps"] = {"tools": True}
    raw["models"]["inkling"]["routes"].insert(0, fast)
    return raw


def test_shipped_registry_gives_every_model_one_unique_port():
    reg = Registry(copy.deepcopy(RAW))
    ports = {mid: {r.port for r in reg.routes(mid)} for mid in reg.models}
    assert all(len(p) == 1 for p in ports.values())
    assert len({next(iter(p)) for p in ports.values()}) == len(ports)
    assert ports["pika"] == {8090}


def test_route_cfg_is_the_flat_dict_backends_get():
    cfg = Registry(copy.deepcopy(RAW)).routes("inkling")[0].cfg
    assert cfg["kind"] == "llm" and cfg["repo"] and cfg["model_path"]
    assert cfg["est_vram_gb"] == 18 and cfg["est_download_gb"] == 17.2
    assert cfg["engine"] == "strata" and cfg["route"] == "strata" and "routes" not in cfg


def test_duplicate_port_is_rejected_at_load():
    raw = copy.deepcopy(RAW)
    raw["models"]["inkling"]["routes"][0]["port"] = 8090
    with pytest.raises(RegistryError, match="8090"):
        Registry(raw)


@pytest.mark.parametrize("edit,match", [
    (lambda r: r["models"]["inkling"]["routes"][0].update(engine="vllm"), "engine"),
    (lambda r: r["models"]["inkling"]["routes"][0].pop("vram_gb"), "vram_gb"),
    (lambda r: r["models"]["inkling"].pop("routes"), "routes"),
    (lambda r: r["models"]["inkling"].update(routes=[]), "routes"),
    (lambda r: r["models"]["inkling"].update(kind="audio"), "kind"),
    (lambda r: r["models"]["inkling"]["routes"][0].update(engine="pika"), "只能跑"),
    (lambda r: r["models"]["inkling"]["routes"][0].update(requires={"sm": "120"}), "sm"),
    (lambda r: r["models"]["inkling"]["routes"][0].update(requires={"arch": 1}), "requires"),
    (lambda r: r["models"]["inkling"]["routes"].append(dict(r["models"]["inkling"]["routes"][0])), "重复"),
])
def test_bad_registry_is_rejected_with_a_message_naming_the_field(edit, match):
    raw = copy.deepcopy(RAW)
    edit(raw)
    with pytest.raises(RegistryError, match=match):
        Registry(raw)


def test_parse_sm():
    assert parse_sm("12.0") == 120 and parse_sm("8.9") == 89 and parse_sm("") is None


def test_first_matching_route_wins_and_falls_back():
    reg = Registry(two_route_raw())
    assert reg.choose("inkling", "12.0").route.name == "fast"
    assert reg.choose("inkling", "8.9").route.name == "strata"
    assert reg.choose("inkling", "").route.name == "strata"


def test_route_caps_override_model_caps():
    reg = Registry(two_route_raw())
    assert reg.routes("inkling")[0].cfg["caps"] == {"tools": True, "vision": True}
    assert reg.routes("inkling")[1].cfg["caps"] == {"tools": True}


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
        assert models["inkling"]["route"] == "fast" and models["pika"]["route"] == "pika"
        assert client.post("/admin/load", json={"id": "inkling"}).status_code == 200
        wait_ready(client, "inkling")
        assert mgr.backends["inkling"].cfg["route"] == "fast"
        assert mgr.backends["inkling"].cfg["model_path"] == "fast.gguf"


def test_load_falls_back_on_older_gpu_and_explicit_route_must_fit():
    client, mgr = make_client(registry=tmp_registry(two_route_raw()), compute_cap="8.9")
    with client:
        r = client.post("/admin/load", json={"id": "inkling", "route": "fast"})
        assert r.status_code == 409 and "sm_120" in r.json()["reason"] and "sm_89" in r.json()["reason"]
        assert mgr.backends == {}
        assert client.post("/admin/load", json={"id": "inkling"}).status_code == 200
        wait_ready(client, "inkling")
        assert mgr.backends["inkling"].cfg["route"] == "strata"
        assert client.post("/admin/load", json={"id": "qwen", "route": "nope"}).status_code == 404


def test_status_reports_why_no_route_fits():
    raw = two_route_raw()
    raw["models"]["inkling"]["routes"].pop()
    client, _ = make_client(registry=tmp_registry(raw), compute_cap="8.9")
    with client:
        m = next(m for m in client.get("/admin/status").json()["models"] if m["id"] == "inkling")
        assert m["route"] is None and "sm_120" in m["route_reason"]
