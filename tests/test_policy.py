import pytest

from controller.policy import DiskView, MaxOnline, Online, RequiresBelow, Spec, plan_load, plan_rules, violations

GB = 10**9
ROOMY = DiskView(free_bytes=10**15, weights_bytes={}, need_bytes=0, reserve_bytes=0, inflight_bytes=0)

LLM_VIDEO = [MaxOnline("llm", 1, True), RequiresBelow("video", "llm", 20)]
LLM_VIDEO_SPECS = {s.id: s for s in (Spec("qwen", "llm", 50), Spec("inkling", "llm", 18), Spec("pika", "video", 69))}

# Three groups with nothing llm- or video-specific: a swaps at 1, b refuses at 2, c is gated by both.
THREE = [MaxOnline("a", 1, True), MaxOnline("b", 2, False), MaxOnline("e", 2, True),
         RequiresBelow("c", "b", 10), RequiresBelow("c", "a", 30)]
THREE_SPECS = {s.id: s for s in (
    Spec("a1", "a", 25), Spec("a2", "a", 40), Spec("a3", "a", 5),
    Spec("b1", "b", 8), Spec("b2", "b", 8), Spec("b3", "b", 8), Spec("b-big", "b", 12),
    Spec("c1", "c", 50), Spec("d1", "d", 3),
    Spec("e1", "e", 1), Spec("e2", "e", 1), Spec("e3", "e", 1))}


def plan(target, online, specs, rules, **kw):
    kw.setdefault("total_gb", None)
    kw.setdefault("disk", ROOMY)
    return plan_load(specs[target], online, specs, rules, **kw)


def disk(free, weights=None, need=0, reserve=0, inflight=0):
    return DiskView(free_bytes=free, weights_bytes=weights or {}, need_bytes=need,
                    reserve_bytes=reserve, inflight_bytes=inflight)


def on(spec_id, vram=None, specs=None):
    s = (specs or {**LLM_VIDEO_SPECS, **THREE_SPECS})[spec_id]
    return Online(s.id, s.group, vram)


# (name, rules, specs, target, online, swap, ok, unload, reason fragments)
RULE_CASES = [
    ("llm onto an empty gpu", LLM_VIDEO, LLM_VIDEO_SPECS, "inkling", [], True, True, [], ()),
    ("already online is a no-op", LLM_VIDEO, LLM_VIDEO_SPECS, "inkling", [on("inkling", 17)], True, True, [], ()),
    ("second llm swaps the first out", LLM_VIDEO, LLM_VIDEO_SPECS, "qwen", [on("inkling", 17)], True, True,
     ["inkling"], ()),
    ("second llm refused when the caller forbids swap", LLM_VIDEO, LLM_VIDEO_SPECS, "qwen", [on("inkling", 17)],
     False, False, [], ("inkling", "最多 1 个")),
    ("video next to a small llm", LLM_VIDEO, LLM_VIDEO_SPECS, "pika", [on("inkling", 17.8)], True, True, [], ()),
    ("video with no llm", LLM_VIDEO, LLM_VIDEO_SPECS, "pika", [], True, True, [], ()),
    ("video refused next to a big llm", LLM_VIDEO, LLM_VIDEO_SPECS, "pika", [on("qwen", 49)], True, False, [],
     ("qwen", "20GB", "实测")),
    ("measured vram beats the estimate", LLM_VIDEO, LLM_VIDEO_SPECS, "pika", [on("inkling", 21)], True, False, [],
     ("inkling",)),
    ("estimate is used while unmeasured (big)", LLM_VIDEO, LLM_VIDEO_SPECS, "pika", [on("qwen", None)], True, False,
     [], ("qwen", "预计")),
    ("estimate is used while unmeasured (small)", LLM_VIDEO, LLM_VIDEO_SPECS, "pika", [on("inkling", None)], True,
     True, [], ()),
    ("big llm refused while video is online", LLM_VIDEO, LLM_VIDEO_SPECS, "qwen", [on("pika", 68)], True, False, [],
     ("qwen", "pika")),
    ("swapping to a big llm refused while video is online", LLM_VIDEO, LLM_VIDEO_SPECS, "qwen",
     [on("inkling", 17), on("pika", 68)], True, False, [], ("qwen",)),
    ("small llm allowed while video is online", LLM_VIDEO, LLM_VIDEO_SPECS, "inkling", [on("pika", 68)], True, True,
     [], ()),
    ("group without rules is unconstrained", THREE, THREE_SPECS, "d1", [on("a1", 25), on("b1", 8)], True, True, [], ()),
    ("refuse-conflict group is full", THREE, THREE_SPECS, "b3", [on("b1", 8), on("b2", 8)], True, False, [],
     ("同组最多 2 个", "规则不允许换出")),
    ("refuse-conflict group has room", THREE, THREE_SPECS, "b2", [on("b1", 8)], True, True, [], ()),
    ("swap group replaces its member", THREE, THREE_SPECS, "a2", [on("a1", 25)], True, True, ["a1"], ()),
    ("swap with limit 2 unloads only the oldest member", THREE, THREE_SPECS, "e3", [on("e1", 1), on("e2", 1)], True,
     True, ["e1"], ()),
    ("gated group refused next to an over-limit b", THREE, THREE_SPECS, "c1", [on("b-big", 12)], True, False, [],
     ("b-big", "10GB")),
    ("gated group refused next to an over-limit a", THREE, THREE_SPECS, "c1", [on("a2", 40)], True, False, [],
     ("a2", "30GB")),
    ("gated group allowed when both are under their limits", THREE, THREE_SPECS, "c1",
     [on("a1", 25), on("b1", 8)], True, True, [], ()),
    ("a load that would break the gate is refused", THREE, THREE_SPECS, "a2", [on("c1", 50)], True, False, [],
     ("a2",)),
    ("a swap that fits the gate is allowed", THREE, THREE_SPECS, "a3", [on("a1", 25), on("c1", 50)], True, True,
     ["a1"], ()),
    ("a swap never unloads the gated group to make room", THREE, THREE_SPECS, "a2", [on("a1", 25), on("c1", 50)],
     True, False, [], ("a2",)),
    ("no rules at all", [], THREE_SPECS, "a1", [on("a2", 40)], True, True, [], ()),
]


@pytest.mark.parametrize("name,rules,specs,target,online,swap,ok,unload,reason", RULE_CASES,
                         ids=[c[0] for c in RULE_CASES])
def test_rule_sets(name, rules, specs, target, online, swap, ok, unload, reason):
    d = plan_rules(specs[target], online, specs, rules, swap)
    assert d.ok is ok and d.unload == unload
    for fragment in reason:
        assert fragment in d.reason
    p = plan(target, online, specs, rules, swap=swap)
    assert p.ok is ok and p.unload == unload
    assert not p.evict


VIOLATION_CASES = [
    ("clean", LLM_VIDEO, [on("inkling", 17), on("pika", 68)], 0),
    ("llm grew past the video limit", LLM_VIDEO, [on("inkling", 22), on("pika", 68)], 1),
    ("two llms online", LLM_VIDEO, [on("inkling", 17), on("qwen", 40)], 1),
    ("unmeasured falls back to the estimate", LLM_VIDEO, [on("qwen", None), on("pika", 68)], 1),
    ("three groups, both gates broken", THREE, [on("a2", 40), on("b-big", 12), on("c1", 50)], 2),
    ("three groups, over the b limit", THREE, [on("b1", 8), on("b2", 8), on("b3", 8)], 1),
    ("gate only applies while the gated group is online", THREE, [on("a2", 40), on("b-big", 12)], 0),
]


@pytest.mark.parametrize("name,rules,online,count", VIOLATION_CASES, ids=[c[0] for c in VIOLATION_CASES])
def test_violations_on_the_live_set(name, rules, online, count):
    assert len(violations(online, {**LLM_VIDEO_SPECS, **THREE_SPECS}, rules)) == count


# ---- total VRAM ---------------------------------------------------------------------------------

def test_total_vram_g4_allows_inkling_plus_pika():
    assert plan("pika", [on("inkling", 17.8)], LLM_VIDEO_SPECS, LLM_VIDEO, total_gb=96, margin_gb=4).ok


def test_total_vram_a100_rejects_inkling_plus_pika():
    d = plan("pika", [on("inkling", 17.8)], LLM_VIDEO_SPECS, LLM_VIDEO, total_gb=80, margin_gb=4)
    assert not d.ok and "显存不够" in d.reason


def test_total_vram_a100_allows_pika_alone_and_counts_swapped_out_llm_as_free():
    assert plan("pika", [], LLM_VIDEO_SPECS, LLM_VIDEO, total_gb=80, margin_gb=4).ok
    d = plan("qwen", [on("inkling", 17.8)], LLM_VIDEO_SPECS, LLM_VIDEO, total_gb=80, margin_gb=4)
    assert d.ok and d.unload == ["inkling"]


def test_total_vram_rejects_model_bigger_than_gpu():
    assert not plan("qwen", [], LLM_VIDEO_SPECS, LLM_VIDEO, total_gb=40, margin_gb=4).ok


def test_total_vram_applies_with_no_rules_at_all():
    d = plan("a2", [on("a1", 25)], THREE_SPECS, [], total_gb=60, margin_gb=4)
    assert not d.ok and "显存不够" in d.reason


# ---- disk ---------------------------------------------------------------------------------------

def test_disk_full_refuses_and_touches_nothing():
    d = plan("qwen", [on("inkling", 17)], LLM_VIDEO_SPECS, LLM_VIDEO,
             disk=disk(free=0, weights={"inkling": 17 * GB}, need=77 * GB))
    assert not d.ok and "磁盘不够" in d.reason
    assert d.unload == [] and d.evict == []


def test_refused_load_does_not_evict_cached_weights():
    d = plan("pika", [], LLM_VIDEO_SPECS, LLM_VIDEO, disk=disk(free=0, weights={"qwen": 77 * GB}, need=148 * GB))
    assert not d.ok and d.evict == []


def test_swap_counts_unloaded_weights_as_freeable_only_when_evicted():
    online = [on("inkling", 17)]
    tight = disk(free=0, weights={"inkling": 17 * GB}, need=10 * GB)
    d = plan("qwen", online, LLM_VIDEO_SPECS, LLM_VIDEO, disk=tight)
    assert d.ok and d.unload == ["inkling"] and d.evict == ["inkling"]

    roomy = disk(free=50 * GB, weights={"inkling": 17 * GB}, need=10 * GB)
    d = plan("qwen", online, LLM_VIDEO_SPECS, LLM_VIDEO, disk=roomy)
    assert d.ok and d.unload == ["inkling"] and d.evict == []


def test_evicts_largest_offline_first_and_only_as_needed():
    weights = {"qwen": 77 * GB, "inkling": 17 * GB}
    d = plan("pika", [], LLM_VIDEO_SPECS, LLM_VIDEO, disk=disk(free=100 * GB, weights=weights, need=148 * GB))
    assert d.ok and d.evict == ["qwen"]
    d = plan("pika", [], LLM_VIDEO_SPECS, LLM_VIDEO, disk=disk(free=60 * GB, weights=weights, need=148 * GB))
    assert d.ok and d.evict == ["qwen", "inkling"]


def test_staying_online_models_are_never_evicted():
    d = plan("pika", [on("inkling", 17)], LLM_VIDEO_SPECS, LLM_VIDEO,
             disk=disk(free=0, weights={"inkling": 17 * GB}, need=10 * GB))
    assert not d.ok and d.evict == []


def test_inflight_downloads_and_reserve_are_subtracted_from_free_space():
    fits = disk(free=100 * GB, need=90 * GB, reserve=5 * GB)
    assert plan("pika", [], LLM_VIDEO_SPECS, LLM_VIDEO, disk=fits).ok
    other_download = disk(free=100 * GB, need=90 * GB, reserve=5 * GB, inflight=20 * GB)
    d = plan("pika", [], LLM_VIDEO_SPECS, LLM_VIDEO, disk=other_download)
    assert not d.ok and "磁盘不够" in d.reason


def test_target_own_weights_are_not_evicted():
    d = plan("qwen", [], LLM_VIDEO_SPECS, LLM_VIDEO, disk=disk(free=0, weights={"qwen": 10 * GB}, need=5 * GB))
    assert not d.ok and d.evict == []
