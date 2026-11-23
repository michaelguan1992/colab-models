from controller.policy import DiskView, Online, Spec, plan_load, violations

SPECS = {
    "qwen": Spec("qwen", "llm", 50),
    "inkling": Spec("inkling", "llm", 18),
    "pika": Spec("pika", "video", 69),
}


GB = 10**9
ROOMY = DiskView(free_bytes=10**15, weights_bytes={}, need_bytes=0, reserve_bytes=0, inflight_bytes=0)


def plan(target, online, specs, **kw):
    kw.setdefault("total_gb", None)
    kw.setdefault("disk", ROOMY)
    return plan_load(target, online, specs, **kw)


def disk(free, weights=None, need=0, reserve=0, inflight=0):
    return DiskView(free_bytes=free, weights_bytes=weights or {}, need_bytes=need,
                    reserve_bytes=reserve, inflight_bytes=inflight)


def test_load_llm_on_empty_gpu():
    d = plan(SPECS["inkling"], [], SPECS)
    assert d.ok and d.unload == []


def test_already_online_is_noop():
    d = plan(SPECS["inkling"], [Online("inkling", "llm", 17)], SPECS)
    assert d.ok and d.unload == []


def test_second_llm_swaps_out_first():
    d = plan(SPECS["qwen"], [Online("inkling", "llm", 17)], SPECS)
    assert d.ok and d.unload == ["inkling"]


def test_second_llm_rejected_without_swap():
    d = plan(SPECS["qwen"], [Online("inkling", "llm", 17)], SPECS, swap=False)
    assert not d.ok


def test_video_allowed_with_small_llm():
    d = plan(SPECS["pika"], [Online("inkling", "llm", 17.8)], SPECS)
    assert d.ok


def test_video_allowed_with_no_llm():
    assert plan(SPECS["pika"], [], SPECS).ok


def test_video_rejected_with_big_llm():
    d = plan(SPECS["pika"], [Online("qwen", "llm", 49)], SPECS)
    assert not d.ok and "qwen" in d.reason


def test_video_uses_measured_not_estimate():
    # Inkling estimated 18GB but measured 21GB (big KV cache) -> reject.
    d = plan(SPECS["pika"], [Online("inkling", "llm", 21)], SPECS)
    assert not d.ok


def test_video_uses_estimate_when_unmeasured():
    assert not plan(SPECS["pika"], [Online("qwen", "llm", None)], SPECS).ok
    assert plan(SPECS["pika"], [Online("inkling", "llm", None)], SPECS).ok


def test_big_llm_rejected_while_video_online():
    d = plan(SPECS["qwen"], [Online("pika", "video", 68)], SPECS)
    assert not d.ok


def test_swap_to_big_llm_rejected_while_video_online():
    online = [Online("inkling", "llm", 17), Online("pika", "video", 68)]
    d = plan(SPECS["qwen"], online, SPECS)
    assert not d.ok


def test_small_llm_allowed_while_video_online():
    d = plan(SPECS["inkling"], [Online("pika", "video", 68)], SPECS)
    assert d.ok


def test_violations():
    assert violations([Online("inkling", "llm", 17), Online("pika", "video", 68)], SPECS) == []
    assert violations([Online("inkling", "llm", 22), Online("pika", "video", 68)], SPECS)
    assert violations([Online("inkling", "llm", 17), Online("qwen", "llm", 40)], SPECS)


def test_total_vram_g4_allows_inkling_plus_pika():
    d = plan(SPECS["pika"], [Online("inkling", "llm", 17.8)], SPECS, total_gb=96, margin_gb=4)
    assert d.ok


def test_total_vram_a100_rejects_inkling_plus_pika():
    d = plan(SPECS["pika"], [Online("inkling", "llm", 17.8)], SPECS, total_gb=80, margin_gb=4)
    assert not d.ok and "显存不够" in d.reason


def test_total_vram_a100_allows_pika_alone_and_counts_swapped_out_llm_as_free():
    assert plan(SPECS["pika"], [], SPECS, total_gb=80, margin_gb=4).ok
    d = plan(SPECS["qwen"], [Online("inkling", "llm", 17.8)], SPECS, total_gb=80, margin_gb=4)
    assert d.ok and d.unload == ["inkling"]


def test_total_vram_rejects_model_bigger_than_gpu():
    assert not plan(SPECS["qwen"], [], SPECS, total_gb=40, margin_gb=4).ok


def test_disk_full_refuses_and_touches_nothing():
    d = plan(SPECS["qwen"], [Online("inkling", "llm", 17)], SPECS,
             disk=disk(free=0, weights={"inkling": 17 * GB}, need=77 * GB))
    assert not d.ok and "磁盘不够" in d.reason
    assert d.unload == [] and d.evict == []


def test_refused_load_does_not_evict_cached_weights():
    d = plan(SPECS["pika"], [], SPECS, disk=disk(free=0, weights={"qwen": 77 * GB}, need=148 * GB))
    assert not d.ok and d.evict == []


def test_swap_counts_unloaded_weights_as_freeable_only_when_evicted():
    online = [Online("inkling", "llm", 17)]
    tight = disk(free=0, weights={"inkling": 17 * GB}, need=10 * GB)
    d = plan(SPECS["qwen"], online, SPECS, disk=tight)
    assert d.ok and d.unload == ["inkling"] and d.evict == ["inkling"]

    roomy = disk(free=50 * GB, weights={"inkling": 17 * GB}, need=10 * GB)
    d = plan(SPECS["qwen"], online, SPECS, disk=roomy)
    assert d.ok and d.unload == ["inkling"] and d.evict == []


def test_evicts_largest_offline_first_and_only_as_needed():
    weights = {"qwen": 77 * GB, "inkling": 17 * GB}
    d = plan(SPECS["pika"], [], SPECS, disk=disk(free=100 * GB, weights=weights, need=148 * GB))
    assert d.ok and d.evict == ["qwen"]
    d = plan(SPECS["pika"], [], SPECS, disk=disk(free=60 * GB, weights=weights, need=148 * GB))
    assert d.ok and d.evict == ["qwen", "inkling"]


def test_staying_online_models_are_never_evicted():
    d = plan(SPECS["pika"], [Online("inkling", "llm", 17)], SPECS,
             disk=disk(free=0, weights={"inkling": 17 * GB}, need=10 * GB))
    assert not d.ok and d.evict == []


def test_inflight_downloads_and_reserve_are_subtracted_from_free_space():
    fits = disk(free=100 * GB, need=90 * GB, reserve=5 * GB)
    assert plan(SPECS["pika"], [], SPECS, disk=fits).ok
    other_download = disk(free=100 * GB, need=90 * GB, reserve=5 * GB, inflight=20 * GB)
    d = plan(SPECS["pika"], [], SPECS, disk=other_download)
    assert not d.ok and "磁盘不够" in d.reason


def test_target_own_weights_are_not_evicted():
    d = plan(SPECS["qwen"], [], SPECS, disk=disk(free=0, weights={"qwen": 10 * GB}, need=5 * GB))
    assert not d.ok and d.evict == []
