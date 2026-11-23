from controller.policy import Online, Spec, plan_load, violations

SPECS = {
    "qwen": Spec("qwen", "llm", 50),
    "inkling": Spec("inkling", "llm", 18),
    "pika": Spec("pika", "video", 69),
}


def test_load_llm_on_empty_gpu():
    d = plan_load(SPECS["inkling"], [], SPECS)
    assert d.ok and d.unload == []


def test_already_online_is_noop():
    d = plan_load(SPECS["inkling"], [Online("inkling", "llm", 17)], SPECS)
    assert d.ok and d.unload == []


def test_second_llm_swaps_out_first():
    d = plan_load(SPECS["qwen"], [Online("inkling", "llm", 17)], SPECS)
    assert d.ok and d.unload == ["inkling"]


def test_second_llm_rejected_without_swap():
    d = plan_load(SPECS["qwen"], [Online("inkling", "llm", 17)], SPECS, swap=False)
    assert not d.ok


def test_video_allowed_with_small_llm():
    d = plan_load(SPECS["pika"], [Online("inkling", "llm", 17.8)], SPECS)
    assert d.ok


def test_video_allowed_with_no_llm():
    assert plan_load(SPECS["pika"], [], SPECS).ok


def test_video_rejected_with_big_llm():
    d = plan_load(SPECS["pika"], [Online("qwen", "llm", 49)], SPECS)
    assert not d.ok and "qwen" in d.reason


def test_video_uses_measured_not_estimate():
    # Inkling estimated 18GB but measured 21GB (big KV cache) -> reject.
    d = plan_load(SPECS["pika"], [Online("inkling", "llm", 21)], SPECS)
    assert not d.ok


def test_video_uses_estimate_when_unmeasured():
    assert not plan_load(SPECS["pika"], [Online("qwen", "llm", None)], SPECS).ok
    assert plan_load(SPECS["pika"], [Online("inkling", "llm", None)], SPECS).ok


def test_big_llm_rejected_while_video_online():
    d = plan_load(SPECS["qwen"], [Online("pika", "video", 68)], SPECS)
    assert not d.ok


def test_swap_to_big_llm_rejected_while_video_online():
    online = [Online("inkling", "llm", 17), Online("pika", "video", 68)]
    d = plan_load(SPECS["qwen"], online, SPECS)
    assert not d.ok


def test_small_llm_allowed_while_video_online():
    d = plan_load(SPECS["inkling"], [Online("pika", "video", 68)], SPECS)
    assert d.ok


def test_violations():
    assert violations([Online("inkling", "llm", 17), Online("pika", "video", 68)], SPECS) == []
    assert violations([Online("inkling", "llm", 22), Online("pika", "video", 68)], SPECS)
    assert violations([Online("inkling", "llm", 17), Online("qwen", "llm", 40)], SPECS)


def test_total_vram_g4_allows_inkling_plus_pika():
    d = plan_load(SPECS["pika"], [Online("inkling", "llm", 17.8)], SPECS, total_gb=96, margin_gb=4)
    assert d.ok


def test_total_vram_a100_rejects_inkling_plus_pika():
    d = plan_load(SPECS["pika"], [Online("inkling", "llm", 17.8)], SPECS, total_gb=80, margin_gb=4)
    assert not d.ok and "显存不够" in d.reason


def test_total_vram_a100_allows_pika_alone_and_counts_swapped_out_llm_as_free():
    assert plan_load(SPECS["pika"], [], SPECS, total_gb=80, margin_gb=4).ok
    d = plan_load(SPECS["qwen"], [Online("inkling", "llm", 17.8)], SPECS, total_gb=80, margin_gb=4)
    assert d.ok and d.unload == ["inkling"]


def test_total_vram_rejects_model_bigger_than_gpu():
    assert not plan_load(SPECS["qwen"], [], SPECS, total_gb=40, margin_gb=4).ok
