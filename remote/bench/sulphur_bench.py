"""Time Sulphur-3 text-to-video in ComfyUI at different step counts (run on the Mac after `lm comfy sulphur-comfy`).

The graph is the official Pika text-to-video template written out in API form, with the model, LoRA,
sampler and sigmas swapped per case. Each case runs twice: the first run (seed --seed, the clip that is
saved) includes loading or re-patching the weights, the second (the next seed) is the warm time.

    python remote/bench/sulphur_bench.py --cases turbo4,base20 --out outputs/sulphur
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROMPT = ("A red fox trots through fresh snow at the edge of a pine forest, stops, looks at the camera and "
          "shakes the snow off its fur; soft morning light, shallow depth of field; snow crunching, light wind.")
TURBO4 = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
TURBO8 = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
PIKA = "pika_h3_flex_fl2va_v1_bf16.safetensors"
PIKA_TURBO4 = "pika_h3_flex_turbo_4step_fl2va.safetensors"
PIKA_TURBO4_LITE = "pika_h3_flex_turbo_4step_fl2va_lite_r32.safetensors"
PIKA_TURBO8_HD = "pika_h3_flex_turbo_8step_hd_fl2va.safetensors"
PIKA_SIGMAS = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"   # the template's 6-step table

# The route downloads only the Pika Turbo 4-step LoRA: the other LoRA cases need theirs put back in loras/.


def shifted(steps: int, shift: float) -> str:
    """The simple scheduler's sigmas for a flow shift, as a ManualSigmas string (the model's own shift is 12)."""
    ts = [1 - i / steps for i in range(steps)]
    return ", ".join(f"{shift * t / (1 + (shift - 1) * t):.6f}" for t in ts) + ", 0.0"


# name -> lora, sampler, sigmas (a ManualSigmas string) or (scheduler, steps), cfg (None = no negative pass)
CASES = {
    "pika6": dict(lora=PIKA, sampler="lcm", sigmas=PIKA_SIGMAS, cfg=None),
    "turbo4": dict(lora=TURBO4, sampler="euler", schedule=("simple", 4), cfg=None),
    "turbo8": dict(lora=TURBO8, sampler="euler", schedule=("simple", 8), cfg=None),
    "pikaturbo4": dict(lora=PIKA_TURBO4, sampler="euler", schedule=("simple", 4), cfg=None),
    # Pika's README asks for shift 6 with FL2VA; _s12 repeats the model's default through ManualSigmas
    "pt4_s6": dict(lora=PIKA_TURBO4, sampler="euler", sigmas=shifted(4, 6), cfg=None),
    "pt4_s12": dict(lora=PIKA_TURBO4, sampler="euler", sigmas=shifted(4, 12), cfg=None),
    "lite4_s6": dict(lora=PIKA_TURBO4_LITE, sampler="euler", sigmas=shifted(4, 6), cfg=None),
    "lite4_s12": dict(lora=PIKA_TURBO4_LITE, sampler="euler", sigmas=shifted(4, 12), cfg=None),
    "hd8_s6": dict(lora=PIKA_TURBO8_HD, sampler="euler", sigmas=shifted(8, 6), cfg=None),
    "hd8_s12": dict(lora=PIKA_TURBO8_HD, sampler="euler", sigmas=shifted(8, 12), cfg=None),
    "base20": dict(lora=None, sampler="euler", schedule=("beta", 20), cfg=None),
    "base20cfg": dict(lora=None, sampler="euler", schedule=("beta", 20), cfg=4.0),
}


def graph(case: dict, a) -> dict:
    frames = max(5, round(a.seconds * 24))
    frames += (5 - frames % 17) % 17
    model = ["unet", 0]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": a.unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vae", 0], "prompt": a.prompt,
            "width": a.width, "height": a.height, "length": frames}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": case["sampler"]}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
            "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "adecode": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "audio": ["adecode", 0], "fps": 24}},
        "save": {"class_type": "SaveVideo", "inputs": {
            "video": ["video", 0], "filename_prefix": f"video/sulphur_{a.tag}", "format": "auto", "codec": "auto"}},
    }
    if case["lora"]:
        g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": case["lora"], "strength_model": 1.0}}
        model = ["lora", 0]
    if "sigmas" in case:
        g["sigmas"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": case["sigmas"]}}
    else:
        scheduler, steps = case["schedule"]
        g["sigmas"] = {"class_type": "BasicScheduler", "inputs": {
            "model": model, "scheduler": scheduler, "steps": steps, "denoise": 1.0}}
    if case["cfg"] is None:
        g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}}
    else:
        g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}}
        g["guider"] = {"class_type": "CFGGuider", "inputs": {
            "model": model, "positive": ["cond", 0], "negative": ["neg", 0], "cfg": case["cfg"]}}
    return g


def call(base: str, path: str, body: dict | None = None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"null")


def run(base: str, g: dict, timeout: float) -> dict:
    try:
        pid = call(base, "/prompt", {"prompt": g})["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"ComfyUI 拒绝了工作流: {e.read().decode()[:2000]}")
    t0, peak = time.time(), 0.0
    while time.time() - t0 < timeout:
        dev = call(base, "/system_stats")["devices"][0]
        peak = max(peak, (dev["vram_total"] - dev["vram_free"]) / 1e9)
        h = call(base, f"/history/{pid}")
        if pid in h:
            status = h[pid]["status"]
            stamps = {m[0]: m[1].get("timestamp") for m in status.get("messages", [])}
            if status.get("status_str") != "success":
                raise SystemExit(f"运行失败: {json.dumps(status, ensure_ascii=False)[:2000]}")
            files = [f for out in h[pid]["outputs"].values() for v in out.values() if isinstance(v, list)
                     for f in v if isinstance(f, dict) and "filename" in f]
            return {"seconds": round((stamps["execution_success"] - stamps["execution_start"]) / 1000, 1),
                    "peak_vram_gb": round(peak, 1), "files": files}
        time.sleep(2)
    raise SystemExit("超时")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8190")
    p.add_argument("--cases", default="pikaturbo4,base20")
    p.add_argument("--unet", default="s3.safetensors")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--out", default="outputs/sulphur")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results, seed = {}, a.seed
    for name in a.cases.split(","):
        a.tag = name
        runs = []
        for i in range(a.repeat):   # a new seed each time: ComfyUI answers an unchanged graph from its cache
            a.seed = seed + i
            runs.append(run(a.base, graph(CASES[name], a), a.timeout))
        results[name] = {"runs_s": [r["seconds"] for r in runs], "peak_vram_gb": max(r["peak_vram_gb"] for r in runs)}
        print(name, json.dumps(results[name]), flush=True)
        for f in runs[0]["files"]:   # the clip made with --seed, so cases compare like for like
            q = urllib.parse.urlencode({k: f.get(k, "") for k in ("filename", "subfolder", "type")})
            with urllib.request.urlopen(f"{a.base}/view?{q}", timeout=300) as r:
                (out / f"{name}_{f['filename']}").write_bytes(r.read())
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
