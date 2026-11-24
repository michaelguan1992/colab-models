"""Pika setups compared on one prompt and seed: adapter, schedule, precision, placement.

Run in the diffusers venv on the runtime, one setup per process (placement can't be undone in-process):
  python pika_bench.py --tag official6 --adapter official
Prints one JSON line per run; the first run of a process includes lazy weight loading.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils.export_utils import encode_video

PROMPT = ("A golden retriever runs along a beach at sunset, splashing through shallow waves; "
          "sound of surf, paws on wet sand, distant gulls.")
ADAPTERS = {
    "official": ("/content/models/pika", "fl2va/adapter.safetensors"),
    "turbo": ("/content/models/pika-lora", "pika_h3_flex_turbo_4step_fl2va.safetensors"),
}
TURBO4 = [1000, 750, 500, 250, 0]  # ComfyUI "simple" 4 steps, as the Turbo card prescribes


def build(a):
    manager = ComponentsManager()
    if not a.resident:
        manager.enable_auto_cpu_offload(device="cuda:0")
    pipe = ModularPipeline.from_pretrained("/content/models/pika", components_manager=manager)
    pipe.load_components(workflow="fl2va",
                         dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16})
    src, name = ADAPTERS[a.adapter]
    pipe.load_lora_weights(src, weight_name=name)
    if a.fuse:
        pipe.fuse_lora()
        pipe.unload_lora_weights()
    if a.fp8:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

        def bf16_linear(m, _fqn):  # PerRow FP8 rejects the model's non-bf16 Linear layers
            return isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.bfloat16

        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        quantize_(pipe.transformer, cfg, filter_fn=bf16_linear)
        if a.fp8_text:
            quantize_(pipe.text_encoder, cfg, filter_fn=bf16_linear)
    if a.steps4:
        pipe.register_to_config(denoising_step_list=TURBO4)
        pipe.scheduler.register_to_config(shift=a.video_shift)
    if a.resident:
        pipe.to("cuda")
    pipe.scheduler.set_timesteps(sigmas=[s / 1000 for s in pipe.config.denoising_step_list], device="cuda")
    print(json.dumps({"tag": a.tag, "steps": pipe.config.denoising_step_list,
                      "video_timesteps": [round(float(t), 1) for t in pipe.scheduler.timesteps]}), flush=True)
    return pipe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--adapter", choices=list(ADAPTERS), default="turbo")
    p.add_argument("--steps4", action="store_true")
    p.add_argument("--video-shift", type=float, default=6.0)
    p.add_argument("--fuse", action="store_true")
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--fp8-text", action="store_true")
    p.add_argument("--resident", action="store_true")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--out", default="/content/bench")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    pipe = build(a)
    print(json.dumps({"tag": a.tag, "load_s": round(time.monotonic() - t0, 1),
                      "vram_after_load_gib": round(torch.cuda.memory_allocated() / 2**30, 1)}), flush=True)
    for i in range(a.runs):
        torch.cuda.reset_peak_memory_stats()
        t = time.monotonic()
        out = pipe(prompt=PROMPT, height=480, width=832, num_frames=121,
                   generator=torch.Generator("cuda").manual_seed(42), output=["videos", "audio", "sampling_rate"])
        torch.cuda.synchronize()
        gen_s = time.monotonic() - t
        path = f"{a.out}/{a.tag}_{i}.mp4"
        encode_video(out["videos"][0], fps=24, output_path=path, audio=out["audio"][0],
                     audio_sample_rate=out["sampling_rate"])
        print(json.dumps({"tag": a.tag, "run": i, "gen_s": round(gen_s, 1),
                          "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1), "file": path}), flush=True)


if __name__ == "__main__":
    main()
