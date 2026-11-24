"""MOSS-Image-SFT setups compared on fixed prompts and seeds: steps, distilled LoRA, text-encoder precision.

Run on the runtime, one setup per process (the LoRA wraps the transformer in place):
  python moss_bench.py --tag base50
  python moss_bench.py --tag akh4 --lora /content/moss/akh/lora.safetensors --steps 4
Prints one JSON line per image; the first image of a process includes lazy weight loading.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from torch import nn

PROMPTS = {
    "menu": "A chalkboard cafe menu on a brick wall. Title 'TODAY'S SPECIAL', lines 'Flat White $4.50', "
            "'Matcha Latte $5.25', 'Croissant $3.80'. Warm morning light, shallow depth of field.",
    "shanghai": "A rainy night street in Shanghai, a glowing red neon sign reading '深夜食堂' above a wooden door, "
                "reflections on wet pavement, a cyclist with an umbrella passing by.",
    "infographic": "A clean flat infographic poster titled 'Remote Work 2026' with a pie chart split 62% blue "
                   "'Hybrid' and 38% orange 'Office', and three icons with captions 'Focus', 'Commute', 'Teams'.",
    "portrait": "A close-up photograph of an elderly fisherman with a weathered face and a grey beard, wearing a "
                "yellow raincoat, harbour boats blurred behind him, overcast daylight, 85mm lens.",
}
SEED = 1234
LORA_TARGETS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


class LoRALinear(nn.Module):
    """fp32 LoRA kept unmerged on the bf16 base; merging into bf16 loses the distillation delta."""

    def __init__(self, base: nn.Linear, rank: int, scale: float) -> None:
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        self.scale = scale

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        return out + (self.up(self.down(x.to(self.down.weight.dtype))) * self.scale).to(out.dtype)


def apply_lora(transformer, path: str, strength: float) -> None:
    sd = load_file(path)
    for i, layer in enumerate(transformer.layers):
        for name in LORA_TARGETS:
            key = f"diffusion_model.layers.{i}.proj.{name}"
            down, up = sd[f"{key}.lora_down.weight"], sd[f"{key}.lora_up.weight"]
            rank = down.shape[0]
            m = LoRALinear(layer.proj[name], rank, float(sd[f"{key}.alpha"]) / rank * strength)
            m.down.weight.data.copy_(down.float())
            m.up.weight.data.copy_(up.float())
            layer.proj[name] = m.to(layer.proj[name].weight.device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default="/content/moss/sft")
    p.add_argument("--lora")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, help="default: 3.0 without a LoRA, 1.0 with one")
    p.add_argument("--fp8-text", action="store_true", help="torchao FP8 text encoder (needs sm_89+)")
    p.add_argument("--resident", action="store_true", help="keep every component on the GPU (no offload)")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--out", default="/content/moss/out")
    a = p.parse_args()
    cfg = a.cfg if a.cfg is not None else (1.0 if a.lora else 3.0)
    Path(a.out).mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    pipe = DiffusionPipeline.from_pretrained(a.model, custom_pipeline="pipeline_moss_image", trust_remote_code=True,
                                             torch_dtype={"default": torch.bfloat16, "vae": torch.float32})
    if a.lora:
        apply_lora(pipe.transformer, a.lora, a.strength)
    if a.fp8_text:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_
        pipe.text_encoder.to("cuda")
        quantize_(pipe.text_encoder, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    if a.resident:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    print(json.dumps({"tag": a.tag, "load_s": round(time.monotonic() - t0, 1), "steps": a.steps, "cfg": cfg,
                      "lora": a.lora, "strength": a.strength, "fp8_text": a.fp8_text, "resident": a.resident}),
          flush=True)

    for name, prompt in PROMPTS.items():
        torch.cuda.reset_peak_memory_stats()
        t = time.monotonic()
        image = pipe(prompt, height=a.size, width=a.size, num_inference_steps=a.steps, guidance_scale=cfg,
                     generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
        torch.cuda.synchronize()
        path = f"{a.out}/{a.tag}_{name}.png"
        image.save(path)
        print(json.dumps({"tag": a.tag, "prompt": name, "gen_s": round(time.monotonic() - t, 1),
                          "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1), "file": path}), flush=True)


if __name__ == "__main__":
    main()
