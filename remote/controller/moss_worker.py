"""MOSS-Image image worker, same /health + /generate + /jobs contract as the Pika worker.

Runs MOSS-Image-SFT with a step-distilled LoRA (lodestones' 8-step DMD by default, CFG 1). The LoRA stays
unmerged in fp32 beside the bf16 weights: merged into bf16 it loses most of the distillation delta.
The comparison behind these defaults is docs/spike.md §12 (remote/bench/moss_bench.py).
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
import uuid
from pathlib import Path
from queue import Queue

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LORA_TARGETS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


class GenReq(BaseModel):
    prompt: str
    height: int | None = Field(None, ge=256, le=2048, multiple_of=16)  # None = the worker's --size
    width: int | None = Field(None, ge=256, le=2048, multiple_of=16)
    seed: int | None = None


def lora_linear(base, rank: int, scale: float):
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.down = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
            self.up = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)

        @property
        def weight(self):  # the pipeline reads proj weights for dtype and device
            return self.base.weight

        def forward(self, x):
            out = self.base(x)
            return out + (self.up(self.down(x.to(torch.float32))) * scale).to(out.dtype)

    return LoRALinear()


def apply_lora(transformer, path: str, strength: float) -> None:
    """Wrap each targeted projection with the kohya-format LoRA in ``path``."""
    from safetensors.torch import load_file

    sd = load_file(path)
    for i, layer in enumerate(transformer.layers):
        for name in LORA_TARGETS:
            key = f"diffusion_model.layers.{i}.proj.{name}"
            down, up = sd[f"{key}.lora_down.weight"], sd[f"{key}.lora_up.weight"]
            rank = down.shape[0]
            m = lora_linear(layer.proj[name], rank, float(sd[f"{key}.alpha"]) / rank * strength)
            m.down.weight.data.copy_(down.float())
            m.up.weight.data.copy_(up.float())
            layer.proj[name] = m.to(layer.proj[name].weight.device)


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pipe = None
        self.ready = False
        self.error: str | None = None
        self.jobs: dict[str, dict] = {}
        self.queue: Queue[tuple[str, GenReq]] = Queue()
        Path(args.outputs_dir).mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        try:
            import torch
            from diffusers import DiffusionPipeline

            a = self.args
            pipe = DiffusionPipeline.from_pretrained(
                a.model_dir, custom_pipeline="pipeline_moss_image", trust_remote_code=True,
                torch_dtype={"default": torch.bfloat16, "vae": torch.float32})
            if a.lora_repo:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(a.lora_repo, a.lora_file, revision=a.lora_revision,
                                       local_dir=str(Path(a.model_dir) / "lora"))
                apply_lora(pipe.transformer, path, a.lora_strength)
            if a.cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            self.pipe = pipe
            self.ready = True
        except Exception:
            self.error = traceback.format_exc(limit=5)

    def run_forever(self) -> None:
        while True:
            job_id, req = self.queue.get()
            job = self.jobs[job_id]
            job.update(state="running", started_at=time.time())
            try:
                job["file"] = self._generate(job_id, req)
                job["state"] = "done"
            except Exception:
                job.update(state="error", error=traceback.format_exc(limit=5))
            job["finished_at"] = time.time()

    def _generate(self, job_id: str, req: GenReq) -> str:
        import torch

        a = self.args
        generator = torch.Generator("cpu").manual_seed(req.seed) if req.seed is not None else None
        image = self.pipe(req.prompt, height=req.height or a.size, width=req.width or a.size,
                          num_inference_steps=a.steps, guidance_scale=a.cfg, generator=generator).images[0]
        name = f"{job_id}.png"
        image.save(Path(a.outputs_dir) / name)
        return name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--lora-repo", help="HF repo holding a step-distilled LoRA; omit for the plain model")
    p.add_argument("--lora-file", default="lora.safetensors")
    p.add_argument("--lora-revision")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0, help="1 = one forward pass per step (distilled LoRAs)")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--outputs-dir", default="/content/outputs")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--cpu-offload", action="store_true")
    args = p.parse_args()

    w = Worker(args)
    threading.Thread(target=w.load, daemon=True).start()
    threading.Thread(target=w.run_forever, daemon=True).start()
    app = FastAPI(title="moss worker")

    @app.get("/health")
    def health():
        depth = sum(1 for j in list(w.jobs.values()) if j["state"] in ("queued", "running"))
        return {"ready": w.ready, "error": w.error, "queue_depth": depth}

    @app.post("/generate")
    def generate(req: GenReq):
        if not w.ready:
            raise HTTPException(503, w.error or "loading")
        job_id = uuid.uuid4().hex[:12]
        w.jobs[job_id] = {"id": job_id, "state": "queued", "prompt": req.prompt, "queued_at": time.time()}
        w.queue.put((job_id, req))
        return w.jobs[job_id]

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in w.jobs:
            raise HTTPException(404, "no such job")
        return w.jobs[job_id]

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
