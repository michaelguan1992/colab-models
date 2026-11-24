"""Pika-H3-Flex video worker (separate process so its VRAM is attributable and killable).

Pipeline code follows the Pika-H3-Flex model card (diffusers ModularPipeline). The card's
own snippets are inconsistent about LoRA loading, so the exact calls are marked
``spike`` and must be confirmed on the G4 runtime (docs/spike.md).
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
import time
import traceback
import uuid
from pathlib import Path
from queue import Queue

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class GenReq(BaseModel):
    prompt: str
    image_b64: str | None = None      # optional first frame (PNG/JPEG, base64)
    height: int | None = None         # None = the worker's --height
    width: int | None = None          # None = the worker's --width
    seconds: float = Field(5, ge=5, le=15)  # Pika-H3-Flex only accepts 5-15 s clips
    steps: int | None = None
    seed: int | None = None


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
            from diffusers import ComponentsManager, ModularPipeline

            a = self.args
            manager = ComponentsManager()
            if a.cpu_offload:
                manager.enable_auto_cpu_offload(device="cuda:0")
            # The workflow picks which components load; fl2va's set serves t2va (no image) and i2va/fl2va alike.
            pipe = ModularPipeline.from_pretrained(a.model_dir, components_manager=manager)
            pipe.load_components(workflow=a.workflow,
                                 dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16})
            # spike: Turbo LoRA is pre-fused with the Flex adapter, so load it instead of fl2va/adapter.
            pipe.load_lora_weights(a.lora_dir, weight_name=a.lora_file)
            if not a.cpu_offload:
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
        from diffusers.utils.export_utils import encode_video

        kwargs = dict(prompt=req.prompt, height=req.height or self.args.height, width=req.width or self.args.width,
                      num_frames=round(req.seconds * self.args.fps) + 1,
                      num_inference_steps=req.steps or self.args.steps, guidance_scale=self.args.guidance,
                      output=["videos", "audio", "sampling_rate"])
        if req.image_b64:
            from PIL import Image
            kwargs["image"] = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
        if req.seed is not None:
            kwargs["generator"] = torch.Generator("cuda").manual_seed(req.seed)
        out = self.pipe(**kwargs)
        name = f"{job_id}.mp4"
        encode_video(out["videos"][0], fps=self.args.fps, output_path=str(Path(self.args.outputs_dir) / name),
                     audio=out["audio"][0], audio_sample_rate=out["sampling_rate"])
        return name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--lora-dir", required=True)
    p.add_argument("--lora-file", required=True)
    p.add_argument("--workflow", default="fl2va")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--outputs-dir", default="/content/outputs")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--cpu-offload", action="store_true")
    args = p.parse_args()

    w = Worker(args)
    threading.Thread(target=w.load, daemon=True).start()
    threading.Thread(target=w.run_forever, daemon=True).start()
    app = FastAPI(title="pika worker")

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
