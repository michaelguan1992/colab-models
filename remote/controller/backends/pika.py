from __future__ import annotations

import os
import sys

import httpx

from .base import Backend


class PikaBackend(Backend):
    """Pika-H3-Flex + Turbo LoRA, run by ``controller.pika_worker`` in its own process."""

    kind = "video"

    @property
    def port(self) -> int:
        return int(self.cfg.get("port", 8090))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def lora_dir(self):
        return self.registry.models_dir / f"{self.id}-lora"

    async def download(self) -> None:
        await self._hf_download(self.cfg["repo"], self.cfg.get("include"), self.local_dir)
        lora = self.cfg["lora"]
        await self._hf_download(lora["repo"], [lora["file"]], self.lora_dir)
        marker = self.registry.models_dir / ".pika_deps_ok"
        req = self.local_dir / "examples" / "requirements-diffusers.txt"
        if req.exists() and not marker.exists():
            await self._run_logged([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])
            marker.touch()

    async def _launch(self) -> None:
        lora = self.cfg["lora"]
        cmd = [
            sys.executable, "-m", "controller.pika_worker",
            "--model-dir", str(self.local_dir),
            "--lora-dir", str(self.lora_dir),
            "--lora-file", lora["file"],
            "--workflow", self.cfg.get("workflow", "fl2va"),
            "--steps", str(self.cfg.get("steps", 4)),
            "--guidance", str(self.cfg.get("guidance", 1.0)),
            "--outputs-dir", str(self.registry.outputs_dir),
            "--port", str(self.port),
        ]
        if self.cfg.get("cpu_offload"):
            cmd.append("--cpu-offload")
        await self._spawn(cmd, env={**os.environ})

    async def _ready(self, client: httpx.AsyncClient) -> bool:
        r = await client.get(f"{self.base_url}/health")
        if r.status_code != 200:
            return False
        body = r.json()
        if body.get("error"):
            raise RuntimeError(body["error"])
        return bool(body.get("ready"))
