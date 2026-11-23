from __future__ import annotations

import asyncio
import os

import httpx

from .base import Backend


class StrataBackend(Backend):
    """One LLM served by Strata on a local port (OpenAI + Anthropic compatible)."""

    kind = "llm"

    @property
    def port(self) -> int:
        return int(self.cfg.get("port") or self.registry.strata.get("port", 8081))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _fmt(self, args: list[str]) -> list[str]:
        model_path = str(self.local_dir / self.cfg["model_path"])
        return [a.format(model_path=model_path, port=self.port) for a in args]

    async def _launch(self) -> None:
        s = self.registry.strata
        cmd = [s.get("bin", "strata"), *self._fmt(s["launch"]), *self.cfg.get("extra", [])]
        await self._spawn(cmd, env={**os.environ})

    async def _ready(self, client: httpx.AsyncClient) -> bool:
        r = await client.get(f"{self.base_url}/v1/models")
        if r.status_code != 200:
            return False
        data = r.json().get("data") or []
        if data:
            self.upstream_model = data[0].get("id")
        return True

    async def _kill(self) -> None:
        # Strata may daemonise its server; ask it to stop on our port first.
        s = self.registry.strata
        if s.get("stop"):
            try:
                p = await asyncio.create_subprocess_exec(
                    s.get("bin", "strata"), *self._fmt(s["stop"]),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await asyncio.wait_for(p.wait(), timeout=60)
            except (OSError, asyncio.TimeoutError):
                pass
        await super()._kill()
