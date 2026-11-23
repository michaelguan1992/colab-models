from __future__ import annotations

import httpx

from .base import Backend


class JobsBackend(Backend):
    """A long-job worker (video, image, ...) whose ready endpoint reports ``ready``, ``error`` and ``queue_depth``."""

    async def _ready(self, client: httpx.AsyncClient) -> bool:
        r = await client.get(f"{self.base_url}{self.engine.ready.path}")
        if r.status_code != 200:
            return False
        body = r.json()
        self.queue_depth = int(body.get("queue_depth") or 0)
        if body.get("error"):
            raise RuntimeError(body["error"])
        return bool(body.get("ready"))
