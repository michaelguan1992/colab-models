"""Controller HTTP API (listens on 127.0.0.1 only; reached from the Mac via ssh -L)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from .manager import Manager, RequestHandle

log = logging.getLogger("lm.controller")

HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding",
               "te", "trailer", "upgrade", "proxy-authorization", "authorization", "x-api-key"}


CHAT = "chat"
VIDEO = "video"


class LoadReq(BaseModel):
    id: str
    swap: bool = True
    route: str | None = None


class UnloadReq(BaseModel):
    id: str | None = None  # None = unload everything


def _openai_error(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": "lm_controller"}}, status_code=status)


async def tracked_stream(handle: RequestHandle, stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Keeps the request counted as in-flight until the response stream is fully consumed or dropped."""
    try:
        async for chunk in stream:
            yield chunk
    finally:
        handle.finish()


def create_app(manager: Manager, idle_poll_s: float = 30, autoload: str | None = None) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10))

        async def idle_loop():
            while True:
                await asyncio.sleep(idle_poll_s)
                with contextlib.suppress(Exception):
                    await manager.health_check()
                with contextlib.suppress(Exception):
                    await manager.idle_check()

        await manager.startup_cleanup()
        task = asyncio.create_task(idle_loop())
        if autoload:
            try:
                plan = await manager.load(autoload)
                if not plan.ok:
                    log.error("启动加载 %s 被拒绝: %s", autoload, plan.reason)
            except KeyError as e:
                log.error("启动加载失败: %s", e.args[0] if e.args else e)
            except Exception:
                log.exception("启动加载 %s 失败", autoload)
        yield
        task.cancel()
        await manager.unload_all()
        await app.state.client.aclose()

    app = FastAPI(title="lm controller", lifespan=lifespan)

    def unavailable(capability: str) -> str:
        loading = manager.loading(capability)
        if loading:
            return f"{capability} 模型 {loading[0].id} 正在加载中"
        ids = manager.registry.ids_serving(capability)
        return f"当前没有在线的 {capability} 模型，请先 {' 或 '.join('/load ' + i for i in ids)}"

    # ---- admin -----------------------------------------------------------
    @app.get("/admin/status")
    async def status():
        return manager.status()

    @app.post("/admin/load")
    async def load(req: LoadReq):
        try:
            d = await manager.load(req.id, swap=req.swap, route=req.route)
        except KeyError as e:
            raise HTTPException(404, e.args[0] if e.args else str(e))
        if not d.ok:
            return JSONResponse({"ok": False, "reason": d.reason}, status_code=409)
        return {"ok": True, "unloaded": d.unload, "evicted": d.evict, "reason": d.reason}

    @app.post("/admin/unload")
    async def unload(req: UnloadReq):
        if req.id is None:
            return {"ok": True, "unloaded": await manager.unload_all()}
        return {"ok": await manager.unload(req.id), "unloaded": [req.id]}

    # ---- OpenAI / Anthropic compatible proxy -----------------------------
    @app.get("/v1/models")
    async def models():
        # No touch(): dsh polls the model list, which must not keep a billed runtime "busy".
        chat = manager.serving(CHAT)
        others = [b.id for b in manager.backends.values() if b.state == "ready" and CHAT not in b.model.serves]
        suffix = f" [+{', '.join(others)}]" if others else ""
        data = [{"id": b.id, "object": "model", "owned_by": "colab", "name": b.model.display + suffix,
                 **({"context_length": b.route.caps["context"]} if "context" in b.route.caps else {})}
                for b in chat]
        return {"object": "list", "data": data}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        inference = request.method != "GET"  # polling GETs must not keep a billed runtime busy
        if inference:
            manager.touch()
        ready = manager.serving(CHAT)
        if not ready:
            return _openai_error(503, unavailable(CHAT))
        llm = ready[0]
        body = await request.body()
        if body and llm.upstream_model and request.headers.get("content-type", "").startswith("application/json"):
            with contextlib.suppress(ValueError):
                payload = json.loads(body)
                if isinstance(payload, dict) and "model" in payload:
                    payload["model"] = llm.upstream_model
                    body = json.dumps(payload).encode()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        client: httpx.AsyncClient = request.app.state.client
        upstream = client.build_request(request.method, f"{llm.base_url}/v1/{path}",
                                        params=request.query_params, headers=headers, content=body)
        handle = manager.begin_request(activity=inference)
        try:
            r = await client.send(upstream, stream=True)
        except httpx.HTTPError as e:
            handle.finish()
            return _openai_error(502, f"上游 {llm.id} 无响应: {e}")
        except BaseException:
            handle.finish()
            raise
        out_headers = {k: v for k, v in r.headers.items() if k.lower() not in HOP_HEADERS | {"content-encoding"}}

        async def close() -> None:
            await r.aclose()
            handle.finish()

        return StreamingResponse(tracked_stream(handle, r.aiter_raw()), status_code=r.status_code,
                                 headers=out_headers, background=BackgroundTask(close))

    # ---- video -----------------------------------------------------------
    async def _video_forward(request: Request, method: str, path: str, body: bytes | None = None):
        manager.touch()
        ready = manager.serving(VIDEO)
        if not ready:
            raise HTTPException(503, unavailable(VIDEO))
        v = ready[0]
        client: httpx.AsyncClient = request.app.state.client
        r = await client.request(method, f"{v.base_url}{path}", content=body,
                                 headers={"content-type": "application/json"})
        if method == "POST" and r.status_code < 300:
            v.queue_depth = max(v.queue_depth, 1)  # until the next health poll reports the real depth
        return JSONResponse(r.json(), status_code=r.status_code)

    @app.post("/video/generate")
    async def video_generate(request: Request):
        return await _video_forward(request, "POST", "/generate", await request.body())

    @app.get("/video/jobs/{job_id}")
    async def video_job(job_id: str, request: Request):
        return await _video_forward(request, "GET", f"/jobs/{job_id}")

    @app.get("/files/{name}")
    async def files(name: str):
        if not re.fullmatch(r"[\w.-]+", name):
            raise HTTPException(400, "bad name")
        path = manager.registry.outputs_dir / name
        if not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path)

    @app.get("/health")
    async def health():
        return {"ok": True, "service": "lm-controller", "version": os.environ.get("LM_VERSION", "dev")}

    return app
