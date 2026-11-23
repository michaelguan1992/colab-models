"""Controller HTTP API (listens on 127.0.0.1 only; reached from the Mac via ssh -L)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from .manager import Manager

HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding",
               "te", "trailer", "upgrade", "proxy-authorization", "authorization", "x-api-key"}


class LoadReq(BaseModel):
    id: str
    swap: bool = True


class UnloadReq(BaseModel):
    id: str | None = None  # None = unload everything


def _openai_error(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": "lm_controller"}}, status_code=status)


def create_app(manager: Manager, idle_poll_s: float = 30, autoload: str | None = None) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10))

        async def idle_loop():
            while True:
                await asyncio.sleep(idle_poll_s)
                with contextlib.suppress(Exception):
                    await manager.idle_check()

        task = asyncio.create_task(idle_loop())
        if autoload:
            await manager.load(autoload)
        yield
        task.cancel()
        await manager.unload_all()
        await app.state.client.aclose()

    app = FastAPI(title="lm controller", lifespan=lifespan)

    # ---- admin -----------------------------------------------------------
    @app.get("/admin/status")
    async def status():
        return manager.status()

    @app.post("/admin/load")
    async def load(req: LoadReq):
        try:
            d = await manager.load(req.id, swap=req.swap)
        except KeyError as e:
            raise HTTPException(404, str(e))
        if not d.ok:
            return JSONResponse({"ok": False, "reason": d.reason}, status_code=409)
        return {"ok": True, "unloaded": d.unload, "reason": d.reason}

    @app.post("/admin/unload")
    async def unload(req: UnloadReq):
        if req.id is None:
            return {"ok": True, "unloaded": await manager.unload_all()}
        return {"ok": await manager.unload(req.id), "unloaded": [req.id]}

    # ---- OpenAI / Anthropic compatible proxy -----------------------------
    @app.get("/v1/models")
    async def models():
        # No touch(): dsh polls the model list, which must not keep a billed runtime "busy".
        llm = manager.active_llm()
        data = []
        if llm:
            cfg = manager.registry.model(llm.id)
            extras = [b.id for b in manager.backends.values() if b.kind == "video" and b.state == "ready"]
            name = cfg.get("display", llm.id) + (f" [+{', '.join(extras)}]" if extras else "")
            data.append({"id": llm.id, "object": "model", "owned_by": "colab", "name": name})
        return {"object": "list", "data": data}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        manager.touch()
        llm = manager.active_llm()
        if not llm:
            loading = [b.id for b in manager.backends.values() if b.kind == "llm"]
            msg = f"LLM {loading[0]} 正在加载中" if loading else "当前没有在线的 LLM，请先 /load inkling 或 /load qwen"
            return _openai_error(503, msg)
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
        try:
            r = await client.send(upstream, stream=True)
        except httpx.HTTPError as e:
            return _openai_error(502, f"上游 {llm.id} 无响应: {e}")
        out_headers = {k: v for k, v in r.headers.items() if k.lower() not in HOP_HEADERS | {"content-encoding"}}
        return StreamingResponse(r.aiter_raw(), status_code=r.status_code, headers=out_headers,
                                 background=BackgroundTask(r.aclose))

    # ---- video -----------------------------------------------------------
    async def _video_forward(request: Request, method: str, path: str, body: bytes | None = None):
        manager.touch()
        v = manager.active_video()
        if not v:
            raise HTTPException(503, "视频模型未在线，请先 /load pika")
        client: httpx.AsyncClient = request.app.state.client
        r = await client.request(method, f"{v.base_url}{path}", content=body,
                                 headers={"content-type": "application/json"})
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
        return {"ok": True}

    return app
