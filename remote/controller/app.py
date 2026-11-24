"""Controller HTTP API (listens on 127.0.0.1 only; reached from the Mac via ssh -L)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from typing import AsyncIterator, Awaitable, TypeVar

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from .backends.base import Backend
from .manager import LOADING_STATES, Manager, RequestHandle

log = logging.getLogger("lm.controller")

T = TypeVar("T")

HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding",
               "te", "trailer", "upgrade", "proxy-authorization", "authorization", "x-api-key"}


CHAT = "chat"


class LoadReq(BaseModel):
    id: str
    swap: bool = True
    route: str | None = None


class JobReq(BaseModel):
    model: str | None = None  # None = the only online jobs model (that serves `serves`, when given)
    serves: str | None = None  # e.g. video, image
    input: dict


class UnloadReq(BaseModel):
    id: str | None = None  # None = unload everything


def _openai_error(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": "lm_controller"}}, status_code=status)


async def until_disconnect(request: Request, work: Awaitable[T]) -> T | None:
    """Await ``work`` unless the client disconnects first; then cancel it and return None.

    A non-streaming generation sends nothing until it is done, so without this the engine
    finishes a long answer nobody is waiting for. Call only after the request body is read.
    """
    async def gone() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    task, watch = asyncio.ensure_future(work), asyncio.ensure_future(gone())
    try:
        await asyncio.wait({task, watch}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        watch.cancel()
        if not task.done():
            task.cancel()
    if task.cancelled():
        return None
    return task.result()


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

    def chat_target(requested: object, path: str) -> Backend | JSONResponse:
        """The online chat model the request names; never a silent substitute for another model."""
        ready = manager.serving(CHAT)
        if requested in (None, "", "default"):
            if len(ready) == 1:
                target = ready[0]
            elif not ready:
                return _openai_error(503, unavailable(CHAT))
            else:
                return _openai_error(400, f"有多个聊天模型在线（{', '.join(b.id for b in ready)}），请在 model 字段指定一个")
        else:
            target = next((b for b in ready if b.id == requested), None)
            if target is None:
                b = manager.backends.get(str(requested))
                online = "、".join(x.id for x in ready) or "无"
                if b is not None and b.state in LOADING_STATES:
                    return _openai_error(503, f"{requested} 正在加载中")
                if str(requested) in manager.registry.models:
                    return _openai_error(404, f"{requested} 不在线（在线的聊天模型：{online}），请先 /load {requested}")
                return _openai_error(404, f"没有叫 {requested} 的模型（在线的聊天模型：{online}）")
        if path.startswith("messages") and "anthropic" not in target.engine.apis:
            return _openai_error(400, f"{target.id} 的引擎 {target.engine.name} 不支持 Anthropic /v1/messages，"
                                      "请改用 OpenAI 接口 /v1/chat/completions")
        return target

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        inference = request.method != "GET"  # polling GETs must not keep a billed runtime busy
        if inference:
            manager.touch()
        body = await request.body()
        payload = None
        if body and request.headers.get("content-type", "").startswith("application/json"):
            with contextlib.suppress(ValueError):
                payload = json.loads(body)
        requested = payload.get("model") if isinstance(payload, dict) else None
        llm = chat_target(requested, path)
        if isinstance(llm, JSONResponse):
            return llm
        if isinstance(payload, dict) and llm.upstream_model:
            payload["model"] = llm.upstream_model  # also fills it in when the client left it out
            body = json.dumps(payload).encode()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        client: httpx.AsyncClient = request.app.state.client
        upstream = client.build_request(request.method, f"{llm.base_url}/v1/{path}",
                                        params=request.query_params, headers=headers, content=body)
        handle = manager.begin_request(activity=inference)
        try:
            r = await until_disconnect(request, client.send(upstream, stream=True))
            if r is None:  # the client left before the engine answered; dropping the send aborts the engine
                handle.finish()
                return Response(status_code=499)
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

    # ---- jobs (video and any other jobs-shaped engine) ----------------------
    job_owner: dict[str, str] = {}  # job id -> model id; job ids come from the workers

    def jobs_target(requested: object, serves: str | None) -> Backend:
        def fits(model) -> bool:
            return serves is None or serves in model.serves
        ready = [b for b in manager.backends.values() if b.state == "ready" and b.engine.shape == "jobs" and fits(b.model)]
        if requested in (None, "", "default"):
            if len(ready) == 1:
                return ready[0]
            if ready:
                raise HTTPException(400, f"有多个任务模型在线（{', '.join(b.id for b in ready)}），请指定 model")
            ids = [m.id for m in manager.registry.models.values()
                   if fits(m) and any(manager.registry.engines[r.engine].shape == "jobs" for r in m.routes)]
            loading = [b.id for b in manager.backends.values() if b.id in ids and b.state in LOADING_STATES]
            if loading:
                raise HTTPException(503, f"{loading[0]} 正在加载中")
            raise HTTPException(503, f"当前没有在线的任务模型，请先 {' 或 '.join('/load ' + i for i in ids)}")
        target = next((b for b in ready if b.id == requested), None)
        if target is None:
            b = manager.backends.get(str(requested))
            if b is not None and b.state in LOADING_STATES:
                raise HTTPException(503, f"{requested} 正在加载中")
            raise HTTPException(404 if str(requested) not in manager.registry.models else 503,
                                f"{requested} 不在线，请先 /load {requested}")
        return target

    async def _job_forward(request: Request, target: Backend, method: str, path: str, body: bytes | None = None):
        client: httpx.AsyncClient = request.app.state.client
        try:
            r = await client.request(method, f"{target.base_url}{path}", content=body,
                                     headers={"content-type": "application/json"})
        except httpx.HTTPError as e:
            raise HTTPException(502, f"{target.id} 无响应: {e}")
        data = r.json()
        if isinstance(data, dict) and "id" in data:
            data["model"] = target.id
        return r, data

    @app.post("/jobs")
    async def submit_job(req: JobReq, request: Request):
        manager.touch()
        target = jobs_target(req.model, req.serves)
        r, data = await _job_forward(request, target, "POST", "/generate", json.dumps(req.input).encode())
        if r.status_code < 300:
            job_owner[data["id"]] = target.id
            target.queue_depth = max(target.queue_depth, 1)  # until the next health poll reports the real depth
        return JSONResponse(data, status_code=r.status_code)

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str, request: Request):
        manager.touch()
        owner = job_owner.get(job_id)
        target = manager.backends.get(owner) if owner else None
        if target is None or target.state != "ready":
            raise HTTPException(404, f"没有任务 {job_id}" if owner is None else f"任务 {job_id} 所属的 {owner} 已下线")
        r, data = await _job_forward(request, target, "GET", f"/jobs/{job_id}")
        return JSONResponse(data, status_code=r.status_code)

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
