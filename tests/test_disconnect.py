"""A client that hangs up mid-generation must close the engine connection, or the GPU keeps generating."""

import asyncio
import socket
import threading
import time

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route

from controller.app import create_app
from ctl_helpers import FakeBackend, make_manager


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            return server
        time.sleep(0.02)
    raise AssertionError("server did not start")


def test_client_disconnect_closes_the_upstream_stream():
    engine = {"chunks": 0, "closed": False}

    async def generate(request):
        async def body():
            try:
                while True:
                    engine["chunks"] += 1
                    yield b"data: tok\n\n"
                    await asyncio.sleep(0.05)
            finally:
                engine["closed"] = True
        return StreamingResponse(body(), media_type="text/event-stream")

    engine_port, ctl_port = free_port(), free_port()
    engine_server = serve(Starlette(routes=[Route("/v1/chat/completions", generate, methods=["POST"])]),
                          engine_port)

    class Upstream(FakeBackend):
        @property
        def base_url(self):
            return f"http://127.0.0.1:{engine_port}"

    mgr = make_manager(factory=Upstream)
    ctl_server = serve(create_app(mgr, idle_poll_s=3600), ctl_port)
    try:
        base = f"http://127.0.0.1:{ctl_port}"
        assert httpx.post(f"{base}/admin/load", json={"id": "inkling"}).status_code == 200
        with httpx.stream("POST", f"{base}/v1/chat/completions", json={"model": "inkling", "stream": True},
                          timeout=5) as r:
            next(r.iter_bytes())
        for _ in range(100):
            if engine["closed"]:
                break
            time.sleep(0.05)
        assert engine["closed"], f"engine kept streaming after the client left ({engine['chunks']} chunks)"
        assert mgr.inflight()["requests"] == 0
    finally:
        ctl_server.should_exit = engine_server.should_exit = True


def test_client_disconnect_before_the_engine_answers_aborts_the_upstream_request():
    engine = {"aborted": False, "finished": False}

    async def generate(request):
        for _ in range(100):  # a non-streaming generation: nothing is sent until it is done
            if await request.is_disconnected():
                engine["aborted"] = True
                return StreamingResponse(iter([b""]))
            await asyncio.sleep(0.05)
        engine["finished"] = True
        return StreamingResponse(iter([b"{}"]), media_type="application/json")

    engine_port, ctl_port = free_port(), free_port()
    engine_server = serve(Starlette(routes=[Route("/v1/chat/completions", generate, methods=["POST"])]),
                          engine_port)

    class Upstream(FakeBackend):
        @property
        def base_url(self):
            return f"http://127.0.0.1:{engine_port}"

    mgr = make_manager(factory=Upstream)
    ctl_server = serve(create_app(mgr, idle_poll_s=3600), ctl_port)
    try:
        base = f"http://127.0.0.1:{ctl_port}"
        httpx.post(f"{base}/admin/load", json={"id": "inkling"})
        try:
            httpx.post(f"{base}/v1/chat/completions", json={"model": "inkling"}, timeout=0.5)
        except httpx.ReadTimeout:
            pass
        for _ in range(60):
            if engine["aborted"] or engine["finished"]:
                break
            time.sleep(0.05)
        assert engine["aborted"], "the engine finished a generation nobody was waiting for"
        assert mgr.inflight()["requests"] == 0
    finally:
        ctl_server.should_exit = engine_server.should_exit = True
