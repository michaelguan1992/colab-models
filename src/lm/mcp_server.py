"""`lm mcp`: the lm commands as MCP tools over stdio, for dsh, Claude Code, Cursor, Codex.

Each tool runs the same `lm` command a terminal would and returns what it printed, so the CLI
and the MCP surface cannot drift apart.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


def _run(*argv: str) -> str:
    from . import cli  # imported late: cli imports this module for `lm mcp`
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = cli.main(list(argv))
    text = out.getvalue().strip()
    if rc != 0:
        raise ToolError(text or f"lm {' '.join(argv)} 失败（退出码 {rc}）")  # its message reaches the model
    return text


server = MCPServer(
    "lm", log_level="WARNING",
    instructions=("Self-hosted models on a Colab GPU. Only one LLM is online at a time; a video or image model "
                  "may need the LLM unloaded first. Check status before loading."))


@server.tool()
def status() -> str:
    """Which models are online on the Colab GPU, their VRAM use, and what is downloading."""
    return _run("status")


@server.tool()
def load(model: str, route: str | None = None) -> str:
    """Start loading a model by id (e.g. inkling, qwen, pika). Another LLM is swapped out.
    Returns immediately; call status until the model shows online."""
    return _run("load", model, "--no-wait", *(["--route", route] if route else []))


@server.tool()
def unload(model: str | None = None) -> str:
    """Unload one model, or every model when no id is given. The Colab session keeps running."""
    return _run("unload", *([model] if model else []))


@server.tool()
def up(model: str | None = None) -> str:
    """Start (or reuse) the Colab GPU session and load a model. Takes minutes; runs in the background."""
    subprocess.Popen([sys.argv[0], "up", *([model] if model else [])], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return "Colab 会话正在后台启动，几分钟后用 status 查看。"


@server.tool()
def down() -> str:
    """Unload every model and stop the Colab session, which stops billing."""
    return _run("down")


@server.tool()
def generate(prompt: str, seconds: float = 5, image_path: str | None = None, seed: int | None = None) -> str:
    """Generate a video with synchronized audio (Pika) and save the MP4 locally. Needs the video model
    online. seconds must be 5 to 15. Returns the saved path. Takes a minute or more."""
    argv = ["video", prompt, "--seconds", str(seconds)]
    if image_path:
        argv += ["--image", image_path]
    if seed is not None:
        argv += ["--seed", str(seed)]
    return _run(*argv)


@server.tool()
def generate_image(prompt: str, width: int | None = None, height: int | None = None, seed: int | None = None) -> str:
    """Generate an image (MOSS-Image, 1024x1024 by default; sides are multiples of 16) and save the PNG locally.
    Needs the image model online. Returns the saved path. Takes under a minute on the G4."""
    argv = ["image", prompt]
    for flag, v in (("--width", width), ("--height", height), ("--seed", seed)):
        if v is not None:
            argv += [flag, str(v)]
    return _run(*argv)


def run() -> int:
    # The protocol owns fd 1. Point fd 1 at stderr so child processes (the colab CLI, ssh) cannot
    # write into the protocol stream, and hand the original fd to the server.
    proto = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = io.TextIOWrapper(os.fdopen(proto, "wb", buffering=0), encoding="utf-8", write_through=True)
    server.run("stdio")
    return 0
