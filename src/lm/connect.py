"""Harness adapters for `lm connect`: pure functions from the controller's model list to config.

Input is ``/admin/status`` (every model with the caps of the route this GPU would run), so the
model list, context windows and thinking levels are never hand-written on the Mac.
"""

from __future__ import annotations

import json

# dsh thinking level -> the reasoning_effort value Basalt/Strata accept
_DSH_LEVEL = {"none": "off", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high",
              "xhigh": "xhigh", "max": "max"}


def chat_models(status: dict) -> list[dict]:
    """Chat models this GPU can run (a model with no fitting route is left out)."""
    return [m for m in status["models"] if "chat" in m["serves"] and m.get("route")]


def dsh_provider(status: dict, base_url: str) -> dict:
    models = []
    for m in chat_models(status):
        caps = m.get("caps") or {}
        entry: dict = {"id": m["id"], "name": m["display"]}
        if "context" in caps:
            entry["contextWindow"] = caps["context"]
        entry["input"] = ["text", "image"] if caps.get("vision") else ["text"]
        levels = caps.get("reasoning") or []
        entry["reasoningEfforts"] = {_DSH_LEVEL[lv]: lv for lv in levels if lv in _DSH_LEVEL} or False
        models.append(entry)
    return {
        "api": "openai-completions",
        "baseURL": f"{base_url}/v1",
        # dsh refuses keyless custom routes ("No API key for provider"); the controller ignores the key.
        "apiKeyEnv": "LM_COLAB_API_KEY",
        "compat": {"supportsDeveloperRole": False, "maxTokensField": "max_tokens",
                   "supportsReasoningEffort": True},
        "models": models,
    }


def mcp_server(lm_bin: str) -> dict:
    # down and unload wait for engines to exit, which can outlast dsh's default 60s tool timeout.
    return {"transport": "stdio", "command": lm_bin, "args": ["mcp"], "toolCallTimeoutMs": 1_800_000}


def merge_dsh_patch(rows: list, provider: dict, server: dict) -> list:
    """Set providers.colab on llm-pi-ai and servers.lm on mcp-manager, keeping every other row and key."""
    rows = [r for r in rows if r is not None]

    def row(row_id: str) -> dict:
        r = next((r for r in rows if isinstance(r, dict) and r.get("id") == row_id), None)
        if r is None:
            r = {"id": row_id}
            rows.append(r)
        return r

    row("llm-pi-ai").setdefault("config", {}).setdefault("providers", {})["colab"] = provider
    row("mcp-manager").setdefault("config", {}).setdefault("servers", {})["lm"] = server
    return rows


def claude_env(status: dict, base_url: str, model: str, lm_bin: str) -> tuple[dict, str]:
    """Environment and --mcp-config for running Claude Code against the controller's /v1/messages."""
    models = {m["id"]: m for m in chat_models(status)}
    if model not in models:
        raise ValueError(f"{model} 不是这块 GPU 上能跑的聊天模型（可选：{', '.join(models) or '无'}）")
    context = (models[model].get("caps") or {}).get("context")
    env = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": "local",  # the controller strips auth headers; any value works
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": model,
        # Claude Code's background calls pick these aliases; route them to the same model, not to Anthropic ids.
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        # A per-request attribution header changes the prompt prefix, defeating the engine's prefix cache.
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "MCP_TOOL_TIMEOUT": "1800000",
    }
    if context:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(context)  # Claude Code cannot know a custom id's window
    mcp = {"mcpServers": {"lm": {"command": lm_bin, "args": ["mcp"]}}}
    return env, json.dumps(mcp)


def openai_text(status: dict, base_url: str) -> str:
    lines = [f"Base URL: {base_url}/v1", "API key:  任意非空值（如 local）", "Models:"]
    for m in chat_models(status):
        ctx = (m.get("caps") or {}).get("context")
        lines.append(f"  {m['id']:<10} {m['display']}" + (f"（上下文 {ctx}）" if ctx else ""))
    lines.append("请求里的 model 必须是在线模型的 id；只有一个聊天模型在线时，model 可以留空或写 default。")
    return "\n".join(lines)
