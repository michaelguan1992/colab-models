"""`lm` — one-command start/stop of models on a Colab G4 runtime."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml

from . import tunnel
from .colab import Colab, ColabError, eprint
from .config import CONFIG_PATH, REPO_ROOT, Config, load

STATE_ICON = {"ready": "● 在线", "offline": "○ 离线", "downloading": "◐ 下载中", "starting": "◑ 启动中",
              "stopping": "◌ 停止中", "error": "✕ 错误", "pending": "◌ 排队"}


# ---- helpers -------------------------------------------------------------
def api(cfg: Config, method: str, path: str, **kw) -> httpx.Response:
    try:
        return httpx.request(method, f"{cfg.base_url}{path}", timeout=kw.pop("timeout", 30), **kw)
    except httpx.HTTPError as e:
        raise SystemExit(f"连不上 controller ({cfg.base_url})：{e}\n运行 `lm up` 或 `lm tunnel`。")


def ensure_tunnel(cfg: Config) -> None:
    if not tunnel.healthy(cfg):
        tunnel.start(cfg, Colab(cfg).proxy_command())


def print_status(s: dict, cfg: Config) -> None:
    g = s["gpu"]
    print(f"Session  {cfg.session} ({cfg.gpu})   API {cfg.base_url}/v1")
    if g.get("available"):
        print(f"GPU      {g['name']}  {g['used_gb']:.1f} / {g['total_gb']:.1f} GB")
    print(f"{'模型':<10}{'状态':<14}{'显存':>8}   说明")
    for m in s["models"]:
        state = STATE_ICON.get(m["state"], m["state"])
        if m.get("download_progress") is not None:
            state += f" {m['download_progress'] * 100:.0f}%"
        vram = f"{m['vram_gb']:.1f}GB" if m.get("vram_gb") else "-"
        note = m.get("error") or m["display"]
        print(f"{m['id']:<10}{state:<14}{vram:>8}   {note}")
    idle_min = s["idle_seconds"] / 60
    print(f"空闲 {idle_min:.0f} 分钟（{s['idle_unload_minutes']:.0f} 分钟后自动卸载模型）")


def wait_model(cfg: Config, model_id: str, timeout: float = 3600) -> bool:
    last = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = api(cfg, "GET", "/admin/status").json()
        m = next(x for x in s["models"] if x["id"] == model_id)
        line = STATE_ICON.get(m["state"], m["state"])
        if m.get("download_progress") is not None:
            line += f" {m['download_progress'] * 100:.0f}%"
        if line != last:
            print(f"  {model_id}: {line}", flush=True)
            last = line
        if m["state"] == "ready":
            return True
        if m["state"] in ("error", "offline"):
            print(f"  失败: {m.get('error') or '已下线'}")
            return False
        time.sleep(3)
    print("  超时")
    return False


# ---- commands ------------------------------------------------------------
def cmd_up(cfg: Config, a) -> int:
    colab = Colab(cfg)
    model = None if a.no_load else (a.model or cfg.default_model)
    if colab.session_active():
        print(f"复用已有 Colab 会话 {cfg.session}")
    else:
        print(f"创建 Colab 会话 {cfg.session}（{cfg.gpu}{', high-mem' if cfg.high_mem else ''}）…")
        colab.new()
    print("部署 controller …")
    colab.deploy(load=model)
    print("建立 ssh 隧道 …")
    tunnel.start(cfg, colab.proxy_command())
    print(f"就绪：{cfg.base_url}/v1")
    if model:
        print(f"加载 {model}（首次需从 HF 下载）…")
        if not wait_model(cfg, model):
            return 1
    if a.open:
        return cmd_dsh(cfg, a)
    return 0


def cmd_down(cfg: Config, a) -> int:
    if tunnel.healthy(cfg):
        print("卸载所有模型 …")
        api(cfg, "POST", "/admin/unload", json={}, timeout=120)
    if tunnel.stop():
        print("已关闭隧道")
    if a.keep_session:
        print("保留 Colab 会话（仍在计费）")
        return 0
    colab = Colab(cfg)
    if colab.session_active():
        colab.stop()
        print(f"已停止 Colab 会话 {cfg.session}，不再消耗计算单元")
    else:
        print("没有活动的 Colab 会话")
    return 0


def cmd_status(cfg: Config, a) -> int:
    if not tunnel.healthy(cfg):
        active = Colab(cfg).session_active()
        print(f"Session  {cfg.session}: {'运行中（隧道未连，运行 lm tunnel）' if active else '未运行'}")
        return 0 if not a.json else 1
    s = api(cfg, "GET", "/admin/status").json()
    if a.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        print_status(s, cfg)
        if a.usage:
            print(Colab(cfg).usage())
    return 0


def cmd_load(cfg: Config, a) -> int:
    ensure_tunnel(cfg)
    r = api(cfg, "POST", "/admin/load", json={"id": a.model, "swap": not a.no_swap})
    body = r.json()
    if r.status_code != 200 or not body.get("ok"):
        print(f"拒绝：{body.get('reason') or body.get('detail')}")
        return 1
    if body.get("unloaded"):
        print(f"已卸载 {', '.join(body['unloaded'])}")
    return 0 if a.no_wait else (0 if wait_model(cfg, a.model) else 1)


def cmd_unload(cfg: Config, a) -> int:
    ensure_tunnel(cfg)
    body = api(cfg, "POST", "/admin/unload", json={"id": a.model}, timeout=120).json()
    print(f"已下线：{', '.join(body.get('unloaded') or []) or '无'}")
    return 0


def cmd_video(cfg: Config, a) -> int:
    ensure_tunnel(cfg)
    req = {"prompt": a.prompt, "num_frames": int(a.seconds * 24) + 1, "height": a.height, "width": a.width}
    if a.image:
        req["image_b64"] = base64.b64encode(Path(a.image).read_bytes()).decode()
    if a.seed is not None:
        req["seed"] = a.seed
    r = api(cfg, "POST", "/video/generate", json=req)
    if r.status_code != 200:
        print(f"失败：{r.json().get('detail')}")
        return 1
    job = r.json()
    print(f"任务 {job['id']} 已提交，生成中 …")
    while job["state"] in ("queued", "running"):
        time.sleep(3)
        job = api(cfg, "GET", f"/video/jobs/{job['id']}").json()
    if job["state"] != "done":
        print(f"生成失败：{job.get('error')}")
        return 1
    out = Path(a.out or cfg.outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / job["file"]
    with httpx.stream("GET", f"{cfg.base_url}/files/{job['file']}", timeout=None) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    print(f"已保存 {dest}（耗时 {job['finished_at'] - job['started_at']:.0f}s）")
    return 0


def cmd_tunnel(cfg: Config, a) -> int:
    tunnel.start(cfg, Colab(cfg).proxy_command())
    print(f"隧道已连通：{cfg.base_url}")
    return 0


def cmd_logs(cfg: Config, a) -> int:
    name = a.name or "controller"
    Colab(cfg).ssh(f"tail -n {a.lines} /content/lm/logs/{name}.log")
    return 0


def cmd_setup_dsh(cfg: Config, a) -> int:
    """Add a `colab` provider to the dsh profile patch (merging into the existing llm-pi-ai row)."""
    patch = Path(cfg.dsh_home).expanduser() / "profiles" / (a.profile or cfg.dsh_profile) / "cordis.patch.yml"
    provider = {
        "api": "openai-completions",
        "baseURL": f"{cfg.base_url}/v1",
        "compat": {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
        "models": [{"id": "inkling"}, {"id": "qwen"}],
    }
    rows = yaml.safe_load(patch.read_text()) if patch.exists() else None
    rows = rows or []
    row = next((r for r in rows if isinstance(r, dict) and r.get("id") == "llm-pi-ai"), None)
    if row is None and not a.force:
        print(
            f"{patch} 中还没有 llm-pi-ai 行。dsh 的 patch 会整行替换 config，直接新建可能覆盖内置 provider。\n"
            "请改用 dsh 界面：Settings → Models → Add model provider → Custom model API\n"
            f"  Provider ID: colab\n  Base URL:    {cfg.base_url}/v1\n"
            "  Protocol:    OpenAI Chat Completions\n  API key:     任意非空值（如 local）\n"
            "  Models:      自动发现（只显示当前在线的模型）\n"
            "或加 --force 强制写入。"
        )
        return 0
    if row is None:
        row = {"id": "llm-pi-ai", "config": {"providers": {}}}
        rows.append(row)
    row.setdefault("config", {}).setdefault("providers", {})["colab"] = provider
    if patch.exists():
        backup = patch.with_suffix(f".yml.bak-{datetime.now():%Y%m%d%H%M%S}")
        shutil.copy2(patch, backup)
        print(f"已备份 {backup}")
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text(yaml.safe_dump(rows, allow_unicode=True, sort_keys=False))
    print(f"已写入 provider colab → {patch}（dsh 下次请求时自动重载）")
    return 0


def cmd_dsh(cfg: Config, a) -> int:
    plugin_patch = REPO_ROOT / "dsh" / "plugin-colab" / "cordis.yml"
    cmd = ["npx", "-y", "@deepseek-ai/dsh", "web", "--patch", str(plugin_patch)]
    print("启动 dsh：" + " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lm", description="Colab G4 模型一键启停")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("up", help="创建/复用 Colab 会话、部署 controller、建隧道、加载模型")
    s.add_argument("model", nargs="?", help="要加载的模型（默认取配置 default_model）")
    s.add_argument("--no-load", action="store_true", help="只开机，不加载模型")
    s.add_argument("--open", action="store_true", help="完成后启动 dsh")
    s.set_defaults(fn=cmd_up)

    s = sub.add_parser("down", help="卸载模型、关隧道、停止 Colab 会话")
    s.add_argument("--keep-session", action="store_true")
    s.set_defaults(fn=cmd_down)

    s = sub.add_parser("status", help="显示在线模型与显存")
    s.add_argument("--json", action="store_true")
    s.add_argument("--usage", action="store_true", help="同时显示计算单元余额")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("load", help="加载模型（LLM 会自动替换当前 LLM）")
    s.add_argument("model")
    s.add_argument("--no-swap", action="store_true", help="已有 LLM 时拒绝而不是替换")
    s.add_argument("--no-wait", action="store_true")
    s.set_defaults(fn=cmd_load)

    s = sub.add_parser("unload", help="卸载模型（不带参数则全部卸载）")
    s.add_argument("model", nargs="?")
    s.set_defaults(fn=cmd_unload)

    s = sub.add_parser("video", help="用 Pika 生成视频并下载到本地")
    s.add_argument("prompt")
    s.add_argument("--image", help="首帧图片")
    s.add_argument("--seconds", type=float, default=5)
    s.add_argument("--height", type=int, default=480)
    s.add_argument("--width", type=int, default=832)
    s.add_argument("--seed", type=int)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_video)

    s = sub.add_parser("tunnel", help="重建 ssh 隧道")
    s.set_defaults(fn=cmd_tunnel)

    s = sub.add_parser("logs", help="查看远端日志（controller / inkling / qwen / pika）")
    s.add_argument("name", nargs="?")
    s.add_argument("-n", "--lines", type=int, default=80)
    s.set_defaults(fn=cmd_logs)

    s = sub.add_parser("setup-dsh", help="把 colab provider 写入 dsh profile")
    s.add_argument("--profile")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_setup_dsh)

    s = sub.add_parser("dsh", help="带 colab 插件启动 dsh web")
    s.set_defaults(fn=cmd_dsh)

    a = p.parse_args(argv)
    cfg = load()
    try:
        return a.fn(cfg, a)
    except (ColabError, RuntimeError) as e:
        eprint(f"错误：{e}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
