"""`lm` — one-command start/stop of models on a Colab G4 runtime."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import posixpath
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import httpx
import yaml

from . import connect, tunnel
from .colab import Colab, ColabError, eprint
from .config import CONFIG_PATH, REPO_ROOT, STATE_DIR, Config, load
from .lock import state_lock

ASSIGN_STATE = STATE_DIR / "assign.json"
SESSION_STATE = STATE_DIR / "session.json"
WATCHDOG_PID = STATE_DIR / "watchdog.pid"
WATCHDOG_LOG = STATE_DIR / "watchdog.log"

STATE_ICON = {"ready": "● 在线", "offline": "○ 离线", "installing": "◐ 安装引擎", "downloading": "◐ 下载中",
              "starting": "◑ 启动中", "stopping": "◌ 停止中", "error": "✕ 错误", "pending": "◌ 排队"}


WATCHDOG_POLL_S = 60
WATCHDOG_WAKE_GAP_S = 3 * WATCHDOG_POLL_S
LOCKED_COMMANDS = {"up", "down", "load", "unload", "tunnel", "video", "connect"}


# ---- helpers -------------------------------------------------------------
class ControllerUnreachable(RuntimeError):
    pass


def api(cfg: Config, method: str, path: str, **kw) -> httpx.Response:
    try:
        return httpx.request(method, f"{cfg.base_url}{path}", timeout=kw.pop("timeout", 30), **kw)
    except httpx.HTTPError as e:
        raise ControllerUnreachable(f"连不上 controller ({cfg.base_url})：{e}\n运行 `lm up` 或 `lm tunnel`。")


def ensure_tunnel(cfg: Config, colab: Colab | None = None, require_health: bool = True) -> None:
    """Reuse a working tunnel; only (re)start it when it is not. require_health=False accepts a live ssh
    master even if the controller is not answering yet."""
    if tunnel.healthy(cfg) or (not require_health and tunnel.master_alive(cfg)):
        return
    tunnel.start(cfg, (colab or Colab(cfg)).proxy_command(), require_health=require_health)


def print_status(s: dict, cfg: Config) -> None:
    g = s["gpu"]
    print(f"Session  {cfg.session}   API {cfg.base_url}/v1")
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
    if s.get("external_work"):
        print(f"外部任务算作活动：{'、'.join(s['external_work'])}")


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


# ---- GPU assignment ------------------------------------------------------
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _attempts_today() -> int:
    st = _read_json(ASSIGN_STATE)
    return int(st.get("count", 0)) if st.get("date") == date.today().isoformat() else 0


def _count_attempt() -> None:
    ASSIGN_STATE.write_text(json.dumps({"date": date.today().isoformat(), "count": _attempts_today() + 1}))


def _label(acc: dict) -> str:
    return acc["gpu"] + (" high-mem" if acc.get("high_mem") else "")


def acquire(cfg: Config, colab: Colab, wait: float | None) -> bool:
    """Get a runtime from cfg.accelerators, in order. One request per accelerator per round.

    Capacity errors (503) move on to the next accelerator; with --wait, a new round starts
    after ~assign_retry_minutes, for at most ``wait`` minutes (inf: until the daily cap). A daily request cap keeps us far from the rapid-retry
    pattern that got other accounts restricted (colabtools#6206).
    """
    accs = list(cfg.accelerators)
    deadline = time.monotonic() + wait * 60 if wait else None
    rnd = 0
    while True:
        rnd += 1
        for acc in list(accs):
            if _attempts_today() >= cfg.assign_daily_limit:
                print(f"今天已发出 {cfg.assign_daily_limit} 次分配请求，达到上限（assign_daily_limit），明天再试。")
                return False
            print(f"申请 {_label(acc)} …", flush=True)
            _count_attempt()
            ok, kind, detail = colab.new(acc["gpu"], bool(acc.get("high_mem")))
            if not ok:
                msg = {"capacity": "暂时没有空闲容量（503）",
                       "rejected": "被拒绝（400，该规格不可用），以后不再尝试",
                       "quota": "已有太多会话（412），先 `colab sessions` 清理",
                       }.get(kind, f"失败：{detail}")
                print(f"  {_label(acc)} {msg}")
                if kind == "rejected":
                    accs.remove(acc)
                if kind == "quota":
                    return False
                continue
            if detail:
                print(f"  {detail}")
            name, vram = colab.gpu_info()
            if vram and vram < cfg.min_vram_gb:
                print(f"  分到的是 {name} {vram:.0f}GB，小于 min_vram_gb={cfg.min_vram_gb:.0f}，停掉换下一个")
                colab.stop()
                continue
            print(f"  已分配 {name}（{vram:.0f}GB）")
            SESSION_STATE.write_text(json.dumps({"gpu": acc["gpu"], "name": name, "vram_gb": vram,
                                                 "since": time.time()}))
            return True
        if not accs:
            print("所有备选 GPU 都被拒绝，请检查 accelerators 配置")
            return False
        if not wait:
            print("这一轮都没分到。稍后再运行 `lm up`，或用 `lm up --wait [分钟]` 自动按间隔重试。")
            return False
        delay = cfg.assign_retry_minutes * 60 * random.uniform(0.8, 1.2)
        if time.monotonic() + delay > deadline:
            print(f"等了 {wait:.0f} 分钟仍没分到 GPU，放弃。")
            return False
        print(f"第 {rnd} 轮没分到，{delay / 60:.0f} 分钟后重试（Ctrl-C 放弃）…", flush=True)
        time.sleep(delay)


# ---- idle watchdog -------------------------------------------------------
def _watchdog_pid() -> int | None:
    try:
        pid = int(WATCHDOG_PID.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def start_watchdog(cfg: Config) -> None:
    if cfg.idle_stop_minutes <= 0 or _watchdog_pid():
        return
    log = open(WATCHDOG_LOG, "ab")
    p = subprocess.Popen([sys.executable, "-m", "lm.cli", "watchdog"], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=log, start_new_session=True)
    WATCHDOG_PID.write_text(str(p.pid))


def stop_watchdog() -> None:
    pid = _watchdog_pid()
    WATCHDOG_PID.unlink(missing_ok=True)
    if pid and pid != os.getpid():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


@dataclass
class WatchState:
    last_tick: float | None = None
    unreachable_since: float | None = None
    grace_until: float = 0.0
    query_failing: bool = False


def _log(msg: str) -> None:
    print(f"{datetime.now():%F %T} {msg}", flush=True)


def _guarded_down(cfg: Config, colab: Colab) -> bool:
    """Stop the runtime unless another lm command holds the lock. True once the runtime is stopped."""
    with state_lock(wait=False) as got:
        if not got:
            _log("另一个 lm 命令正在运行，这一轮跳过停止")
            return False
        try:
            return shutdown(cfg, colab, keep_session=False) == 0
        except (ColabError, RuntimeError) as e:
            _log(f"自动停止失败，继续守护：{e}")
            return False


def watchdog_tick(cfg: Config, colab: Colab, st: WatchState, now: float, status: dict | None) -> bool:
    """One guard step. status is /admin/status or None when unreachable. True when the watchdog is done."""
    limit = cfg.idle_stop_minutes * 60
    if st.last_tick is not None and now - st.last_tick > WATCHDOG_WAKE_GAP_S:
        _log("检测到电脑刚睡醒，空闲时间从现在重新计算")
        st.grace_until = now + limit
        st.unreachable_since = None
    st.last_tick = now
    if status is None:
        st.unreachable_since = st.unreachable_since or now
        active = colab.session_active()
        if active is False:
            _log("会话已不存在，watchdog 退出")
            return True
        if active is None and not st.query_failing:
            _log("colab CLI 查询失败（授权过期或断网？），无法确认会话状态，继续守护")
        st.query_failing = active is None
        if now >= st.grace_until and now - st.unreachable_since >= limit:
            _log(f"连续 {limit / 60:.0f} 分钟连不上 controller，自动停止")
            return _guarded_down(cfg, colab)
        if now - st.unreachable_since > 120 and not tunnel.is_running():  # a live keeper is already retrying
            with state_lock(wait=False) as got:
                if got:
                    try:
                        tunnel.start(cfg, colab.proxy_command())
                    except RuntimeError as e:
                        _log(f"重连隧道失败：{e}")
        return False
    st.unreachable_since = None
    st.query_failing = False
    if status.get("busy"):
        return False
    if now >= st.grace_until and status.get("idle_seconds", 0) >= limit:
        _log(f"空闲 {status['idle_seconds'] / 60:.0f} 分钟，自动停止")
        return _guarded_down(cfg, colab)
    return False


def cmd_watchdog(cfg: Config, a) -> int:
    """Runs detached on the Mac: stops the runtime once it has been idle for idle_stop_minutes.

    The controller only unloads models when idle; the Colab session itself keeps billing
    (~9 CU/h on G4) until stopped. It never stops while the controller reports busy work, and
    after the Mac wakes from sleep it waits a full idle window before stopping.
    """
    colab = Colab(cfg)
    st = WatchState()
    while True:
        time.sleep(WATCHDOG_POLL_S)
        try:
            status = httpx.get(f"{cfg.base_url}/admin/status", timeout=10).json()
        except (httpx.HTTPError, ValueError):
            status = None
        if watchdog_tick(cfg, colab, st, time.time(), status):
            break
    WATCHDOG_PID.unlink(missing_ok=True)
    return 0


# ---- commands ------------------------------------------------------------
def cmd_up(cfg: Config, a) -> int:
    colab = Colab(cfg)
    model = None if a.no_load else (a.model or cfg.default_model)
    active = colab.session_active()
    if active is None:
        print("无法查询 Colab 会话（colab CLI 出错，授权过期或断网？）。先运行 `colab sessions` 检查。")
        return 1
    if active:
        print(f"复用已有 Colab 会话 {cfg.session}")
    elif not acquire(cfg, colab, wait=a.wait):
        return 1
    start_watchdog(cfg)
    health = tunnel.Probe("down")
    if active:
        ensure_tunnel(cfg, colab, require_health=False)
        health = tunnel.probe(cfg)
        if health.state == "foreign":
            raise tunnel.PortConflict(tunnel.conflict_message(cfg))
    if health.state == "ours" and health.version == colab.version:
        print(f"controller 代码版本 {health.version} 与本地一致，跳过部署")
        if model and not request_load(cfg, model, swap=True):
            return 1
    else:
        print("部署 controller …")
        colab.deploy(load=model)
        print("建立 ssh 隧道 …")
        ensure_tunnel(cfg, colab)
    print(f"就绪：{cfg.base_url}/v1")
    if model and not wait_model(cfg, model):
        return 1
    return 0


def shutdown(cfg: Config, colab: Colab, keep_session: bool) -> int:
    if tunnel.healthy(cfg):
        print("卸载所有模型 …")
        try:
            api(cfg, "POST", "/admin/unload", json={}, timeout=120)
        except ControllerUnreachable as e:
            print(f"卸载失败：{e}\n继续关闭。")
    if tunnel.stop():
        print("已关闭隧道")
    if keep_session:
        print("保留 Colab 会话（仍在计费）")
        return 0
    active = colab.session_active()
    if active is False:
        print("没有活动的 Colab 会话")
        return 0
    if active is None:
        print("无法确认会话状态（colab CLI 查询失败），仍尝试停止 …")
    try:
        colab.stop()
    except ColabError as e:
        if active is None:
            print(f"停止失败，会话可能仍在计费：{e}")
            return 1
        raise
    SESSION_STATE.unlink(missing_ok=True)
    print(f"已停止 Colab 会话 {cfg.session}，不再消耗计算单元")
    return 0


def cmd_down(cfg: Config, a) -> int:
    stop_watchdog()
    return shutdown(cfg, Colab(cfg), a.keep_session)


def cmd_status(cfg: Config, a) -> int:
    probe = tunnel.probe(cfg)
    if probe.state != "ours":
        if probe.state == "foreign":
            print(tunnel.conflict_message(cfg))
            return 1
        active = Colab(cfg).session_active()
        label = {True: "运行中（隧道未连，运行 lm tunnel）", False: "未运行",
                 None: "状态未知（colab CLI 查询失败）"}[active]
        print(f"Session  {cfg.session}: {label}")
        return 0 if not a.json else 1
    s = api(cfg, "GET", "/admin/status").json()
    if a.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        print_status(s, cfg)
        if a.usage:
            print(Colab(cfg).usage())
    return 0


def request_load(cfg: Config, model: str, swap: bool, route: str | None = None) -> bool:
    req = {"id": model, "swap": swap}
    if route:
        req["route"] = route
    r = api(cfg, "POST", "/admin/load", json=req)
    body = r.json()
    if r.status_code != 200 or not body.get("ok"):
        print(f"拒绝：{body.get('reason') or body.get('detail')}")
        return False
    if body.get("unloaded"):
        print(f"已卸载 {', '.join(body['unloaded'])}")
    return True


def cmd_load(cfg: Config, a) -> int:
    ensure_tunnel(cfg)
    if not request_load(cfg, a.model, swap=not a.no_swap, route=a.route):
        return 1
    return 0 if a.no_wait else (0 if wait_model(cfg, a.model) else 1)


def cmd_unload(cfg: Config, a) -> int:
    ensure_tunnel(cfg)
    body = api(cfg, "POST", "/admin/unload", json={"id": a.model}, timeout=120).json()
    print(f"已下线：{', '.join(body.get('unloaded') or []) or '无'}")
    return 0


def _detail(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return r.text
    d = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(d, list):  # pydantic validation errors from the worker
        return "；".join(f"{'.'.join(map(str, e.get('loc', [])[1:]))}: {e.get('msg')}" for e in d)
    return str(d)


def cmd_video(cfg: Config, a) -> int:
    ensure_tunnel(cfg)
    inp: dict = {"prompt": a.prompt, "seconds": a.seconds}
    for k in ("height", "width", "seed"):
        if getattr(a, k) is not None:
            inp[k] = getattr(a, k)
    if a.image:
        inp["image_b64"] = base64.b64encode(Path(a.image).expanduser().read_bytes()).decode()
    r = api(cfg, "POST", "/jobs", json={"model": a.model, "input": inp})
    if r.status_code != 200:
        print(f"失败：{_detail(r)}")
        return 1
    job = r.json()
    print(f"任务 {job['id']}（{job['model']}）已提交，生成中 …")
    while job["state"] in ("queued", "running"):
        time.sleep(3)
        job = api(cfg, "GET", f"/jobs/{job['id']}").json()
    if job["state"] != "done":
        print(f"生成失败：{job.get('error') or job.get('detail')}")
        return 1
    out = Path(a.out or cfg.outputs_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dest = (out / job["file"]).resolve()
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
    logs_dir = posixpath.join(posixpath.dirname(cfg.remote_dir.rstrip("/")), "logs")
    Colab(cfg).ssh(f"tail -n {a.lines} {shlex.quote(f'{logs_dir}/{name}.log')}")
    return 0


def lm_bin() -> str:
    return shutil.which("lm") or os.path.abspath(sys.argv[0])


def _write_with_backup(path: Path, text: str) -> None:
    if path.exists():
        for old in path.parent.glob(f"{path.name}.bak-*"):
            old.unlink()
        backup = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d%H%M%S}")
        shutil.copy2(path, backup)
        print(f"已备份 {backup}（只保留最近一份）")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def cmd_connect(cfg: Config, a) -> int:
    """Write or print a harness's config from the controller's model list (needs `lm up` first)."""
    ensure_tunnel(cfg)
    status = api(cfg, "GET", "/admin/status").json()
    if a.harness == "openai":
        print(connect.openai_text(status, cfg.base_url))
        return 0
    if a.harness == "claude-code":
        env, mcp = connect.claude_env(status, cfg.base_url, a.model or cfg.default_model, lm_bin())
        print("# 用 `lm claude` 直接启动；或者把下面这些放进你自己的 shell：")
        for k, v in env.items():
            print(f"export {k}={shlex.quote(v)}")
        print(f"# MCP：claude --mcp-config {shlex.quote(mcp)}")
        return 0
    patch = Path(cfg.dsh_home).expanduser() / "profiles" / (a.profile or cfg.dsh_profile) / "cordis.patch.yml"
    rows = (yaml.safe_load(patch.read_text()) if patch.exists() else None) or []
    if not any(isinstance(r, dict) and r.get("id") == "llm-pi-ai" for r in rows) and not a.force:
        print(f"{patch} 中还没有 llm-pi-ai 行。dsh 的 patch 会整行替换 config，直接新建可能覆盖内置 provider。\n"
              "先在 dsh 界面里随便加一个 Custom model API（Settings → Models），再重跑；或加 --force 强制写入。")
        return 1
    provider = connect.dsh_provider(status, cfg.base_url)
    rows = connect.merge_dsh_patch(rows, provider, connect.mcp_server(lm_bin()))
    _write_with_backup(patch, yaml.safe_dump(rows, allow_unicode=True, sort_keys=False))
    ids = ", ".join(f"{m['id']}（上下文 {m.get('contextWindow', '?')}）" for m in provider["models"])
    print(f"已写入 {patch}：provider colab 的模型 {ids}；MCP server lm。dsh 下次请求时自动重载。")
    return 0


def cmd_claude(cfg: Config, a) -> int:
    ensure_tunnel(cfg)
    status = api(cfg, "GET", "/admin/status").json()
    env, mcp = connect.claude_env(status, cfg.base_url, a.model or cfg.default_model, lm_bin())
    cmd = [cfg.claude_bin, "--mcp-config", mcp, *a.args]
    return subprocess.call(cmd, env={**os.environ, **env})


def cmd_mcp(cfg: Config, a) -> int:
    from . import mcp_server
    return mcp_server.run()


def cmd_dsh(cfg: Config, a) -> int:
    plugin_patch = REPO_ROOT / "dsh" / "plugin-colab" / "cordis.yml"
    cmd = ["npx", "-y", "@deepseek-ai/dsh", "web", "--patch", str(plugin_patch)]
    print("启动 dsh：" + " ".join(cmd))
    env = {**os.environ}
    env.setdefault("LM_COLAB_API_KEY", "local")  # any non-empty value; see connect.dsh_provider
    return subprocess.call(cmd, env=env)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lm", description="Colab G4 模型一键启停")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("up", help="创建/复用 Colab 会话、部署 controller、建隧道、加载模型")
    s.add_argument("model", nargs="?", help="要加载的模型（默认取配置 default_model）")
    s.add_argument("--no-load", action="store_true", help="只开机，不加载模型")
    s.add_argument("--open", action="store_true", help="完成后启动 dsh")
    s.add_argument("--wait", nargs="?", type=float, const=math.inf, default=None, metavar="分钟",
                   help="分不到 GPU 时按 assign_retry_minutes 间隔继续重试；可给最长等待分钟数，不给则等到当天次数上限")
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
    s.add_argument("--route", help="指定引擎路线（如 strata / basalt）")
    s.set_defaults(fn=cmd_load)

    s = sub.add_parser("unload", help="卸载模型（不带参数则全部卸载）")
    s.add_argument("model", nargs="?")
    s.set_defaults(fn=cmd_unload)

    s = sub.add_parser("video", help="用视频模型（Pika）生成视频并下载到本地")
    s.add_argument("prompt")
    s.add_argument("--image", help="首帧图片")
    s.add_argument("--seconds", type=float, default=5, help="5 到 15 秒")
    s.add_argument("--height", type=int, help="默认由模型决定")
    s.add_argument("--width", type=int, help="默认由模型决定")
    s.add_argument("--model", help="任务模型 id；只有一个在线时可省略")
    s.add_argument("--seed", type=int)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_video)

    s = sub.add_parser("tunnel", help="重建 ssh 隧道")
    s.set_defaults(fn=cmd_tunnel)

    s = sub.add_parser("logs", help="查看远端日志（controller / inkling / qwen / pika）")
    s.add_argument("name", nargs="?")
    s.add_argument("-n", "--lines", type=int, default=80)
    s.set_defaults(fn=cmd_logs)

    s = sub.add_parser("connect", help="按 controller 的模型列表生成 harness 配置（dsh / claude-code / openai）")
    s.add_argument("harness", choices=["dsh", "claude-code", "openai"])
    s.add_argument("--model", help="claude-code 用哪个模型（默认 default_model）")
    s.add_argument("--profile", help="dsh profile（默认 dsh_profile）")
    s.add_argument("--force", action="store_true", help="dsh：patch 里没有 llm-pi-ai 行也写入")
    s.set_defaults(fn=cmd_connect)

    s = sub.add_parser("claude", help="用 Colab 上的模型启动 Claude Code（带 lm 的 MCP 工具）")
    s.add_argument("--model", help="默认 default_model")
    s.set_defaults(fn=cmd_claude)  # every other argument is passed to claude unchanged

    s = sub.add_parser("mcp", help="以 stdio MCP server 运行（给 harness 调用）")
    s.set_defaults(fn=cmd_mcp)

    s = sub.add_parser("dsh", help="带 colab 插件启动 dsh web")
    s.set_defaults(fn=cmd_dsh)

    s = sub.add_parser("watchdog", help=argparse.SUPPRESS)  # spawned by `lm up`
    s.set_defaults(fn=cmd_watchdog)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    a, rest = parser.parse_known_args(argv)
    if a.cmd == "claude":
        a.args = rest
    elif rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    cfg = load()
    try:
        if a.cmd in LOCKED_COMMANDS:
            with state_lock():
                rc = a.fn(cfg, a)
        else:
            rc = a.fn(cfg, a)
        if rc == 0 and getattr(a, "open", False):
            rc = cmd_dsh(cfg, a)
        return rc
    except (ColabError, RuntimeError) as e:
        eprint(f"错误：{e}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
