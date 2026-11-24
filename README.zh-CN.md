# LocalModels

在 Google Colab 的 GPU 上运行自己选的模型，在 Mac 上使用，一条命令启动或停止。

[English](README.md)

目标运行时是 Colab G4（RTX PRO 6000 Blackwell，96GB）。A100 80GB 和更小的 GPU 也能用，走备用路线。在 Mac 上，`lm` 命令负责创建 Colab 会话、部署 controller、打开 ssh 隧道。之后就可以在 DeepSeek Harness（dsh）、Claude Code 或任何兼容 OpenAI 的客户端里使用这些模型。

## 模型

| id | 模型 | 引擎 | 显存 |
|---|---|---|---|
| `inkling` | G4：OpenUncensored/Inkling-2-Mini-Uncensored 的 Basalt 包（1M 上下文）。其他 GPU：AtomicChat 的 GGUF（Q4_K，32K） | Basalt / Strata | ~55GB / ~18GB |
| `qwen` | SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF（IQ3_XXS）。G4：转成 Basalt 包（262K 上下文，能看图）。其他 GPU：GGUF 走 Strata（同样 262K） | Basalt / Strata | ~50GB / ~62GB |
| `qwen-tiny` | Qwen/Qwen3-0.6B-GGUF（Q8_0，演示用） | llama.cpp | ~2GB |
| `pika` | Pika-H3-Flex 加官方 fl2va adapter（6 步，**视频加音频**）。G4 上跑 FP8，整个模型常驻显存，5 秒视频约 36 秒 | diffusers | ~78GB |
| `moss` | OpenMOSS-Team/MOSS-Image-SFT 加 lodestones 的 8 步 LoRA（CFG 1，**图像**）。A100/H100/G4 上常驻显存，其他 GPU 用 CPU offload。L4 上 1024px 每张约 44 秒 | diffusers（模型自带的 pipeline） | ~32GB（估计，G4 上未实测） |

模型都写在 [`remote/models.yaml`](remote/models.yaml) 里。每个模型有一条或多条路线，controller 选第一条 GPU 要求匹配的路线。加模型或引擎只需要改这个文件。

## 调度规则

规则写在 `remote/models.yaml` 的 `rules` 里。controller 用它们检查"加载之后会在线的模型集合"（[`policy.py`](remote/controller/policy.py)）。

1. **同一时间只有一个 LLM。** `llm` 组最多一个模型在线（`max_online: 1, on_conflict: swap`）。加载另一个 LLM 时，会先卸载当前的。
2. **Pika 要求 LLM 足够小。** `pika`（`video` 组）只有在在线 LLM 的**实测**显存低于 22GB 时才能加载（`requires_online_below`）。G4 上 Inkling 走 Basalt、开 1M 上下文，实测 54.7GB，所以 Pika 只能单独在线（演示用的 `qwen-tiny` 除外）。
   `moss`（`image` 组）没有专门的规则，只受显存检查约束：在线模型加上新模型，不能超过总显存减去余量。在 G4 上，它能和任一 LLM 同时在线，但不能和 Pika 同时在线。
3. **空闲卸载。** 45 分钟没有请求，卸载所有模型（`idle_unload_minutes`）。
   在运行时上手动跑的任务也算活动，包括：不属于任何已加载模型的 GPU 进程，以及 `/content/lm/holds/` 里的任何文件。转换、下载这类不用 GPU 的任务，开始时放一个 hold 文件，跑完删掉。`lm status` 会列出这些外部任务。

## 一次性准备

安装 Colab CLI，然后按提示在浏览器里授权：

```bash
uv tool install google-colab-cli
```

在本仓库里安装 `lm` 命令：

```bash
uv tool install -e .
```

接入客户端。先运行 `lm up`：客户端配置按 controller 报告的模型和能力生成，不用手写。

- **dsh：** `lm connect dsh` 把两样东西合并写入 `~/.dsh/profiles/web/cordis.patch.yml`。一是 provider `colab`，模型、上下文长度、思考档位、能否看图都来自 controller 的 `caps`。二是 MCP server `lm`。写之前会备份，只保留最近一份。换了 GPU 或改了 `models.yaml` 之后再运行一次。
- **Claude Code：** `lm claude` 启动 `claude`，其余参数原样传过去，比如 `lm claude -p "hi"`。它把 `ANTHROPIC_BASE_URL` 指向 controller，把所有模型别名指向 `default_model`，按 `caps` 设置上下文窗口，并用 `--mcp-config` 挂上 `lm mcp`。它不改 Claude Code 的全局设置。`claude` 不在 PATH 上时，在配置里设 `claude_bin`。`lm connect claude-code` 只打印这些环境变量。
- **其他 OpenAI 客户端**（Open WebUI、Continue、Cursor 等）：`lm connect openai` 打印 Base URL 和模型 id。

第一次在真实 G4 上运行时，按 [docs/spike.md](docs/spike.md) 逐项核对。

## 日常使用

开会话、部署、建隧道，并加载 `inkling`（`lm up qwen` 则加载 Qwen）：

```bash
lm up
```

卸载模型、关隧道、停止 Colab 会话（停止计费）：

```bash
lm down
```

### 分不到 GPU 时

G4 经常返回 503，意思是没有空闲容量，不是账号的问题。`lm up` 按 `accelerators` 的顺序每种申请一次：先 G4，再 A100 80GB（high-mem）。都没有就退出。

- `lm up --wait [分钟]` 每隔 `assign_retry_minutes`（默认 12 分钟，±20% 随机）再申请一轮。给了分钟数就最多等这么久（比如 `lm up --wait 120`），不给就一直等到当天的 `assign_daily_limit` 用完。
- 每天最多发 `assign_daily_limit`（默认 12）次申请，多次运行 `lm up` 共用这个额度。短时间内大量重试曾让别的账号被限制（colabtools#6206）。
- 分到的显存小于 `min_vram_gb`（默认 75）就立刻停掉，换下一种。
- 在 A100 80GB 上，因为显存检查，Inkling 和 Pika 不能同时在线。

### 计费与自动关机

- controller 空闲 `idle_unload_minutes`（45）分钟后只卸载模型，Colab 会话仍然按小时计费（G4 约 8.9 计算单元/小时）。
- `lm up` 还会在 Mac 上起一个 watchdog，空闲 `idle_stop_minutes`（默认 60）分钟后自动运行 `lm down`。日志在 `~/.local/state/lm/watchdog.log`。
- Mac 睡眠时 watchdog 不运行，所以离开前最好手动 `lm down`。
- dsh 轮询模型列表不算活动，不会让会话一直开着。

### 断线

- 隧道由一个守护进程看着。ssh 退出后（断网、换 Wi-Fi、Mac 睡醒），1 秒后重连，连不上就把间隔逐次加倍，最长 30 秒。链路悄悄断掉时，ssh 约 30 秒内就能发现。日志在 `~/.local/state/lm/tunnel.log`。
- 客户端中途断开（关掉对话、按停止、超时）时，controller 会立刻断开到引擎的连接，GPU 不会继续生成没人要的回答。流式和非流式请求都是这样。

### 配置

`~/.config/lm/config.toml` 里的都是可选项：

```toml
accelerators = [{gpu = "G4"}, {gpu = "A100", high_mem = true}]
min_vram_gb = 75
assign_retry_minutes = 12
assign_daily_limit = 12
idle_stop_minutes = 60
default_model = "inkling"
```

### 命令

| 命令 | 作用 |
|---|---|
| `lm status [--usage]` | GPU、在线模型和显存、空闲时长；`--usage` 同时显示计算单元余额 |
| `lm load <id>` / `lm unload [id]` | 加载模型（LLM 会替换当前的 LLM）/ 卸载一个或全部 |
| `lm video "提示词" [--seconds 5到15] [--image 首帧.png]` | 用 Pika 生成视频并下载 |
| `lm image "提示词" [--width 1024 --height 1024] [--seed N]` | 用 MOSS-Image 生成图片并下载 |
| `lm tunnel` | 重建 ssh 隧道 |
| `lm logs [controller\|inkling\|qwen\|pika\|moss]` | 查看远端日志 |
| `lm connect dsh\|claude-code\|openai` | 按 controller 的模型列表生成客户端配置 |
| `lm claude [参数]` | 用 Colab 上的模型启动 Claude Code |
| `lm dsh` | 带 colab 插件启动 dsh web |
| `lm mcp` | 以 stdio MCP server 运行 |

## 请求怎么路由

- 请求里的 `model` 是哪个在线模型的 id，就发给哪个。模型不在线或不存在时返回 404 并列出在线的模型，不会悄悄换成别的模型。只有一个聊天模型在线时，`model` 可以留空或写 `default`。
- `/v1/messages`（Anthropic 协议）直通给声明了 `anthropic` 的引擎（Basalt、Strata）。llama.cpp 这类引擎会返回明确的错误。
- 视频和图片任务走 `POST /jobs {"model": "pika", "input": {"prompt": "...", "seconds": 5}}`。也可以不写 `model`，改写 `"serves": "video"` 或 `"image"`，由 controller 挑选在线的对应模型。用 `GET /jobs/{id}` 查状态，结果在 `/files/` 下。帧数由 worker 按自己的 fps 计算，5 到 15 秒以外的时长会被拒绝。

## 给模型用的工具（MCP）

`lm mcp` 是一个 stdio MCP server，工具有 `status`、`load`、`unload`、`up`、`down`、`generate`（视频）和 `generate_image`。每个工具运行的就是对应的 `lm` 命令。`lm connect dsh` 和 `lm claude` 会自动挂上它；其他支持 MCP 的 harness 把命令配成 `lm mcp` 即可。`generate` 要等视频生成完，所以 dsh 那边的工具超时设成了 30 分钟。

## 在 dsh 里使用

`lm dsh` 会带上 [`dsh/plugin-colab`](dsh/plugin-colab) 插件启动 dsh web。插件只提供斜杠命令 `/up [model]`、`/down`、`/load <id>`、`/unload [id]`、`/status`，每个都直接运行 `lm`。dsh 的模型选择器通过 `/v1/models` 发现模型，这个接口**只列出在线且能 `chat` 的模型**。如果别的模型（比如 Pika）也在线，名字后面会带 `[+pika]`。

## 目录结构

```
src/lm/                Mac 端 CLI（colab CLI 封装、ssh 隧道、各个命令）
remote/bootstrap.sh    在 Colab 上安装 controller 依赖并启动它（可重复执行）。
                       引擎在第一次加载用到它的模型时才安装。
remote/models.yaml     注册表：engines（怎么装、怎么起）、models（仓库、路线、caps、显存估计）、
                       rules（调度规则）。加模型或引擎只改这个文件。
remote/controller/     FastAPI :9000：按 model 路由的 /v1 反向代理、/jobs、/admin、调度规则
remote/bench/          docs/spike.md 里数据对应的 bench 和转换脚本
dsh/plugin-colab/      dsh 插件（斜杠命令，调用 lm）
```

Mac 和 Colab 之间只走 `ssh -L`（以 `colab ssh --proxy-mode` 作为 ProxyCommand），没有任何公网入口。

## 测试

```bash
uv run pytest
```
