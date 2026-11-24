# LocalModels：Colab G4 上的自托管模型，一键启停

在 Google Colab G4 运行时（RTX PRO 6000，96GB）上跑模型，在 Mac 上用 DeepSeek Harness（dsh）调用。

| id | 模型 | 引擎 | 显存估计 |
|---|---|---|---|
| `inkling` | G4：OpenUncensored/Inkling-2-Mini-Uncensored（Basalt 包，1M 上下文）；其他 GPU：AtomicChat GGUF（Q4_K，32K） | Basalt / Strata | ~55GB / ~18GB |
| `qwen` | SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF（IQ3_XXS） | Strata | ~50GB |
| `qwen-tiny` | Qwen/Qwen3-0.6B-GGUF（Q8_0，演示用） | llama.cpp | ~2GB |
| `pika` | Pika-H3-Flex + Turbo LoRA（4 步，**视频+音频**） | diffusers | ~69GB |

**调度规则**（写在 `remote/models.yaml` 的 `rules`，由 controller 对"加载之后的在线集合"强制执行，`remote/controller/policy.py`）
1. 同一时间最多一个 `llm` 组的模型在线（`max_online: 1, on_conflict: swap`）。加载另一个时，会先把同组当前的卸载掉。
2. 只有在线 `llm` 组模型的**实测**显存低于 22GB 时，`video` 组的 `pika` 才能上线（`requires_online_below`）。在 G4 上 Inkling 走 Basalt、开 1M 上下文，实测 54.7GB，所以 Pika 只能单独在线（演示用的 qwen-tiny 除外）。
3. 45 分钟没有请求，自动卸载所有模型（`idle_unload_minutes`）。
   在运行时上手动跑的任务也算活动，不会被当成空闲：GPU 上有不属于任何已加载模型的进程，或 `/content/lm/holds/` 里有文件（给转换、下载这类不用 GPU 的任务用，跑完删掉）。`lm status` 会列出这些外部任务。

## 一次性准备
```bash
uv tool install google-colab-cli      # 然后按提示在浏览器里授权
```
```bash
uv tool install -e .                  # 在本仓库里执行，安装 lm 命令
```
接入 harness。先运行 `lm up`，配置会按 controller 报告的模型和能力生成，不在 Mac 上手写：
- dsh：`lm connect dsh`。它合并写入 `~/.dsh/profiles/web/cordis.patch.yml` 两样东西。一是 provider `colab`，模型、上下文、思考档位、能否看图都来自 `caps`。二是 MCP server `lm`。写之前会备份，只留最近一份。换了 GPU 或改了 `models.yaml` 之后重跑一次。
- Claude Code：`lm claude`，其余参数原样传给 `claude`，如 `lm claude -p "hi"`。它把 `ANTHROPIC_BASE_URL` 指向 controller，把所有模型别名指向 `default_model`，按 caps 设置上下文窗口，并用 `--mcp-config` 挂上 `lm mcp`。它不改你的 Claude Code 全局设置。`claude` 不在 PATH 上时，在配置里设 `claude_bin`。`lm connect claude-code` 只打印这些环境变量。
- 其他 OpenAI 客户端（Open WebUI、Continue、Cursor 等）：`lm connect openai` 打印 Base URL 和模型 id。

第一次在真实 G4 上运行时，按 [docs/spike.md](docs/spike.md) 逐项核对。

## 日常使用
```bash
lm up            # 开机 + 部署 + 隧道 + 加载 inkling（lm up qwen 则加载 Qwen）
```
```bash
lm down          # 卸载模型 + 关闭隧道 + 停止 Colab 会话（停止计费）
```
### 分不到 GPU 时
G4 经常返回 503（没有空闲容量，不是账号问题）。`lm up` 按 `accelerators` 顺序每种只申请一次：先 G4，再 A100 80GB（high-mem）。都没有就退出。
- `lm up --wait [分钟]`：每隔约 `assign_retry_minutes`（默认 12 分钟，±20% 随机）再申请一轮。给了分钟数就最多等这么久（如 `lm up --wait 120`），不给则一直等到当天的 `assign_daily_limit`。
- 每天最多发 `assign_daily_limit`（默认 12）次申请，多次运行 `lm up` 共用这个额度。短时间内大量重试曾导致别的账号被限制（colabtools#6206）。
- 分到的显存小于 `min_vram_gb`（默认 75）就立刻停掉，换下一种。
- 在 A100 80GB 上，Inkling 和 Pika 不能同时在线（始终生效的显存检查：在线模型加目标模型的显存，不能超过 GPU 总显存减余量）。

### 计费与自动关机
- controller 空闲 `idle_unload_minutes`（45）分钟后只卸载模型，Colab 会话仍按小时计费（G4 约 8.9 计算单元/小时）。
- `lm up` 会在 Mac 上起一个 watchdog，空闲 `idle_stop_minutes`（默认 60）分钟后自动 `lm down`。日志在 `~/.local/state/lm/watchdog.log`。
- Mac 睡眠时 watchdog 不运行，所以离开前最好手动 `lm down`。
- dsh 轮询模型列表不算活动，不会让会话一直开着。

### 断线
- 隧道由一个守护进程看着。ssh 退出（断网、换 Wi-Fi、Mac 睡醒）后，它 1 秒后重连，连不上就把间隔逐次加倍，最长 30 秒。链路悄悄断掉时，ssh 约 30 秒内就能发现。日志在 `~/.local/state/lm/tunnel.log`。
- 客户端中途断开（关掉对话、按停止、超时）时，controller 会立刻断开到引擎的连接，GPU 不会继续生成没人要的回答。流式和非流式请求都是这样。

### 配置（`~/.config/lm/config.toml`，都是可选项）
```toml
accelerators = [{gpu = "G4"}, {gpu = "A100", high_mem = true}]
min_vram_gb = 75
assign_retry_minutes = 12
assign_daily_limit = 12
idle_stop_minutes = 60
default_model = "inkling"
```

其他命令：`lm status [--usage]`、`lm load <id>`、`lm unload [id]`、`lm video "提示词" [--seconds 5到15] [--image 首帧.png]`、`lm tunnel`、`lm logs [controller|inkling|qwen|pika]`、`lm dsh`、`lm mcp`。

### 请求怎么路由
- 请求里的 `model` 是哪个在线模型的 id，就发给哪个。模型不在线或不存在时返回 404 并列出在线的模型，不会悄悄换成别的模型。只有一个聊天模型在线时，`model` 可以留空或写 `default`。
- `/v1/messages`（Anthropic 协议）直通给声明了 `anthropic` 的引擎（Basalt、Strata）。llama.cpp 这类引擎会返回明确的错误。
- 视频这类任务走 `POST /jobs {"model": "pika", "input": {"prompt": "...", "seconds": 5}}`，用 `GET /jobs/{id}` 查状态，结果在 `/files/`。帧数由 worker 按自己的 fps 计算，5 到 15 秒以外的时长会被拒绝。

### 给模型用的工具（MCP）
`lm mcp` 是一个 stdio MCP server，工具有 `status`、`load`、`unload`、`up`、`down`、`generate`。每个工具运行的就是对应的 `lm` 命令。`lm connect dsh` 和 `lm claude` 会自动挂上它。其他支持 MCP 的 harness 把命令配成 `lm mcp` 即可。`generate` 要等视频生成完，所以 dsh 那边的工具超时设成了 30 分钟。

### 在 dsh 里使用
`lm dsh` 会带上 `dsh/plugin-colab` 插件启动 dsh web。插件只提供斜杠命令 `/up [model]`、`/down`、`/load <id>`、`/unload [id]`、`/status`，每个都直接运行 `lm`。dsh 的模型选择器通过 `/v1/models` 发现模型，这个接口**只列出在线且 `serves: [chat]` 的模型**。如果别的模型（如 Pika）也在线，名字后面会带 `[+pika]`。

## 结构
```
src/lm/                Mac 端 CLI（colab CLI 封装、ssh 隧道、命令）
remote/bootstrap.sh    在 Colab 上安装 controller 依赖并启动它（可重复执行）；引擎在第一次加载用到它的模型时才安装
remote/models.yaml     注册表：engines（怎么装、怎么起）、models（仓库、路线、caps、显存估计）、rules（调度规则）。加模型或引擎只改它
remote/controller/     FastAPI :9000 —— /v1 按 model 路由的反向代理、/jobs、/admin、调度规则
dsh/plugin-colab/      dsh 插件（斜杠命令，调用 lm）
```
Mac 和 Colab 之间只走 `ssh -L`（以 `colab ssh --proxy-mode` 作为 ProxyCommand），没有公网入口。

## 测试
```bash
uv run pytest
```
