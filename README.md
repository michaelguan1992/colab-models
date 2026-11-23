# LocalModels：Colab G4 上的自托管模型，一键启停

在 Google Colab G4 运行时（RTX PRO 6000，96GB）上跑模型，在 Mac 上用 DeepSeek Harness（dsh）调用。

| id | 模型 | 引擎 | 显存估计 |
|---|---|---|---|
| `inkling` | AtomicChat/Inkling-2-Mini-Uncensored-GGUF（Q4_K） | Strata | ~18GB |
| `qwen` | SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF（IQ3_XXS） | Strata | ~50GB |
| `pika` | Pika-H3-Flex + Turbo LoRA（4 步，**视频+音频**） | diffusers | ~69GB |

**调度规则**（由 controller 强制执行，`remote/controller/policy.py`）
1. 同一时间最多一个 LLM 在线。加载另一个 LLM 时，会先把当前的卸载掉。
2. 只有在线 LLM 的**实测**显存低于 20GB 时，`pika` 才能上线。所以 Inkling 可以和 Pika 同时在线，Qwen 不行。
3. 45 分钟没有请求，自动卸载所有模型（`idle_unload_minutes`）。

## 一次性准备
```bash
uv tool install google-colab-cli      # 然后按提示在浏览器里授权
```
```bash
uv tool install -e .                  # 在本仓库里执行，安装 lm 命令
```
在 dsh 里添加 provider（任选一种）：
- 界面操作：Settings → Models → Add model provider → Custom model API。Provider ID 填 `colab`，Base URL 填 `http://127.0.0.1:9000/v1`，协议选 OpenAI Chat Completions，key 随便填。
- 或者运行 `lm setup-dsh`：合并写入 `~/.dsh/profiles/web/cordis.patch.yml`，写之前会先备份。

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
- `lm up --wait`：每隔约 `assign_retry_minutes`（默认 12 分钟，±20% 随机）再申请一轮。
- 每天最多发 `assign_daily_limit`（默认 12）次申请，多次运行 `lm up` 共用这个额度。短时间内大量重试曾导致别的账号被限制（colabtools#6206）。
- 分到的显存小于 `min_vram_gb`（默认 75）就立刻停掉，换下一种。
- 在 A100 80GB 上，Inkling 和 Pika 不能同时在线（规则 3：在线模型加目标模型的显存，不能超过 GPU 总显存减余量）。

### 计费与自动关机
- controller 空闲 `idle_unload_minutes`（45）分钟后只卸载模型，Colab 会话仍按小时计费（G4 约 8.9 计算单元/小时）。
- `lm up` 会在 Mac 上起一个 watchdog，空闲 `idle_stop_minutes`（默认 60）分钟后自动 `lm down`。日志在 `~/.local/state/lm/watchdog.log`。
- Mac 睡眠时 watchdog 不运行，所以离开前最好手动 `lm down`。
- dsh 轮询模型列表不算活动，不会让会话一直开着。

### 配置（`~/.config/lm/config.toml`，都是可选项）
```toml
accelerators = [{gpu = "G4"}, {gpu = "A100", high_mem = true}]
min_vram_gb = 75
assign_retry_minutes = 12
assign_daily_limit = 12
idle_stop_minutes = 60
default_model = "inkling"
```

其他命令：`lm status [--usage]`、`lm load <id>`、`lm unload [id]`、`lm video "提示词" [--image 首帧.png]`、`lm tunnel`、`lm logs [controller|inkling|qwen|pika]`、`lm dsh`。

### 在 dsh 里使用
`lm dsh` 会带上 `dsh/plugin-colab` 插件启动 dsh web。插件提供：
- 斜杠命令：`/up [model]`、`/down`、`/load <id>`、`/unload [id]`、`/status`
- 给模型调用的工具：`colab_status`、`generate_video`（生成的 mp4 保存到 `./outputs/`）
- 在线状态的显示：dsh 的模型选择器通过 `/v1/models` 发现模型，而这个接口**只列出在线的 LLM**；如果 Pika 也在线，名字后面会带 `[+pika]`。完整状态（下载进度、显存）用 `/status` 查看。

## 结构
```
src/lm/                Mac 端 CLI（colab CLI 封装、ssh 隧道、命令）
remote/bootstrap.sh    在 Colab 上安装依赖和 Strata，并启动 controller（可重复执行）
remote/models.yaml     模型注册表：仓库、量化文件、启动参数、显存估计
remote/controller/     FastAPI :9000 —— /v1 反向代理、/admin、/video、调度规则
dsh/plugin-colab/      dsh 插件（斜杠命令 + 工具）
```
Mac 和 Colab 之间只走 `ssh -L`（以 `colab ssh --proxy-mode` 作为 ProxyCommand），没有公网入口。

## 测试
```bash
uv run pytest
```
