# G4 首次实机验证清单（spike）

代码按文档写好，但下面这些只有在真实 Colab G4 会话上才能确认。每项记下结果；不符时改 `remote/models.yaml` 或 `~/.config/lm/config.toml`，通常不用改代码。

## 1. Colab 会话
```bash
colab new -s localmodels --gpu G4
colab ssh -s localmodels -- bash -lc 'nvidia-smi; df -h /content; free -g'
```
- [x] colab CLI 0.8.0 授权成功，余额 1061.29 计算单元（2026-11-22）
- [x] `--gpu G4 --high-mem` 被拒（HTTP 400）。原因是 G4 没有 high-mem 规格，标准规格已有约 185GB 内存；CLI 提示的「无权限」只是它对所有 400 的固定说法。配置里 G4 不带 high_mem
- [ ] 不带 `--high-mem` 的 G4：首次尝试返回 HTTP 503 Service Unavailable（CLI 已自动重试 3 次，每次间隔 10s），可能是暂时没有空闲容量。2026-11-22 再试约 3.5 小时后仍是 503；没有分配到会话，也没有扣计算单元。注意：`COLAB_CLI_RETRY_BACKOFF` 是指数退避的基数（第 n 次等 `backoff × 2^n` 秒），不要设太大
- [x] 2026-11-22 T4 探测：立即分到（us-east1），随即 `colab stop` → 账号没有被限制，G4 的 503 是容量问题（同 google-colab-cli#236）。G4 应低频重试（10–15 分钟一次、每天有上限）
- [ ] 2026-11-22 回退测试：`--gpu A100 --high-mem`（80GB）和 `--gpu A100`（40GB）各试 1 次，都是 503 → 当时高端 GPU 普遍没容量，备选链不能只靠 A100
- [ ] 2026-11-22 23:35 再试 G4、A100 80GB 各 1 次：仍是 503
- [ ] 2026-11-23 02:10、06:11（PST）用 `lm up --no-load` 自动申请两轮（每轮 G4、A100 80GB 各 1 次）：仍全部 503。`lm` 正确把失败识别为「容量」并退出，没有留下会话
- [x] `colab status -s NAME` 在会话不存在时也返回 0 → `lm` 的 `session_active()` 改为解析 `colab sessions` 输出
- [x] 2026-11-23 08:2x 每小时轮询（G4 → A100 80GB → L4）第一轮：G4、A100 仍 503，**L4 分到**（NVIDIA L4 22.5GB）。整个 L4 测试约 10 分钟，花费 0.11 计算单元
- [x] L4 实测：驱动 580.82.07、CUDA 13.0、compute 8.9；`/content` 236GB（overlay，只用了 419MB）；内存 52GB；12 核。G4 的同类数据仍待测
- [ ] Ultra 账号能分到 G4（RTX PRO 6000 96GB）；驱动 ≥ 580（Strata 的 CUDA 13 包需要）
- [ ] `/content` 剩余空间：____ GB（社区数据约 236GB；三个模型合计约 242GB 放不下，controller 会在下载前删除离线模型的权重）
- [ ] 内存：____ GB

- [x] 调度规则 3 在 L4 上生效：加载 Pika 被拒（「pika 预计 69GB + 在线的 inkling ≈ 87GB，超过本机可用 18GB」），没有开始下载
- [x] huggingface_hub 1.x 已不用 hf_transfer → 改用 `HF_XET_HIGH_PERFORMANCE=1`

### 1b. 回退到 A100 80GB 时（`lm up` 分不到 G4 会自动尝试）
- [ ] `nvidia-smi` 显示 80GB，驱动版本 ____。驱动 < 580 时 Strata 的 CUDA 13 包跑不了；v0.2.20 也发布了 `strata-engine-…-cuda12` 包，需要确认安装脚本会不会自动选它
- [ ] Strata 在 sm_80 上能跑 Inkling 和 Qwen
- [ ] Pika 单独在线：实测显存 ____ GB（必须 ≤ 76，即 80 − 4 余量）

## 2. 隧道
```bash
ssh -N -L 9000:127.0.0.1:9000 -o ProxyCommand="colab ssh -s localmodels --proxy-mode" root@colab-localmodels
```
- [x] 能连上（L4，root，无需密钥）。若用户名不是 root、或需要密钥（`colab ssh -i KEY`），写进 config.toml：`ssh_user`、`ssh_extra = ["-i", "~/.ssh/colab"]`
- [x] `colab ssh -- CMD` 退出后，`setsid nohup` 起的 controller 仍在运行
- [x] **Colab 每个运行时只允许一条 SSH 连接**：隧道连着时，再 `colab ssh` 返回 HTTP 429。→ 隧道改为 OpenSSH ControlMaster（`~/.local/state/lm/cm.sock`），`lm logs` 等远程命令走同一条连接；没有隧道时才用 `colab ssh`

## 3. Strata
```bash
lm up --no-load && lm load inkling && lm logs inkling
```
- [x] 安装脚本能跑通（v0.2.20：校验和、签名都通过），装到 `/root/.local/bin/strata`；登录 shell 的 PATH 里没有它，bootstrap 已自行加上
- [x] `strata launch` **会转成后台进程**（runtime → serve.server → strata-inkling 引擎，父进程变成 init），launch 本身随即退出。`launch --stop --port 8081` 能停掉它，卸载后显存回到 0。→ 显存统计改为按端口找监听进程，再统计其整个进程树；实测 Inkling 17.5GB
- [x] `/v1/models`、`/v1/chat/completions` 正常；tool calling 返回标准 `tool_calls`；流式首 token 0.85s，约 30 chunk/s（L4）。冷启动（下载 17GB + 启动）2 分 15 秒，从磁盘重新加载 48 秒
- [ ] Inkling 视觉：Strata 会自动找到同目录的 mmproj（`--vision …mmproj…`，配置里显示 `vision: cpu`），不用另加参数。但发图片返回 400：「this Inkling-2-Mini engine does not read images yet; update Strata or attach the Vision shard」→ 要么等 Strata 新版，要么查 Vision shard 是什么
- [ ] Qwen：能直接读分片目录 `IQ3_XXS/`；n-gram 表（shard 2）是 mmap 而不是整个加载进显存；实测显存 ____ GB
- [ ] `--drafting auto`：Inkling 没有启用投机解码（引擎参数里没有 draft/mtp）。Qwen 待测

## 4. Pika
```bash
lm load inkling && lm load pika && lm logs pika
lm video "a red fox running through snow, crunching footsteps"
```
- [ ] `examples/requirements-diffusers.txt` 能装上（固定的 diffusers commit）
- [ ] `ModularPipeline.from_pretrained(<本地目录>, workflow="fl2va")` 只用 include 列出的子集就能加载
- [ ] Turbo LoRA 用 `load_lora_weights(lora_dir, weight_name="pika_h3_flex_turbo_4step_fl2va.safetensors")` 能加载（模型卡里的 `fl2va/adapter.safetensors` 写法和实际文件对不上）
- [ ] 没下载 `fl2va/` 时 pipeline 也能正常加载（Turbo LoRA 已经融合了 Flex adapter）。如果 `ModularPipeline` 强制要求这个目录，就把 `fl2va/*` 加回 include（多 2.8GB）
- [ ] 下载子集的实际大小 ____ GB（预计 144GB + LoRA 3.9GB）、下载耗时 ____ 分钟
- [ ] pipeline 的输出字段和 `encode_video(...)` 的参数签名 → 按需修改 `remote/controller/pika_worker.py::_generate`
- [ ] Inkling 和 Pika 同时在线时的显存峰值：____ GB（必须低于 96）

## 5. dsh
```bash
lm dsh        # = npx -y @deepseek-ai/dsh web --patch dsh/plugin-colab/cordis.yml
```
- [ ] 通过 npx 安装时，能加载 `.ts` 插件，相对路径 `name: ./index.ts` 能正确解析
- [ ] `inject: ['tools', 'commands']` 写法正确；`/status /up /down /load /unload` 在输入框里能用
- [ ] 斜杠命令返回的 `text` 显示在聊天界面里
- [ ] 两个工具 `colab_status`、`generate_video` 能被模型调用
- [ ] 默认 DSH_HOME 是 `~/.dsh`（不是的话在 config.toml 里设置 `dsh_home`）

## 6. 待测：Pika NVFP4（FastVideo/Pika-H3-Flex-NVFP4-QAD）
目前用的是 BF16 + Turbo（约 148GB 下载）。NVFP4 版下载只要约 68GB：transformer 38.6GB、NVFP4 文本编码器 16.5GB、vae 10.4GB、NVFP4 解码器 1.4GB、audio_vae 0.6GB。但截至 2026-11，它只在 GB200（sm_100）上实测过，社区也几乎没有反馈。

在 G4 上用一个单独的 venv 测，不要动现有环境：
```bash
colab ssh -s localmodels -- bash -lc 'python3 -m venv /content/fv && /content/fv/bin/pip install fastvideo && \
  hf download FastVideo/Pika-H3-Flex-NVFP4-QAD --exclude "transformer_ref/*" --local-dir /content/models/pika-nvfp4'
```
然后按 FastVideo 的 `docs/inference/pika_h3_flex.md` 运行 FL2VA/T2VA 480p 5s，提示词和种子与 BF16 + Turbo 保持一致。

- [ ] 能在 sm_120（RTX PRO 6000）上运行；FA4 能否启用，还是退回到 TORCH_SDPA（文档说 SDPA 在 768p 下很慢，且可能 OOM）
- [ ] 480p 5s 的 GPU 显存峰值 ____ GB。只有 Inkling（约 18GB）加上它不超过约 90GB，才能遵守「Pika 与 <20GB 的 LLM 共存」这条规则
- [ ] 768p 5s 是否 OOM（GB200 上的峰值是 104.6 GiB）
- [ ] 端到端耗时 ____ s，对比 BF16 + Turbo 4 步的 ____ s
- [ ] 画质和音频与 BF16 + Turbo 并排对比（固定种子、3–5 个提示词）
- [ ] 结论：值得切换的话，在 `pika_worker.py` 里新增 FastVideo 后端，并把 models.yaml 换成 NVFP4 仓库（步数固定为 6，不再挂 Turbo LoRA）

## 7. 第 1 阶段真机验证（2026-11-23，A100-SXM4-80GB，compute_cap 8.0）

- 旧 controller 的 `/health` 只返回 `{"ok": true}`；新 `lm` 把它识别为旧版并重新部署，没有误报端口冲突。
- 部署后 `/health` 返回 `{"service": "lm-controller", "version": "19c7209e16fd"}`，与本地 `code_version` 一致。
- 三个模型都选中唯一路线（strata / pika），`compute_cap` 正确读到 8.0。
- Inkling 加载期间 `busy: true`；上线后实测显存 18.4GB；对话正常。
- 版本一致时 `lm up` 跳过部署，在线模型不受影响。
- 强制重新部署（模拟 controller 重启）后：GPU 上没有残留进程，8081 不再监听，`/content/lm/state.json` 清空。旧代码在这里会留下约 18GB 的孤儿 Strata。
- 之后再次加载 Inkling 正常，端口和显存都已释放。
- 修了一个旧 bug：`bootstrap.sh` 在把 `~/.local/bin` 加进 PATH 之前检查 `strata`，导致每次部署都重新下载安装 Strata（90MB）。
- 没在真机上验证：磁盘不够时的切换、HF 受限仓库（只对真实 HF API 跑过预检）、空闲卸载等待进行中的请求、HF token 经 `colab ssh` 首次传输。

## 8. 第 2 阶段真机验证（2026-11-23）

### L4（sm_89）
- llama.cpp 路线（qwen-tiny，b12651 CUDA 12.8 预编译包）：按需安装约 20 秒，下载约 10 秒，对话正常，`/v1/models` 带 `context_length`。
- 第一次跑在 CPU 上（53 tok/s，`nvidia-smi` 看不到进程）：cudart 附加包解压在自己的目录里，不在 `LD_LIBRARY_PATH`。修复后 GPU 显存 1.7GB，生成 251 tok/s。
- 同组切换：加载 inkling 时自动卸载 qwen-tiny（`"unloaded": ["qwen-tiny"]`）。
- Strata 按需安装通过（installing → starting），之后会话被手动 `lm down`，没等到 ready。

### G4（第一次拿到）
- 硬件：NVIDIA RTX PRO 6000 Blackwell Server Edition，驱动 580.82.07，compute_cap 12.0，显存 97887 MiB；磁盘 236GB；内存 176GB；48 核。`hf` 在 `/usr/local/bin/hf`。
- Inkling（strata 路线，AD-Q4_K，`--context 32k --drafting auto`）：下载约 45 秒，启动约 30 秒；显存 18.6GB；261 token 用时 2.7 秒（约 96 tok/s，含 prefill）；工具调用正常。
- 结论：Strata v0.2.20 在 sm_120 上可用；Basalt 路线需要的 G4 条件（驱动 ≥580、CUDA 13）满足。

### G4：Qwen（strata 路线）
- 下载 77GB 约 4 分钟（约 320MB/s），启动约 3 分钟；显存 53.9GB。
- 370 token 用时 1.9 秒（约 194 tok/s，`reasoning_effort: none`）；`high` 产生思考内容、`none` 不产生；工具调用正常。
- Strata 自己往 `/content/models/` 根目录下载了 866MB 的 `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf`，不在模型目录里，磁盘清理算不到它。

### G4：调度与磁盘（真机）
- Qwen 在线（54GB）时加载 Pika 被规则拒绝；之后切回 Inkling 时，规划器为 Pika（148GB）删除了离线的 Qwen 权重，空闲从 138GB 变成 210GB。

### G4：Pika
- 依次修了：Colab 的 python 没有 ensurepip（venv 改用 `--without-pip` 借系统 pip）；venv 继承了 Colab 的 torchao 0.10，peft 要求 >0.16（引擎 pip 列表加 `torchao>=0.16`）；`workflow` 应传给 `load_components`，不是 `from_pretrained`；输出要 `output=["videos","audio","sampling_rate"]` 再交给 `encode_video`。
- 不开 offload 时 transformer（62GB）和文本编码器（62GB）一起上卡，单独加载也 OOM。开 `--cpu-offload` 后峰值 64.4GB。
- 5 秒 480p 带音频：首次请求 498 秒（权重从磁盘懒加载），之后 80 秒；去噪每步约 3 秒。
- diffusers 的 Pika pipeline 固定 6 步（`denoising_step_list`），不接受 `num_inference_steps` / `guidance_scale`；社区 Turbo LoRA 是按 ComfyUI 4 步 euler 调的，现在在官方 6 步调度下运行，属于非标准组合。
- Pika 只接受 5–15 秒。
- 与 Inkling 共存：先上 Inkling 再上 Pika 可以，生成期间 Inkling 照常回答；生成后 Pika 实测占用降到 13.8GB。反过来先上 Pika、用过一次后，它的自动 offload 会占到约 82GB，此时再加载 Inkling 会被规则 3 拒绝。

### G4：Basalt v0.5.6（Inkling）
- 转换：AtomicChat 的 GGUF 不能转（Basalt 的 GGUF 转换只支持 qwen4exp / deepseek41 / hy4flash）。改用 `OpenUncensored/Inkling-2-Mini-Uncensored`（另一份去审查，safetensors，文件结构和官方一致，含 DSpark `mtp.safetensors`），`--checkpoint` 转换 70 秒，18.3GB。转换器不在 release 包里，要从源码 `tools/convert` 取。
- 启动：约 15 秒；显存 19.8GB；`--vision dynamic`（Inkling 的图像编码器在引擎内，`--vision 0` 会被拒绝）。
- `/v1/models` 在加载权重时就返回 200（`status.value: loading`），就绪要等 `loaded`。
- 速度：端到端约 260 tok/s（Strata 约 96 tok/s，约 2.7 倍）；引擎侧 317–407 tok/s，DSpark 接受率 73–96%。
- 工具调用、看图都正常。Basalt 下 Inkling 默认开思考，`reasoning_effort` 有效（none / low / high 的思考量逐级增加）。
- 私有仓库 `mgresearch/colab-models-basalt`（commit `4c0f85f3`）存放转换好的 `Inkling-2-Mini-Uncensored-OU.basalt`（19.6GB）。HF_TOKEN 经 ssh stdin 送到运行时（ControlMaster 路径，37 字节，权限 600）。从仓库下载约 1 分钟，启动约 10 秒，端到端约 248 tok/s。
- 上下文：模型原生 1M（`model_max_length` 1048576；24 层里 20 层是 1024 窗口的滑动注意力，只有 4 层全局）。`--max-context` 32K 时 19.9GB，64K 时 20.9GB，1M 时 54.7GB。1M 下 239K token 的提示 26.6 秒读完，能找回开头埋的口令。代价是 G4 上 Pika 不能再和 Inkling 同时在线。

## 9. 第 4 阶段（G4，2026-11-23）

- 按 `model` 路由：`inkling` 正常。`gpt-4o` 返回 404"没有叫 gpt-4o 的模型（在线的聊天模型：inkling）"。离线的 `qwen` 返回 404 并提示 `/load qwen`。不带 `model` 时路由到唯一在线的 Inkling，并补上 Basalt 自己的模型名。第一次部署漏了补名这一步，Basalt 回了 "model is required"，已修。
- `/v1/messages` 直通 Basalt：工具调用返回 `tool_use get_weather {"city": "Paris"}`。流式 SSE 正常，先是 thinking 块，再是 text。
- `/jobs`：没有任务模型在线时返回 503 并提示 `/load pika`。卸载 Inkling、单独加载 Pika 后：
  - 经 MCP 工具 `generate`（MCP 客户端走 stdio 调 `lm mcp`）生成 5 秒视频：432 秒，是加载后的第一次请求，权重懒加载。得到 H.264 832×480，5.17 秒，带 AAC 音轨。
  - 再用 `lm video --seconds 6` 生成：92 秒。
  - 3 秒和 20 秒被 worker 拒绝。原因要用 MCP 的 `ToolError` 抛出才能传给模型，否则客户端只看到 "Error executing tool generate"，已修。
  - 没能让 LLM 在 harness 里调用 `generate`：G4 上 Inkling（1M 上下文）和 Pika 放不下，而能和 Pika 共存的 qwen-tiny 只有 8K 上下文，装不下 dsh 或 Claude Code 的系统提示。工具本身和 harness 的接线都已分别验证。
- Claude Code 2.1.330（`lm claude -p`）：经 Inkling 回答正常，调用 MCP 工具 `mcp__lm__status` 也正常。自定义模型 id 要设 `CLAUDE_CODE_MAX_CONTEXT_TOKENS`，否则它按 1M 估算并打警告。`lm claude` 现在按 caps 设置这个值。
- dsh：`lm connect dsh` 写入后，正在运行的 dsh 热加载了 provider 和 MCP server。新会话里能看到 `mcp__lm__status` 等 6 个工具，并成功调用。MCP 管理器在 patch 里的行 id 是 `mcp-manager`，不是它 README 示例里的 `mcp`。dsh 的 MCP 工具默认超时 60 秒，不够 `generate` 用，所以写入了 `toolCallTimeoutMs: 1800000`。

## 10. 第 5 阶段：隧道和连接（G4，2026-11-23）

只做了隧道和连接。`Provider` 接口和远端自释放按你的决定不做。

- 隧道重连：杀掉 ssh 进程后，守护进程 1 秒后重开 ssh，3.3 秒后 `/health` 恢复。之前要等 watchdog 发现连不上超过 2 分钟才重连。
- 客户端断开：让 Inkling 写 8000 token 的长文，客户端 4 秒后放弃。非流式和流式两种请求，GPU 利用率都从 99% 在 5 秒内降到 0，`inflight.requests` 回到 0。修之前，非流式请求断开后引擎会继续生成到结束；流式请求本来就会停。
- 远端自释放的可行性（没有实现，记下备查）：`google.colab.runtime.unassign()` 只是向环境变量 `TBE_RUNTIME_ADDR` 指向的地址发 `POST /unassign`，不依赖 notebook 前端，所以从 ssh 里也应该能调用。没有实际调用过。

## 11. 第 3 阶段：Qwen 的 Basalt 包和 Pika 加速（2026-11-23）

### 空闲判断漏掉了 ssh 上手动跑的任务
- 第一次做这一阶段时，G4 在 19:56 被 Mac 上的 watchdog 按"空闲 60 分钟"停掉。当时 FP8 基准和 Qwen 草稿层下载都在跑，但它们是从 ssh 直接启动的，不经过 controller，controller 收不到请求，就把会话当成了空闲。运行时磁盘不保留，Pika 缓存（约 141GB）、下载到一半的草稿层和基准结果都丢了。
- 修法：controller 在空闲检查时把两种外部任务算作活动。一是 GPU 上不属于任何已加载模型的进程（按 nvidia-smi 的 pid 和各后端的进程树比对）；二是 `/content/lm/holds/` 下有文件，给下载、转换这类不用 GPU 的任务用。`/admin/status` 和 `lm status` 会列出这些任务。`remote/bench/phase3.sh` 的每一步开始时建 hold 文件，结束时删掉。在 A100 上实测：拉取草稿层期间 `external_work` 是 `['hold mtp']`，`idle_seconds` 一直在 30 以内。
- 另一个坑：从非交互 shell 运行 `lm up` 时，`~/.zshrc` 里的 `HF_TOKEN` 不会加载，token 没送到运行时，之前也没有任何提示。现在 deploy 找不到 token 时会打印警告。

### Qwen 转成 Basalt 包（A100 上完成；转换只用 CPU 和磁盘）
- 草稿层：`mtp_fetch.py fetch` 加 `verify`，用范围请求只取官方 Qwen3.8 的 MTP 张量，共 4.9GB；另外取 `jesdga/Qwen3.8-Flash-Next-Basalt` 里调好的 `convert/mtp/rt/dense.bin`（111MB）。
- GGUF：SC117 的 `IQ3_XXS/*` 加 `mmproj-Qwen3.8-Flash-Next-BF16.gguf`。注意 `hf download` 同时给了文件名和 `--include` 时会忽略 `--include`，要分两次下载。
- 转换 658 秒，输出 74.7GiB（80.2GB）一个文件，17 个段：experts 40.0GiB，gguf（n-gram 表）30.9GiB，dense 1.4GiB，mtp 0.77GiB，vision 0.85GiB。转换器提醒自己转的包"明显慢于官方包"，所以要实测。
- 原生上下文 262144（GGUF 里的 `qwen4exp.context_length`）。
- 上传到 `mgresearch/colab-models-basalt`，commit `7f958d1b`，文件名 `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.basalt`。Xet 去重后实际只传了 10.3GB。

### Qwen：Basalt 和 Strata 在 G4 上对比
- 下载私有包：controller 的 `hf download` 开着 `HF_XET_HIGH_PERFORMANCE`，HF 的 Xet CDN 两次都在几分钟内返回 429，而且每次都卡在同一个块上；不开高性能模式也一样。失败后半截文件会被删掉，所以重试会从头开始。改成固定 4 个连接（`HF_XET_FIXED_DOWNLOAD_CONCURRENCY=4`）、Xet 自己重试 30 次（`HF_XET_CLIENT_RETRY_MAX_ATTEMPTS=30`）之后，80GB 一次下完。同一时间下载公开的 SC117 GGUF（77GB）没有遇到 429。controller 现在第一次用高性能模式，失败后按这组设置最多再试 3 次。超过大小上限的文件不能关掉 Xet 走普通 HTTP（`hf` 会直接拒绝）。
- 另一个 bug：模型已经在线时，`lm load qwen --route basalt` 直接回"已在线"，不会切换路线。已修：指定了不同路线时，先确认新路线可用，再停掉旧的、加载新的。
- 显存：Basalt 加载后 46.0GB，读完 150K token 的提示后 49.6GB；Strata（32K 上下文）53.9GB。
- 生成速度（同 4 个提示词各跑两轮，`remote/bench/chat_bench.py`）：Basalt 230–410 tok/s，平均约 294；Strata 232–372 tok/s，平均约 289。在误差范围内一样快，没有看到转换器警告的"自己转的包明显更慢"。首 token 都是 0.27–0.5 秒。Strata 两轮的输出逐 token 相同，Basalt 每轮不同。
- 读长提示（`remote/bench/prefill_bench.py`，开头埋一个口令）：约 24K token，Strata 2.7–2.8 秒；Basalt 第一次 31 秒（像是一次性的预热），第二次 4.1 秒。150K token 只有 Basalt 能跑（Strata 只开了 32K），14.6 秒，口令找回正确。
- 回答：代码、中文解释、算术（15:35 到达）两边都对。
- 结论：G4 上 Qwen 默认走 Basalt。好处是 262K 上下文（dsh 的系统提示就有约 23K token）、少用约 8GB 显存、能看图；代价是第一次长提示要预热，包放在私有仓库，下载要 HF_TOKEN。

### Pika：官方 adapter、FP8、常驻显存（G4）
- 默认改用官方 `fl2va/adapter.safetensors`。它是按 pipeline 自带的 6 步调度蒸馏的；社区 Turbo LoRA 是按 4 步、shift 6 调的，而 diffusers 的 pipeline 只跑自己的 6 步表。之前在 G4 上测过三组（offload，bf16，5 秒）：Turbo LoRA + 6 步 77.4 秒，官方 adapter + 6 步 76.0 秒，Turbo LoRA + 4 步 70.1 秒。速度差不多，所以选两边配套的组合。worker 去掉了 pipeline 根本不认的 `--steps` / `--guidance`，`/jobs` 也不再接受 `steps`。注册表里不再有模型用 `lora` 字段，这套机制已删除。
- torchao 的 PerRow FP8 只接受 bf16 权重，模型里有非 bf16 的 Linear，直接量化会报错；现在只量化 bf16 的 Linear（`filter_fn`）。
- 5 秒 480p，种子 42（`remote/bench/pika_bench.py`）：

| 方案 | 热启动每段 | 第一次请求 | 峰值显存 | 加载 |
|---|---|---|---|---|
| bf16 + offload（原来的默认） | 78 秒 | 354 秒 | 78.9GB（生成后仍占着） | 很快，权重在第一次请求时才读 |
| FP8 + 融合 adapter + 全部常驻 | 35–37 秒 | 37 秒 | 77.5GB | 约 400 秒（先读权重再量化） |
| FP8 + 融合 adapter + offload | 55 秒 | 90 秒 | 45.4GB | 约 500 秒 |

- 画质：第 0、60、120 帧对比（`docs/pika-fp8-frames.png`，上到下：bf16、FP8 常驻、FP8 offload），构图和动作和 bf16 接近，两种 FP8 几乎一样，没看到伪影。
- 经 controller 端到端（`lm load pika` 后 `lm video`）：加载 539 秒（权重已在磁盘上），5 秒视频 39 秒和 35 秒，10 秒视频 83 秒。10 秒视频之后进程占 87.3GB，注册表按 88GB 估。
- 结论：sm_89 以上（G4、L4、H100）走 `fp8` 路线：融合、FP8、常驻，快 2.2 倍，而且第一次请求不再等 6 分钟。A100（sm_80）没有 FP8，走 `bf16` 路线（offload）。显存占用和原来差不多，所以 Pika 在 G4 上仍然只能单独在线。

## 12. MOSS-Image 选型（L4，2026-11-24）

在单独的 L4 会话 `moss` 上做（不需要 G4；A100 和 H100 当时都申请不到）。脚本是 `remote/bench/moss_fetch.sh`、`moss_bench.py`、`moss_runs.sh`。对比图在 `docs/moss/compare-l4.jpg`。

- 底座：`OpenMOSS-Team/MOSS-Image-SFT`。transformer 是 5B（仓库里存 fp32，20.3GB），文本编码器 Qwen3.5-9B（16.8GB），FLUX.2 VAE。用官方 `custom_pipeline="pipeline_moss_image"`，Colab 自带的 diffusers 0.42、transformers 5.22 就够。这个 pipeline 没有 `load_lora_weights`，LoRA 用 kohya 格式的 key 自己包一层，fp32 不合并（作者说蒸馏的增量比 bf16 的舍入误差还小，合并进 bf16 会丢掉）。
- 4 个提示词（英文菜单、中文招牌"深夜食堂"、带数字的信息图、人像），同一个种子，1024²，L4 上开 `enable_model_cpu_offload`，峰值 15.7GiB。下面的时间去掉了每组第一张（含权重加载）：

| 方案 | 步数 / CFG | 每张 | 质量 |
|---|---|---|---|
| base50（官方默认） | 50 / 3.0 | 约 176 秒 | 皮肤质感和光线最自然；但信息图把 38% 画成了 33% |
| lode8（lodestones `lora.safetensors`） | 8 / 1.0 | 约 44 秒 | 数字对；图例重复了一个 Office；人像最接近 base50 |
| lode8l（`lora_latest` 0.7） | 8 / 1.0 | 约 44 秒 | 图例出现乱码"Reed" |
| akh4（akhaliq v2） | 4 / 1.0 | 约 35 秒 | 数字对；小字有乱码（"Fendnea"）；人像更平滑 |
| akh3（akhaliq 3 步） | 3 / 1.0 | 约 32 秒 | 小字乱码最多，饼图多一道线 |

- 菜单和"深夜食堂"五组都写对了。蒸馏版颜色更暖、更饱和，人像更像图库照片。
- 4 步只比 8 步快 9 秒，因为 L4 上每张图大约 25 秒花在文本编码器和 DiT 在 CPU 和 GPU 之间来回搬（每步约 2.3 秒）。G4 上全部常驻显存时这部分会没有，步数的差别会更明显，没有实测。
- FP8 文本编码器加全部常驻：L4 的 22GB 在 VAE 解码时 OOM，没有结果。要在 G4 上再测。
- 结论：默认用 lode8（8 步，CFG 1，fp32 LoRA 不合并），质量最接近官方 50 步，快约 4 倍（L4、offload）。akh4 可以做快速档。

## 13. Sulphur-3 的步数对比（A100，2026-11-25）

`sulphur-comfy` 的 `resident` 路线：`s3.safetensors`（bf16，40.2GB）、NVFP4 文本编码器、int8 视频 VAE，全部常驻。脚本是 `remote/bench/sulphur_bench.py`，节点图照官方 Pika 文生视频模板写成 API 格式。文生视频，1344×768，5 秒（124 帧，带音轨），同一个提示词（雪地里的狐狸）和种子。每组跑两次，第二次换种子，记热启动时间。

| 方案 | LoRA | 采样器 / 调度 | 步数 / CFG | 第一次 | 热启动 |
|---|---|---|---|---|---|
| turbo4 | `minimax_h3_fl2v_turbo_4step_v1.0_768p` | euler / simple | 4 / 无 | 100 秒（含加载） | 83 秒 |
| pikaturbo4 | `pika_h3_flex_turbo_4step_fl2va` | euler / simple | 4 / 无 | 84 秒 | 83 秒 |
| pika6 | `pika_h3_flex_fl2va_v1` | lcm / 模板的 sigma 表 | 6 / 无 | 118 秒 | 117 秒 |
| turbo8 | `minimax_h3_fl2v_turbo_8step_v1.0` | euler / simple | 8 / 无 | 152 秒 | 152 秒 |
| base20 | 无 | euler / beta | 20 / 无 | 358 秒 | 358 秒 |
| base20cfg | 无 | euler / beta | 20 / 4.0 | 702 秒 | 702 秒 |

- 每次前向约 17 秒，文本编码加解码等固定开销约 14 秒。带 CFG 时每步两次前向，时间翻倍。
- 显存：峰值 63.4GB，生成后占 54.4GB。
- 画面：第 10、60、110 帧对比在 `docs/sulphur-a100-frames.png`（从上到下按表格顺序）。六组都连贯、符合提示词，缩略图上看不出蒸馏版有伪影；不挂 LoRA 的两组构图不同（侧面走过，机位更远）。只测了这一个提示词，没有测 Sulphur 微调针对的内容，也没有逐帧看细节和听音轨。
- 结论：4 步 Turbo 比 20 步快 4.3 倍，比 20 步带 CFG 快 8.5 倍。两个 4 步 LoRA 一样快，lightx2v 的不带 Pika 的运动风格。
- L4 上只确认了 `offload` 路线能跑：`quantized/s3_w4a8.safetensors` 在 ComfyUI v0.46.3 里能加载，832×480 的 4 步首次 159 秒。之后会话就换成了 A100，没有热启动数字。G4 没测。

### 人物打斗场面（G4，2026-11-25）

同一条路线和脚本，这次分到的是 G4（RTX PRO 6000 Blackwell，95.6GB）。提示词是雨夜巷子里两人对打（红夹克女子直拳、摆拳、转身后踢，灰色连帽衫男子格挡、下潜、扫腿，手持环绕镜头）。1344×768，5 秒，种子 42，每组只跑一次，所以 turbo4 的时间含加载。

| 方案 | 步数 / CFG | 耗时 | 画面（第 8、30、52、74、96、118 帧，`docs/sulphur-fight-g4-frames.jpg`，从上到下） |
|---|---|---|---|
| turbo4 | 4 / 无 | 55 秒（含加载） | 两人始终在画内，出拳、格挡、踢腿、下潜都有；肢体和衣服稳定，快动作处有正常的运动模糊 |
| pikaturbo4 | 4 / 无 | 46 秒 | 和 turbo4 接近，最后才出踢腿；机位移动更多 |
| pika6 | 6 / 无 | 67 秒 | 画面偏灰、偏电影感；最后一帧两人身体缠在一起，分不清肢体 |
| turbo8 | 8 / 无 | 87 秒 | 女子双手变成白色团块（像缠了绷带），其余稳定 |
| base20 | 20 / 无 | 207 秒 | 手机实拍感；男子的脸和发型前后不一致 |
| base20cfg | 20 / 4.0 | 407 秒 | 雨和镜头运动最强，动作最激烈；镜头绕到背面后两人的衣服细节变了（夹克背后多出字，卫衣变浅） |

- 每次前向约 10 秒，G4 比 A100 快约 1.75 倍。峰值显存 63.6GB。
- 这个场景里 4 步 Turbo 没有比 20 步差：肢体最干净的反而是两组 4 步。20 步带 CFG 动作幅度最大，但人物一致性最差。
- 没有一组完整做出提示词里的全部动作（扫腿和跳起躲避都没有）。只看了抽帧，没有听音轨；每组只有一个种子，上面的差别有一部分可能是种子造成的。

### 另外五个场景（G4，2026-11-25）

只比 turbo4 和 base20cfg，条件同上。五个场景的耗时几乎一样：turbo4 46–47 秒，base20cfg 406–407 秒。抽帧（第 8、45、82、118 帧）在 `docs/sulphur-more-g4-frames.jpg`，每个场景两行，上 turbo4、下 base20cfg。

| 场景 | turbo4 | base20cfg |
|---|---|---|
| 咖啡馆对话特写 | 脸稳定，有口型和撩头发；皮肤偏光滑 | 皮肤纹理更真实，但整体偏橙红、过饱和 |
| 街舞（全身） | 脚步、转身、单膝落地都有，镜子里的倒影基本对得上 | 同样完整，多了一顶帽子 |
| 海边情侣 | 牵手、说笑、最后抱住；没有抱起转圈 | 接近，同样没有抱起转圈 |
| 滑板下台阶 | 滑下台阶后人滑出画面，朋友举手；没有明显的跳 | 中途切了镜头，最后滑板的人换成了另一个 |
| 切菜下锅 | 彩椒、洋葱、下锅三步都对，手和刀正常 | 颜色过饱和，彩椒片叠成不合理的一摞，手直接按进热锅里 |

- 五个场景里 turbo4 都不比 base20cfg 差，滑板和切菜明显更好。base20cfg 多处偏色、过饱和，CFG 4 对这个模型可能偏高（RL LoRA 的训练元数据里有引导蒸馏），没有试更低的 CFG。
- 仍然每个场景一个种子，只看了抽帧，没有听音轨，对话场景的台词有没有说对没有确认。

### 同样五个场景加上两种 Pika 加速（G4，2026-11-26）

同样的提示词、种子和分辨率，补跑 pikaturbo4 和 pika6。耗时：pikaturbo4 46–47 秒（第一条含加载 55 秒），pika6 66–67 秒，显存峰值 63.6GB。四行对比的抽帧（自上而下 turbo4、pikaturbo4、pika6、base20cfg）在 `outputs/sulphur-more/<场景>/compare4-frames.jpg`，其中海边和滑板两张放在 `docs/sulphur-more-4way-beach.jpg`、`docs/sulphur-more-4way-skate.jpg`。

| 场景 | turbo4 | pikaturbo4 | pika6 |
|---|---|---|---|
| 咖啡馆对话特写 | 脸稳定，撩头发出现得晚 | 脸稳定，撩头发清楚，手和戒指正常 | 脸稳定，表情变化最多，手只露出一部分 |
| 街舞（全身） | 动作完整，结尾是双膝跪地；多面镜子里倒影较乱 | 动作完整，结尾也是双膝跪地；画面最干净 | 结尾是单膝落地，最贴提示词；多了一顶帽子 |
| 海边情侣 | 牵手、说笑、最后抱住，没有抱起 | 同左，但女方衣服变成了短上衣加牛仔短裤 | 唯一一个把人抱起来的 |
| 滑板下台阶 | 中途切到近景，结尾主角只剩远处一个小点 | 中途切到另一个角度，结尾滑走、朋友在旁 | 中途切到近景，结尾滑走、朋友举手 |
| 切菜下锅 | 三步都对 | 三步都对 | 三步都对，但开头的彩椒像两个小番茄，锅里彩椒偏少 |

- 三种少步数方案画质接近，都明显好于 base20cfg。pikaturbo4 和 turbo4 一样快，画面最干净；pika6 慢约 40%（多两步），但街舞和海边两个场景的动作更贴提示词，代价是偶尔多出提示词里没有的东西（帽子）。
- 滑板这一条三种方案都切了镜头，没有一个拍出完整的"跳下台阶"。
- 仍然每个场景一个种子，只看了抽帧，没有听音轨；上面的差异有多少来自种子、多少来自方案，这个样本量分不出来。

当时先定的是 Pika Turbo 4 步完整版，最终方案见本节末尾。

### shift 6 对 12，精简版和 8 步 HD（G4，2026-11-26）

Pika Turbo 的 README 建议 FL2VA 用 flow shift 6，而上面所有跑分用的是模型在 ComfyUI 里的默认值 12。同样五个场景、同样的种子，用 ManualSigmas 写出两种 shift 的 sigma 表，各跑三个 LoRA：完整 4 步版（pt4）、rank 32 精简版（lite4，0.31GB）、8 步 HD（hd8）。耗时：4 步的四组都是 46–47 秒，hd8 两组 86 秒，显存峰值 63.6GB。抽帧在 `outputs/sulphur-shift/<场景>/compare6-frames.jpg`，自上而下 pt4_s12、pt4_s6、lite4_s12、lite4_s6、hd8_s12、hd8_s6。

- **预设的 sigma 表与实测一致**：pt4_s12（手写表）和前一节的 pikaturbo4（BasicScheduler simple 4 步）逐帧比较，五个场景的平均 PSNR 是 31–36 dB，构图和动作相同，差别在编码和浮点误差的量级。预设「Sulphur 文生视频（4 步）」在 ComfyUI 里打开没有报缺文件，直接运行 47 秒出片。
- **精简版和完整版几乎一样**：同一 shift 下两者构图、动作、人物都对得上，抽帧里只有背景细节不同。
- **shift 6 对 12 差别不大**：同一个 LoRA 换 shift，画面是同一条视频的小变体。街舞一条 shift 6 落成了提示词要的单膝（shift 12 是双膝），海边一条 shift 6 把女方头发变成了马尾；其余看不出谁更好。
- **8 步 HD 没有体现出优势**：慢一倍，抽帧的缩略图上看不出更多细节；切菜一条反而更差（shift 12 洋葱变成紫色，shift 6 到结尾也没有下锅）。它的构图和 4 步版不同，不是同一条视频的精修版。
- 海边一条六组都没有抱起来，滑板一条六组都切了镜头。
- 仍然每个场景一个种子，只看了缩小的抽帧，没有听音轨；README 说音频在 6 步以下会变差，这一点没有验证。

用户先凭缩小的并排视频选了 8 步 HD、shift 6，之后看了全分辨率、带音轨的原始视频，认为完整版 4 步、shift 12 最均衡。

**最终方案：Pika Turbo 4 步（完整版，3.9GB），shift 12。** `sulphur-comfy` 路由只下载这一个 LoRA，预设「Sulphur 文生视频（4 步）」是官方 Pika 文生视频模板换掉模型、LoRA、采样器（euler）和 sigma 表（`1.0, 0.972973, 0.923077, 0.8, 0.0`）。G4 上一条 5 秒视频 46–47 秒，A100 上 83 秒。
