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
