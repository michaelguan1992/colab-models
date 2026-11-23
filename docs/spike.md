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
- [x] `colab status -s NAME` 在会话不存在时也返回 0 → `lm` 的 `session_active()` 改为解析 `colab sessions` 输出
- [ ] Ultra 账号能分到 G4（RTX PRO 6000 96GB）；驱动 ≥ 580（Strata 的 CUDA 13 包需要）
- [ ] `/content` 剩余空间：____ GB（社区数据约 236GB；三个模型合计约 242GB 放不下，controller 会在下载前删除离线模型的权重）
- [ ] 内存：____ GB

### 1b. 回退到 A100 80GB 时（`lm up` 分不到 G4 会自动尝试）
- [ ] `nvidia-smi` 显示 80GB，驱动版本 ____。驱动 < 580 时 Strata 的 CUDA 13 包跑不了；v0.2.20 也发布了 `strata-engine-…-cuda12` 包，需要确认安装脚本会不会自动选它
- [ ] Strata 在 sm_80 上能跑 Inkling 和 Qwen
- [ ] Pika 单独在线：实测显存 ____ GB（必须 ≤ 76，即 80 − 4 余量）

## 2. 隧道
```bash
ssh -N -L 9000:127.0.0.1:9000 -o ProxyCommand="colab ssh -s localmodels --proxy-mode" root@colab-localmodels
```
- [ ] 能连上。若用户名不是 root、或需要密钥（`colab ssh -i KEY`），写进 config.toml：`ssh_user`、`ssh_extra = ["-i", "~/.ssh/colab"]`
- [ ] `colab ssh -- CMD` 退出后，`setsid nohup` 起的 controller 仍在运行

## 3. Strata
```bash
lm up --no-load && lm load inkling && lm logs inkling
```
- [ ] 安装脚本在 Colab（Linux x86_64、root）能跑通，`strata` 在 PATH 里
- [ ] `strata launch --model-path … --port 8081 --access local --no-api-key --yes --no-tui` 留在前台运行（如果它会自己转成后台进程，`launch --stop --port` 必须能停掉它）
- [ ] 提供 `/v1/models` 和 `/v1/chat/completions`；带 tool calling 的请求能正常返回
- [ ] Inkling：视觉输入用的 mmproj 参数名 → 加到 `models.inkling.extra`
- [ ] Qwen：能直接读分片目录 `IQ3_XXS/`；n-gram 表（shard 2）是 mmap 而不是整个加载进显存；实测显存 ____ GB
- [ ] `--drafting auto` 对两个模型分别选中了什么（mtp / dflash2）

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
