# G4 首次实机验证清单（spike）

代码按文档写好，但下面这些只有在真实 Colab G4 会话上才能确认。每项记下结果；不符时改 `remote/models.yaml` 或 `~/.config/lm/config.toml`，通常不用改代码。

## 1. Colab 会话
```bash
colab new -s localmodels --gpu G4 --high-mem
colab ssh -s localmodels -- bash -lc 'nvidia-smi; df -h /content; free -g'
```
- [ ] Ultra 账号能分到 G4（RTX PRO 6000 96GB）；驱动 ≥ 580（Strata 的 CUDA 13 包需要）
- [ ] `/content` 剩余空间：____ GB（Qwen 约 77GB + Inkling 约 17GB + Pika 子集约 150GB？）
- [ ] 内存：____ GB

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
