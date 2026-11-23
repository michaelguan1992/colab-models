# 推理加速方案调研

调研日期：2026-11-23。硬件：Colab G4，RTX PRO 6000 Blackwell 96GB，sm_120。

每条结论后面标了依据：**已核实**表示我自己用 `gh` 或 HF API 查过；**有出处**表示调研 agent 给了链接但我没有逐条打开；**未确认**表示没找到一手来源。

## 结论先行

- 你的两个 LLM 都不再只能用 Strata。llama.cpp 主线已经合并了两者的架构：`qwen4exp` 在 2026-08-27 合并（PR #27742），`inkling2_loop` 在 2026-11-20 合并（PR #32694）。**已核实**。
- 两个模型都有非 GGUF 版本，可以走 vLLM 或 SGLang，但只有 Inkling 有去审查版：`OpenUncensored/Inkling-2-Mini-Uncensored`（约 20GB，NVFP4/MXFP8 safetensors）。Qwen3.8 的去审查版只有 GGUF。**已核实**仓库存在。
- Pika 最值得试的三件事是：融合 LoRA 并给 DiT 做 FP8、换注意力后端、拿 SGLang-Diffusion 做对照。SGLang 文档里有 Pika-H3-Flex 在 RTX PRO 6000 上的实测配方。**已核实**页面存在且提到这两个名字。
- sm_120 是共同的风险。大多数引擎的 Blackwell 支持是针对 B200（sm_100）的，sm_120 的证据大多来自论坛帖子和 GitHub issue。

## LLM：推理引擎

| 引擎 | 能加载的格式 | sm_120 | API | 适合你的情况吗 |
|---|---|---|---|---|
| Strata（现在用的） | GGUF + 自有 pack | 能跑（我们在 L4 上验证过，G4 未验证） | OpenAI + Anthropic + Responses | 是。MoE 专家放内存、DFlash2/MTP 都有 |
| llama.cpp（llama-server） | GGUF，含 NVFP4/MXFP4 | 原生 NVFP4，有出处 | OpenAI + Anthropic `/v1/messages`，有出处 | 是。最稳的 GGUF 方案，支持 MTP 草稿模型和 `--n-cpu-moe` |
| ik_llama.cpp | GGUF + 自有量化 | 未确认 | OpenAI | 只在模型放不进 96GB、需要 CPU/GPU 混合跑 MoE 时有意义。两个模型的架构支持都未确认 |
| ExLlamaV3 / TabbyAPI | 只认 EXL3 | 有人用 4 张 RTX PRO 6000 跑，有出处 | OpenAI；Anthropic 未确认 | Qwen3.8 有官方作者 turboderp 做的 EXL3（1.66 到 6.05 bpw），但不是去审查版 |
| vLLM | safetensors（FP8、NVFP4、AWQ）；GGUF 是实验性插件 | 能用，但 NVFP4 MoE 要打补丁，有人遇到 CUDA graph 让整机卡死 | OpenAI + Anthropic | 适合 Inkling 的 safetensors 版。Inkling 需要 vLLM main/nightly，0.35.0 还没发布 |
| SGLang | safetensors（FP8、NVFP4） | NVIDIA 容器写明支持 RTX PRO 6000 | OpenAI + Anthropic，Claude Code 接入文档最全 | 适合 Inkling（有专用镜像 `lmsysorg/sglang:inkling-2-mini`） |
| TensorRT-LLM | NVIDIA checkpoint | 有人在 RTX PRO 6000 上报"Arch not supported" | OpenAI | 不适合 |
| LMDeploy、MLC-LLM、mistral.rs | 各不相同 | 未确认 | 各不相同 | 等 sm_120 支持确认后再看 |

## Basalt（你选中的路线）

以下全部**已核实**，来源是仓库 README、`docs/platforms.md`、`docs/model-file.md` 和 `docs/models/inkling2.md`。

- 仓库是 `jesdga95/basalt`，MIT 许可。它是 Strata 的分支，借用了 Infernix 和 llama.cpp 的内核。仓库 2026-10-06 创建，有 1914 星，最新版 v0.5.6 于 2026-11-23 发布，更新很频繁。
- 主卡只支持 Blackwell sm_120（G4 正好是）。第二张卡可以是 Ada sm_89，但主卡不能。所以 **L4 和 A100 上跑不了**。
- 官方支持你的两个模型，而且默认开启投机解码：
  - Qwen3.8-Flash-Next 用 MTP。
  - Inkling 2 Mini 用 DSpark，包里自带草稿模型。
  - 两个模型都能看图，Qwen3.8 还能看视频。
- 同时提供 OpenAI 和 Anthropic 接口，只监听 127.0.0.1。
- 官方模型包是从**原版**权重做的，不是去审查版：
  - Qwen 是 `Qwen3.8-FN-NVFP4-DQ6_K`。
  - Inkling 是 `Inkling-2-Mini-NVFP4-MX8`，19.6GB。
- 想用去审查版，需要自己转换：
  - `tools/convert/convert.py --gguf ... --mmproj ...` 能把 GGUF 转成 `.basalt`。
  - Inkling 也可以用 `--checkpoint DIR` 从 safetensors 目录转换，正好对上 `OpenUncensored/Inkling-2-Mini-Uncensored`。
  - 你那两个 GGUF 的具体量化类型（`GSQ-RCO` IQ3_XXS、`AD-Q4_K`）能不能转，**未确认**。
- 有 66MB 的 Linux 预编译包。它需要 R580 以上的驱动和 CUDA 13 运行库，自带的 Python 依赖要单独装进 venv。release 里没有校验和文件。
- 它自带多模型路由，以及 `idle-unload`、`idle-sleep`（Inkling 睡眠时只占 20MiB，唤醒约 2 秒）。这些功能和我们的 controller 职责重叠。
- 没有找到在 RTX PRO 6000 上的速度数字，文档里的基准测试都是 RTX 5090 和 5060 Ti 的。

## LLM：每个模型能走哪些路

### Qwen3.8-Flash-Next（SC117 abliterated GGUF）

| 路线 | 格式 | 投机解码 | 去审查 | 备注 |
|---|---|---|---|---|
| Strata（现状） | GGUF + Strata pack | `mtp-q2_0.gguf`、DFlash2（只有 Strata 能读） | 是 | |
| llama.cpp 主线 | 同一份 GGUF，不用改 | 仓库里的 `mtp-...-Q4_0.gguf`（2.2GB），用 `-md` 加载 | 是 | n-gram 分片（28.8GB）在 llama.cpp 里怎么处理未确认 |
| ExLlamaV3 | `turboderp/Qwen3.8-Flash-Next-exl3` | 自带 MTP 头 | 否 | 2 到 4 bpw 能放进 96GB |
| vLLM / SGLang | `nvidia/Qwen3.8-Flash-Next-NVFP4` | 原生 MTP | 否 | 只在 B200/B300 上测过；能否放进 96GB 未确认 |

没有任何 DFlash、EAGLE-3、DSpark 草稿模型适配 Qwen3.8。DFlash2 是 Strata 内部的东西。

### Inkling-2-Mini（AtomicChat Uncensored GGUF）

| 路线 | 格式 | 投机解码 | 去审查 | 备注 |
|---|---|---|---|---|
| Strata（现状） | GGUF | 无（没有 MTP 头） | 是 | 不支持图片输入（我们实测过） |
| llama.cpp 主线 | 同一份 GGUF，需要 b12600 以上 | DSpark 需要 PR #32218，仍未合并，维护者要求修改 | 是 | 模型卡说支持 mmproj 图片输入 |
| vLLM main | `OpenUncensored/Inkling-2-Mini-Uncensored` | DSpark 内置在 checkpoint 里，一次草拟 7 个 token | 是 | 去审查版是否保留了 DSpark 权重未确认 |
| SGLang | 同上 | DSpark 未确认 | 是 | 有专用镜像 |

## LLM：投机解码

| 方法 | 哪些引擎支持 | 有出处的加速 |
|---|---|---|
| MTP 头 | vLLM、llama.cpp、Strata、TensorRT-LLM | 取决于接受率；llama.cpp 有一例全部拒绝反而变慢 |
| EAGLE-3 | vLLM、SGLang、TensorRT-LLM | vLLM 最高约 2.5x |
| DFlash | vLLM、SGLang、TensorRT-LLM | Qwen3-8B 最高 6.1x，单并发 vLLM 约 4x |
| DFlash2 | 只有 Strata | 比 MTP 快 1% 到 12% |
| Suffix decoding | vLLM | 智能体类任务 2.5x 到 5.3x（8B 模型，H100） |
| DSpark | vLLM（Inkling 内置）；llama.cpp 待合并 | 未找到单卡数字 |

注意：这些数字大多来自别的模型和别的卡。对你的模型能有多少提升，只能实测。

## LLM：其他手段

- FP8 KV cache：在 RTX PRO 6000 上能用；有人在低并发下测到慢 10% 到 15%。单用户场景不一定划算。
- Strata 的 KV cache 默认 int8；q4_0 长文档困惑度变差 8% 到 12%。
- FlashAttention 3 没有 Blackwell 内核，FA4 需要 sm_120 没有的指令，社区有移植版。
- SGLang 接 Claude Code 时要设 `CLAUDE_CODE_ATTRIBUTION_HEADER=0`，否则每轮都重新 prefill 全部历史。

## 视频：Pika-H3-Flex

Pika-H3-Flex 是 MiniMax-H3 底模（33B 稠密 DiT）加两个 LoRA。官方只在 H100 上测过，没有提供 NVFP4 或 FP8 的 transformer 权重。之前记录的"NVFP4 变体"在模型卡里找不到，**未确认**。

| 方案 | sm_120 | 效果（有出处） | 画质风险 | 接入难度 |
|---|---|---|---|---|
| 融合 Turbo LoRA（`fuse_lora`） | 是 | 去掉每步的 LoRA 矩阵乘，提升幅度未确认 | 几乎没有 | 一行 |
| DiT 在线 FP8/MXFP8 | 是，SGLang 在 SM120 上默认就用 MXFP8 | 峰值显存 87→53GiB，时间少 5%（B300） | 低到中 | 中：torchao 或 ModelOpt，这个模型上没人试过 |
| 换注意力后端（`flash_4_hub`，再试 `sage_blackwell_hub`） | 是，diffusers 文档列出 SM120 | SageAttention 在 4090 上约 1.7x | FA4 无损；Sage 中等 | 一行 `set_attention_backend` |
| 2 步草稿 LoRA | 是 | 比 4 步快约 1.6x | 高：场景出错、音频变差 | 一行 |
| NVFP4（ModelOpt、社区权重） | 需要 CC≥10.0，sm_120 的 FP4 内核有缺口 | 社区版 5090 上 12.9s vs 18.1s | 中到高 | 高 |
| 缓存（FBCache、MagCache 等） | 是 | 50 步时省 30%，但只有 4 步时收益很小 | 高 | 中 |
| torch.compile | 是 | 低于测量误差，而且会改变输出 | 低 | 低，不值得 |
| 分组 offload | 是 | 768p/15 秒能压到 29 到 32GiB | 无，但更慢 | 低。可以让 Pika 和 Qwen 同时在线，但规则 2 目前不允许 |

可替换的服务栈：

- **SGLang-Diffusion**：原生支持 `Pika-Labs/Pika-H3-Flex`，有 RTX PRO 6000 单卡实测配方（MXFP8 + FA4 窗口注意力）。不能用 `--lora-path` 替换内置的 adapter，所以 Turbo LoRA 可能用不了（未确认）。
- **vLLM-Omni**：支持 MiniMax-H3 底模，有 RTX PRO 6000 的 NVFP4 配置；Pika 的支持未确认。
- **FastVideo、LightX2V**：用的是它们自己的蒸馏版本，不是 Pika 的权重。
- **ComfyUI**：原生支持 Pika，但它是图形界面，不适合当后端服务。

## 对设计的影响

这些方案都能装进 [generalization.md](generalization.md) 里的 `engines` / `models` 结构，不需要新增第三种引擎形状：

- llama.cpp、vLLM、SGLang、ExLlamaV3 都属于 `server` 形状，各写一个 `engines:` 条目。
- 同一个模型可以有多条路线，比如 `inkling` 走 Strata、`inkling-vllm` 走 vLLM。它们放在同一个 `group: llm` 里，规则 1 自动保证同时只有一个在线。
- 量化格式和投机解码属于"模型 + 引擎"的组合，写在模型条目的 `source` 和 `extra` 里，不用单独建一个维度。
- 视频的 FP8、注意力后端、LoRA 融合是 `diffusers` worker 的参数，写在 `params` 里。SGLang-Diffusion 是另一个 `server` 引擎，它的接口是否符合 `/jobs` 需要验证。
