# 通用化设计：任意模型、任意 harness

状态：草案，待确认后分阶段实施。

## 目标

现在加一个模型或接一个新前端，要同时改 `models.yaml`、`policy.py`、`manager.py`、`app.py`、`cli.py` 和 dsh 插件。目标是：

- 加模型只改 `models.yaml`，不写 Python。
- 接新 harness 只写一个几十行的配置生成函数，或者什么都不写（走 MCP）。
- 你定的调度规则（同时只一个 LLM、Pika 只在 LLM < 20GB 时上线）变成配置里的数据，不再是代码常量。
- 已知的边界情况在设计里就有归宿，而不是出事后打补丁。

## 现在哪里写死了

两个只读排查把耦合点列全了，这里只列决定设计的几处：

- `kind` 同时表示"是什么"和"用什么引擎跑"。`manager.py:31-35` 用 `{"llm": StrataBackend, "video": PikaBackend}[kind]`，`policy.py:109` 拒绝任何其他 kind。图像、语音、embedding 模型完全加不进来。
- 规则 1、2 写在 `policy.py` 里，20GB 是常量 `VIDEO_LLM_LIMIT_GB`（`policy.py:17`）。
- 代理只认一个在线 LLM，并且把请求里的 `model` 字段无条件改写成它（`app.py:99-104`）。前端选哪个模型都会被悄悄送到同一个。
- 视频只有 Pika 专用的 `/video/*` 路由，24fps 在 `cli.py`、`index.ts`、`pika_worker.py` 各写了一次。
- 模型 id 硬编码在 `cmd_setup_dsh`（`cli.py:359-360`）、dsh 插件提示、`app.py:96` 的报错文案和 `default_model` 里。改名要改六处。
- 所有 harness 集成都写在 dsh 专用代码里：provider 配置、slash 命令、`generate_video` 工具、启动器。接第二个 harness 要全部重写一遍。
- 端口：controller 9000 在 `bootstrap.sh:31` 写死（`remote_port` 配置是死的），所有 Strata 模型共用 8081，插件和 `cordis.yml` 各自写死 9000。

## 核心数据结构

按 Model the Domain，把领域装进一个注册表，代码只解释它。新的 `models.yaml` 分三块：

```yaml
engines:
  strata:
    shape: server            # 起一个 HTTP 服务，controller 反代
    apis: [openai, anthropic]
    install: {script: install-strata.sh, version: v0.2.20}
    launch: ["strata", "launch", "--model-path", "{model_path}", "--port", "{port}", ...]
    ready: /v1/models
    stop: ["strata", "launch", "--stop", "--port", "{port}"]   # 守护进程化的引擎才需要
  llama-cpp:
    shape: server
    apis: [openai]
    install: {pip: ["llama-cpp-python[server]==…"], venv: true}
    launch: ["{venv}/bin/python", "-m", "llama_cpp.server", "--model", "{model_path}", "--port", "{port}"]
    ready: /v1/models
  diffusers:
    shape: jobs              # 长任务：提交、轮询、取文件
    install: {pip_file: "{repo_dir}/examples/requirements-diffusers.txt", venv: true}
    launch: ["{venv}/bin/python", "-m", "controller.workers.diffusers", ...]

models:
  inkling:
    engine: strata
    group: llm
    source: {hf: AtomicChat/Inkling-2-Mini-Uncensored-GGUF, include: [...], revision: <sha>}
    vram_gb: 18          # 预估；上线后用实测覆盖
    disk_gb: 17
    serves: [chat]
    caps: {tools: true, vision: false, reasoning: []}
  qwen:
    engine: strata
    group: llm
    caps: {tools: true, reasoning: [none, low, medium, high]}
    ...
  pika:
    engine: diffusers
    group: video
    serves: [video]
    params: {fps: 24, width: 480, height: 832, steps: 4}
    ...

rules:
  - {group: llm, max_online: 1, on_conflict: swap}
  - {group: video, requires_online_below: {group: llm, vram_gb: 20}}
```

几个刻意的选择：

- **`engine` 和 `group` 分开。** `group` 只用于调度规则，`engine` 只决定怎么装、怎么起、怎么停。加一个用 llama.cpp 跑的 LLM，只要 `engine: llama-cpp, group: llm`，规则 1 自动管住它。
- **引擎只有两种形状。** `server`（vLLM、llama.cpp、SGLang、Strata、TEI、whisper 服务都属于这类）和 `jobs`（视频、图像、音乐生成）。Laziness Protocol：两种形状能覆盖目前所有候选，再多一种就是为假想需求加层。引擎差异全写成数据（命令模板、就绪路径、停止命令），只有 Strata 这种守护进程化的需要 `stop` 字段。
- **规则只有两种原语，再加一条永远生效的显存总量检查。** `max_online` 表达"同组最多 N 个"，`requires_online_below` 表达"只有某组显存低于阈值时才能上线"。你的两条规则正好是这两种。不做通用表达式语言，等出现第三种真实规则再加。
- **规则是对"加载之后的在线集合"的约束，不是只在加载某组时检查。** 审查用原型证明过：如果把 `requires_online_below` 只挂在 video 组上，Pika 在线时加载一个 22GB 的 LLM 会被放行，而现在的 `policy.py` 会拒绝。所以规划器先算出"卸载谁、加载谁"之后的集合，再对这个集合检查全部规则。`on_conflict: swap` 只会卸载同一组里的模型，不会为了让 LLM 上线去卸 Pika。
- **`caps` 包含 `context` 和 `max_output`。** 审查实测过：Claude Code 不认识 `inkling` 时假设上下文有 100 万 token，而 Strata 只开了 32k，所以它永远不会压缩上下文，直到请求溢出。dsh 也会读 `context_length`，现在没人提供。
- **每个引擎独立 venv。** vLLM、SGLang、diffusers 的依赖版本互相冲突，也会冲掉 controller 自己的 FastAPI/httpx。现在 `pika.py:32-36` 把 diffusers 装进 controller 的解释器，加第二个 diffusers 模型就可能出事。

### 加速路线（routes）

同一个模型可以有几条跑法，每条叫一条路线：引擎、权重来源、参数，加上对硬件的要求。模型条目里的 `routes` 按优先级排列，controller 选第一条当前 GPU 满足要求的路线：

```yaml
engines:
  basalt:
    shape: server
    apis: [openai, anthropic]
    install: {archive: "https://github.com/jesdga95/basalt/releases/download/v0.5.6/basalt-v0.5.6-linux-x64.tar.gz",
              sha256: <我们自己记录>, venv: true}
    requires: {sm: [120]}          # 主卡必须是 Blackwell sm_120
    ...

models:
  inkling:
    group: llm
    caps: {tools: true, context: 32768}
    routes:
      - engine: basalt
        source: {hf: <你的私有仓库>/Inkling-2-Mini-Uncensored-Basalt}
        vram_gb: 20
        caps: {vision: true, drafting: dspark}   # 覆盖模型级 caps
      - engine: strata
        source: {hf: AtomicChat/Inkling-2-Mini-Uncensored-GGUF, include: [...]}
        vram_gb: 18
  pika:
    group: video
    routes:
      - engine: diffusers
        params: {fuse_lora: true, dit_precision: fp8}
```

- 没有新增维度的代码，只是 `models` 条目多了一层列表。引擎的硬件要求写在 `engines.<name>.requires`，路线可以再加限制。
- `caps` 和 `vram_gb` 可以按路线覆盖，因为同一个模型在不同引擎上能力不同（Inkling 在 Basalt 上能看图，在 Strata 上不能）。
- `lm load inkling --route strata` 可以手动指定路线。`/admin/status` 显示每个在线模型实际用的路线。
- 同组规则按模型算，不按路线算：Inkling 不管走哪条路线，都占 `group: llm` 的那一个名额。

注册表在 controller 启动时解析和校验一次（Boundary Discipline）。拼错字段、端口冲突、`rules` 引用不存在的 group、`engine` 不存在，都在启动时报错，而不是在加载时抛 `KeyError`。

## 请求怎么走

```
harness ──HTTP──► 127.0.0.1:9000 (ssh -L) ──► controller
                                               ├─ /v1/chat/completions, /v1/messages, /v1/embeddings …
                                               │     按请求里的 model 路由到对应 server 引擎
                                               ├─ /v1/models   列出所有在线模型和 caps
                                               ├─ /jobs        提交到 jobs 引擎，按 model 路由
                                               └─ /admin/*     状态、加载、卸载
```

- **按 `model` 路由，不再改写。** 请求里的 `model` 是在线模型 id 就转发过去。不在线就返回错误，列出在线的模型和加载命令。唯一的兼容例外是 `model` 为空或为 `default` 时，路由到 `group: llm` 里唯一在线的那个。悄悄换模型比报错更糟，因为用户以为自己在跟 Qwen 说话。
- **`/v1/models` 只列出能聊天的在线模型**，带 `context_length`。审查把 dsh 的模型列表解析代码单独跑过：它不看 `caps`，会把 Pika 当成聊天模型显示出来。所以完整列表（所有模型、所有 `caps`、在线状态）放在 `/admin/models`，harness 配置生成器读这个，不再在 Mac 上硬编码模型列表。没有聊天模型在线时 `/v1/models` 返回空列表。部分 harness（Open WebUI、Continue）会缓存空列表，这在每个 adapter 的说明里写明。
- **Anthropic 接口直通。** 引擎的 `apis` 里有 `anthropic` 时，`/v1/messages` 直接转发，Claude Code 这类只说 Anthropic 协议的 harness 就能用。没有的引擎返回明确错误，不做协议翻译（翻译层是一个独立项目，等真有需要再引入 LiteLLM 之类的现成东西）。
- **`/jobs` 取代 `/video/*`。** `POST /jobs {model, input}` 返回 job id，`GET /jobs/{id}` 查状态，结果文件走 `/files/`。worker 自己声明输入参数和默认值（fps、分辨率），Mac 端和 harness 端不再各写一遍 24fps。

## harness 怎么接

harness 需要的东西拆成三层，只有第三层每个 harness 各写一份：

1. **数据面**：OpenAI/Anthropic HTTP，地址 `http://127.0.0.1:9000/v1`。任何 OpenAI 客户端今天就能用。
2. **控制面**：一个 MCP server，`lm mcp`（stdio）。工具有 `status`、`load`、`unload`、`up`、`down`、`generate`（提交 job 并等待结果下载到本地）。Claude Code、Cursor、Codex、OpenCode 都支持 MCP，一份实现全部能用。`lm` CLI 仍是同一套操作的命令行入口，MCP 工具直接调用相同的 Python 函数。
3. **接入适配器**：`lm connect <harness>`。每个 adapter 是一个纯函数，输入是 `/v1/models` 的结果和 base URL，输出是要写入的配置文件内容。写之前备份，只保留最近一份备份（现在 `~/.dsh/profiles/web` 已经堆了 5 个 `.bak`）。首批适配器：
   - `dsh`：就是现在的 `cmd_setup_dsh`，但模型列表和 `reasoningEfforts` 从 `caps` 生成。
   - `claude-code`：写 `ANTHROPIC_BASE_URL` 等环境变量，加上 MCP server 注册。
   - `openai`：只打印 base URL 和模型 id，给 Open WebUI、Continue、Cursor 等手动填。

dsh 插件的去留取决于一个还没验证的事实：dsh 是否能当 MCP 客户端。如果能，删掉插件里的工具，只保留 slash 命令或者整个删掉。如果不能，插件改成调用 `lm` 的薄壳，不再自己维护状态类型和模型 id。

## GPU 提供方

`Colab` 类在 `cli.py` 里被直接构造了约 10 次。`tests/test_acquire.py` 里的 `FakeColab` 已经说明了需要的接口：`session_active`、`new`、`gpu_info`、`stop`、`deploy`、`ssh`、`proxy_command`、`usage`。把它定成 `Provider` 协议，`config.toml` 里加 `provider = "colab"`。这一步只立接口，不实现第二个提供方。等真要用 RunPod 或自己的机器时再写。

## 边界情况

下面每一条都标了来源：**已复现**表示我跑过代码看到了错误行为，**读代码**表示从代码推断但没跑，**猜测**表示依赖还没验证的外部行为。

### 加载与卸载

| 情况 | 现状 | 设计里的处理 |
|---|---|---|
| 切换 LLM 时磁盘不够 | **已复现**：先卸了 inkling 才检查磁盘，加载 qwen 被拒后 `online == []` | 规划器是纯函数，一次算出卸载集合、要删除的权重和磁盘是否够用。要卸载的模型的权重算作可释放空间（审查证明，不算的话会拒绝今天能成功的切换）。全部通过才开始卸载 |
| 加载被拒，别的模型权重却已经删了 | **已复现**（审查）：磁盘为 0 时加载 pika，返回 409，qwen 的缓存权重已经被删 | 同上。删除只在规划通过之后执行；被拒绝时什么都不删、什么都不卸 |
| 下载中途取消（切换、卸载、空闲） | **读代码**：`_run_logged` 的子进程没存到 `self.proc`，取消后 `hf download` 继续跑，之后磁盘清理可能删掉它正在写的目录 | 下载进程和引擎进程统一记录在 backend 上，用独立进程组，取消时整组杀掉 |
| 旧 Strata 停不掉，新模型用同一个端口 | **读代码**：`--stop` 的错误被吞掉，新模型的就绪检查会命中旧服务，代理把请求发给错误的模型 | 每个模型分配独立端口；启动前确认端口空闲；就绪检查核对返回的模型身份 |
| 停止后显存还没释放就启动下一个 | **读代码**：没有等待 | 卸载后轮询 `nvidia-smi`，等显存降到预期再启动，超时则报错 |
| 引擎上线后崩溃 | **读代码**：上线后不再检查，状态一直是 ready，代理返回 502 | 每 30 秒健康检查，失败则标记 error 并释放 |
| 上线后违反规则被自动卸载 | **读代码**：先 pop 再设 error，状态里显示成离线，原因只在事件日志里 | 保留一个 error 状态的条目，带原因，直到用户再次操作 |
| 同时发起两个下载 | **读代码**：磁盘检查只看当前剩余空间，两个都通过，然后写满 | 磁盘检查扣除正在下载的模型还要写入的字节 |
| 受限的 HF 仓库 | **读代码**：没有 token 处理，失败只给退出码，而且失败前可能已经删掉了别的模型 | `HF_TOKEN` 从 Mac 的配置随部署传过去，不落盘到仓库；下载前先用 API 检查能否访问，再做磁盘清理 |
| 模型版本漂移 | **读代码**：没有固定 revision，作者更新仓库后行为会变 | `source.revision` 固定 commit |
| 注册表写错 | **读代码**：加载时才抛 `KeyError`；`--load` 写错 id 会让 controller 启动失败 | 启动时校验，错误指出字段 |

### 加速路线

| 情况 | 处理 |
|---|---|
| 拿到的是 L4 或 A100，首选路线要求 sm_120 | 自动退到下一条路线（Strata）。`lm up` 和 `/admin/status` 都写明"因 GPU 是 L4，用 strata 路线"。没有任何路线满足时，拒绝加载并列出每条路线的要求 |
| 首选路线加载失败（下载失败、引擎崩溃） | 不自动换路线，报错并提示可以用 `--route` 指定。自动换会掩盖问题，而且换路线要重新下载十几到几十 GB |
| 去审查版权重需要先转换格式（GGUF → `.basalt`） | 不在每次启动时转换。转换一次，上传到你自己的私有 HF 仓库，以后直接下载转换好的包。转换步骤做成 `lm convert` 命令，在 Colab 上跑 |
| 引擎自带调度器（Basalt 有自己的多模型路由和 `idle-unload`、`idle-sleep`） | 每个 Basalt 进程只配一个模型，关掉它自己的空闲卸载和睡眠，避免两个调度器互相打架。显存和空闲只由 controller 管 |
| 引擎的 release 没有校验和（Basalt 只有两个压缩包） | 固定版本，sha256 由我们自己第一次下载时记录到 `models.yaml`，之后不匹配就拒绝安装 |
| 引擎版本升级改变包格式（Basalt 的 `.basalt` 有 v1/v2） | 引擎版本和转换好的包一起固定；升级引擎时重新跑 `lm convert` |
| 视频 FP8 量化后画质变差 | `dit_precision` 是参数，可以改回 `bf16`；第一次启用时用同一提示词和种子出一对 BF16/FP8 视频做对比，结果记录在 `docs/spike.md` |

### 会话与连接

| 情况 | 现状 | 设计里的处理 |
|---|---|---|
| 对已在运行的会话再执行 `lm up` | **读代码**：每次都重新部署，`bootstrap.sh:29` 杀掉 controller，守护进程化的 Strata 变成孤儿，继续占显存，新 controller 看不到它 | controller 健康且代码版本（`remote/` 的哈希）相同时跳过部署；需要重启时，controller 启动先读远端状态文件，接管或清理旧引擎 |
| 两个 `lm` 同时运行 | **读代码**：没有锁，可能同时 `colab new`，两个隧道互相杀，两个 watchdog | 所有改状态的命令拿 `~/.local/state/lm/lock` 文件锁 |
| 本地 9000 端口被占 | **读代码**：如果占用方也回应 `/health` 200，`lm` 会以为连上了 | `/health` 返回 `{service: "lm-controller", session, version}` 并核对；端口只在 `config.toml` 配一处，插件和 harness 配置都从它生成 |
| Mac 睡眠 | **读代码**：watchdog 不运行，会话持续计费；醒来后第一次检查可能立刻执行 `lm down` | 远端加一个计时器：超过 N 分钟没有来自 Mac 的心跳并且空闲，就自己释放运行时。从 ssh 里能否释放 Colab 运行时是**猜测**，需要先验证 |
| `colab` CLI 自身失败（授权过期、断网） | **读代码**：被当成"会话不存在"，watchdog 退出，会话可能还在计费 | 区分"确认不存在"和"查询失败"；查询失败时继续守护并重试 |
| 隧道在流式输出中途断开 | **读代码**：客户端看到连接重置；watchdog 要 3 到 4 分钟后才重连 | 由连接失败触发重连，不等 watchdog 周期；controller 在客户端断开时取消上游生成 |

### 空闲与计费

| 情况 | 现状 | 设计里的处理 |
|---|---|---|
| 下载很久（Pika 约 148GB） | **读代码**：下载不算活动，45 分钟后可能被空闲卸载，60 分钟后可能被 `lm down` | 空闲的定义改成"没有进行中的工作并且距上次请求超过 N 分钟"；进行中的工作包括下载、流式响应、排队或运行中的 job |
| 视频生成时客户端不再轮询 | **读代码**：job 不算活动，可能在生成中被卸载 | 同上；job 状态写盘，卸载前检查 |
| 某个 harness 轮询其他 `GET /v1/*` | **读代码**：除 `/v1/models` 外的 GET 都会刷新活动时间，可能让会话永远不空闲 | 只有推理请求和 job 算活动 |

### 多个 harness 同时使用

| 情况 | 现状 | 设计里的处理 |
|---|---|---|
| A 加载 qwen 时 B 正在用 inkling | **读代码**：直接换掉，B 下一次请求悄悄变成 qwen | 按 `model` 路由后，B 会收到"inkling 不在线"的明确错误，而不是答非所问 |
| 任一 harness 执行 down | 所有人都断 | 保持现状。这是一人一机的工具，加引用计数不值得 |
| 输出目录不同 | dsh 写到它的工作目录，`lm` 写到自己的工作目录 | 统一到 `config.toml` 里的 `outputs_dir` |

## 不做的事

- 多 GPU、多节点。Colab 一个运行时只有一块卡。
- harness 之间的协议翻译（OpenAI ↔ Anthropic）。用直通，需要时引入现成工具。
- API 鉴权。端口只绑在 127.0.0.1，经 ssh 隧道访问。如果以后要给别人用，再加 token。
- 第二个 GPU 提供方的实现。只立 `Provider` 接口。
- 通用规则表达式。两种规则原语够用，等出现第三种真实规则再扩展。

## 分阶段实施

每一阶段都以一次可以验证的检查结束，前一阶段通过才开始下一阶段。

1. **修已知的正确性问题，不改结构。** 下面这些都要在第 2 阶段改端口之前完成。原因是第 2 阶段会改 `remote/`，`lm up` 必然重新部署；如果这时还不能接管旧引擎，守护进程化的 Strata 会留在 8081 上占着显存，而新 controller 根本不会去看 8081。
   - 纯函数规划器：卸载、删除、磁盘检查一次算完，被拒绝时什么都不动。
   - 下载进程可取消。
   - error 状态保留原因。
   - 远端状态文件，记录引擎的 pid 和端口；controller 启动时接管或清理。
   - `lm up` 在代码版本没变时不重新部署。
   - 卸载后等显存释放再启动下一个；上线后定期做健康检查。
   - 空闲按进行中的工作计算。
   - Mac 端文件锁。
   - 下载前检查 HF 仓库能否访问。

   验证：每个问题一个先失败后通过的测试，再加一次 L4 真机 `lm up` → 切换 → `lm up` 再执行一次，并确认 `nvidia-smi` 里没有孤儿进程。
2. **注册表 v2 和通用引擎。** 新的 `models.yaml` 结构、两种引擎形状、规则写成数据、每个模型独立端口、每个引擎独立 venv。把 Inkling、Qwen、Pika 迁移过去，在同一次改动里删掉 `StrataBackend`/`PikaBackend` 的 kind 分支和 `VIDEO_LLM_LIMIT_GB`。验证：规则用表驱动测试覆盖；L4 真机加载 Inkling；再加一个用 llama.cpp 跑的小模型，证明加模型不用写 Python。
   本阶段同时加入 `routes`：先只有 Strata 一条路线，把结构立起来。

3. **Basalt 路线和视频加速。** 安装 Basalt 预编译包；用 `lm convert` 把两个去审查模型转成 `.basalt` 并上传到你的私有 HF 仓库；Pika 加 `fuse_lora` 和 `dit_precision: fp8`。验证：在 G4 上对同一组提示词测 Strata 和 Basalt 的 prefill/decode 速度和回答；在 L4 上确认自动退回 Strata；Pika 出一对 BF16/FP8 对比视频并记录显存和耗时。G4 一直申请不到，所以这一阶段可能要等。

4. **按 model 路由、`/jobs`、MCP、`lm connect`。** 删掉 `/video/*` 和 dsh 插件里重复的部分。验证：在 dsh 和 Claude Code 里各完成一次对话和一次 `generate` 调用。
5. **`Provider` 接口、远端自释放和隧道恢复。** 先验证从 ssh 里能否释放 Colab 运行时，能则实现心跳加自释放，不能则在文档里写明 Mac 睡眠时的计费风险。同时实现：连接失败时立即重连隧道，客户端断开时取消上游生成，区分"会话确认不存在"和"`colab` 查询失败"。

## 待验证的外部事实

- Strata 的 `/v1/messages` 是否完整支持 tool use 和流式（决定 Claude Code 能否直连）。
- dsh 能否作为 MCP 客户端（决定 dsh 插件删不删）。
- 从 Colab 运行时内部能否释放自己（决定远端自释放能否实现）。
- `strata launch --stop` 返回时显存是否已经释放。
- Claude Code 用什么设置告诉它自定义模型的上下文窗口（决定 `claude-code` adapter 写什么）。
- Basalt 的转换器能否处理 SC117 的 `GSQ-RCO` IQ3_XXS GGUF（文档只列了 Unsloth 的量化类型）和 AtomicChat 的 `AD-Q4_K` GGUF。Inkling 也可以改用 `OpenUncensored/Inkling-2-Mini-Uncensored` 的 safetensors，走转换器的 `--checkpoint DIR`。
- Basalt 的 GGUF 转换在 Colab 上需要多少内存和磁盘（Qwen 的源文件约 76GB）。
- Pika 的 DiT 用 torchao 或 ModelOpt 做 FP8 后，在 sm_120 上的画质和速度。
