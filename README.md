# LocalModels

Run your own models on a Google Colab GPU and use them from a Mac, starting and stopping everything with one command.

[中文说明](README.zh-CN.md)

The target runtime is a Colab G4 (RTX PRO 6000 Blackwell, 96GB). A100 80GB and smaller GPUs work too, with fallback routes. On the Mac, the `lm` CLI creates the Colab session, deploys a controller and opens an ssh tunnel. You then use the models from DeepSeek Harness (dsh), Claude Code or any OpenAI-compatible client.

## Models

| id | Model | Engine | VRAM |
|---|---|---|---|
| `inkling` | G4: OpenUncensored/Inkling-2-Mini-Uncensored as a Basalt pack (1M context). Other GPUs: the AtomicChat GGUF (Q4_K, 32K) | Basalt / Strata | ~55GB / ~18GB |
| `qwen` | SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF (IQ3_XXS). G4: converted to a Basalt pack (262K context, vision). Other GPUs: the GGUF (32K) | Basalt / Strata | ~50GB / ~54GB |
| `qwen-tiny` | Qwen/Qwen3-0.6B-GGUF (Q8_0, for demos) | llama.cpp | ~2GB |
| `pika` | Pika-H3-Flex with the official fl2va adapter (6 steps, **video and audio**). On the G4 it runs in FP8, fully resident: about 36 s for a 5 s clip | diffusers | ~78GB |
| `moss` | OpenMOSS-Team/MOSS-Image-SFT with lodestones' 8-step LoRA (CFG 1, **images**). Resident on A100/H100/G4, CPU offload elsewhere. About 44 s per 1024px image on an L4 | diffusers (model's own pipeline) | ~32GB (estimate, not measured on the G4) |

Models are declared in [`remote/models.yaml`](remote/models.yaml). Each model has one or more routes, and the controller picks the first route whose GPU requirement matches. Adding a model or an engine only touches that file.

## Scheduling rules

The rules live in the `rules` section of `remote/models.yaml`. The controller checks them against the set of models that would be online after a load ([`policy.py`](remote/controller/policy.py)).

1. **One LLM at a time.** At most one model of the `llm` group is online (`max_online: 1, on_conflict: swap`). Loading another LLM unloads the current one first.
2. **Pika needs a small LLM.** `pika` (group `video`) loads only while the online LLM's *measured* VRAM is under 22GB (`requires_online_below`). On the G4, Inkling on Basalt with 1M context measures 54.7GB, so Pika runs alone (or next to the `qwen-tiny` demo model).
   `moss` (group `image`) has no rule of its own. Only the VRAM check applies: online models plus the new one must fit in total VRAM minus a margin. On the G4 that lets it share with either LLM, but not with Pika.
3. **Idle unload.** After 45 minutes without requests, all models are unloaded (`idle_unload_minutes`).
   Work you start by hand on the runtime counts as activity. That means a GPU process that belongs to no loaded model, or any file in `/content/lm/holds/`. Use a hold file for CPU or network work such as conversions and downloads, and delete it when done. `lm status` lists this external work.

## Setup (once)

Install the Colab CLI, then authorize it in the browser when prompted:

```bash
uv tool install google-colab-cli
```

Install the `lm` command from this repository:

```bash
uv tool install -e .
```

Connect a client. Run `lm up` first: client configs are generated from the models and capabilities the controller reports, not written by hand.

- **dsh:** `lm connect dsh` merges two things into `~/.dsh/profiles/web/cordis.patch.yml`. One is the provider `colab`, whose models, context length, reasoning levels and vision support come from the controller's `caps`. The other is the MCP server `lm`. The file is backed up first, and only the latest backup is kept. Run it again after switching GPUs or editing `models.yaml`.
- **Claude Code:** `lm claude` starts `claude` and passes any other arguments through, for example `lm claude -p "hi"`. It points `ANTHROPIC_BASE_URL` at the controller, maps every model alias to `default_model`, sets the context window from `caps` and attaches `lm mcp` with `--mcp-config`. It does not change your global Claude Code settings. If `claude` is not on your PATH, set `claude_bin` in the config. `lm connect claude-code` only prints the environment variables.
- **Other OpenAI clients** (Open WebUI, Continue, Cursor and so on): `lm connect openai` prints the base URL and model ids.

On a first run on a real G4, check each item in [docs/spike.md](docs/spike.md) (in Chinese).

## Daily use

Start a session, deploy, open the tunnel and load `inkling` (`lm up qwen` loads Qwen instead):

```bash
lm up
```

Unload models, close the tunnel and stop the Colab session, which stops billing:

```bash
lm down
```

### When no GPU is available

The G4 often answers 503. That means no free capacity, not a problem with your account. `lm up` asks once for each entry in `accelerators`, in order: G4 first, then A100 80GB (high-mem). If none is free, it exits.

- `lm up --wait [minutes]` asks again every `assign_retry_minutes` (default 12, ±20% jitter). With a number, it gives up after that many minutes (for example `lm up --wait 120`). Without one, it keeps trying until the day's `assign_daily_limit`.
- At most `assign_daily_limit` (default 12) requests are sent per day, shared by every `lm up` run. Rapid retries have gotten other accounts restricted (colabtools#6206).
- A GPU with less than `min_vram_gb` (default 75) of VRAM is stopped at once, and the next accelerator is tried.
- On an A100 80GB, Inkling and Pika cannot be online together, because of the VRAM check.

### Billing and auto-stop

- When the controller has been idle for `idle_unload_minutes` (45), it unloads the models only. The Colab session keeps billing by the hour (about 8.9 compute units per hour on the G4).
- `lm up` also starts a watchdog on the Mac that runs `lm down` after `idle_stop_minutes` (default 60) of idleness. Its log is `~/.local/state/lm/watchdog.log`.
- The watchdog does not run while the Mac sleeps, so run `lm down` yourself before you leave.
- dsh polling the model list does not count as activity and will not keep the session alive.

### Disconnects

- A supervisor process watches the tunnel. When ssh exits (network loss, a Wi-Fi change, the Mac waking up), it reconnects after 1 second. If that fails, it doubles the delay each time, up to 30 seconds. ssh notices a silently dead link within about 30 seconds. The log is `~/.local/state/lm/tunnel.log`.
- When a client disconnects mid-request (a closed chat, a stop button, a timeout), the controller drops the engine connection immediately, so the GPU does not keep generating an answer nobody will read. This holds for streaming and non-streaming requests.

### Configuration

All keys in `~/.config/lm/config.toml` are optional:

```toml
accelerators = [{gpu = "G4"}, {gpu = "A100", high_mem = true}]
min_vram_gb = 75
assign_retry_minutes = 12
assign_daily_limit = 12
idle_stop_minutes = 60
default_model = "inkling"
```

### Commands

| Command | What it does |
|---|---|
| `lm status [--usage]` | GPU, online models and their VRAM, idle time; `--usage` adds the compute unit balance |
| `lm load <id>` / `lm unload [id]` | Load a model (an LLM replaces the current LLM) / unload one, or all |
| `lm video "prompt" [--seconds 5-15] [--image first-frame.png]` | Generate a video with Pika and download it |
| `lm image "prompt" [--width 1024 --height 1024] [--seed N]` | Generate an image with MOSS-Image and download it |
| `lm tunnel` | Rebuild the ssh tunnel |
| `lm logs [controller\|inkling\|qwen\|pika\|moss]` | Show remote logs |
| `lm connect dsh\|claude-code\|openai` | Generate client config from the controller's model list |
| `lm claude [args]` | Start Claude Code on the Colab models |
| `lm dsh` | Start dsh web with the colab plugin |
| `lm mcp` | Run as a stdio MCP server |

## How requests are routed

- The `model` field of a request picks the online model with that id. If the model is offline or unknown, the controller returns 404 and lists the online models. It never silently substitutes another model. When only one chat model is online, `model` may be empty or `default`.
- `/v1/messages` (the Anthropic protocol) is passed straight through to engines that declare `anthropic` (Basalt, Strata). Engines such as llama.cpp return a clear error.
- Video and image jobs go to `POST /jobs {"model": "pika", "input": {"prompt": "...", "seconds": 5}}`. Instead of `model` you can send `"serves": "video"` or `"image"`, and the controller picks the online model with that capability. Poll `GET /jobs/{id}` for status; results are served under `/files/`. The worker turns seconds into frames at its own fps, and durations outside 5 to 15 seconds are rejected.

## Tools for models (MCP)

`lm mcp` is a stdio MCP server with the tools `status`, `load`, `unload`, `up`, `down`, `generate` (video) and `generate_image`. Each tool runs the matching `lm` command. `lm connect dsh` and `lm claude` attach it automatically; for any other MCP-capable harness, configure the command `lm mcp`. `generate` waits for the video to finish, so the dsh tool timeout is set to 30 minutes.

## Using it in dsh

`lm dsh` starts dsh web with the [`dsh/plugin-colab`](dsh/plugin-colab) plugin. The plugin only adds slash commands, `/up [model]`, `/down`, `/load <id>`, `/unload [id]` and `/status`, and each one runs `lm`. The dsh model picker discovers models through `/v1/models`, which **lists only online models that serve `chat`**. If another model such as Pika is also online, the name carries a `[+pika]` suffix.

## Layout

```
src/lm/                Mac-side CLI (colab CLI wrapper, ssh tunnel, commands)
remote/bootstrap.sh    Installs the controller's dependencies on Colab and starts it (safe to rerun).
                       Engines are installed the first time a model that needs them loads.
remote/models.yaml     Registry: engines (how to install and launch), models (repos, routes, caps,
                       VRAM estimates) and rules (scheduling). Adding a model or engine only touches this file.
remote/controller/     FastAPI on :9000: a /v1 reverse proxy routed by model, /jobs, /admin, scheduling rules
remote/bench/          Benchmark and conversion scripts behind the numbers in docs/spike.md
dsh/plugin-colab/      dsh plugin (slash commands that call lm)
```

The Mac and Colab talk only over `ssh -L`, with `colab ssh --proxy-mode` as the ProxyCommand. Nothing is exposed to the public internet.

## Tests

```bash
uv run pytest
```
