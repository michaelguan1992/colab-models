// dsh-plugin-colab: control the Colab G4 models from inside DeepSeek Harness.
//
// Slash commands (run without a model turn):
//   /up [model]   /down   /load <id>   /unload [id]   /status
// Agent tools:
//   colab_status, generate_video
//
// API surface used (dsh developer preview — re-check on upgrade):
//   ctx.commands.register({ name, description, input, handler }) -> CommandResult
//   ctx.tools.register(defineTool({ name, description, parameters, output, execute }))
// Outbound calls use Node's global fetch because ctx.web.fetch is GET-only.

import type { Context } from '@deepseek-ai/cordis'
import Schema from '@deepseek-ai/schemastery'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { spawn } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

export const name = 'colab-models'
export const inject = ['tools', 'commands']

export interface Config {
  baseURL: string
  lmBin: string
  outputsDir: string
}

export const Config: Schema<Config> = Schema.object({
  baseURL: Schema.string().default('http://127.0.0.1:9000'),
  lmBin: Schema.string().default('lm'),
  outputsDir: Schema.string().default('outputs'),
})

type ModelStatus = {
  id: string; kind: string; display: string; state: string
  vram_gb?: number | null; error?: string | null; download_progress?: number | null
}
type Status = {
  gpu: { name: string; used_gb: number; total_gb: number; available: boolean }
  models: ModelStatus[]; online: string[]; idle_seconds: number; idle_unload_minutes: number
}

const ICON: Record<string, string> = {
  ready: '● 在线', offline: '○ 离线', downloading: '◐ 下载中', starting: '◑ 启动中',
  stopping: '◌ 停止中', error: '✕ 错误', pending: '◌ 排队',
}

export function apply(ctx: Context, config: Config) {
  const api = async (path: string, init?: RequestInit): Promise<Response> =>
    fetch(`${config.baseURL}${path}`, {
      ...init,
      headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) },
    })

  const getStatus = async (): Promise<Status | null> => {
    try {
      const r = await api('/admin/status')
      return r.ok ? ((await r.json()) as Status) : null
    } catch {
      return null
    }
  }

  const formatStatus = (s: Status | null): string => {
    if (!s) return 'Colab 未连接（运行 /up 或在终端 `lm up`）'
    const lines = [
      `GPU ${s.gpu.name} ${s.gpu.used_gb.toFixed(1)}/${s.gpu.total_gb.toFixed(0)}GB`,
      ...s.models.map((m) => {
        let st = ICON[m.state] ?? m.state
        if (m.download_progress != null) st += ` ${Math.round(m.download_progress * 100)}%`
        const vram = m.vram_gb ? ` ${m.vram_gb.toFixed(1)}GB` : ''
        return `${m.id.padEnd(8)} ${st}${vram}${m.error ? ` — ${m.error}` : ''}`
      }),
      `在线：${s.online.length ? s.online.join(', ') : '无'}`,
    ]
    return lines.join('\n')
  }

  const runLm = (args: string[], detached = false): Promise<string> =>
    new Promise((resolve) => {
      const child = spawn(config.lmBin, args, {
        detached, stdio: detached ? 'ignore' : ['ignore', 'pipe', 'pipe'],
      })
      if (detached) {
        child.unref()
        resolve('')
        return
      }
      let out = ''
      child.stdout?.on('data', (d) => (out += d))
      child.stderr?.on('data', (d) => (out += d))
      child.on('error', (e) => resolve(`无法运行 ${config.lmBin}: ${e.message}`))
      child.on('close', () => resolve(out.trim()))
    })

  const ok = (text: string) => ({ kind: 'success' as const, text })
  const err = (text: string) => ({ kind: 'error' as const, text })
  const arg = (raw: string) => raw.trim().split(/\s+/).filter(Boolean)[0]

  // ---- slash commands ------------------------------------------------------
  ctx.commands.register({
    name: 'status',
    description: '显示 Colab 上在线的模型与显存',
    handler: async () => ok(formatStatus(await getStatus())),
  })

  ctx.commands.register({
    name: 'up',
    description: '启动 Colab G4 会话并加载模型（默认 inkling）',
    input: { hint: '[inkling|qwen]' },
    handler: async (inv) => {
      const model = arg(inv.rawInput)
      await runLm(['up', ...(model ? [model] : [])], true)
      return ok(`正在后台启动 Colab 会话${model ? `并加载 ${model}` : ''}，几分钟后用 /status 查看。`)
    },
  })

  ctx.commands.register({
    name: 'down',
    description: '卸载所有模型并停止 Colab 会话（停止计费）',
    handler: async () => ok(await runLm(['down'])),
  })

  ctx.commands.register({
    name: 'load',
    description: '加载模型；LLM 同时只能一个，会自动替换',
    input: { hint: 'inkling | qwen | pika' },
    handler: async (inv) => {
      const id = arg(inv.rawInput)
      if (!id) return err('用法：/load <inkling|qwen|pika>')
      try {
        const r = await api('/admin/load', { method: 'POST', body: JSON.stringify({ id, swap: true }) })
        const body = (await r.json()) as { ok?: boolean; reason?: string; detail?: string; unloaded?: string[] }
        if (!r.ok || !body.ok) return err(`拒绝：${body.reason ?? body.detail}`)
        const swapped = body.unloaded?.length ? `已卸载 ${body.unloaded.join(', ')}；` : ''
        return ok(`${swapped}${id} 开始加载（首次需下载），用 /status 查看进度。`)
      } catch {
        return err('Colab 未连接，先 /up')
      }
    },
  })

  ctx.commands.register({
    name: 'unload',
    description: '卸载模型（不带参数则全部卸载）',
    input: { hint: '[inkling|qwen|pika]' },
    handler: async (inv) => {
      const id = arg(inv.rawInput) ?? null
      try {
        const r = await api('/admin/unload', { method: 'POST', body: JSON.stringify({ id }) })
        const body = (await r.json()) as { unloaded?: string[] }
        return ok(`已下线：${body.unloaded?.join(', ') || '无'}`)
      } catch {
        return err('Colab 未连接')
      }
    },
  })

  // ---- agent tools ---------------------------------------------------------
  ctx.tools.register(defineTool({
    name: 'colab_status',
    description: 'Show which self-hosted models are online on the Colab GPU and their VRAM use.',
    parameters: {},
    output: { schema: { type: 'string' }, render: (_a: unknown, v: string) => [{ type: 'text', text: v }] },
    async execute() {
      return formatStatus(await getStatus())
    },
  }))

  ctx.tools.register(defineTool({
    name: 'generate_video',
    description:
      'Generate a short video with synchronized audio using Pika-H3-Flex (Turbo, 4 steps) on the Colab GPU. ' +
      'Requires the pika model to be online. Returns the local path of the saved MP4.',
    parameters: {
      prompt: { type: 'string', required: true, description: 'Scene description, including sound.' },
      seconds: { type: 'number', description: 'Duration in seconds (default 5).' },
    },
    output: { schema: { type: 'string' }, render: (_a: unknown, v: string) => [{ type: 'text', text: v }] },
    async execute(args: { prompt: string; seconds?: number }, exec: { signal: AbortSignal }) {
      const frames = Math.round((args.seconds ?? 5) * 24) + 1
      const r = await api('/video/generate', {
        method: 'POST', signal: exec.signal,
        body: JSON.stringify({ prompt: args.prompt, num_frames: frames }),
      })
      if (!r.ok) throw new Error(`视频模型不可用：${((await r.json()) as { detail?: string }).detail}`)
      let job = (await r.json()) as { id: string; state: string; file?: string; error?: string }
      while (job.state === 'queued' || job.state === 'running') {
        await new Promise((res) => setTimeout(res, 3000))
        if (exec.signal.aborted) throw new Error('aborted')
        job = (await (await api(`/video/jobs/${job.id}`, { signal: exec.signal })).json()) as typeof job
      }
      if (job.state !== 'done' || !job.file) throw new Error(`生成失败：${job.error}`)
      const file = await fetch(`${config.baseURL}/files/${job.file}`, { signal: exec.signal })
      await mkdir(config.outputsDir, { recursive: true })
      const dest = join(config.outputsDir, job.file)
      await writeFile(dest, Buffer.from(await file.arrayBuffer()))
      return `已生成视频：${dest}`
    },
  }))
}
