// dsh-plugin-colab: slash commands for the Colab models, run without a model turn.
//
//   /up [model]   /down   /load <id>   /unload [id]   /status   /comfy [model]
//
// Each command runs the `lm` CLI, so model ids, state names and rules live in one place.
// The agent tools (status, load, ...) come from `lm mcp`, which `lm connect dsh`
// registers with dsh's MCP manager.
//
// API surface used (dsh developer preview — re-check on upgrade):
//   ctx.commands.register({ name, description, input, handler }) -> CommandResult

import type { Context } from '@deepseek-ai/cordis'
import Schema from '@deepseek-ai/schemastery'
import { spawn } from 'node:child_process'

export const name = 'colab-models'
export const inject = ['commands']

export interface Config {
  lmBin: string
}

export const Config: Schema<Config> = Schema.object({
  lmBin: Schema.string().default('lm'),
})

export function apply(ctx: Context, config: Config) {
  const runLm = (args: string[]): Promise<{ code: number; out: string }> =>
    new Promise((resolve) => {
      const child = spawn(config.lmBin, args, { stdio: ['ignore', 'pipe', 'pipe'] })
      let out = ''
      child.stdout?.on('data', (d) => (out += d))
      child.stderr?.on('data', (d) => (out += d))
      child.on('error', (e) => resolve({ code: 1, out: `无法运行 ${config.lmBin}: ${e.message}` }))
      child.on('close', (code) => resolve({ code: code ?? 1, out: out.trim() }))
    })

  const words = (raw: string) => raw.trim().split(/\s+/).filter(Boolean)
  const result = ({ code, out }: { code: number; out: string }) =>
    ({ kind: code === 0 ? ('success' as const) : ('error' as const), text: out || '(无输出)' })

  const command = (name: string, description: string, argv: (args: string[]) => string[] | string,
                   hint?: string) =>
    ctx.commands.register({
      name, description, ...(hint ? { input: { hint } } : {}),
      handler: async (inv) => {
        const args = argv(words(inv.rawInput))
        return typeof args === 'string' ? { kind: 'error' as const, text: args } : result(await runLm(args))
      },
    })

  command('status', '显示 Colab 上在线的模型与显存', () => ['status'])
  command('down', '卸载所有模型并停止 Colab 会话（停止计费）', () => ['down'])
  command('load', '加载模型；LLM 同时只能一个，会自动替换', (a) => (a[0] ? ['load', a[0], '--no-wait'] : '用法：/load <模型 id>，可用的 id 见 /status'), '<模型 id>')
  command('unload', '卸载模型（不带参数则全部卸载）', (a) => ['unload', ...a.slice(0, 1)], '[模型 id]')

  // Images and video are made in ComfyUI, not through an agent tool: the generation models do not fit
  // beside the chat model. Loading takes minutes on a fresh runtime, so this runs detached like /up.
  ctx.commands.register({
    name: 'comfy',
    description: '在浏览器里打开 ComfyUI（moss-comfy 生图，pika-comfy 生视频）',
    input: { hint: '[模型 id]' },
    handler: async (inv) => {
      const model = words(inv.rawInput)[0]
      const child = spawn(config.lmBin, ['comfy', ...(model ? [model] : [])], { detached: true, stdio: 'ignore' })
      child.unref()
      return { kind: 'success' as const, text: `正在加载 ${model ?? '默认的 ComfyUI 模型'}，就绪后会自动打开浏览器；进度用 /status 查看。` }
    },
  })

  ctx.commands.register({
    name: 'up',
    description: '启动 Colab 会话并加载模型（默认取 lm 配置的 default_model）',
    input: { hint: '[模型 id]' },
    handler: async (inv) => {
      const model = words(inv.rawInput)[0]
      const child = spawn(config.lmBin, ['up', ...(model ? [model] : [])], { detached: true, stdio: 'ignore' })
      child.unref()
      return { kind: 'success' as const, text: `正在后台启动 Colab 会话${model ? `并加载 ${model}` : ''}，几分钟后用 /status 查看。` }
    },
  })
}
