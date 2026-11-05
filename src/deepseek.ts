/**
 * Wire translation for DeepSeek models (DeepSeek-V3 and DeepSeek-R1) deployed
 * on Google Cloud Vertex AI (Model Garden / Serverless MaaS).
 *
 * Unlike Claude (Anthropic publisher) and Gemini (Google publisher), DeepSeek
 * in Vertex Model Garden provides a standard OpenAI-compatible completions
 * endpoint:
 *   https://${location}-aiplatform.googleapis.com/v1beta1/projects/${project}/locations/${location}/endpoints/openapi/chat/completions
 *
 * This module translates between the harness GenerateOptions/StreamChunk shapes
 * and the OpenAI chat completions payload and Server-Sent Events (SSE) stream.
 *
 * @module dsh-google-vertex/deepseek
 */

import {
  EMPTY_RESPONSE_CODE,
  failureForEvent,
} from './wire.ts'
import type {
  FinishReason,
  GenerateOptions,
  StreamChunk,
  TokenUsage,
} from './host.ts'
import type { VertexModel } from './adapter.ts'
import type { VertexWireConfig } from './wire.ts'

/** Context capacity reported for DeepSeek models in Model Garden (128k). */
export const DEFAULT_DEEPSEEK_CONTEXT_WINDOW = 128_000

/** Output cap applied when a caller omits one (16k). */
export const DEFAULT_DEEPSEEK_MAX_TOKENS = 16_384

/**
 * Built-in DeepSeek models served on Vertex AI Model Garden.
 */
export const DEFAULT_DEEPSEEK_MODELS: readonly VertexModel[] = [
  { id: 'deepseek-ai/DeepSeek-V3', name: 'DeepSeek-V3 (Vertex Model Garden)' },
  { id: 'deepseek-ai/DeepSeek-R1', name: 'DeepSeek-R1 (Vertex Model Garden)' },
  { id: 'deepseek-ai/deepseek-v4.1-flash-maas', name: 'DeepSeek-V4.1-Flash (Vertex Model Garden)' },
]

/**
 * Normalize DeepSeek model aliases to their canonical Vertex Model Garden identifiers.
 * Maps aliases like `deepseek-v3`, `deepseek-r1`, and `deepseek-v4.1-flash`.
 * @param model - model identifier or alias.
 * @returns canonical Vertex Model Garden model id.
 */
export function normalizeDeepSeekModel(model: string): string {
  const trimmed = model.trim()
  if (trimmed.length === 0) return DEFAULT_DEEPSEEK_MODELS[0]!.id
  const lower = trimmed.toLowerCase()
  if (lower === 'deepseek-v3' || lower === 'deepseek-ai/deepseek-v3') {
    return 'deepseek-ai/DeepSeek-V3'
  }
  if (lower === 'deepseek-r1' || lower === 'deepseek-ai/deepseek-r1') {
    return 'deepseek-ai/DeepSeek-R1'
  }
  if (
    lower === 'deepseek-v4.1-flash'
    || lower === 'deepseek-v4-1-flash'
    || lower === 'deepseek-v4.1-flash-maas'
    || lower === 'deepseek-ai/deepseek-v4.1-flash'
    || lower === 'deepseek-ai/deepseek-v4.1-flash-maas'
  ) {
    return 'deepseek-ai/deepseek-v4.1-flash-maas'
  }
  return trimmed
}

/**
 * Endpoint for Vertex Model Garden OpenAI-compatible chat completions.
 * @param project - Google Cloud project id.
 * @param location - Vertex AI region (e.g. us-central1).
 * @returns the absolute request URL.
 */
export function deepseekEndpointFor(project: string, location: string): string {
  const host = location === 'global' ? 'aiplatform.googleapis.com' : `${location}-aiplatform.googleapis.com`
  return `https://${host}/v1beta1/projects/${encodeURIComponent(project)}/locations/${encodeURIComponent(location)}/endpoints/openapi/chat/completions`
}

/** Configuration needed for DeepSeek wire requests. */
export interface DeepSeekWireConfig extends VertexWireConfig {
  readonly contextWindow?: number
}

/** OpenAI tool function schema. */
interface OpenAIFunction {
  name: string
  description: string
  parameters: Record<string, unknown>
}

/** OpenAI tool object. */
interface OpenAITool {
  type: 'function'
  function: OpenAIFunction
}

/** OpenAI tool call in message history. */
interface OpenAIToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

/** OpenAI chat message format. */
export type OpenAIMessage =
  | { role: 'system'; content: string }
  | { role: 'user'; content: string }
  | { role: 'assistant'; content: string | null; tool_calls?: OpenAIToolCall[] }
  | { role: 'tool'; tool_call_id: string; content: string }

/** DeepSeek OpenAI-compatible Chat Completions request body. */
export interface DeepSeekRequestBody {
  model: string
  messages: OpenAIMessage[]
  stream: true
  stream_options?: { include_usage: boolean }
  max_tokens: number
  temperature?: number
  stop?: string[]
  tools?: OpenAITool[]
}

/**
 * Format messages from GenerateOptions into standard OpenAI Chat Completions messages.
 */
function formatMessages(options: GenerateOptions): OpenAIMessage[] {
  const messages: OpenAIMessage[] = []

  if (options.system !== undefined && options.system.length > 0) {
    messages.push({ role: 'system', content: options.system })
  }

  for (const message of options.messages) {
    if (message.role === 'system' || message.role === 'developer') {
      const text = message.content
        .filter(b => b.type === 'text')
        .map(b => (b as { text: string }).text)
        .join('\n')
      if (text.length > 0) messages.push({ role: 'system', content: text })
    } else if (message.role === 'user') {
      const text = message.content
        .filter(b => b.type === 'text')
        .map(b => (b as { text: string }).text)
        .join('\n')
      messages.push({ role: 'user', content: text })
    } else if (message.role === 'assistant') {
      const text = message.content
        .filter(b => b.type === 'text')
        .map(b => (b as { text: string }).text)
        .join('\n')
      const toolCalls: OpenAIToolCall[] = message.content
        .filter(b => b.type === 'tool-call')
        .map(b => {
          const tc = b as { id: string; name: string; arguments: string }
          return {
            id: tc.id,
            type: 'function' as const,
            function: { name: tc.name, arguments: tc.arguments },
          }
        })

      if (toolCalls.length > 0) {
        messages.push({
          role: 'assistant',
          content: text.length > 0 ? text : null,
          tool_calls: toolCalls,
        })
      } else {
        messages.push({
          role: 'assistant',
          content: text,
        })
      }
    } else if (message.role === 'tool') {
      const text = message.content
        .filter(b => b.type === 'text')
        .map(b => (b as { text: string }).text)
        .join('')
      messages.push({
        role: 'tool',
        tool_call_id: message.toolCallId ?? '',
        content: text,
      })
    }
  }

  return messages
}

/**
 * Strip DeepSeek-specific markup language (DSML) tags and special tokens that
 * can leak into tool arguments or content from Vertex Model Garden's backend parser.
 *
 * Examples of leaked artifacts:
 * - `</｜DSML｜ parameter>` or `<｜DSML｜ parameter ...>`
 * - `<｜tool call begin｜>`, `<｜tool call end｜>`, `<｜tool sep｜>`
 * - `<｜...｜>` and `<|...|>`
 * - `</parameter>` or `</DSML>`
 * @param text - string that may contain leaked DSML tokens.
 * @returns cleaned string.
 */
export function sanitizeDeepSeekText(text: string): string {
  if (text.length === 0) return text
  return text
    // Strip DSML parameter tags (with fullwidth or standard pipe, self-closing or not)
    .replace(/<[/／]?[｜|]?\s*DSML\s*[｜|]?[^>]*>/gi, '')
    // Strip standalone </parameter> tags
    .replace(/<[/／]parameter\s*>/gi, '')
    // Strip DeepSeek special token delimiters <｜...｜>
    .replace(/<[｜|][^｜|>]+[｜|]>/g, '')
}

/**
 * Filter out tools requiring vision/multimodal capabilities that text-only DeepSeek
 * on Vertex Model Garden cannot execute (e.g. `read_image`). Exposing `read_image`
 * biases DeepSeek to call it instead of `read`, triggering repetitive failure loops.
 * @param tool - tool definition from harness.
 * @returns true if the tool can be safely served to DeepSeek.
 */
export function isSupportedDeepSeekTool(tool: { name: string; description: string }): boolean {
  const name = tool.name.toLowerCase()
  if (name === 'read_image' || name === 'view_image' || name === 'inspect_image') {
    return false
  }
  return true
}

/**
 * Build the OpenAI Chat Completions payload for Vertex Model Garden DeepSeek.
 * @param options - harness generate options.
 * @param config - project wire configuration.
 * @returns OpenAI-compatible request body.
 */
export function buildDeepSeekRequest(options: GenerateOptions, config: DeepSeekWireConfig): DeepSeekRequestBody {
  const model = normalizeDeepSeekModel(options.model)
  const messages = formatMessages(options)

  const tools: OpenAITool[] | undefined = options.tools && options.tools.length > 0
    ? options.tools
        .filter(isSupportedDeepSeekTool)
        .map(tool => ({
          type: 'function' as const,
          function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters,
          },
        }))
    : undefined

  const maxTokens = options.maxTokens ?? config.maxTokens

  return {
    model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: maxTokens,
    ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
    ...(options.stop && options.stop.length > 0 ? { stop: [...options.stop] } : {}),
    ...(tools ? { tools } : {}),
  }
}

/**
 * Map OpenAI finish_reason to harness FinishReason.
 * @param reason - raw finish_reason string from provider.
 * @param sawToolCall - whether the response contained a tool call.
 * @returns harness FinishReason.
 */
export function mapOpenAIFinishReason(reason: string | undefined, sawToolCall: boolean): FinishReason {
  if (sawToolCall || reason === 'tool_calls' || reason === 'function_call') {
    return { kind: 'tool-calls' }
  }
  switch (reason) {
    case 'length':
      return { kind: 'max-tokens' }
    case 'content_filter':
      return {
        kind: 'error',
        failure: { message: 'google-vertex: content filter triggered', code: 'SAFETY_BLOCKED' },
      }
    default:
      return { kind: 'stop' }
  }
}

/**
 * Map OpenAI usage counters to harness TokenUsage.
 * @param raw - raw usage object from provider SSE event.
 * @returns disjoint TokenUsage.
 */
export function mapOpenAIUsage(raw: unknown): TokenUsage {
  if (typeof raw !== 'object' || raw === null) {
    return { inputTokens: 0, outputTokens: 0 }
  }
  const usage = raw as Record<string, unknown>
  const prompt = typeof usage['prompt_tokens'] === 'number' && Number.isFinite(usage['prompt_tokens'])
    ? usage['prompt_tokens'] : 0
  const completion = typeof usage['completion_tokens'] === 'number' && Number.isFinite(usage['completion_tokens'])
    ? usage['completion_tokens'] : 0
  const total = typeof usage['total_tokens'] === 'number' && Number.isFinite(usage['total_tokens'])
    ? usage['total_tokens'] : prompt + completion

  const promptDetails = typeof usage['prompt_tokens_details'] === 'object' && usage['prompt_tokens_details'] !== null
    ? usage['prompt_tokens_details'] as Record<string, unknown> : undefined
  const cached = typeof promptDetails?.['cached_tokens'] === 'number' && Number.isFinite(promptDetails['cached_tokens'])
    ? promptDetails['cached_tokens'] : 0

  const completionDetails = typeof usage['completion_tokens_details'] === 'object' && usage['completion_tokens_details'] !== null
    ? usage['completion_tokens_details'] as Record<string, unknown> : undefined
  const reasoning = typeof completionDetails?.['reasoning_tokens'] === 'number' && Number.isFinite(completionDetails['reasoning_tokens'])
    ? completionDetails['reasoning_tokens'] : 0

  return {
    inputTokens: Math.max(0, prompt - cached),
    outputTokens: completion,
    totalTokens: total,
    ...(cached > 0 ? { cacheReadTokens: cached } : {}),
    ...(reasoning > 0 ? { reasoningTokens: reasoning } : {}),
  }
}

/** An active streaming tool call being assembled. */
interface ActiveToolCall {
  blockIndex: number
  id: string
  name: string
  arguments: string
}

/** An active content block being assembled. */
interface ActiveBlock {
  type: 'reasoning' | 'text'
  index: number
  text: string
}

/**
 * Statefully translates OpenAI-compatible SSE events from DeepSeek into harness StreamChunks.
 *
 * Handles:
 * - Direct `delta.reasoning_content` (DeepSeek-R1 standard field)
 * - Inline `<think>...</think>` tags in `delta.content` (streaming tag parsing across chunk boundaries)
 * - Streaming tool calls (`delta.tool_calls`) with argument delta accumulation
 * - `finish_reason` mapping ('stop', 'tool_calls', 'length')
 * - Usage accounting
 * - `data: [DONE]` terminal event
 */
export class DeepSeekStreamTranslator {
  readonly #model: string
  #activeBlock: ActiveBlock | undefined
  #nextIndex = 0
  #toolCalls = new Map<number, ActiveToolCall>()
  #sawToolCall = false
  #usage: TokenUsage | undefined
  #usageReported = false
  #rawFinishReason: string | undefined
  #sawFinish = false
  #terminal = false
  #failed = false
  #blocksEmitted = false
  #inThinkTag = false
  #tagBuffer = ''

  constructor(model: string) {
    this.#model = model
  }

  /** The model identifier this translator was constructed for. */
  get model(): string {
    return this.#model
  }

  /** Whether the stream has completed with a terminal chunk. */
  get terminal(): boolean {
    return this.#terminal
  }

  /** Whether a finish event or indicator was observed. */
  get sawFinish(): boolean {
    return this.#sawFinish
  }

  /**
   * Feed one decoded SSE record.
   * @param event - parsed SSE payload or [DONE] indicator.
   * @returns array of completed harness chunks.
   */
  handle(event: Record<string, unknown>): StreamChunk[] {
    if (this.#terminal || this.#failed) return []

    // Provider error envelope
    if (event['error'] !== undefined) {
      this.#failed = true
      this.#terminal = true
      return [{ type: 'finish', reason: { kind: 'error', failure: failureForEvent(event['error']) } }]
    }

    // Done signal ([DONE])
    if (event['done'] === true || event['type'] === 'done') {
      return this.finish()
    }

    const out: StreamChunk[] = []

    // Usage reporting
    if (event['usage'] !== undefined && typeof event['usage'] === 'object' && event['usage'] !== null) {
      this.#usage = mapOpenAIUsage(event['usage'])
    }

    const choices = event['choices']
    if (Array.isArray(choices)) {
      for (const choice of choices) {
        if (typeof choice !== 'object' || choice === null) continue
        const fields = choice as Record<string, unknown>

        const finishReason = fields['finish_reason']
        if (typeof finishReason === 'string') {
          this.#rawFinishReason = finishReason
          this.#sawFinish = true
        }

        const delta = fields['delta']
        if (typeof delta === 'object' && delta !== null) {
          const deltaObj = delta as Record<string, unknown>

          // 1. Check reasoning_content (DeepSeek-R1 field)
          const reasoning = deltaObj['reasoning_content'] ?? deltaObj['reasoning']
          if (typeof reasoning === 'string' && reasoning.length > 0) {
            // Strip any accidental <think> / </think> wrappers
            const cleaned = reasoning.replace(/<\/?think>/g, '')
            if (cleaned.length > 0) {
              out.push(...this.#appendReasoning(cleaned))
            }
          }

          // 2. Check regular text content
          const content = deltaObj['content']
          if (typeof content === 'string' && content.length > 0) {
            // If we were previously emitting via reasoning_content and now get regular content,
            // ensure the reasoning block is closed
            if (this.#activeBlock?.type === 'reasoning' && !this.#inThinkTag) {
              out.push(...this.#closeActiveBlock())
            }
            out.push(...this.#processContentWithThinkTags(content))
          }

          // 3. Check tool calls
          const toolCalls = deltaObj['tool_calls']
          if (Array.isArray(toolCalls)) {
            out.push(...this.#closeActiveBlock())
            for (const tc of toolCalls) {
              if (typeof tc !== 'object' || tc === null) continue
              out.push(...this.#handleToolCallDelta(tc as Record<string, unknown>))
            }
          }
        }
      }
    }

    return out
  }

  /**
   * Finalize the stream and emit terminal blocks, usage, and finish reason.
   */
  finish(): StreamChunk[] {
    if (this.#terminal) return []

    const out: StreamChunk[] = []

    // Flush any leftover text in the tag buffer
    if (this.#tagBuffer.length > 0) {
      if (this.#inThinkTag) {
        out.push(...this.#appendReasoning(this.#tagBuffer))
      } else {
        out.push(...this.#appendText(this.#tagBuffer))
      }
      this.#tagBuffer = ''
    }

    // Close any open content block
    out.push(...this.#closeActiveBlock())

    // Close all open tool calls
    for (const active of this.#toolCalls.values()) {
      const sanitized = sanitizeDeepSeekText(active.arguments)
      out.push({
        type: 'block-end',
        index: active.blockIndex,
        block: {
          type: 'tool-call',
          id: active.id,
          name: active.name,
          arguments: sanitized.length > 0 ? sanitized : '{}',
        },
      })
      this.#blocksEmitted = true
    }
    this.#toolCalls.clear()

    // Emit usage if present and not yet emitted
    if (this.#usage !== undefined && !this.#usageReported) {
      out.push({ type: 'usage', usage: this.#usage })
      this.#usageReported = true
    }

    // Check for empty response
    if (!this.#blocksEmitted) {
      this.#terminal = true
      out.push({
        type: 'finish',
        reason: {
          kind: 'error',
          failure: {
            message: 'google-vertex: model produced no content',
            code: EMPTY_RESPONSE_CODE,
          },
        },
      })
      return out
    }

    // Emit terminal finish chunk
    const reason = mapOpenAIFinishReason(this.#rawFinishReason, this.#sawToolCall)
    out.push({ type: 'finish', reason })
    this.#terminal = true
    return out
  }

  /** Close the currently open reasoning or text block. */
  #closeActiveBlock(): StreamChunk[] {
    if (this.#activeBlock === undefined) return []
    const block = this.#activeBlock
    this.#activeBlock = undefined
    this.#blocksEmitted = true
    return [{
      type: 'block-end',
      index: block.index,
      block: block.type === 'reasoning'
        ? { type: 'reasoning', text: block.text }
        : { type: 'text', text: block.text },
    }]
  }

  /** Ensure a content block of the given type is active. */
  #ensureBlock(type: 'reasoning' | 'text'): StreamChunk[] {
    if (this.#activeBlock !== undefined && this.#activeBlock.type === type) {
      return []
    }
    const out: StreamChunk[] = []
    if (this.#activeBlock !== undefined) {
      out.push(...this.#closeActiveBlock())
    }
    const index = this.#nextIndex++
    this.#activeBlock = { type, index, text: '' }
    out.push({ type: 'block-start', index, blockType: type })
    return out
  }

  /** Append reasoning text to an active reasoning block. */
  #appendReasoning(text: string): StreamChunk[] {
    if (text.length === 0) return []
    const out: StreamChunk[] = []
    out.push(...this.#ensureBlock('reasoning'))
    this.#activeBlock!.text += text
    out.push({ type: 'reasoning-delta', index: this.#activeBlock!.index, text })
    return out
  }

  /** Append visible text to an active text block. */
  #appendText(text: string): StreamChunk[] {
    if (text.length === 0) return []
    const out: StreamChunk[] = []
    out.push(...this.#ensureBlock('text'))
    this.#activeBlock!.text += text
    out.push({ type: 'text-delta', index: this.#activeBlock!.index, text })
    return out
  }

  /**
   * Stream-parse `<think>...</think>` tags inside delta.content.
   * Handles tags split across multiple chunks.
   */
  #processContentWithThinkTags(chunk: string): StreamChunk[] {
    const cleanChunk = sanitizeDeepSeekText(chunk)
    if (cleanChunk.length === 0) return []
    let input = this.#tagBuffer + cleanChunk
    this.#tagBuffer = ''
    const out: StreamChunk[] = []

    while (input.length > 0) {
      if (!this.#inThinkTag) {
        // Outside <think>: looking for <think> tag
        const openIdx = input.indexOf('<think>')
        if (openIdx !== -1) {
          const before = input.slice(0, openIdx)
          if (before.length > 0) {
            out.push(...this.#appendText(before))
          }
          out.push(...this.#closeActiveBlock())
          this.#inThinkTag = true
          input = input.slice(openIdx + 7) // 7 = '<think>'.length
          continue
        }

        // Check if input ends with a partial prefix of '<think>'
        let matchedPrefix = false
        const maxLen = Math.min(6, input.length)
        for (let len = maxLen; len >= 1; len--) {
          const tail = input.slice(-len)
          if ('<think>'.startsWith(tail)) {
            this.#tagBuffer = tail
            const before = input.slice(0, -len)
            if (before.length > 0) {
              out.push(...this.#appendText(before))
            }
            matchedPrefix = true
            break
          }
        }
        if (!matchedPrefix) {
          out.push(...this.#appendText(input))
        }
        break
      } else {
        // Inside <think>: looking for </think> tag
        const closeIdx = input.indexOf('</think>')
        if (closeIdx !== -1) {
          const before = input.slice(0, closeIdx)
          if (before.length > 0) {
            out.push(...this.#appendReasoning(before))
          }
          out.push(...this.#closeActiveBlock())
          this.#inThinkTag = false
          input = input.slice(closeIdx + 8) // 8 = '</think>'.length
          continue
        }

        // Check if input ends with a partial prefix of '</think>'
        let matchedPrefix = false
        const maxLen = Math.min(7, input.length)
        for (let len = maxLen; len >= 1; len--) {
          const tail = input.slice(-len)
          if ('</think>'.startsWith(tail)) {
            this.#tagBuffer = tail
            const before = input.slice(0, -len)
            if (before.length > 0) {
              out.push(...this.#appendReasoning(before))
            }
            matchedPrefix = true
            break
          }
        }
        if (!matchedPrefix) {
          out.push(...this.#appendReasoning(input))
        }
        break
      }
    }

    return out
  }

  /**
   * Handle one delta.tool_calls item.
   */
  #handleToolCallDelta(tc: Record<string, unknown>): StreamChunk[] {
    const out: StreamChunk[] = []
    const tIndex = typeof tc['index'] === 'number' ? tc['index'] : 0
    const fn = typeof tc['function'] === 'object' && tc['function'] !== null
      ? tc['function'] as Record<string, unknown> : undefined
    const rawArgsDelta = typeof fn?.['arguments'] === 'string' ? fn['arguments'] : ''
    const argsDelta = sanitizeDeepSeekText(rawArgsDelta)
    const fnName = typeof fn?.['name'] === 'string' ? fn['name'] : undefined
    const tcId = typeof tc['id'] === 'string' ? tc['id'] : undefined

    let active = this.#toolCalls.get(tIndex)
    if (active === undefined) {
      const blockIndex = this.#nextIndex++
      const id = tcId ?? `call_${blockIndex}`
      const name = fnName ?? ''
      active = { blockIndex, id, name, arguments: argsDelta }
      this.#toolCalls.set(tIndex, active)
      this.#sawToolCall = true

      out.push({ type: 'block-start', index: blockIndex, blockType: 'tool-call' })
      out.push({
        type: 'tool-call-delta',
        index: blockIndex,
        id,
        ...(name.length > 0 ? { name } : {}),
        argumentsDelta: argsDelta,
      })
    } else {
      if (tcId !== undefined && active.id.startsWith('call_')) {
        active.id = tcId
      }
      if (fnName !== undefined && active.name.length === 0) {
        active.name = fnName
      }
      active.arguments += argsDelta
      out.push({
        type: 'tool-call-delta',
        index: active.blockIndex,
        id: active.id,
        ...(fnName !== undefined ? { name: fnName } : {}),
        argumentsDelta: argsDelta,
      })
    }

    return out
  }
}
