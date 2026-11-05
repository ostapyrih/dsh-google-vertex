/**
 * DeepSeek adapter tests over a stubbed transport: request URL and authorization,
 * streaming reasoning and text deltas, tool calls, and failure handling.
 *
 * @module dsh-google-vertex/tests/deepseek-adapter
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import type { FetchLike, ServiceAccount } from '../src/auth.ts'
import {
  DEFAULT_DEEPSEEK_CONTEXT_WINDOW,
  DEFAULT_DEEPSEEK_MAX_TOKENS,
  DEFAULT_DEEPSEEK_MODELS,
} from '../src/deepseek.ts'
import {
  GoogleVertexDeepSeekAdapter,
  type DeepSeekAdapterConfig,
} from '../src/deepseek_adapter.ts'
import type { GenerateOptions } from '../src/host.ts'
import { DEFAULT_STREAM_IDLE_TIMEOUT_MS } from '../src/wire.ts'
import { collect, streamResponse, tokens } from './support.ts'

const CONFIG: DeepSeekAdapterConfig = {
  serviceAccount: { client_email: 'x@y', private_key: 'unused' } as ServiceAccount,
  project: 'my-project',
  location: 'us-central1',
  models: DEFAULT_DEEPSEEK_MODELS,
  contextWindow: DEFAULT_DEEPSEEK_CONTEXT_WINDOW,
  maxTokens: DEFAULT_DEEPSEEK_MAX_TOKENS,
  streamIdleTimeoutMs: DEFAULT_STREAM_IDLE_TIMEOUT_MS,
}

const OPTIONS_V3: GenerateOptions = {
  model: 'deepseek-v3',
  messages: [{ id: '1', role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
}

test('model metadata normalizes aliases and reports capacities', async () => {
  const adapter = new GoogleVertexDeepSeekAdapter(CONFIG, { tokens: tokens() })

  const flash = await adapter.resolveModel('google-vertex-deepseek', 'deepseek-v4.1-flash')
  assert.equal(flash.id, 'deepseek-v4.1-flash')
  assert.equal(flash.name, 'DeepSeek-V4.1-Flash (Vertex Model Garden)')
  assert.equal(flash.context?.contextWindow, 1_000_000)
  assert.equal(flash.defaultMaxTokens, 16_384)

  const v4 = await adapter.resolveModel('google-vertex-deepseek', 'deepseek-v4.1')
  assert.equal(v4.id, 'deepseek-v4.1')
  assert.equal(v4.name, 'DeepSeek-V4.1-Flash (Vertex Model Garden)')
  assert.equal(v4.context?.contextWindow, 1_000_000)

  // Verify canonical MaaS ID
  const canonical = await adapter.resolveModel('google-vertex-deepseek', 'deepseek-ai/deepseek-v4.1-flash-maas')
  assert.equal(canonical.id, 'deepseek-ai/deepseek-v4.1-flash-maas')
  assert.equal(canonical.name, 'DeepSeek-V4.1-Flash (Vertex Model Garden)')

  // Aliases outside catalog preserve ID and fall back to canonical model name
  const v3 = await adapter.resolveModel('google-vertex-deepseek', 'deepseek-v3')
  assert.equal(v3.id, 'deepseek-v3')
  assert.equal(v3.context?.contextWindow, 128_000)
  assert.equal(v3.defaultMaxTokens, 16_384)
})

test('prepareCall binds canonical model and stream function', async () => {
  const adapter = new GoogleVertexDeepSeekAdapter(CONFIG, { tokens: tokens() })
  const prep = await adapter.prepareCall('google-vertex-deepseek', 'deepseek-v4.1')

  assert.equal(prep.model.id, 'deepseek-v4.1')
  assert.equal(typeof prep.stream, 'function')
})

test('streams DeepSeek text through the OpenAPI endpoint with bearer token', async () => {
  const requests: { url: string; init: RequestInit }[] = []
  const fetch: FetchLike = (url, init) => {
    requests.push({ url, init })
    return Promise.resolve(streamResponse([
      'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
      'data: {"choices":[{"delta":{"content":" there!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\n',
      'data: [DONE]\n\n',
    ]))
  }

  const adapter = new GoogleVertexDeepSeekAdapter(CONFIG, { tokens: tokens('tok-123'), fetch })
  const chunks = await collect(adapter, OPTIONS_V3)

  assert.equal(requests.length, 1)
  const req = requests[0]!
  assert.equal(
    req.url,
    'https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/endpoints/openapi/chat/completions',
  )
  const headers = req.init.headers as Record<string, string>
  assert.equal(headers['authorization'], 'Bearer tok-123')
  assert.equal(headers['content-type'], 'application/json')
  assert.equal(headers['accept'], 'text/event-stream')

  const parsedBody = JSON.parse(String(req.init.body))
  assert.equal(parsedBody.model, 'deepseek-ai/DeepSeek-V3')
  assert.equal(parsedBody.stream, true)

  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hi' },
    { type: 'text-delta', index: 0, text: ' there!' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hi there!' } },
    { type: 'usage', usage: { inputTokens: 5, outputTokens: 3, totalTokens: 8 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('streams DeepSeek-R1 reasoning and answer through adapter', async () => {
  const fetch: FetchLike = () => Promise.resolve(streamResponse([
    'data: {"choices":[{"delta":{"reasoning_content":"Thinking deeply"},"finish_reason":null}]}\n\n',
    'data: {"choices":[{"delta":{"content":"Answer"},"finish_reason":"stop"}]}\n\n',
    'data: [DONE]\n\n',
  ]))

  const adapter = new GoogleVertexDeepSeekAdapter(CONFIG, { tokens: tokens(), fetch })
  const chunks = await collect(adapter, {
    model: 'deepseek-r1',
    messages: [{ id: '1', role: 'user', content: [{ type: 'text', text: 'Solve x' }] }],
  })

  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'Thinking deeply' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'Thinking deeply' } },
    { type: 'block-start', index: 1, blockType: 'text' },
    { type: 'text-delta', index: 1, text: 'Answer' },
    { type: 'block-end', index: 1, block: { type: 'text', text: 'Answer' } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('a provider 401 retires its bearer token before the next call', async () => {
  let invalidated: string | undefined
  const tokenSource = {
    get: () => Promise.resolve('stale-token'),
    invalidate: (token: string) => { invalidated = token },
  }
  const adapter = new GoogleVertexDeepSeekAdapter(CONFIG, {
    tokens: tokenSource,
    fetch: () => Promise.resolve(new Response('unauthorized', { status: 401 })),
  })

  const chunks = await collect(adapter, OPTIONS_V3)
  assert.equal(invalidated, 'stale-token')
  assert.equal(chunks.length, 1)
  const finish = chunks[0]
  assert.equal(finish?.type === 'finish' && finish.reason.kind === 'error' ? finish.reason.failure.code : undefined, 'AUTH')
})

test('a body that ends without finish reason is reported as transport truncation', async () => {
  const adapter = new GoogleVertexDeepSeekAdapter(CONFIG, {
    tokens: tokens(),
    fetch: () => Promise.resolve(streamResponse([
      'data: {"choices":[{"delta":{"content":"part"},"finish_reason":null}]}\n\n',
    ])),
  })

  const chunks = await collect(adapter, OPTIONS_V3)
  assert.equal(chunks.at(-1)?.type, 'finish')
  const last = chunks.at(-1) as { reason: { kind: string; failure: { code: string } } }
  assert.equal(last.reason.kind, 'error')
  assert.equal(last.reason.failure.code, 'TRANSPORT')
})
