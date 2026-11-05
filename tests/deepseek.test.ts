/**
 * Unit tests for DeepSeek wire translation and stream handling: request
 * serialization, reasoning translation (reasoning_content and <think> tags),
 * streaming tool calls, usage accounting, and finish mapping.
 *
 * @module dsh-google-vertex/tests/deepseek
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { EMPTY_RESPONSE_CODE } from '@deepseek-ai/dsh-llm'

import {
  buildDeepSeekRequest,
  DeepSeekStreamTranslator,
  DEFAULT_DEEPSEEK_CONTEXT_WINDOW,
  DEFAULT_DEEPSEEK_MAX_TOKENS,
  DEFAULT_DEEPSEEK_MODELS,
  deepseekEndpointFor,
  isSupportedDeepSeekTool,
  mapOpenAIFinishReason,
  mapOpenAIUsage,
  normalizeDeepSeekModel,
  sanitizeDeepSeekText,
} from '../src/deepseek.ts'
import type { GenerateOptions } from '../src/host.ts'
import type { VertexWireConfig } from '../src/wire.ts'

const WIRE: VertexWireConfig = {
  project: 'test-project',
  location: 'us-central1',
  maxTokens: DEFAULT_DEEPSEEK_MAX_TOKENS,
}

test('DEFAULT_DEEPSEEK_MODELS and capacities are exported correctly', () => {
  assert.equal(DEFAULT_DEEPSEEK_CONTEXT_WINDOW, 128_000)
  assert.equal(DEFAULT_DEEPSEEK_MAX_TOKENS, 16_384)
  assert.equal(DEFAULT_DEEPSEEK_MODELS.length, 3)
  assert.equal(DEFAULT_DEEPSEEK_MODELS[0]?.id, 'deepseek-ai/DeepSeek-V3')
  assert.equal(DEFAULT_DEEPSEEK_MODELS[1]?.id, 'deepseek-ai/DeepSeek-R1')
  assert.equal(DEFAULT_DEEPSEEK_MODELS[2]?.id, 'deepseek-ai/deepseek-v4.1-flash-maas')
})

test('normalizeDeepSeekModel handles aliases and preserves custom IDs', () => {
  assert.equal(normalizeDeepSeekModel('deepseek-v3'), 'deepseek-ai/DeepSeek-V3')
  assert.equal(normalizeDeepSeekModel('DEEPSEEK-V3'), 'deepseek-ai/DeepSeek-V3')
  assert.equal(normalizeDeepSeekModel('deepseek-ai/deepseek-v3'), 'deepseek-ai/DeepSeek-V3')
  assert.equal(normalizeDeepSeekModel('deepseek-r1'), 'deepseek-ai/DeepSeek-R1')
  assert.equal(normalizeDeepSeekModel('DeepSeek-R1'), 'deepseek-ai/DeepSeek-R1')
  assert.equal(normalizeDeepSeekModel('deepseek-ai/deepseek-r1'), 'deepseek-ai/DeepSeek-R1')
  assert.equal(normalizeDeepSeekModel('deepseek-v4.1-flash'), 'deepseek-ai/deepseek-v4.1-flash-maas')
  assert.equal(normalizeDeepSeekModel('deepseek-ai/deepseek-v4.1-flash-maas'), 'deepseek-ai/deepseek-v4.1-flash-maas')
  assert.equal(normalizeDeepSeekModel('custom-model-id'), 'custom-model-id')
  assert.equal(normalizeDeepSeekModel(''), 'deepseek-ai/DeepSeek-V3')
})

test('deepseekEndpointFor formats the Vertex Model Garden endpoint', () => {
  const url = deepseekEndpointFor('my-project', 'us-central1')
  assert.equal(
    url,
    'https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/endpoints/openapi/chat/completions',
  )
})

test('buildDeepSeekRequest serializes options into OpenAI chat completions payload', () => {
  const options: GenerateOptions = {
    model: 'deepseek-r1',
    system: 'You are a helpful assistant.',
    temperature: 0.7,
    maxTokens: 4096,
    stop: ['###'],
    messages: [
      { id: '1', role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      {
        id: '2',
        role: 'assistant',
        content: [
          { type: 'reasoning', text: 'user said hello' },
          { type: 'text', text: 'Hi there!' },
        ],
      },
      {
        id: '3',
        role: 'assistant',
        content: [
          { type: 'tool-call', id: 'call_1', name: 'lookup', arguments: '{"q":"weather"}' },
        ],
      },
      {
        id: '4',
        role: 'tool',
        toolCallId: 'call_1',
        content: [{ type: 'text', text: 'sunny' }],
      },
    ],
    tools: [
      {
        name: 'lookup',
        description: 'Lookup info',
        parameters: { type: 'object', properties: { q: { type: 'string' } }, required: ['q'] },
      },
    ],
  }

  const body = buildDeepSeekRequest(options, WIRE)
  assert.equal(body.model, 'deepseek-ai/DeepSeek-R1')
  assert.equal(body.stream, true)
  assert.deepEqual(body.stream_options, { include_usage: true })
  assert.equal(body.temperature, 0.7)
  assert.equal(body.max_tokens, 4096)
  assert.deepEqual(body.stop, ['###'])
  assert.equal(body.tools?.length, 1)
  assert.equal(body.tools[0]?.function.name, 'lookup')

  assert.deepEqual(body.messages, [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hello' },
    { role: 'assistant', content: 'Hi there!' },
    {
      role: 'assistant',
      content: null,
      tool_calls: [
        {
          id: 'call_1',
          type: 'function',
          function: { name: 'lookup', arguments: '{"q":"weather"}' },
        },
      ],
    },
    { role: 'tool', tool_call_id: 'call_1', content: 'sunny' },
  ])
})

test('mapOpenAIFinishReason maps stop, length, and tool calls', () => {
  assert.deepEqual(mapOpenAIFinishReason('stop', false), { kind: 'stop' })
  assert.deepEqual(mapOpenAIFinishReason('length', false), { kind: 'max-tokens' })
  assert.deepEqual(mapOpenAIFinishReason('tool_calls', false), { kind: 'tool-calls' })
  assert.deepEqual(mapOpenAIFinishReason('function_call', false), { kind: 'tool-calls' })
  assert.deepEqual(mapOpenAIFinishReason('stop', true), { kind: 'tool-calls' })
  assert.equal(mapOpenAIFinishReason('content_filter', false).kind, 'error')
})

test('mapOpenAIUsage maps prompt, completion, cached, and reasoning tokens', () => {
  const usage = mapOpenAIUsage({
    prompt_tokens: 100,
    completion_tokens: 50,
    total_tokens: 150,
    prompt_tokens_details: { cached_tokens: 40 },
    completion_tokens_details: { reasoning_tokens: 30 },
  })

  assert.deepEqual(usage, {
    inputTokens: 60,
    outputTokens: 50,
    totalTokens: 150,
    cacheReadTokens: 40,
    reasoningTokens: 30,
  })
})

test('DeepSeekStreamTranslator streams regular text chunks', () => {
  const translator = new DeepSeekStreamTranslator('deepseek-ai/DeepSeek-V3')
  const chunks = [
    ...translator.handle({ choices: [{ delta: { content: 'Hello' }, finish_reason: null }] }),
    ...translator.handle({ choices: [{ delta: { content: ' world!' }, finish_reason: null }] }),
    ...translator.handle({
      choices: [{ delta: {}, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    }),
    ...translator.finish(),
  ]

  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hello' },
    { type: 'text-delta', index: 0, text: ' world!' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello world!' } },
    { type: 'usage', usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
  assert.equal(translator.terminal, true)
})

test('DeepSeekStreamTranslator translates delta.reasoning_content for DeepSeek-R1', () => {
  const translator = new DeepSeekStreamTranslator('deepseek-ai/DeepSeek-R1')
  const chunks = [
    ...translator.handle({ choices: [{ delta: { reasoning_content: 'Let me think' }, finish_reason: null }] }),
    ...translator.handle({ choices: [{ delta: { reasoning_content: ' step by step.' }, finish_reason: null }] }),
    ...translator.handle({ choices: [{ delta: { content: 'The result is 42.' }, finish_reason: null }] }),
    ...translator.handle({
      choices: [{ delta: {}, finish_reason: 'stop' }],
      usage: {
        prompt_tokens: 20,
        completion_tokens: 15,
        total_tokens: 35,
        completion_tokens_details: { reasoning_tokens: 8 },
      },
    }),
    ...translator.finish(),
  ]

  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'Let me think' },
    { type: 'reasoning-delta', index: 0, text: ' step by step.' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'Let me think step by step.' } },
    { type: 'block-start', index: 1, blockType: 'text' },
    { type: 'text-delta', index: 1, text: 'The result is 42.' },
    { type: 'block-end', index: 1, block: { type: 'text', text: 'The result is 42.' } },
    {
      type: 'usage',
      usage: { inputTokens: 20, outputTokens: 15, totalTokens: 35, reasoningTokens: 8 },
    },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('DeepSeekStreamTranslator translates inline <think> tags split across chunks', () => {
  const translator = new DeepSeekStreamTranslator('deepseek-ai/DeepSeek-R1')
  const chunks = [
    ...translator.handle({ choices: [{ delta: { content: '<th' }, finish_reason: null }] }),
    ...translator.handle({ choices: [{ delta: { content: 'ink>Pondering deeply</th' }, finish_reason: null }] }),
    ...translator.handle({ choices: [{ delta: { content: 'ink>\nHere is the answer.' }, finish_reason: null }] }),
    ...translator.handle({ choices: [{ delta: {}, finish_reason: 'stop' }] }),
    ...translator.finish(),
  ]

  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'Pondering deeply' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'Pondering deeply' } },
    { type: 'block-start', index: 1, blockType: 'text' },
    { type: 'text-delta', index: 1, text: '\nHere is the answer.' },
    { type: 'block-end', index: 1, block: { type: 'text', text: '\nHere is the answer.' } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
})

test('DeepSeekStreamTranslator closes unclosed <think> tag on max-tokens truncation', () => {
  const translator = new DeepSeekStreamTranslator('deepseek-ai/DeepSeek-R1')
  const chunks = [
    ...translator.handle({ choices: [{ delta: { content: '<think>Still thinking...' }, finish_reason: null }] }),
    ...translator.handle({ choices: [{ delta: {}, finish_reason: 'length' }] }),
    ...translator.finish(),
  ]

  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'Still thinking...' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'Still thinking...' } },
    { type: 'finish', reason: { kind: 'max-tokens' } },
  ])
})

test('DeepSeekStreamTranslator handles streaming tool calls with argument accumulation', () => {
  const translator = new DeepSeekStreamTranslator('deepseek-ai/DeepSeek-V3')
  const chunks = [
    ...translator.handle({
      choices: [{
        delta: {
          tool_calls: [{
            index: 0,
            id: 'call_abc123',
            type: 'function',
            function: { name: 'get_weather', arguments: '{"city":' },
          }],
        },
        finish_reason: null,
      }],
    }),
    ...translator.handle({
      choices: [{
        delta: {
          tool_calls: [{
            index: 0,
            function: { arguments: '"Kyiv"}' },
          }],
        },
        finish_reason: null,
      }],
    }),
    ...translator.handle({
      choices: [{ delta: {}, finish_reason: 'tool_calls' }],
    }),
    ...translator.finish(),
  ]

  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'tool-call' },
    { type: 'tool-call-delta', index: 0, id: 'call_abc123', name: 'get_weather', argumentsDelta: '{"city":' },
    { type: 'tool-call-delta', index: 0, id: 'call_abc123', argumentsDelta: '"Kyiv"}' },
    {
      type: 'block-end',
      index: 0,
      block: { type: 'tool-call', id: 'call_abc123', name: 'get_weather', arguments: '{"city":"Kyiv"}' },
    },
    { type: 'finish', reason: { kind: 'tool-calls' } },
  ])
})

test('DeepSeekStreamTranslator handles [DONE] terminal chunk', () => {
  const translator = new DeepSeekStreamTranslator('deepseek-ai/DeepSeek-V3')
  const chunks = [
    ...translator.handle({ choices: [{ delta: { content: 'Done text' }, finish_reason: 'stop' }] }),
    ...translator.handle({ done: true }),
  ]

  assert.equal(translator.terminal, true)
  assert.equal(chunks.at(-1)?.type, 'finish')
})

test('DeepSeekStreamTranslator handles provider error envelope', () => {
  const translator = new DeepSeekStreamTranslator('deepseek-ai/DeepSeek-V3')
  const chunks = translator.handle({
    error: { message: 'Quota exceeded', code: 429, status: 'RESOURCE_EXHAUSTED' },
  })

  assert.equal(translator.terminal, true)
  assert.equal(chunks.length, 1)
  assert.equal(chunks[0]?.type, 'finish')
  assert.equal((chunks[0] as { reason: { kind: string } }).reason.kind, 'error')
})

test('DeepSeekStreamTranslator reports empty response when model produces no content', () => {
  const translator = new DeepSeekStreamTranslator('deepseek-ai/DeepSeek-V3')
  const chunks = [
    ...translator.handle({ choices: [{ delta: {}, finish_reason: 'stop' }] }),
    ...translator.finish(),
  ]

  assert.equal(translator.terminal, true)
  assert.equal(chunks.length, 1)
  assert.equal(chunks[0]?.type, 'finish')
  const failure = (chunks[0] as { reason: { failure?: { code: string } } }).reason.failure
  assert.equal(failure?.code, EMPTY_RESPONSE_CODE)
})

test('sanitizeDeepSeekText strips leaked DSML tags and special token delimiters', () => {
  assert.equal(
    sanitizeDeepSeekText('  readonly contextWindow: number</｜DSML｜ parameter>, '),
    '  readonly contextWindow: number, ',
  )
  assert.equal(sanitizeDeepSeekText('</｜DSML｜ parameter>'), '')
  assert.equal(sanitizeDeepSeekText('<｜tool call begin｜>function<｜tool sep｜>edit'), 'functionedit')
  assert.equal(
    sanitizeDeepSeekText('<｜DSML｜ parameter name=path>src/index.ts</｜DSML｜ parameter>'),
    'src/index.ts',
  )
  assert.equal(sanitizeDeepSeekText('const x = 1;</parameter>'), 'const x = 1;')
  assert.equal(sanitizeDeepSeekText('regular text without tags'), 'regular text without tags')
})

test('DeepSeekStreamTranslator strips leaked DSML tags from tool arguments and content', () => {
  const translator = new DeepSeekStreamTranslator('deepseek-ai/deepseek-v4.1-flash-maas')
  const chunks = [
    ...translator.handle({
      choices: [{
        delta: {
          tool_calls: [{
            index: 0,
            id: 'call_edit1',
            type: 'function',
            function: {
              name: 'edit',
              arguments: '{"new_string": "readonly contextWindow: number</｜DSML｜ parameter>, "}',
            },
          }],
        },
        finish_reason: null,
      }],
    }),
    ...translator.handle({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] }),
    ...translator.finish(),
  ]

  const delta = chunks.find(c => c.type === 'tool-call-delta') as { argumentsDelta: string } | undefined
  assert.ok(delta)
  assert.equal(delta.argumentsDelta, '{"new_string": "readonly contextWindow: number, "}')

  const end = chunks.find(c => c.type === 'block-end') as { block: { arguments: string } } | undefined
  assert.ok(end)
  assert.equal(end.block.arguments, '{"new_string": "readonly contextWindow: number, "}')
})

test('buildDeepSeekRequest filters out read_image to prevent hallucination loops', () => {
  const options: GenerateOptions = {
    model: 'deepseek-v4.1-flash',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'Read index.ts' }] }],
    tools: [
      { name: 'read', description: 'Read text file', parameters: { type: 'object' } },
      { name: 'read_image', description: 'Read image file', parameters: { type: 'object' } },
      { name: 'edit', description: 'Edit file', parameters: { type: 'object' } },
    ],
  }
  const body = buildDeepSeekRequest(options, WIRE)
  assert.equal(body.tools?.length, 2)
  assert.deepEqual(body.tools?.map(t => t.function.name), ['read', 'edit'])
  assert.equal(isSupportedDeepSeekTool({ name: 'read_image', description: '' }), false)
  assert.equal(isSupportedDeepSeekTool({ name: 'read', description: '' }), true)
})
