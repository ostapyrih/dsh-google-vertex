/**
 * Dynamic model discovery for Vertex AI publisher endpoints.
 *
 * All three routes discover their catalogs from one API: the wildcard publisher
 * list `publishers/* /models`, which is the same call `gcloud ai model-garden
 * models list` makes. It answers for every publisher at once, so a single
 * paginated walk yields the Gemini, Claude, and DeepSeek catalogs instead of
 * three hardcoded lists.
 *
 * Two server-side filters matter:
 *
 * - `is_hf_wildcard(false)` drops the thousands of deploy-your-own
 *   Hugging Face catalog entries (`publishers/hf-* /models/...`), which are not
 *   servable over these routes.
 * - a publisher filter per route keeps each catalog small: `publishers/google`
 *   and `publishers/anthropic` are distinct publishers, while DeepSeek is
 *   published by `deepseek-ai`.
 *
 * `listAllVersions=True` is required, or the list returns one row per model and
 * the version-qualified MaaS ids never appear.
 *
 * Results are cached for five minutes so the model picker does not make a
 * network call on every open.
 *
 * @module dsh-google-vertex/discovery
 */

import type { FetchLike } from './auth.ts'
import type { TokenProvider, VertexModel } from './adapter.ts'
import { DEFAULT_GEMINI_MODELS } from './gemini.ts'
import { endpointOrigin } from './wire.ts'

/** How long a cached model list stays valid, in milliseconds. */
export const DEFAULT_CACHE_TTL_MS = 5 * 60 * 1000

/**
 * Vertex API version for the model list endpoint. `publishers.models.list`
 * exists only in v1beta1; v1 serves `get` alone.
 */
const API_VERSION = 'v1beta1'

/** Most catalog pages read before giving up on a misbehaving endpoint. */
const MAX_PAGES = 20

/**
 * Page size. Vertex caps `page_size` at 100 and refuses anything larger with
 * `INVALID_ARGUMENT: Page size should be non-negative and maximum size is 100`.
 */
const PAGE_SIZE = 100

/**
 * Region whose catalog is authoritative for DeepSeek MaaS.
 *
 * Measured: the `deepseek-ai` publisher appears under `us-central1` but not
 * under `global`, yet its models answer only on the global endpoint. The two
 * facts are independent, so the listing location is pinned here rather than
 * taken from configuration.
 */
const MAAS_CATALOG_LOCATION = 'us-central1'

/**
 * Publisher that serves Claude through Vertex's Anthropic route.
 */
export const ANTHROPIC_PUBLISHER = 'anthropic'

/**
 * Publisher that serves Gemini.
 */
export const GOOGLE_PUBLISHER = 'google'

/**
 * Publisher that serves DeepSeek MaaS models.
 */
export const DEEPSEEK_PUBLISHER = 'deepseek-ai'

/**
 * Id segments naming a Gemini variant this text route cannot drive: embedding,
 * speech, image output, transcription, the Live API, and the Omni/nano-banana
 * media models. Each is published under `publishers/google` beside the text
 * models but speaks a different API shape than `:streamGenerateContent`.
 */
const NON_TEXT_SEGMENTS: ReadonlySet<string> = new Set([
  'embedding',
  'tts',
  'image',
  'live',
  'audio',
  'transcribe',
  'translate',
  'omni',
  'banana',
  'vision',
  'veo',
  'imagen',
])

/**
 * Id segments naming a DeepSeek variant that is not a chat completion model.
 * The Model Garden catalog lists these beside the chat models.
 */
const NON_CHAT_DEEPSEEK_SEGMENTS: ReadonlySet<string> = new Set(['ocr', 'coder', 'math', 'vl', 'embedding', 'guard'])

/**
 * Claude families this route serves. The Anthropic publisher also lists
 * non-Claude experiment ids; an id must name a Claude family to be advertised.
 */
const CLAUDE_PREFIX = 'claude-'

/** `ListPublisherModelsResponse`, narrowed to the fields read here. */
interface PublisherModelsResponse {
  publisherModels?: PublisherModelEntry[]
  nextPageToken?: string
}

/**
 * One `PublisherModel`, narrowed to the fields read here. It carries no display
 * name, and its `supportedActions` describes console actions rather than API
 * verbs, so neither can name or filter a model.
 */
interface PublisherModelEntry {
  /** Resource name, e.g. `publishers/google/models/gemini-2.5-pro`. */
  name?: string
  /**
   * Full resource template, e.g.
   * `projects/{project}/locations/{location}/publishers/google/models/deepseek-v4.1-flash-maas@001`.
   * Its final segment is the id the MaaS completions endpoint expects.
   */
  publisherModelTemplate?: string
}

/** Cached model list with an expiry timestamp. */
interface CachedModels {
  models: readonly VertexModel[]
  expiresAt: number
}

/**
 * Extract the model id from a Vertex resource name.
 *
 * A resource name looks like `publishers/google/models/gemini-2.5-pro`: the
 * model id is the last segment. A name without a `/` is returned as-is.
 * @param resourceName - the `name` field from the catalog entry.
 * @returns the bare model id.
 */
function modelIdFromResource(resourceName: string): string {
  const lastSlash = resourceName.lastIndexOf('/')
  return lastSlash >= 0 ? resourceName.slice(lastSlash + 1) : resourceName
}

/**
 * Extract the publisher segment from a Vertex resource name.
 * @param resourceName - the `name` field, e.g. `publishers/anthropic/models/claude-opus-5`.
 * @returns the publisher, or `undefined` when the name is not publisher-shaped.
 */
function publisherFromResource(resourceName: string): string | undefined {
  const parts = resourceName.split('/')
  return parts[0] === 'publishers' && parts.length >= 4 ? parts[1] : undefined
}

/**
 * True for a Gemini id this route can serve: the publisher catalog also lists
 * Imagen, Veo, embedding, and speech models.
 * @param id - bare model id.
 */
function isGeminiTextModel(id: string): boolean {
  return id.startsWith('gemini-') && !id.split('-').some(segment => NON_TEXT_SEGMENTS.has(segment))
}

/**
 * True for a Claude id this route can serve.
 * @param id - bare model id.
 */
function isClaudeTextModel(id: string): boolean {
  return id.startsWith(CLAUDE_PREFIX) && !id.split('-').some(segment => NON_TEXT_SEGMENTS.has(segment))
}

/**
 * True for a DeepSeek id the OpenAI-compatible Serverless MaaS route can drive.
 * Filters out models that require dedicated GKE / Vertex GPU cluster deployment.
 * @param id - bare model id.
 */
function isDeepSeekChatModel(id: string): boolean {
  const lower = id.toLowerCase()
  return (lower.includes('flash-maas') || lower.endsWith('-maas'))
    && !id.split('-').some(segment => NON_CHAT_DEEPSEEK_SEGMENTS.has(segment))
}

/**
 * Build the URL that lists publisher models.
 *
 * For known publishers like `google` and `anthropic`, Vertex provides dedicated
 * endpoints `publishers/{publisher}/models`. For Model Garden open models like
 * DeepSeek, the wildcard `publishers/* /models` is used with a server-side filter.
 * @param location - region whose catalog to read.
 * @param publisher - publisher name, or `*` for the Model Garden wildcard.
 * @returns the absolute list URL.
 */
export function listModelsUrl(location: string, publisher?: string): string {
  const origin = endpointOrigin(location)
  if (publisher === undefined || publisher === '*') {
    const filter = 'is_hf_wildcard(false)'
    return `${origin}/${API_VERSION}/publishers/*/models`
      + `?pageSize=${PAGE_SIZE}&listAllVersions=True&filter=${encodeURIComponent(filter)}`
  }
  return `${origin}/${API_VERSION}/publishers/${encodeURIComponent(publisher)}/models`
}

/**
 * Fetch one page of publisher models from the Vertex Model Garden catalog.
 * @param url - the list endpoint URL, possibly with a pageToken.
 * @param token - bearer token for authentication.
 * @param fetchFn - transport.
 * @param signal - caller cancellation.
 * @returns the parsed response.
 */
async function fetchPage(
  url: string,
  token: string,
  fetchFn: FetchLike,
  signal?: AbortSignal,
): Promise<PublisherModelsResponse> {
  const response = await fetchFn(url, {
    method: 'GET',
    headers: {
      'authorization': `Bearer ${token}`,
      'accept': 'application/json',
    },
    ...signal === undefined ? {} : { signal },
  })
  if (!response.ok) {
    throw new Error(`google-vertex: model list failed (HTTP ${response.status})`)
  }
  const body: unknown = await response.json()
  if (typeof body !== 'object' || body === null) return {}
  return body as PublisherModelsResponse
}

/**
 * Walk the paginated publisher catalog for one publisher.
 *
 * Every entry is passed through `keep`, and a kept id is turned into a model
 * with `label`. Duplicate ids are collapsed, because `listAllVersions=True`
 * returns one row per published version.
 * @param publisher - publisher to keep, e.g. `google`.
 * @param location - configured Vertex region, or `global`.
 * @param tokens - token source for bearer authentication.
 * @param fetchFn - transport.
 * @param keep - predicate deciding whether an id is servable on this route.
 * @param label - display name for a kept id.
 * @param signal - caller cancellation.
 * @returns model ids and display names, in catalog order.
 */
export async function fetchPublisherModels(
  publisher: string,
  location: string,
  tokens: TokenProvider,
  fetchFn: FetchLike,
  keep: (id: string) => boolean,
  label: (id: string) => string,
  signal?: AbortSignal,
): Promise<readonly VertexModel[]> {
  const bearer = await tokens.get(signal)
  const models: VertexModel[] = []
  const seen = new Set<string>()
  const isWildcard = publisher === '*' || publisher === DEEPSEEK_PUBLISHER
  const url = isWildcard
    ? listModelsUrl(location, '*')
    : listModelsUrl(location, publisher)
  let pageToken: string | undefined

  for (let page = 0; page < MAX_PAGES; page++) {
    const separator = url.includes('?') ? '&' : '?'
    const pageUrl = pageToken !== undefined ? `${url}${separator}pageToken=${encodeURIComponent(pageToken)}` : url
    const response = await fetchPage(pageUrl, bearer, fetchFn, signal)

    for (const entry of response.publisherModels ?? []) {
      if (typeof entry.name !== 'string') continue
      // For wildcard endpoints, partition by publisher:
      if (isWildcard && publisherFromResource(entry.name) !== publisher) continue
      const id = modelIdFromResource(entry.name)
      if (id.length === 0 || seen.has(id) || !keep(id)) continue
      seen.add(id)
      models.push({ id, name: label(id) })
    }

    pageToken = response.nextPageToken
    if (pageToken === undefined || pageToken.length === 0) break
  }

  return models
}

/**
 * Fetch the Gemini text models from Vertex's publisher catalog.
 *
 * Keeps `gemini-` ids minus the variants named in {@link NON_TEXT_SEGMENTS}. A
 * built-in id keeps its built-in name; any other is named by its id.
 * @param location - configured Vertex region, or `global`.
 * @param tokens - token source for bearer authentication.
 * @param fetchFn - transport.
 * @param signal - caller cancellation.
 * @returns model ids and display names, in catalog order.
 */
export function fetchGeminiModels(
  location: string,
  tokens: TokenProvider,
  fetchFn: FetchLike,
  signal?: AbortSignal,
): Promise<readonly VertexModel[]> {
  return fetchPublisherModels(
    GOOGLE_PUBLISHER,
    location,
    tokens,
    fetchFn,
    isGeminiTextModel,
    id => DEFAULT_GEMINI_MODELS.find(model => model.id === id)?.name ?? `${id} (Vertex)`,
    signal,
  )
}

/**
 * Fetch the Claude models from Vertex's Anthropic publisher catalog.
 *
 * Vertex exposes `publishers/anthropic/models`, which is what makes an
 * auto-discovered Claude catalog possible; the picker therefore surfaces a
 * family Google has promoted without a plugin release.
 * @param location - configured Vertex region, or `global`.
 * @param tokens - token source for bearer authentication.
 * @param fetchFn - transport.
 * @param signal - caller cancellation.
 * @returns model ids and display names, in catalog order.
 */
export function fetchClaudeModels(
  location: string,
  tokens: TokenProvider,
  fetchFn: FetchLike,
  signal?: AbortSignal,
): Promise<readonly VertexModel[]> {
  return fetchPublisherModels(
    ANTHROPIC_PUBLISHER,
    location,
    tokens,
    fetchFn,
    isClaudeTextModel,
    id => `${claudeDisplayName(id)} (Vertex)`,
    signal,
  )
}

/**
 * Fetch the DeepSeek chat models published in Vertex Model Garden.
 *
 * These are MaaS models under the `deepseek-ai` publisher. Two measured facts
 * pull in opposite directions and both are load-bearing: the publisher is
 * listed under `us-central1` and absent from `global`, while the models answer
 * only on the global endpoint (a regional call returns
 * `FAILED_PRECONDITION: only available via global endpoint`). The listing is
 * therefore read from {@link MAAS_CATALOG_LOCATION}, never from configuration.
 * @param tokens - token source for bearer authentication.
 * @param fetchFn - transport.
 * @param signal - caller cancellation.
 * @returns model ids and display names, in catalog order.
 */
export async function fetchDeepSeekModels(
  tokens: TokenProvider,
  fetchFn: FetchLike,
  signal?: AbortSignal,
): Promise<readonly VertexModel[]> {
  const models = await fetchPublisherModels(
    DEEPSEEK_PUBLISHER,
    MAAS_CATALOG_LOCATION,
    tokens,
    fetchFn,
    isDeepSeekChatModel,
    id => `${deepseekDisplayName(id)} (Vertex Model Garden)`,
    signal,
  )
  return models.map(m => ({
    id: m.id.startsWith('deepseek-ai/') ? m.id : `deepseek-ai/${m.id}`,
    name: m.name,
  }))
}

/**
 * Turn a Claude model id into a picker label, e.g. `claude-opus-4-6` into
 * `Claude Opus 4.6`.
 * @param id - bare Claude model id.
 * @returns the human-readable family name, without the route suffix.
 */
export function claudeDisplayName(id: string): string {
  const rest = id.startsWith(CLAUDE_PREFIX) ? id.slice(CLAUDE_PREFIX.length) : id
  const words = rest.split('-').map((word) => {
    // A trailing version number stays attached to the word before it: `4-6`
    // becomes `4.6`, while `haiku-4-5` becomes `Haiku 4.5`.
    return /^\d+$/.test(word) ? word : word.charAt(0).toUpperCase() + word.slice(1)
  })
  // Join trailing numeric groups with dots: ['Opus','4','6'] -> 'Opus 4.6'.
  let name = ''
  for (const word of words) {
    if (/^\d+$/.test(word) && name.length > 0 && /\d$/.test(name)) name += `.${word}`
    else name += name.length === 0 ? word : ` ${word}`
  }
  return `Claude ${name}`
}

/**
 * Turn a DeepSeek model id into a picker label, e.g. `deepseek-v4.1-flash-maas`
 * into `DeepSeek-V4.1-Flash`.
 * @param id - bare DeepSeek model id.
 * @returns the human-readable name, without the route suffix.
 */
export function deepseekDisplayName(id: string): string {
  const rest = id.replace(/^deepseek-/i, '').replace(/-maas$/i, '')
  const parts = rest.split('-').map(part => (/^v?\d/.test(part) ? part.toUpperCase() : part.charAt(0).toUpperCase() + part.slice(1)))
  return `DeepSeek-${parts.join('-')}`
}

/**
 * A cached, TTL-bounded model list that falls back to a static default when
 * the remote fetch fails.
 */
export class ModelCache {
  readonly #fallback: readonly VertexModel[]
  #cached: CachedModels | undefined
  #inflight: Promise<readonly VertexModel[]> | undefined

  /**
   * @param fallback - static default returned when the fetch fails or is not
   *   attempted.
   */
  constructor(fallback: readonly VertexModel[]) {
    this.#fallback = fallback
  }

  /**
   * Return the cached model list, or fetch a fresh one.
   *
   * When a fetch function is provided and the cache is stale, it is called to
   * produce a fresh list. On failure, the fallback is returned. Concurrent
   * callers share one in-flight fetch.
   * @param fetchFn - optional async function that returns a fresh model list.
   * @returns the model list, from cache, fetch, or fallback.
   */
  async get(fetchFn?: () => Promise<readonly VertexModel[]>): Promise<readonly VertexModel[]> {
    // Return cached if still valid.
    const cached = this.#cached
    if (cached !== undefined && Date.now() < cached.expiresAt) return cached.models

    // No fetch function means static-only.
    if (fetchFn === undefined) return this.#fallback

    // Share one in-flight fetch.
    if (this.#inflight !== undefined) return this.#inflight

    const operation = fetchFn().then((models) => {
      if (models.length > 0) {
        if (this.#inflight === operation) this.#cached = { models, expiresAt: Date.now() + DEFAULT_CACHE_TTL_MS }
        return models
      }
      // Empty result: use fallback rather than showing nothing.
      return this.#fallback
    }).catch(() => {
      // Network failure: serve fallback silently.
      return this.#cached?.models ?? this.#fallback
    }).finally(() => {
      if (this.#inflight === operation) this.#inflight = undefined
    })
    this.#inflight = operation
    return operation
  }

  /** Force the next `get` to re-fetch. */
  invalidate(): void {
    this.#cached = undefined
    this.#inflight = undefined
  }
}
