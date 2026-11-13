/**
 * dsh-google-vertex: use Google-hosted models from Vertex AI inside DeepSeek
 * Harness, authenticated with a service-account file.
 *
 * Two capabilities, one configuration row: an `ctx.llm` provider adapter for
 * Vertex's Anthropic publisher endpoint (Claude) and one for its Google
 * publisher endpoint (Gemini). Registering them makes both routes selectable in
 * the Web client's model picker, because `buildModelCatalog()` enumerates
 * `ctx.llm.listProviders()` and asks each adapter for `listModels()` /
 * `resolveModel()`.
 *
 * A third: the `google-vertex` settings namespace. The Gemini adapter discovers
 * its catalog at runtime behind a five-minute cache, and a browser half has no
 * other way to reach this process, so the namespace's one write (the Refresh
 * control on this plugin's row page under Plugins) is what drops that cache
 * on demand.
 *
 * The credential is the service-account JSON itself: a path in configuration,
 * or `GOOGLE_APPLICATION_CREDENTIALS` in the launch environment. Nothing is
 * copied into the harness credential store, and the file is read once at mount
 * so a typo fails loudly instead of on the first message.
 *
 * @module dsh-google-vertex
 */

import type { Volatile } from '@deepseek-ai/cordis'
import Schema from '@deepseek-ai/schemastery'
import { GoogleVertexAnthropicAdapter } from './adapter.ts'
import type { VertexAnthropicConfig, VertexModel } from './adapter.ts'
import { loadServiceAccount } from './auth.ts'
import { ServiceAccountTokens } from './auth.ts'
import {
  DEFAULT_DEEPSEEK_CONTEXT_WINDOW,
  DEFAULT_DEEPSEEK_MAX_TOKENS,
  DEFAULT_DEEPSEEK_MODELS,
  normalizeDeepSeekModel,
} from './deepseek.ts'
import { GoogleVertexDeepSeekAdapter } from './deepseek_adapter.ts'
import type { DeepSeekAdapterConfig } from './deepseek_adapter.ts'
import { fetchClaudeModels, fetchDeepSeekModels, fetchGeminiModels } from './discovery.ts'
import { DEFAULT_GEMINI_CONTEXT_WINDOW, DEFAULT_GEMINI_MAX_TOKENS, DEFAULT_GEMINI_MODELS } from './gemini.ts'
import { GoogleVertexGeminiAdapter } from './gemini_adapter.ts'
import type { GeminiAdapterConfig } from './gemini_adapter.ts'
import { DEFAULT_LOCATION, DEFAULT_STREAM_IDLE_TIMEOUT_MS, MAX_TIMER_DELAY_MS } from './wire.ts'
import type { HostContext } from './host.ts'

/** Plugin name as it appears in the loader. */
export const name = 'google-vertex'

/** The `ctx.llm` route serving Google-hosted Anthropic (Claude) models. */
export const PROVIDER = 'google-vertex-anthropic'

/** The `ctx.llm` route serving Gemini models. */
export const GEMINI_PROVIDER = 'google-vertex-gemini'

/** The `ctx.llm` route serving DeepSeek models in Vertex Model Garden. */
export const DEEPSEEK_PROVIDER = 'google-vertex-deepseek'

export {
  DEFAULT_DEEPSEEK_CONTEXT_WINDOW,
  DEFAULT_DEEPSEEK_MAX_TOKENS,
  DEFAULT_DEEPSEEK_MODELS,
}

/**
 * Settings namespace the browser half's card edits: the join key between the
 * two halves. The card registers into `plugins.row.config` under this namespace,
 * and the settings tab pairs the two without knowing what the namespace means.
 */
export const GOOGLE_VERTEX_SETTINGS_NAMESPACE = 'google-vertex'

/** The one service this plugin needs mounted. */
export const inject = ['llm']

/**
 * Claude models Vertex serves, in picker order. Ids are the provider's own
 * aliases, which track the newest dated release of each family and need no
 * edit when Vertex promotes one; the catalog is overridable from configuration
 * for a deployment that pins dated versions or uses a different listing.
 */
export const DEFAULT_MODELS: readonly VertexModel[] = [
  { id: 'claude-opus-4-6', name: 'Claude Opus 4.6 (Vertex)' },
  { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6 (Vertex)' },
  { id: 'claude-opus-4-5', name: 'Claude Opus 4.5 (Vertex)' },
  { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5 (Vertex)' },
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5 (Vertex)' },
]

/** Context capacity reported for every Claude model; every listed family serves 200k. */
export const DEFAULT_CONTEXT_WINDOW = 200_000

/**
 * Output cap used when a caller omits one. Well under the models' 64k ceiling:
 * an agent turn rarely needs more, and the harness enforces its own budget.
 */
export const DEFAULT_MAX_TOKENS = 32_000

/**
 * Configuration received by the plugin: the row as the exported schema emits
 * it, so the volatile stamp arrives as a live reference rather than a string.
 *
 * A profile patch supplies `Options`; `resolveConfig` normalizes one into the
 * adapter configuration.
 */
export interface Config {
  /** Vertex AI Express / project API key; defaults to `GOOGLE_CLOUD_API_KEY`. */
  readonly apiKey?: string
  /** Service-account JSON path, `~` allowed; defaults to `GOOGLE_APPLICATION_CREDENTIALS`. */
  readonly serviceAccountFile?: string
  /**
   * Project id; defaults to `GOOGLE_CLOUD_PROJECT`, then `GCLOUD_PROJECT`, then
   * the credentials file's own `project_id`.
   */
  readonly project?: string
  /** Region, or `global` (the default); defaults to `GOOGLE_CLOUD_LOCATION`. */
  readonly location: string
  /** Claude model ids to advertise, replacing the built-in catalog. */
  readonly models: string[]
  /** Gemini model ids to advertise, replacing the built-in catalog. */
  readonly geminiModels: string[]
  /** DeepSeek model ids to advertise, replacing the built-in catalog. */
  readonly deepseekModels: string[]
  /**
   * Claude context window reported for every model; defaults to 200000. The
   * Gemini route reports {@link DEFAULT_GEMINI_CONTEXT_WINDOW} for every model
   * it serves, which is why this key has no Gemini counterpart.
   */
  readonly contextWindow: number
  /** Claude output cap applied when a caller omits one; defaults to 32000. */
  readonly maxTokens: number
  /**
   * Bound on the interval between two stream reads on either route; defaults to
   * 300000. A provider that stops sending is reported as `TIMEOUT` instead of
   * holding the turn open forever.
   */
  readonly streamIdleTimeoutMs: number
  /** Default reasoning effort for Gemini models (`low`, `medium`, `high`). */
  readonly reasoningEffort: string
  /**
   * Stamp written by the Refresh control; absent until the first manual refresh.
   * Volatile in the schema, so the settings document (which accepts only
   * volatile fields) commits the write into this running reference and its
   * `loader/volatile-update` drops both cached catalogs. The host never reads
   * the value; the write itself is the signal.
   */
  readonly revalidatedAt: Volatile<string | undefined>
}

/** Raw row values accepted from a profile patch, with live references unwrapped. */
export type Options = { [K in keyof Config]?: Config[K] extends Volatile<infer T> ? T : Config[K] }

/**
 * Row schema: defaults live here, so a deployment only states what it changes.
 *
 * `serviceAccountFile`, `project`, `models`, and `geminiModels` carry no
 * `.default()`: the first two fall back to the environment, and an omitted
 * catalog materializes empty, which `resolveConfig` treats exactly like an
 * absent one and replaces with the built-in list.
 *
 * `revalidatedAt` is volatile (the only kind of field the settings document
 * accepts) and carries no default: absence means "never refreshed manually".
 */
export const Config = Schema.object({
  apiKey: Schema.string(),
  serviceAccountFile: Schema.string(),
  project: Schema.string(),
  location: Schema.string().default(DEFAULT_LOCATION),
  models: Schema.array(Schema.string()),
  geminiModels: Schema.array(Schema.string()),
  deepseekModels: Schema.array(Schema.string()),
  contextWindow: Schema.number().step(1).min(1).default(DEFAULT_CONTEXT_WINDOW),
  maxTokens: Schema.number().step(1).min(1).default(DEFAULT_MAX_TOKENS),
  streamIdleTimeoutMs: Schema.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
  reasoningEffort: Schema.string().default('medium'),
  revalidatedAt: Schema.string().volatile(),
})

/** Validated configuration plus the credential source it was read from. */
interface ResolvedConfig {
  readonly anthropic: VertexAnthropicConfig
  readonly gemini: GeminiAdapterConfig
  readonly deepseek: DeepSeekAdapterConfig | undefined
  /** Vertex AI API key, when configured. */
  readonly apiKey?: string
  /** Absolute path of the service-account file, for the mount log line. */
  readonly serviceAccountFile?: string
}

/**
 * Validate and normalize one configuration row.
 *
 * Invalid values throw rather than being silently defaulted: a typo'd path or
 * project would otherwise present as an opaque provider error mid-turn.
 * @param config - raw row values, as a profile patch or an unwrapped row.
 * @param env - environment consulted for the credential and region defaults.
 * @returns the resolved adapter configuration.
 */
export function resolveConfig(config: Options = {}, env: NodeJS.ProcessEnv = process.env): ResolvedConfig {
  // The exported schema is the one source of the numeric defaults and bounds.
  // The credential path, the project, and the region keep their environment
  // fallbacks, so they are read off the raw row instead.
  const filled = Config(config)
  const apiKey = (filled.apiKey ?? (filled.serviceAccountFile ? undefined : env['GOOGLE_CLOUD_API_KEY']))?.trim() || undefined
  const serviceAccountFile = filled.serviceAccountFile ?? (apiKey !== undefined ? undefined : env['GOOGLE_APPLICATION_CREDENTIALS'])
  if (apiKey === undefined && (serviceAccountFile === undefined || serviceAccountFile.trim().length === 0)) {
    throw new Error(
      'google-vertex: no credentials configured: set apiKey or serviceAccountFile in this plugin\'s row,'
      + ' or export GOOGLE_CLOUD_API_KEY / GOOGLE_APPLICATION_CREDENTIALS',
    )
  }
  const serviceAccount = serviceAccountFile !== undefined && serviceAccountFile.trim().length > 0
    ? loadServiceAccount(serviceAccountFile)
    : undefined

  const project = filled.project
    ?? env['GOOGLE_CLOUD_PROJECT']
    ?? env['GCLOUD_PROJECT']
    ?? serviceAccount?.project_id
    ?? (apiKey !== undefined ? '' : undefined)
  if (project === undefined || (apiKey === undefined && project.trim().length === 0)) {
    throw new Error(
      `google-vertex: no project configured: set project in this plugin's row, or export`
      + ` GOOGLE_CLOUD_PROJECT, or use a service-account file that carries "project_id"`,
    )
  }

  const location = config.location ?? env['GOOGLE_CLOUD_LOCATION'] ?? DEFAULT_LOCATION
  if (location.trim().length === 0 || !/^[a-z0-9-]+$/.test(location)) {
    throw new Error(`google-vertex: location "${location}" is not a valid Vertex region`)
  }

  const ids: readonly VertexModel[] = filled.models === undefined || filled.models.length === 0
    ? DEFAULT_MODELS
    : filled.models.map(id => ({ id, name: id }))
  for (const model of ids) {
    if (model.id.trim().length === 0) throw new Error('google-vertex: model ids must be non-empty strings')
  }

  const geminiIds: readonly string[] = filled.geminiModels === undefined || filled.geminiModels.length === 0
    ? DEFAULT_GEMINI_MODELS.map(model => model.id)
    : filled.geminiModels
  const geminiModels: readonly VertexModel[] = geminiIds.map((id) => {
    if (id.trim().length === 0) throw new Error('google-vertex: geminiModels ids must be non-empty strings')
    // A configured id keeps the built-in entry's wording when it is one of them;
    // every model serves the same capacities, which live on the wire config.
    return DEFAULT_GEMINI_MODELS.find(model => model.id === id) ?? { id, name: id }
  })

  const deepseekIds: readonly string[] = filled.deepseekModels === undefined || filled.deepseekModels.length === 0
    ? DEFAULT_DEEPSEEK_MODELS.map(model => model.id)
    : filled.deepseekModels
  const deepseekModels: readonly VertexModel[] = deepseekIds.map((rawId) => {
    if (rawId.trim().length === 0) throw new Error('google-vertex: deepseekModels ids must be non-empty strings')
    const id = normalizeDeepSeekModel(rawId)
    return DEFAULT_DEEPSEEK_MODELS.find(model => model.id === id) ?? { id, name: id }
  })

  const streamIdleTimeoutMs = filled.streamIdleTimeoutMs

  // Express/API-key mode has no service account, so the DeepSeek adapters are
  // never mounted; leave the section unset rather than fabricate one.
  const deepseek: DeepSeekAdapterConfig | undefined = serviceAccount === undefined
    ? undefined
    : {
        serviceAccount,
        project,
        location,
        models: deepseekModels,
        contextWindow: DEFAULT_DEEPSEEK_CONTEXT_WINDOW,
        maxTokens: DEFAULT_DEEPSEEK_MAX_TOKENS,
        streamIdleTimeoutMs,
      }

  return {
    apiKey,
    serviceAccountFile,
    anthropic: {
      serviceAccount,
      project,
      location,
      models: ids,
      contextWindow: filled.contextWindow,
      maxTokens: filled.maxTokens,
      streamIdleTimeoutMs,
    },
    gemini: {
      apiKey,
      serviceAccount,
      project,
      location,
      models: geminiModels,
      maxTokens: DEFAULT_GEMINI_MAX_TOKENS,
      streamIdleTimeoutMs,
      reasoningEffort: filled.reasoningEffort,
    },
    deepseek,
  }
}

/**
 * Mount the configured adapters.
 * @param ctx - host context; `ctx.llm` must be mounted (`inject` guarantees it).
 * @param config - this plugin's row, as the schema emits it.
 */
export function apply(ctx: HostContext, config: Config): void {
  // The volatile stamp stays a live reference: the schema refuses one, and the
  // host never reads the value, so the plain row is what gets resolved.
  const { revalidatedAt: _revalidation, ...row } = config
  const resolved = resolveConfig(row)
  const { anthropic, gemini, deepseek } = resolved

  // Catalogs are discovered from Vertex at runtime unless explicitly pinned
  // by configuration. An empty or omitted list enables live discovery.
  const fetchFn: (input: string, init: RequestInit) => Promise<Response>
    = (input, init) => globalThis.fetch(input, init)
  const tokenSource = anthropic.serviceAccount
    ? new ServiceAccountTokens(anthropic.serviceAccount, { fetch: fetchFn })
    : undefined

  const pinnedClaude = row.models !== undefined && row.models.length > 0
  const discoverClaude = pinnedClaude || tokenSource === undefined
    ? undefined
    : () => fetchClaudeModels(anthropic.location, tokenSource, fetchFn, AbortSignal.timeout(Math.ceil(anthropic.streamIdleTimeoutMs)))

  const pinnedGemini = row.geminiModels !== undefined && row.geminiModels.length > 0
  // Express API keys do not support the v1beta1 publisher model list endpoint (401 UNAUTHENTICATED).
  const discoverGemini = pinnedGemini || gemini.apiKey !== undefined || tokenSource === undefined
    ? undefined
    : () => fetchGeminiModels(gemini.location, tokenSource, fetchFn, AbortSignal.timeout(Math.ceil(gemini.streamIdleTimeoutMs)))

  const pinnedDeepSeek = row.deepseekModels !== undefined && row.deepseekModels.length > 0
  const discoverDeepSeek = pinnedDeepSeek || tokenSource === undefined || deepseek === undefined
    ? undefined
    : () => fetchDeepSeekModels(tokenSource, fetchFn, AbortSignal.timeout(Math.ceil(deepseek.streamIdleTimeoutMs)))

  let anthropicAdapter: GoogleVertexAnthropicAdapter | undefined
  if (tokenSource !== undefined) {
    anthropicAdapter = new GoogleVertexAnthropicAdapter(anthropic, discoverClaude === undefined ? {} : { discover: discoverClaude })
    ctx.llm.registerAdapter([PROVIDER], anthropicAdapter)
  }

  const geminiAdapter = new GoogleVertexGeminiAdapter(gemini, discoverGemini === undefined ? {} : { discover: discoverGemini })
  ctx.llm.registerAdapter([GEMINI_PROVIDER], geminiAdapter)

  let deepseekAdapter: GoogleVertexDeepSeekAdapter | undefined
  if (tokenSource !== undefined && deepseek !== undefined) {
    deepseekAdapter = new GoogleVertexDeepSeekAdapter(deepseek, discoverDeepSeek === undefined ? {} : { discover: discoverDeepSeek })
    ctx.llm.registerAdapter([DEEPSEEK_PROVIDER], deepseekAdapter)
  }

  // A profile edit of `revalidatedAt` drops all cached catalogs.
  if (typeof ctx.on === 'function') {
    ctx.on('loader/volatile-update', () => {
      anthropicAdapter?.invalidateModels()
      geminiAdapter.invalidateModels()
      deepseekAdapter?.invalidateModels()
    })
  }

  const projectLabel = gemini.project.length > 0 ? `project ${gemini.project}` : 'express mode'
  const authLabel = [
    resolved.apiKey !== undefined ? 'apiKey' : undefined,
    resolved.serviceAccountFile !== undefined ? `credentials ${resolved.serviceAccountFile}` : undefined,
  ].filter(Boolean).join(' + ')
  const registered = tokenSource !== undefined
    ? `"${PROVIDER}", "${GEMINI_PROVIDER}", and "${DEEPSEEK_PROVIDER}"`
    : `"${GEMINI_PROVIDER}"`
  const modelSummary = [
    tokenSource !== undefined ? `Claude: ${anthropic.models.map(model => model.id).join(', ')}${pinnedClaude ? '' : ' (+ live discovery)'}` : undefined,
    `Gemini: ${gemini.models.map(model => model.id).join(', ')}${discoverGemini !== undefined ? ' (+ live discovery)' : ''}`,
    deepseek !== undefined ? `DeepSeek: ${deepseek.models.map(model => model.id).join(', ')}${pinnedDeepSeek ? '' : ' (+ live discovery)'}` : undefined,
  ].filter(Boolean).join('. ')

  ctx.logger.info(
    `google-vertex: providers ${registered} registered for ${projectLabel}`
    + ` (location ${gemini.location}, ${authLabel}). ${modelSummary}`,
  )
}
