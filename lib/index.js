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
import Schema from '@deepseek-ai/schemastery';
import { GoogleVertexAnthropicAdapter } from './adapter.js';
import { loadServiceAccount } from './auth.js';
import { ServiceAccountTokens } from './auth.js';
import { DEFAULT_DEEPSEEK_CONTEXT_WINDOW, DEFAULT_DEEPSEEK_MAX_TOKENS, DEFAULT_DEEPSEEK_MODELS, normalizeDeepSeekModel, } from './deepseek.js';
import { GoogleVertexDeepSeekAdapter } from './deepseek_adapter.js';
import { fetchClaudeModels, fetchDeepSeekModels, fetchGeminiModels } from './discovery.js';
import { DEFAULT_GEMINI_CONTEXT_WINDOW, DEFAULT_GEMINI_MAX_TOKENS, DEFAULT_GEMINI_MODELS } from './gemini.js';
import { GoogleVertexGeminiAdapter } from './gemini_adapter.js';
import { DEFAULT_LOCATION, DEFAULT_STREAM_IDLE_TIMEOUT_MS, MAX_TIMER_DELAY_MS } from './wire.js';
/** Plugin name as it appears in the loader. */
export const name = 'google-vertex';
/** The `ctx.llm` route serving Google-hosted Anthropic (Claude) models. */
export const PROVIDER = 'google-vertex-anthropic';
/** The `ctx.llm` route serving Gemini models. */
export const GEMINI_PROVIDER = 'google-vertex-gemini';
/** The `ctx.llm` route serving DeepSeek models in Vertex Model Garden. */
export const DEEPSEEK_PROVIDER = 'google-vertex-deepseek';
export { DEFAULT_DEEPSEEK_CONTEXT_WINDOW, DEFAULT_DEEPSEEK_MAX_TOKENS, DEFAULT_DEEPSEEK_MODELS, };
/**
 * Settings namespace the browser half's card edits: the join key between the
 * two halves. The card registers into `plugins.row.config` under this namespace,
 * and the settings tab pairs the two without knowing what the namespace means.
 */
export const GOOGLE_VERTEX_SETTINGS_NAMESPACE = 'google-vertex';
/** The one service this plugin needs mounted. */
export const inject = ['llm'];
/**
 * Claude models Vertex serves, in picker order. Ids are the provider's own
 * aliases, which track the newest dated release of each family and need no
 * edit when Vertex promotes one; the catalog is overridable from configuration
 * for a deployment that pins dated versions or uses a different listing.
 */
export const DEFAULT_MODELS = [
    { id: 'claude-opus-4-6', name: 'Claude Opus 4.6 (Vertex)' },
    { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6 (Vertex)' },
    { id: 'claude-opus-4-5', name: 'Claude Opus 4.5 (Vertex)' },
    { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5 (Vertex)' },
    { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5 (Vertex)' },
];
/** Context capacity reported for every Claude model; every listed family serves 200k. */
export const DEFAULT_CONTEXT_WINDOW = 200_000;
/**
 * Output cap used when a caller omits one. Well under the models' 64k ceiling:
 * an agent turn rarely needs more, and the harness enforces its own budget.
 */
export const DEFAULT_MAX_TOKENS = 32_000;
/**
 * Row schema: defaults live here, so a deployment only states what it changes.
 *
 * All fields are volatile so they can be edited in the GUI settings panel.
 * `serviceAccountFile`, `project`, `models`, `geminiModels`, and `deepseekModels`
 * carry no `.default()`: the first two fall back to the environment, and an omitted
 * catalog materializes empty, which `resolveConfig` treats exactly like an
 * absent one and replaces with the built-in list.
 */
export const Config = Schema.object({
    serviceAccountFile: Schema.string().volatile(),
    project: Schema.string().volatile(),
    location: Schema.string().default(DEFAULT_LOCATION).volatile(),
    models: Schema.array(Schema.string()).volatile(),
    geminiModels: Schema.array(Schema.string()).volatile(),
    deepseekModels: Schema.array(Schema.string()).volatile(),
    contextWindow: Schema.number().step(1).min(1).default(DEFAULT_CONTEXT_WINDOW).volatile(),
    maxTokens: Schema.number().step(1).min(1).default(DEFAULT_MAX_TOKENS).volatile(),
    streamIdleTimeoutMs: Schema.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS).volatile(),
    revalidatedAt: Schema.string().volatile(),
});
/**
 * Validate and normalize one configuration row.
 *
 * Invalid values throw rather than being silently defaulted: a typo'd path or
 * project would otherwise present as an opaque provider error mid-turn.
 * @param config - raw row values, as a profile patch or an unwrapped row.
 * @param env - environment consulted for the credential and region defaults.
 * @returns the resolved adapter configuration.
 */
export function resolveConfig(config = {}, env = process.env) {
    // The exported schema is the one source of the numeric defaults and bounds.
    // The credential path, the project, and the region keep their environment
    // fallbacks, so they are read off the raw row instead.
    const filled = Config(config);
    // Unwrap volatile references to get actual values
    // Volatile.get() returns VolatileSnapshot<T> which for primitives is T
    const getValue = (v) => v?.get();
    const serviceAccountFile = getValue(filled.serviceAccountFile) ?? env['GOOGLE_APPLICATION_CREDENTIALS'];
    if (serviceAccountFile === undefined || serviceAccountFile.trim().length === 0) {
        throw new Error('google-vertex: no service account configured: set serviceAccountFile in this plugin\'s row,'
            + ' or export GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON path');
    }
    const serviceAccount = loadServiceAccount(serviceAccountFile);
    const project = getValue(filled.project)
        ?? env['GOOGLE_CLOUD_PROJECT']
        ?? env['GCLOUD_PROJECT']
        ?? serviceAccount.project_id;
    if (project === undefined || project.trim().length === 0) {
        throw new Error(`google-vertex: no project configured: set project in this plugin's row, or export`
            + ` GOOGLE_CLOUD_PROJECT, or use a service-account file that carries "project_id"`);
    }
    const location = getValue(filled.location) ?? env['GOOGLE_CLOUD_LOCATION'] ?? DEFAULT_LOCATION;
    if (location.trim().length === 0 || !/^[a-z0-9-]+$/.test(location)) {
        throw new Error(`google-vertex: location "${location}" is not a valid Vertex region`);
    }
    const modelsArray = getValue(filled.models);
    const ids = modelsArray === undefined || modelsArray.length === 0
        ? DEFAULT_MODELS
        : modelsArray.map(id => ({ id, name: id }));
    for (const model of ids) {
        if (model.id.trim().length === 0)
            throw new Error('google-vertex: model ids must be non-empty strings');
    }
    const geminiModelsArray = getValue(filled.geminiModels);
    const geminiIds = geminiModelsArray === undefined || geminiModelsArray.length === 0
        ? DEFAULT_GEMINI_MODELS.map(model => model.id)
        : geminiModelsArray;
    const geminiModels = geminiIds.map((id) => {
        if (id.trim().length === 0)
            throw new Error('google-vertex: geminiModels ids must be non-empty strings');
        // A configured id keeps the built-in entry's wording when it is one of them;
        // every model serves the same capacities, which live on the wire config.
        return DEFAULT_GEMINI_MODELS.find(model => model.id === id) ?? { id, name: id };
    });
    const deepseekModelsArray = getValue(filled.deepseekModels);
    const deepseekIds = deepseekModelsArray === undefined || deepseekModelsArray.length === 0
        ? DEFAULT_DEEPSEEK_MODELS.map(model => model.id)
        : deepseekModelsArray;
    const deepseekModels = deepseekIds.map((rawId) => {
        if (rawId.trim().length === 0)
            throw new Error('google-vertex: deepseekModels ids must be non-empty strings');
        const id = normalizeDeepSeekModel(rawId);
        return DEFAULT_DEEPSEEK_MODELS.find(model => model.id === id) ?? { id, name: id };
    });
    const contextWindow = getValue(filled.contextWindow) ?? DEFAULT_CONTEXT_WINDOW;
    const maxTokens = getValue(filled.maxTokens) ?? DEFAULT_MAX_TOKENS;
    const streamIdleTimeoutMs = getValue(filled.streamIdleTimeoutMs) ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS;
    return {
        serviceAccountFile,
        anthropic: {
            serviceAccount,
            project,
            location,
            models: ids,
            contextWindow,
            maxTokens,
            streamIdleTimeoutMs,
        },
        gemini: {
            serviceAccount,
            project,
            location,
            models: geminiModels,
            maxTokens: DEFAULT_GEMINI_MAX_TOKENS,
            streamIdleTimeoutMs,
        },
        deepseek: {
            serviceAccount,
            project,
            location,
            models: deepseekModels,
            contextWindow: DEFAULT_DEEPSEEK_CONTEXT_WINDOW,
            maxTokens: DEFAULT_DEEPSEEK_MAX_TOKENS,
            streamIdleTimeoutMs,
        },
    };
}
/**
 * Mount all three adapters.
 * @param ctx - host context; `ctx.llm` must be mounted (`inject` guarantees it).
 * @param config - this plugin's row, as the schema emits it.
 */
export function apply(ctx, config) {
    // Track adapter disposables for cleanup on config change
    let disposables = [];
    // Helper to unwrap volatile refs
    const getValue = (v) => v?.get();
    // Function to register adapters with current config
    const registerAdapters = () => {
        // Clean up previous adapters
        for (const dispose of disposables) {
            dispose();
        }
        disposables = [];
        // The volatile stamp stays a live reference: the schema refuses one, and the
        // host never reads the value, so the plain row is what gets resolved.
        const { revalidatedAt: _revalidation, ...row } = config;
        // Create unwrapped config for resolveConfig (which expects plain values)
        const unwrappedConfig = {
            serviceAccountFile: getValue(row.serviceAccountFile),
            project: getValue(row.project),
            location: getValue(row.location),
            models: getValue(row.models),
            geminiModels: getValue(row.geminiModels),
            deepseekModels: getValue(row.deepseekModels),
            contextWindow: getValue(row.contextWindow),
            maxTokens: getValue(row.maxTokens),
            streamIdleTimeoutMs: getValue(row.streamIdleTimeoutMs),
        };
        const resolved = resolveConfig(unwrappedConfig);
        const { anthropic, gemini, deepseek } = resolved;
        // Catalogs are discovered from Vertex at runtime unless explicitly pinned
        // by configuration. An empty or omitted list enables live discovery.
        const fetchFn = (input, init) => globalThis.fetch(input, init);
        const tokenSource = new ServiceAccountTokens(anthropic.serviceAccount, { fetch: fetchFn });
        // Check pinned status using unwrapped values
        const modelsArray = getValue(row.models);
        const pinnedClaude = modelsArray !== undefined && modelsArray.length > 0;
        const discoverClaude = pinnedClaude
            ? undefined
            : () => fetchClaudeModels(anthropic.location, tokenSource, fetchFn, AbortSignal.timeout(Math.ceil(anthropic.streamIdleTimeoutMs)));
        const geminiModelsArray = getValue(row.geminiModels);
        const pinnedGemini = geminiModelsArray !== undefined && geminiModelsArray.length > 0;
        const discoverGemini = pinnedGemini
            ? undefined
            : () => fetchGeminiModels(gemini.location, tokenSource, fetchFn, AbortSignal.timeout(Math.ceil(gemini.streamIdleTimeoutMs)));
        const deepseekModelsArray = getValue(row.deepseekModels);
        const pinnedDeepSeek = deepseekModelsArray !== undefined && deepseekModelsArray.length > 0;
        const discoverDeepSeek = pinnedDeepSeek
            ? undefined
            : () => fetchDeepSeekModels(tokenSource, fetchFn, AbortSignal.timeout(Math.ceil(deepseek.streamIdleTimeoutMs)));
        const anthropicAdapter = new GoogleVertexAnthropicAdapter(anthropic, discoverClaude === undefined ? {} : { discover: discoverClaude });
        const geminiAdapter = new GoogleVertexGeminiAdapter(gemini, discoverGemini === undefined ? {} : { discover: discoverGemini });
        const deepseekAdapter = new GoogleVertexDeepSeekAdapter(deepseek, discoverDeepSeek === undefined ? {} : { discover: discoverDeepSeek });
        disposables.push(ctx.llm.registerAdapter([PROVIDER], anthropicAdapter));
        disposables.push(ctx.llm.registerAdapter([GEMINI_PROVIDER], geminiAdapter));
        disposables.push(ctx.llm.registerAdapter([DEEPSEEK_PROVIDER], deepseekAdapter));
        // A profile edit of any volatile field triggers re-registration via loader/volatile-update
        if (typeof ctx.on === 'function') {
            ctx.on('loader/volatile-update', () => {
                anthropicAdapter.invalidateModels();
                geminiAdapter.invalidateModels();
                deepseekAdapter.invalidateModels();
                registerAdapters();
            });
        }
        ctx.logger.info(`google-vertex: providers "${PROVIDER}", "${GEMINI_PROVIDER}", and "${DEEPSEEK_PROVIDER}" registered for project ${anthropic.project}`
            + ` (location ${anthropic.location}, credentials ${resolved.serviceAccountFile})`
            + `. Claude: ${anthropic.models.map(model => model.id).join(', ')}${pinnedClaude ? '' : ' (+ live discovery)'}`
            + `. Gemini: ${gemini.models.map(model => model.id).join(', ')}${pinnedGemini ? '' : ' (+ live discovery)'}`
            + `. DeepSeek: ${deepseek.models.map(model => model.id).join(', ')}${pinnedDeepSeek ? '' : ' (+ live discovery)'}`);
    };
    // Initial registration
    registerAdapters();
}
