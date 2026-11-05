/**
 * Provider adapter for DeepSeek models (DeepSeek-V3 and DeepSeek-R1) on Google
 * Cloud Vertex AI (Model Garden / Serverless MaaS).
 *
 * Uses the same service-account auth and idle watchdog pipeline as Claude and
 * Gemini, connecting to Vertex's OpenAI-compatible completions endpoint.
 *
 * @module dsh-google-vertex/deepseek-adapter
 */
import { streamVertex, VertexPublisherAdapter, } from './adapter.js';
import { buildDeepSeekRequest, deepseekEndpointFor, DeepSeekStreamTranslator, DEFAULT_DEEPSEEK_CONTEXT_WINDOW, DEFAULT_DEEPSEEK_MAX_TOKENS, normalizeDeepSeekModel, } from './deepseek.js';
/**
 * Duck-typed adapter over Vertex's Model Garden DeepSeek OpenAI endpoint.
 */
export class GoogleVertexDeepSeekAdapter extends VertexPublisherAdapter {
    /**
     * @param config - the resolved configuration this adapter serves.
     * @param options - transport, token-source, and discovery overrides for tests.
     */
    constructor(config, options = {}) {
        super({
            providerName: 'Google Vertex AI (DeepSeek)',
            capacity: {
                contextWindow: config.contextWindow || DEFAULT_DEEPSEEK_CONTEXT_WINDOW,
                defaultMaxTokens: config.maxTokens || DEFAULT_DEEPSEEK_MAX_TOKENS,
            },
            describe: row => `DeepSeek on Google Vertex AI Model Garden (project ${row.project}, ${row.location}).`,
        }, config, options);
    }
    /**
     * Resolve model metadata, normalizing aliases like `deepseek-v3`, `deepseek-r1`, and `deepseek-v4.1-flash`.
     */
    async resolveModel(provider, model, signal) {
        const canonical = normalizeDeepSeekModel(model);
        const resolved = await super.resolveModel(provider, canonical, signal);
        const isFlash = canonical.toLowerCase().includes('flash');
        return {
            ...resolved,
            context: { contextWindow: isFlash ? 1_000_000 : (this.config.contextWindow || DEFAULT_DEEPSEEK_CONTEXT_WINDOW) },
        };
    }
    /**
     * Prepare a call for the specified model, binding normalization and the stream generator.
     */
    async prepareCall(provider, model, signal) {
        const canonical = normalizeDeepSeekModel(model);
        const resolved = await this.resolveModel(provider, canonical, signal);
        return {
            model: resolved,
            stream: (options) => this.stream({
                ...options,
                model: normalizeDeepSeekModel(options.model || canonical),
            }),
        };
    }
    /**
     * Stream one completion through Vertex Model Garden's OpenAI completions endpoint.
     */
    stream(options) {
        const normalizedOptions = {
            ...options,
            model: normalizeDeepSeekModel(options.model),
        };
        return streamVertex(this.config, normalizedOptions, this.fetch, this.tokens, model => new DeepSeekStreamTranslator(model), {
            endpoint: (_model, config) => deepseekEndpointFor(config.project, config.location),
            body: (request, config) => buildDeepSeekRequest(request, config),
            truncatedMessage: model => `google-vertex: model "${model}" stream ended before a finish reason`,
        });
    }
}
