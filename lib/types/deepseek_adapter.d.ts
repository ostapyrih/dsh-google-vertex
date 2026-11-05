/**
 * Provider adapter for DeepSeek models (DeepSeek-V3 and DeepSeek-R1) on Google
 * Cloud Vertex AI (Model Garden / Serverless MaaS).
 *
 * Uses the same service-account auth and idle watchdog pipeline as Claude and
 * Gemini, connecting to Vertex's OpenAI-compatible completions endpoint.
 *
 * @module dsh-google-vertex/deepseek-adapter
 */
import { VertexPublisherAdapter } from './adapter.ts';
import type { TokenProvider, VertexModel } from './adapter.ts';
import type { FetchLike, ServiceAccount } from './auth.ts';
import type { VertexWireConfig } from './wire.ts';
import type { GenerateOptions, LlmResolvedModelInfo, PreparedAdapterCall, StreamChunk } from './host.ts';
/** Resolved adapter configuration for the DeepSeek route. */
export interface DeepSeekAdapterConfig extends VertexWireConfig {
    readonly serviceAccount: ServiceAccount;
    readonly models: readonly VertexModel[];
    readonly contextWindow: number;
    /** Bound on the interval between two stream reads, in milliseconds. */
    readonly streamIdleTimeoutMs: number;
}
/**
 * Duck-typed adapter over Vertex's Model Garden DeepSeek OpenAI endpoint.
 */
export declare class GoogleVertexDeepSeekAdapter extends VertexPublisherAdapter<DeepSeekAdapterConfig> {
    /**
     * @param config - the resolved configuration this adapter serves.
     * @param options - transport, token-source, and discovery overrides for tests.
     */
    constructor(config: DeepSeekAdapterConfig, options?: {
        fetch?: FetchLike;
        tokens?: TokenProvider;
        discover?: () => Promise<readonly VertexModel[]>;
    });
    /**
     * Resolve model metadata, normalizing aliases like `deepseek-v3`, `deepseek-r1`, and `deepseek-v4.1-flash`.
     */
    resolveModel(provider: string, model: string, signal?: AbortSignal): Promise<LlmResolvedModelInfo>;
    /**
     * Prepare a call for the specified model, binding normalization and the stream generator.
     */
    prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall>;
    /**
     * Stream one completion through Vertex Model Garden's OpenAI completions endpoint.
     */
    stream(options: GenerateOptions): AsyncIterable<StreamChunk>;
}
