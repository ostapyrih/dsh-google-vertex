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
import type { FinishReason, GenerateOptions, StreamChunk, TokenUsage } from './host.ts';
import type { VertexModel } from './adapter.ts';
import type { VertexWireConfig } from './wire.ts';
/** Context capacity reported for DeepSeek models in Model Garden (128k). */
export declare const DEFAULT_DEEPSEEK_CONTEXT_WINDOW = 128000;
/** Output cap applied when a caller omits one (16k). */
export declare const DEFAULT_DEEPSEEK_MAX_TOKENS = 16384;
/**
 * Built-in DeepSeek models served on Vertex AI Model Garden.
 */
export declare const DEFAULT_DEEPSEEK_MODELS: readonly VertexModel[];
/**
 * Normalize DeepSeek model aliases to their canonical Vertex Model Garden identifiers.
 * Maps aliases like `deepseek-v3`, `deepseek-r1`, and `deepseek-v4.1-flash`.
 * @param model - model identifier or alias.
 * @returns canonical Vertex Model Garden model id.
 */
export declare function normalizeDeepSeekModel(model: string): string;
/**
 * Endpoint for Vertex Model Garden OpenAI-compatible chat completions.
 * @param project - Google Cloud project id.
 * @param location - Vertex AI region (e.g. us-central1).
 * @returns the absolute request URL.
 */
export declare function deepseekEndpointFor(project: string, location: string): string;
/** Configuration needed for DeepSeek wire requests. */
export interface DeepSeekWireConfig extends VertexWireConfig {
    readonly contextWindow?: number;
}
/** OpenAI tool function schema. */
interface OpenAIFunction {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
}
/** OpenAI tool object. */
interface OpenAITool {
    type: 'function';
    function: OpenAIFunction;
}
/** OpenAI tool call in message history. */
interface OpenAIToolCall {
    id: string;
    type: 'function';
    function: {
        name: string;
        arguments: string;
    };
}
/** OpenAI chat message format. */
export type OpenAIMessage = {
    role: 'system';
    content: string;
} | {
    role: 'user';
    content: string;
} | {
    role: 'assistant';
    content: string | null;
    tool_calls?: OpenAIToolCall[];
} | {
    role: 'tool';
    tool_call_id: string;
    content: string;
};
/** DeepSeek OpenAI-compatible Chat Completions request body. */
export interface DeepSeekRequestBody {
    model: string;
    messages: OpenAIMessage[];
    stream: true;
    stream_options?: {
        include_usage: boolean;
    };
    max_tokens: number;
    temperature?: number;
    stop?: string[];
    tools?: OpenAITool[];
}
/**
 * Strip DeepSeek-specific markup language (DSML) tags and special tokens that
 * can leak into tool arguments or content from Vertex Model Garden's backend parser.
 *
 * Handles both complete and unclosed/truncated tags, such as:
 * - `</｜DSML｜ parameter>` or `<｜DSML｜ parameter ...>`
 * - `<｜DSML｜ parameter name=` (unclosed tag leaked into content/arguments)
 * - `<｜tool call begin｜>`, `<｜tool call end｜>`, `<｜tool sep｜>`
 * - `<｜...｜>` and `<|...|>`
 * - `</parameter>` or `<parameter name="...">`
 * @param text - string that may contain leaked DSML tokens.
 * @returns cleaned string.
 */
export declare function sanitizeDeepSeekText(text: string): string;
/**
 * Filter out tools requiring vision/multimodal capabilities that text-only DeepSeek
 * on Vertex Model Garden cannot execute (e.g. `read_image`). Exposing `read_image`
 * biases DeepSeek to call it instead of `read`, triggering repetitive failure loops.
 * @param tool - tool definition from harness.
 * @returns true if the tool can be safely served to DeepSeek.
 */
export declare function isSupportedDeepSeekTool(tool: {
    name: string;
    description: string;
}): boolean;
/**
 * Build the OpenAI Chat Completions payload for Vertex Model Garden DeepSeek.
 * @param options - harness generate options.
 * @param config - project wire configuration.
 * @returns OpenAI-compatible request body.
 */
export declare function buildDeepSeekRequest(options: GenerateOptions, config: DeepSeekWireConfig): DeepSeekRequestBody;
/**
 * Map OpenAI finish_reason to harness FinishReason.
 * @param reason - raw finish_reason string from provider.
 * @param sawToolCall - whether the response contained a tool call.
 * @returns harness FinishReason.
 */
export declare function mapOpenAIFinishReason(reason: string | undefined, sawToolCall: boolean): FinishReason;
/**
 * Map OpenAI usage counters to harness TokenUsage.
 * @param raw - raw usage object from provider SSE event.
 * @returns disjoint TokenUsage.
 */
export declare function mapOpenAIUsage(raw: unknown): TokenUsage;
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
export declare class DeepSeekStreamTranslator {
    #private;
    constructor(model: string);
    /** The model identifier this translator was constructed for. */
    get model(): string;
    /** Whether the stream has completed with a terminal chunk. */
    get terminal(): boolean;
    /** Whether a finish event or indicator was observed. */
    get sawFinish(): boolean;
    /**
     * Feed one decoded SSE record.
     * @param event - parsed SSE payload or [DONE] indicator.
     * @returns array of completed harness chunks.
     */
    handle(event: Record<string, unknown>): StreamChunk[];
    /**
     * Finalize the stream and emit terminal blocks, usage, and finish reason.
     */
    finish(): StreamChunk[];
}
export {};
