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
import type { FetchLike } from './auth.ts';
import type { TokenProvider, VertexModel } from './adapter.ts';
/** How long a cached model list stays valid, in milliseconds. */
export declare const DEFAULT_CACHE_TTL_MS: number;
/**
 * Publisher that serves Claude through Vertex's Anthropic route.
 */
export declare const ANTHROPIC_PUBLISHER = "anthropic";
/**
 * Publisher that serves Gemini.
 */
export declare const GOOGLE_PUBLISHER = "google";
/**
 * Publisher that serves DeepSeek MaaS models.
 */
export declare const DEEPSEEK_PUBLISHER = "deepseek-ai";
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
export declare function listModelsUrl(location: string, publisher?: string): string;
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
export declare function fetchPublisherModels(publisher: string, location: string, tokens: TokenProvider, fetchFn: FetchLike, keep: (id: string) => boolean, label: (id: string) => string, signal?: AbortSignal): Promise<readonly VertexModel[]>;
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
export declare function fetchGeminiModels(location: string, tokens: TokenProvider, fetchFn: FetchLike, signal?: AbortSignal): Promise<readonly VertexModel[]>;
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
export declare function fetchClaudeModels(location: string, tokens: TokenProvider, fetchFn: FetchLike, signal?: AbortSignal): Promise<readonly VertexModel[]>;
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
export declare function fetchDeepSeekModels(tokens: TokenProvider, fetchFn: FetchLike, signal?: AbortSignal): Promise<readonly VertexModel[]>;
/**
 * Turn a Claude model id into a picker label, e.g. `claude-opus-4-6` into
 * `Claude Opus 4.6`.
 * @param id - bare Claude model id.
 * @returns the human-readable family name, without the route suffix.
 */
export declare function claudeDisplayName(id: string): string;
/**
 * Turn a DeepSeek model id into a picker label, e.g. `deepseek-v4.1-flash-maas`
 * into `DeepSeek-V4.1-Flash`.
 * @param id - bare DeepSeek model id.
 * @returns the human-readable name, without the route suffix.
 */
export declare function deepseekDisplayName(id: string): string;
/**
 * A cached, TTL-bounded model list that falls back to a static default when
 * the remote fetch fails.
 */
export declare class ModelCache {
    #private;
    /**
     * @param fallback - static default returned when the fetch fails or is not
     *   attempted.
     */
    constructor(fallback: readonly VertexModel[]);
    /**
     * Return the cached model list, or fetch a fresh one.
     *
     * When a fetch function is provided and the cache is stale, it is called to
     * produce a fresh list. On failure, the fallback is returned. Concurrent
     * callers share one in-flight fetch.
     * @param fetchFn - optional async function that returns a fresh model list.
     * @returns the model list, from cache, fetch, or fallback.
     */
    get(fetchFn?: () => Promise<readonly VertexModel[]>): Promise<readonly VertexModel[]>;
    /** Force the next `get` to re-fetch. */
    invalidate(): void;
}
