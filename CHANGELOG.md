# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.16.4] - 2026-11-05

### Fixed
- **Filter non-serverless DeepSeek models from catalog**: The Vertex Model Garden publisher catalog returned recipes for dedicated GKE/GPU deployments (e.g. `deepseek-r1`, `deepseek-v3`, `deepseek-v4`), which are not available on Serverless MaaS and failed with HTTP 400/404. Model discovery and defaults now filter to only valid Serverless MaaS models (`deepseek-ai/deepseek-v4.1-flash-maas`).
- **Normalized bare DeepSeek model IDs**: All variants of `deepseek-v4*` and bare names without a publisher prefix are now properly normalized to canonical `<publisher>/<model>` identifiers to avoid `Malformed publisher model` errors from Vertex.

## [0.16.3] - 2026-11-05

### Fixed
- **Comprehensive DSML sanitization for unclosed / truncated tags**: Strips unclosed or malformed DeepSeek Markup Language tags (e.g. `<｜DSML｜ parameter name=`) that leak into tool arguments (such as `write` or `edit`) without a closing `>` delimiter.
- Added support for unclosed parameter tags and trailing special tokens at string boundaries.

## [0.16.2] - 2026-11-05

### Fixed
- **DeepSeek Adapter exactModelMetadata ID matching**: Fixed an issue where DeepSeek Harness failed to load with `adapter returned invalid exact model metadata for provider "google-vertex-deepseek" model "deepseek-r1"`.
  - Preserved the caller's requested model identifier in `resolveModel()` and `prepareCall()` to strictly conform to DeepSeek Harness's runtime contract (`exactModelMetadata.id === model`).
  - Canonical model normalization (`deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3`, etc.) is cleanly bound to the execution and streaming layer.

## [0.16.1] - 2026-11-05

### Fixed
- **Sanitize DSML / special token leakage**: Strips DeepSeek Markup Language (DSML) closing and delimiter tags (`</｜DSML｜ parameter>`, `<｜tool...｜>`, etc.) that Vertex AI Model Garden occasionally leaks into streaming tool arguments and text content.
- **Filter image tools on text-only DeepSeek**: Automatically excludes tools requiring visual/image input (such as `read_image`) when sending tool definitions to DeepSeek on Vertex. Because DeepSeek on Vertex Model Garden is text-only, advertising `read_image` triggered heavy tokenization bias and repetitive hallucination/failure loops when reading source files.

## [0.16.0] - 2026-11-05

### Added
- **Universal dynamic auto-discovery**: All three routes (Claude, Gemini, DeepSeek) now automatically discover available models at runtime from Vertex AI.
  - **Claude**: Discovered live from `publishers/anthropic/models`.
  - **Gemini**: Discovered live from `publishers/google/models` (filtering out embedding/media variants).
  - **DeepSeek MaaS**: Discovered live from Vertex Model Garden (`publishers/*/models` with `is_hf_wildcard(false)` filter in `us-central1`).
- **Automatic fallback**: If discovery encounters an error, adapters seamlessly fall back to built-in default models.
- **Optional model pinning**: `models`, `geminiModels`, and `deepseekModels` in `cordis.patch.yml` are optional. Specifying them pins the catalog and disables live discovery.
- **Refresh models**: The settings control drops all cached catalogs and re-queries Vertex AI.

## [0.15.0] - 2026-11-05

### Added
- Added Google Cloud Vertex DeepSeek provider (`google-vertex-deepseek`).
- Models: DeepSeek-V3, DeepSeek-R1, DeepSeek-V4.1-Flash.
- Streaming reasoning / thought translation and tool calling.
- Disjoint token usage reporting.
