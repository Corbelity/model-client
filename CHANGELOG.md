# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Pre-1.0 versioning:** while the major version is 0, the public API may change between
minor versions. Breaking changes are listed under `Changed` or `Removed` and called out
explicitly, so a minor bump is worth reading before you take it.

## [Unreleased]

Everything below becomes `0.1.0` when the first tag is cut.

### Added

- Reference images as input to image generation: `generate_image(prompt, images=None)`,
  mirroring `complete()` — same `ImageInput` type, same validation, same immutable-tuple
  contract. Capability is declared per provider by two new `ProviderSpec` fields,
  `image_input` and `max_reference_images`; a service without it raises
  `UnsupportedImageInputError` before any network call, and exceeding the cap raises
  `TooManyImagesError` before the upload. On OpenAI, references switch the call from
  `images.generate` to `images.edit`. Text-only generation is unchanged. (SA-347)
- Component breakdown of input tokens on `ModelResult`: `input_text_tokens`,
  `input_image_tokens` and `input_cached_tokens`, all defaulting to `None`. The
  components bill at different rates, so a caller costing a call from the flat
  `prompt_tokens` alone overstates a cached call and understates one carrying image
  input. Populated from `usage.input_tokens_details` on the OpenAI image path and from
  `usage.prompt_tokens_details.cached_tokens` on the chat path; `prompt_tokens` and
  `completion_tokens` are unchanged. The trace record carries the split too, since that
  is where a cost monitor reads usage back from. (SA-341)
- `openai` and `gemini` providers. `openai` covers text, image and speech; `gemini`
  reaches Google's models through their OpenAI-compatibility endpoint and is text-only
  for now — image generation through that shim is unverified, and audio runs over the
  bidirectional Live API, which is a streaming session rather than a request/response
  call. Both wait for a native `google-genai` provider, which will arrive as a separate
  service rather than a change to this one.
- `OpenAICompatibleClient`, an intermediate base holding the chat-completions request and
  response handling now shared by `openrouter`, `openai` and `gemini`. Subclass it to
  point at another OpenAI-compatible gateway; a `ProviderSpec` and a `SPEC` attribute are
  the whole job.
- `max_tokens_param` catalog flag, naming the key a model wants for its completion cap.
  Newer OpenAI models reject `max_tokens` and require `max_completion_tokens`; this is a
  spelling difference, so it is data rather than a prefix rule. Defaults to `max_tokens`
  for unlisted models.
- Catalog entries for `gpt-5.6-terra`, `gpt-image-2.5-flare`, `gpt-4o-mini-tts` and
  `gemini-3.8-flash`, and `[openai]` / `[gemini]` extras. Both resolve to the `openai`
  SDK, so adding either provider pulls in no new dependency.
- `catalog_for(config)` and `available_services()`, plus a `services` field on
  `ModelConfig` (`CORBELITY_SERVICES`, comma-separated). A user catalog merges over the
  built-in one and so cannot remove an entry; this narrows what gets **listed** instead,
  which is what a UI needs when only one provider's credentials are present.
  `available_services()` reports which services have their credentials and endpoints set,
  making the common case `ModelConfig(services=available_services())`. Filtering is never
  applied automatically, affects listing only, and does not change how any call behaves:
  provider code continues to read the unfiltered catalog so capability flags stay findable
  for every model. Aliases fold; `None` means no filter and `()` means list nothing.
- `ModelCatalog.for_services()`, returning a narrowed catalog rather than a tuple so it
  can be filtered again or passed anywhere a catalog is expected.
- `ModelClient`, a provider-agnostic base class for text, image and speech model calls,
  with four provider implementations: `anthropic`, `openrouter`, `ollama-local`,
  `ollama` (cloud) and `huggingface`.
- `make_model_client(service, **kwargs)` plus `client_class()`, `provider_spec()`,
  `known_services()`, `resolve_service()` and `supported_modalities()` — the last of which
  answers without importing a provider SDK or requiring a credential.
- `ModelConfig`, an immutable, injectable settings object with a fixed resolution order:
  per-call argument, injected config, environment, package default. Reads
  `CORBELITY_DEFAULT_SERVICE`, `CORBELITY_DEFAULT_MODEL`, `CORBELITY_MAX_TOKENS`,
  `CORBELITY_TEMPERATURE`, `CORBELITY_TOP_P`, `CORBELITY_NUM_CTX` and
  `CORBELITY_MODEL_CATALOG`.
- `ProviderSpec` and a provider registry, making service identity data rather than code:
  credential and endpoint environment variable names, default endpoint, supported
  modalities, aliases, and the implementing class as a lazily-resolved import path.
  `register_provider()` accepts third-party providers.
- `ModelCatalog` / `ModelInfo` and a shipped `models.json`, loaded via
  `importlib.resources`. A user catalog merges over the built-in one by model id, so a
  newly released model needs a JSON edit rather than a new version of this package.
- `TraceSink`, a structural `Protocol` for observability, with `NullTrace` and
  `JsonlTraceLogger` implementations and a `make_trace_logger()` environment helper
  (`TRACE_ENABLED`, `TRACE_FILE`). Binary payloads are written beside the trace as named
  artifacts rather than inlined as base64.
- Conversation history and image-attachment validation (`validate_history`,
  `validate_images`) that runs before the provider is touched and refuses to auto-repair
  malformed input.
- Magic-byte MIME sniffing for images and audio (`sniff_image_mime`, `sniff_audio_mime`).
- Optional per-provider dependencies (`[anthropic]`, `[openrouter]`, `[ollama]`,
  `[huggingface]`, `[all]`) with lazy SDK imports, so the package imports and answers
  capability questions with zero provider SDKs installed.
- A typed public API: `py.typed` ships inline annotations to downstream type checkers.
- Test suite covering validation, sniffing, configuration, credential resolution, the
  catalog, the registry, service filtering and the observability contract, using a fake
  provider rather than network calls.

### Changed

- `ModelClient._invoke_image()` now takes `(prompt, images)`. Protected, but external
  provider subclasses override it, so this breaks them — permitted while pre-1.0 and
  noted here rather than discovered. `images` is not defaulted, for the same reason
  `_invoke()`'s parameters are not.

### Fixed

- Provider client construction uses `cast()` rather than `# type: ignore[no-any-return]`.
  The ignore could not be correct in both CI environments at once: required with a
  provider SDK installed, reported as unused without one. The lint and type-check steps
  now run in both environments, so a construct that is only valid in one will fail.
- GitHub Actions moved to the Node 24 majors (`checkout@v6`, `setup-uv@v7`,
  `upload-artifact@v6`), and Dependabot now watches actions and dev dependencies monthly.

### Notes on provenance

Extracted from an internal prototype. Behaviour is preserved except where listed below;
the changes exist to make the code safe to embed in someone else's application.

- Configuration is no longer read at import time. The package does not call
  `load_dotenv()`, does not write to `os.environ`, and does not configure logging — it
  attaches only a `NullHandler` to its own logger.
- HuggingFace token discovery is a lookup across several environment variable names, in
  order, rather than a bridge that assigned one name onto another in `os.environ`.
- Anthropic's sampling-parameter support is read from the catalog when stated, falling
  back to a model-name prefix match only when it is not.
- Errors that were previously raised as built-ins now use named subclasses that still
  derive from those built-ins (`MissingCredentialsError` is a `ValueError`), so web layers
  mapping `ValueError` to a 400 keep working.
- Ollama clients resolve configuration before importing their SDK, so a missing host
  raises the same error whether or not the optional SDK is installed.

[Unreleased]: https://github.com/Corbelity/model-client/commits/main
