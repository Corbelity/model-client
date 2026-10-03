# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Pre-1.0 versioning:** while the major version is 0, the public API may change between
minor versions. Breaking changes are listed under `Changed` or `Removed` and called out
explicitly, so a minor bump is worth reading before you take it.

## [Unreleased]

### Added

- `gemini-native` service: Gemini through Google's own SDK (`google-genai`), installed with
  the new `[gemini-native]` extra. Text, with history and image attachments, under the
  same contract as every other provider. A separate service from `gemini`, which is
  unchanged. Aliases `google-genai` and `gemini-genai`; endpoint override
  `GEMINI_NATIVE_BASE_URL`.
- `gemini-3.5-flash-lite` in the built-in catalog under `gemini`, as a worked example of
  the compatibility route alongside the native one. Each route has its own model, because
  a model id is both the catalog's key and the name sent to the API, so one id cannot
  appear twice.
- `ModelResult.output_reasoning_tokens`: tokens a model spent thinking. Billed at the
  output rate but never in the text, so a cost computed from `output_text_tokens` alone
  undercounts a reasoning model. Recorded in the trace's `usage` block.

### Changed

- `INCOMPLETE_FINISH_REASONS` gains Gemini's unclean finishes (`safety`, `recitation`,
  `blocklist`, `prohibited_content`, `spii`, `image_safety`, `language`, `other`) and
  `prompt_blocked`, the value `gemini-native` reports when the prompt itself is refused.
  Each now logs a warning, as a truncated answer already did.
- `available_services()` reports `gemini-native` alongside `gemini` when
  `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set, since one key serves both.
- **`gemini-3.8-flash` now names `gemini-native` in the built-in catalog.** Anything that
  builds its client from a catalog entry (a model picker, `make_model_client(entry.service,
  ...)`) now reaches it through Google's own SDK, and needs the `[gemini-native]` extra
  installed; `[all]` already includes it. Code that names `gemini` explicitly is
  unaffected and still reaches the compatibility shim. A user catalog that sets
  `"service": "gemini"` on an entry keeps overriding the built-in one, as before. One
  knock-on for allow-lists: `ModelConfig(services=("gemini",))` no longer lists
  `gemini-3.8-flash`, only the compatibility example below; name `gemini-native` as well
  (or use `available_services()`, which reports both for one key).

## [0.3.0] - 2026-09-30

Completes the image-generation parameter surface: every output control the provider
accepts can now be set, every one of them is refused rather than substituted when the
service cannot honour it, and the result reports what was actually produced.

### Added

- Output encoding on image generation: `background` ("transparent" / "opaque" / "auto"),
  `output_format` ("png" / "jpeg" / "webp") and `output_compression` (0-100). Transparency
  over a separately generated background is the substantive one; format and compression are
  storage and fidelity. Two combinations are contradictory and raise rather than producing
  something that looks right: transparency needs a format with an alpha channel, and a
  compression quality needs a lossy one. Both are judged against the format in effect --
  the one named, or the service's default, newly recorded as
  `ProviderSpec.image_default_format` -- so `output_compression=80` with no format named is
  refused rather than silently doing nothing. Format facts (which encodings carry alpha,
  which are lossy) live in `media.py` as `ALPHA_FORMATS` and `COMPRESSIBLE_FORMATS`, since
  they are properties of the formats and not of any provider. New
  `UnsupportedBackgroundError` and `UnsupportedFormatError` (both `ValueError`s),
  `supported_image_backgrounds()` and `supported_output_formats()`. Recorded in the trace
  among the requested settings. Omitting them leaves behaviour unchanged. (SA-361)
- `input_fidelity` on image generation: `generate_image(prompt, images=[...],
  input_fidelity="high")`. Controls how strictly reference images are adhered to, and is
  therefore only meaningful alongside them — passing it without `images` raises rather than
  being silently dropped, since a dropped setting is indistinguishable from a model that
  ignored it. Declared per provider by a new `ProviderSpec.image_fidelities`, answered
  without a client or a credential by `supported_input_fidelities()`, and refused by
  `UnsupportedFidelityError` (a `ValueError`) before the provider is touched. Reaches
  OpenAI's edit endpoint only, which is the same condition as having references at all.
  Recorded in the trace among the requested settings. Omitting it leaves behaviour
  unchanged. (SA-356)
- A repository check that fails the build when repo-bound text carries a term the
  maintainers keep private. It scans tracked file contents, tracked paths, every commit
  message in the pushed or proposed range, and the pull request title, body and branch name
  -- the last of those being reachable by no other check, and the route by which text has
  actually escaped before. The term list is a repository secret rather than a file, since a
  public workflow naming what it protects publishes it; in CI nothing matched is printed,
  because the log is public too. `scripts/guard.py` runs the same scan locally and does
  print what it found, and `.githooks/pre-commit` runs it against staged content. Fork pull
  requests get no secrets, so the check skips there.
- Output size, aspect ratio and quality for image generation:
  `generate_image(prompt, images=None, *, aspect_ratio=None, size=None, quality=None)`.
  `aspect_ratio` ("16:9") is the framing decision as stated and resolves to a concrete
  resolution — the smallest at that ratio, since pixels drive cost and a caller wanting
  more names a `size`. `aspect_ratio` and `size` are mutually exclusive. Nothing is
  silently substituted: `UnsupportedSizeError` and `UnsupportedQualityError` raise before
  any network call, and `supported_image_sizes()`, `supported_aspect_ratios()` and
  `supported_image_qualities()` answer without a client or a credential. Declared per
  provider by three new `ProviderSpec` fields (`image_sizes`, `image_custom_size`,
  `image_qualities`); OpenAI offers true 16:9 natively at 2048x1152 and 3840x2160. The
  trace records the requested settings separately from the produced ones. Calls passing
  none of these are unchanged. (SA-355)

- The image response is now reported rather than discarded. `MediaResult` carries the
  `size`, `quality`, `output_format`, `background` and `created` the provider says it
  **used** — read from the response, never echoed from the request — plus `extra` for
  response fields this package does not model yet. `ModelResult` gains
  `output_text_tokens` and `output_image_tokens`, the counterpart to SA-341's input split;
  output tokens dominate image cost, so without them a per-image figure cannot be known to
  be exact. All optional, so a provider reporting none of it yields the result it did
  before. The trace record carries the produced settings beside the artifact and the output
  split in its usage block. (SA-357)

### Changed

- `ModelClient._invoke_image()` now takes `(prompt, images, options)`, where `options` is
  an `ImageOptions`. An object rather than further positional parameters, because each
  image capability that lands wants another setting and the seam should stop churning: a
  field added to `ImageOptions` reaches every provider without another signature change.
  Breaks external provider subclasses, which is permitted pre-1.0. (SA-355)

### Fixed

- An image whose bytes the sniffer does not recognise is no longer labelled `image/png` on
  no evidence. The MIME type still describes the bytes in hand and a requested
  `output_format` never overrides it -- if they disagree, the bytes are what a caller will
  render. But where recognition fails, the fallback is now the format that was requested,
  then the service's declared default, and only then PNG. (SA-361)
- HuggingFace reports an unroutable model actionably again on the text path. The hub has
  two ways of declining: nothing serves the model for that task, or nothing the account has
  *enabled* serves it. Until now only the first was translated, via the bare
  `StopIteration` that `huggingface_hub` raised from an empty provider mapping.
  `huggingface-hub` 2.0.0 routes `conversational` through an auto-router that never fetches
  a mapping, so text began surfacing a raw `BadRequestError` instead. Both conditions now
  raise `ValueError` with distinct messages linking to their own remedy — a served-model
  search, or the account's inference-provider settings. Detection keys on the HTTP status
  plus the API's `model_not_supported` code, not on message text. Image and speech are
  unchanged; they still resolve a mapping and still raise `StopIteration`.

## [0.2.0] - 2026-09-25

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

### Changed

- `ModelClient._invoke_image()` previously took `(prompt, images)`. Protected, but external
  provider subclasses override it, so this breaks them — permitted while pre-1.0 and
  noted here rather than discovered. `images` is not defaulted, for the same reason
  `_invoke()`'s parameters are not.

## [0.1.0] - 2026-09-19

First public release: the library extracted from internal prototype work, with the
provider seam, the registry, the catalog and the observability contract in place.

### Added

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
