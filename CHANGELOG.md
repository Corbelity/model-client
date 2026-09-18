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
  catalog, the registry and the observability contract, using a fake provider rather than
  network calls.

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
