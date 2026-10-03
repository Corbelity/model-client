"""The provider-agnostic base class.

The public entry points -- complete(), generate_image(), generate_speech() -- are TEMPLATE
METHODS: this class owns validation, timing, uniform logging, and tracing via the shared
_run() wrapper, and each provider subclass implements _invoke*(), which makes the provider
call and returns a normalized result (content + token usage + finish reason). So EVERY
provider and EVERY modality is observed identically -- one INFO line per call (service /
model / modality / tokens / latency / finish), the full prompts and response at DEBUG, and,
when a TraceSink is injected, a full structured record of the call. A failure is logged and
traced with its latency before being re-raised, so a call is never silently lost.

Routing has no global local/cloud flag. The SERVICE NAME passed to make_model_client()
picks the provider, so every call site says plainly which provider it is using.

This module configures no logging. It takes a logger and attaches nothing to it; how those
records are formatted, filtered or written is the application's business.
"""
from __future__ import annotations

import importlib
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, ClassVar

from .catalog import load_catalog
from .config import ModelConfig, get_default_config
from .errors import (
    MissingBaseUrlError,
    MissingCredentialsError,
    MissingDependencyError,
    TooManyImagesError,
    UnsupportedBackgroundError,
    UnsupportedFidelityError,
    UnsupportedFormatError,
    UnsupportedImageInputError,
    UnsupportedModalityError,
    UnsupportedQualityError,
    UnsupportedSizeError,
)
from .media import (
    ALPHA_FORMATS,
    COMPRESSIBLE_FORMATS,
    IMAGE,
    SOUND,
    TEXT,
    ImageInput,
    ImageOptions,
    LLMResult,
    MediaResult,
    ModelResult,
    VideoInputs,
    VideoOptions,
    aspect_ratios_of,
    parse_size,
    sizes_for_ratio,
    validate_images,
)
from .messages import History, Message, validate_history
from .registry import ProviderSpec
from .trace import TraceSink
from .video import resolve_request

# finish_reason values that mean the model did NOT stop on its own -- the output is likely
# truncated or filtered, so complete() warns on them. The vocabulary is PROVIDER-SPECIFIC,
# which is why this is a union rather than a single equality check:
#   OpenAI / OpenRouter : clean = "stop";               bad = "length", "content_filter", "error"
#   Anthropic           : clean = "end_turn";           bad = "max_tokens", "refusal"
#   Ollama              : clean = "stop";               bad = "length"
#   HuggingFace / TGI   : clean = "stop" / "eos_token"; bad = "length"
#   Gemini (native)     : clean = "STOP";               bad = "MAX_TOKENS", "SAFETY",
#                         "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
#                         "IMAGE_SAFETY", "LANGUAGE", "OTHER", and "prompt_blocked" -- not a
#                         Gemini value, but what the provider reports when the PROMPT was
#                         refused and no candidate came back at all.
# "tool_use" / "tool_calls" / "stop_sequence" are normal completions, intentionally absent.
# "error" is OpenRouter failing MID-STREAM: it still returns the partial text it had, with
# no usage block, so the only signal that the answer is cut short is this finish reason.
# Compared lower-cased, which is why Gemini's upper-case MAX_TOKENS already matched.
INCOMPLETE_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "content_filter", "refusal", "error",
     "safety", "recitation", "blocklist", "prohibited_content", "spii", "image_safety",
     "language", "other", "prompt_blocked"}
)


def load_sdk(module: str, extra: str) -> ModuleType:
    """Import a provider SDK on demand, turning a missing optional dependency into a
    message with the install command in it.

    Imports live here rather than at module top so that installing this package for one
    provider never requires the other three, and so a broken release of any single SDK
    cannot make `import corbelity.model_client` fail."""
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise MissingDependencyError(module, extra) from err


def get_field(obj: Any, *names: str) -> Any:
    """Best-effort read of a field that may be a dict key OR an attribute, returning the
    first present non-None value or None. Lets token-usage extraction work across the
    providers' differing response shapes (dicts, pydantic models, SimpleNamespace) without
    ever raising on a missing field."""
    for name in names:
        try:
            if isinstance(obj, dict):
                if obj.get(name) is not None:
                    return obj[name]
            else:
                value = getattr(obj, name, None)
                if value is not None:
                    return value
        except Exception:
            continue
    return None


class ModelClient[ClientT](ABC):
    """Provider-agnostic model caller. One concrete subclass per provider;
        ClientT is that provider's SDK type;
        One instance per (model, temperature, max_tokens) configuration.

    `api_key` / `base_url` override the provider's environment variables for a single
    instance -- that is how a UI passes a per-session key through without mutating
    os.environ. Both fall back to the environment when None.

    Inject a TraceSink (`trace`) to capture every call into a structured trace; omit it
    and the client still logs each call (the trace is purely additive)."""

    # Set by each subclass from the registry. Carries the service name, the credential
    # environment variable names, the endpoint, and the supported modalities.
    SPEC: ClassVar[ProviderSpec]

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        trace: TraceSink | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        self._config = config if config is not None else get_default_config()
        self._model = model if model is not None else self._config.default_model
        self._temperature = (
            temperature if temperature is not None else self._config.default_temperature
        )
        self._top_p = top_p if top_p is not None else self._config.default_top_p
        self._max_tokens = (
            max_tokens if max_tokens is not None else self._config.default_max_tokens
        )
        # Ollama context window (local and cloud); providers that manage their own
        # context ignore it.
        self._num_ctx = num_ctx if num_ctx is not None else self._config.num_ctx
        self._api_key = api_key or None          # "" from an empty UI field means "not set"
        self._base_url = base_url or None
        self.last_result: ModelResult | None = None
        self._trace = trace
        self._logger = logging.getLogger(__name__)
        self._client: ClientT = self._build_client()

    # ----------------------------------------------------------------------- #
    # Identity and capability
    # ----------------------------------------------------------------------- #
    @property
    def service(self) -> str:
        return self.SPEC.name

    @property
    def model(self) -> str:
        return self._model

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def supported_modalities(self) -> frozenset[str]:
        return self.SPEC.modalities

    def supports(self, modality: str) -> bool:
        return modality in self.SPEC.modalities

    @property
    def prompt_tokens(self) -> int | None:
        return self.last_result.prompt_tokens if self.last_result else None

    @property
    def completion_tokens(self) -> int | None:
        return self.last_result.completion_tokens if self.last_result else None

    @property
    def total_tokens(self) -> int | None:
        return self.last_result.total_tokens if self.last_result else None

    @property
    def finish_reason(self) -> str | None:
        return self.last_result.finish_reason if self.last_result else None

    # ----------------------------------------------------------------------- #
    # Credential and endpoint resolution
    # ----------------------------------------------------------------------- #
    def _resolve_key(self) -> str:
        """Per-instance override first, then each configured environment variable in
        order. Raises naming all of them, so a missing credential is self-explanatory."""
        if self._api_key:
            return self._api_key
        for name in self.SPEC.key_env:
            value = os.getenv(name, "").strip()
            if value:
                return value
        raise MissingCredentialsError(self.SPEC.name, self.SPEC.key_env)

    def _resolve_base_url(self) -> str | None:
        """Per-instance override, then environment, then the spec's default. Returns None
        when the provider has no endpoint of its own to configure."""
        if self._base_url:
            return self._base_url
        for name in self.SPEC.base_url_env:
            value = os.getenv(name, "").strip()
            if value:
                return value
        if self.SPEC.default_base_url:
            return self.SPEC.default_base_url
        if self.SPEC.requires_base_url:
            raise MissingBaseUrlError(self.SPEC.name, self.SPEC.base_url_env)
        return None

    def _require(self, modality: str) -> None:
        if modality not in self.SPEC.modalities:
            raise UnsupportedModalityError(self.SPEC.name, modality, self.SPEC.modalities)

    # ----------------------------------------------------------------------- #
    # Provider seam
    # ----------------------------------------------------------------------- #
    @abstractmethod
    def _build_client(self) -> ClientT: ...

    @abstractmethod
    def _invoke(self, system: str, user: str, history: tuple[Message, ...],
                images: tuple[ImageInput, ...]) -> LLMResult:
        """Make the provider call and return a normalized LLMResult. Subclasses build the
        request from self._model / self._temperature / self._max_tokens and parse their
        provider's response; timing, logging, tracing and error capture are handled by
        _run() so every provider gets identical observability.

        `history` is prior turns, already validated and normalized to a tuple by
        complete() -- never None, so no subclass needs its own guard. It goes BETWEEN the
        system prompt and the current user turn; where the system prompt itself lives is
        provider-specific (see each subclass).

        `images` attaches to the CURRENT user turn only and is likewise never None.
        It is NOT defaulted here on purpose: a subclass that silently dropped an
        attachment would answer confidently about a picture the model never saw, and a
        TypeError at import time is strictly better than that."""
        ...

    def _invoke_image(self, prompt: str, images: tuple[ImageInput, ...],
                      options: ImageOptions) -> MediaResult:
        """Overridden only by providers that generate images; the default keeps the
        contract honest for the rest.

        `images` are reference images to condition generation on, already validated and
        normalized to a tuple by generate_image() -- never None. Not defaulted, for the
        same reason `_invoke()`'s parameters are not: a provider that silently dropped a
        reference would return a confident generation that ignored it, and a TypeError at
        import is strictly better than that. A provider reaching this method with a
        non-empty tuple is a bug -- generate_image() rejects reference images for any
        service whose spec does not declare image_input.

        `options` carries the requested output settings, already validated against this
        service. A setting left as None was not requested and must be OMITTED from the
        provider call, so the provider's own default applies rather than one invented
        here. Not defaulted, for the same reason `images` is not: a provider that
        silently ignored a requested size would return a confidently wrong image."""
        raise UnsupportedModalityError(self.SPEC.name, IMAGE, self.SPEC.modalities)

    def _invoke_speech(self, text: str) -> MediaResult:
        raise UnsupportedModalityError(self.SPEC.name, SOUND, self.SPEC.modalities)

    # ----------------------------------------------------------------------- #
    # Shared observability wrapper. Every public entry point goes through this, so
    # timing / logging / error capture exist in exactly ONE place rather than once
    # per modality.
    # ----------------------------------------------------------------------- #
    def _run[R: ModelResult](self, modality: str, invoke: Callable[[], R],
                             request: Mapping[str, Any] | None = None) -> R:
        t0 = time.perf_counter()
        try:
            result = invoke()
        except Exception as err:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._logger.exception(
                "Model call FAILED (%s/%s, %s) after %.0f ms",
                self.SPEC.name, self._model, modality, latency_ms,
            )
            # Traced BEFORE re-raising, with the request that caused it -- a failure you
            # can't reproduce is the one you most needed the trace for.
            self._trace_call(modality, latency_ms, request, error=repr(err))
            raise
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_result = result
        self._logger.info(
            "Model call %s/%s (%s): %s prompt + %s completion = %s tokens, finish=%s, %.0f ms",
            self.SPEC.name, self._model, modality, result.prompt_tokens,
            result.completion_tokens, result.total_tokens, result.finish_reason, latency_ms,
        )
        self._trace_call(modality, latency_ms, request, result=result)
        return result

    def _trace_call(self, modality: str, latency_ms: float,
                    request: Mapping[str, Any] | None, *,
                    result: ModelResult | None = None, error: str | None = None) -> None:
        """Emit one trace record, if a tracer was injected.

        Swallows its own failures on purpose: observability must never be load-bearing.
        A full disk should not turn a working model call into a failed one."""
        if self._trace is None:
            return
        try:
            self._trace.llm_call(
                service=self.SPEC.name, model=self._model, modality=modality,
                latency_ms=latency_ms, request=request or {}, result=result, error=error,
            )
        except Exception:
            self._logger.warning(
                "Trace write failed for %s/%s -- the call itself was unaffected.",
                self.SPEC.name, self._model, exc_info=True,
            )

    # ----------------------------------------------------------------------- #
    # Public entry points
    # ----------------------------------------------------------------------- #
    def complete(self, system: str, user: str, history: History | None = None,
                 images: Sequence[ImageInput] | None = None) -> str:
        """Provider-agnostic entry point for TEXT. Times the call, records full visibility
        of it (model, service, token usage, latency and -- via the injected TraceSink --
        the prompts and response), then returns the raw text. A failure is logged and
        traced with its latency before being re-raised, so no call is silently lost.

        `history` is prior user/assistant turns (see validate_history). `images` are
        attachments for the CURRENT turn (see validate_images); they never enter history.
        Omit both and the request built is byte-identical to a single-turn text call.
        Both are validated BEFORE the provider is touched, so a malformed request costs
        nothing."""
        self._require(TEXT)
        turns = validate_history(history)
        pictures = validate_images(images)
        request: dict[str, Any] = {"system": system, "user": user, "history": list(turns)}
        if pictures:
            # Only recorded when something was actually sent -- the trace mirrors the
            # request, and an always-present empty list is noise in every text record.
            request["images"] = list(pictures)
        result = self._run(TEXT, lambda: self._invoke(system, user, turns, pictures), request)
        self._logger.debug(
            "Model call %s/%s prompts+response (%d prior turns, %d image(s)):"
            "\n--- system ---\n%s\n--- user ---\n%s\n--- response ---\n%s",
            self.SPEC.name, self._model, len(turns), len(pictures), system, user, result.text,
        )
        # Surface a degraded response so a weak or truncated model is LOUD, not silent.
        # The text and finish_reason are already normalized by _invoke; only the SET of
        # "bad" finish reasons is provider-specific (see INCOMPLETE_FINISH_REASONS).
        if not result.text.strip():
            self._logger.warning(
                "Model call %s/%s returned an EMPTY response (finish=%s).",
                self.SPEC.name, self._model, result.finish_reason,
            )
        elif str(result.finish_reason or "").lower() in INCOMPLETE_FINISH_REASONS:
            self._logger.warning(
                "Model call %s/%s did not finish cleanly (finish=%s) -- the response may "
                "be truncated or filtered.",
                self.SPEC.name, self._model, result.finish_reason,
            )
        return result.text

    def generate_image(self, prompt: str,
                       images: Sequence[ImageInput] | None = None,
                       *,
                       aspect_ratio: str | None = None,
                       size: str | None = None,
                       quality: str | None = None,
                       input_fidelity: str | None = None,
                       background: str | None = None,
                       output_format: str | None = None,
                       output_compression: int | None = None) -> MediaResult:
        """Text-to-image. Returns raw bytes plus the MIME type needed to render them.

        `images` are REFERENCE images to condition the generation on -- a character sheet,
        a set, a prop -- so a face or a place stays the same between generations. They are
        input to the model, the mirror of the bytes coming back. Not every service can
        take them; ask its spec, or catch UnsupportedImageInputError.

        `aspect_ratio` ("16:9") is the framing decision as a director states it; the client
        resolves it to a concrete resolution on this service. `size` ("2048x1152") names
        the pixels outright. They are MUTUALLY EXCLUSIVE -- passing both is an error rather
        than a precedence puzzle. `quality` is independent of both.

        `input_fidelity` ("high" / "low") is how strictly the REFERENCE images are
        adhered to, so it is only meaningful when `images` are supplied, and passing it
        without them raises. That is deliberate: a caller who set it and saw no effect
        would have no way to tell whether the model ignored it or this client dropped it.

        `background` ("transparent" / "opaque" / "auto"), `output_format` ("png" / "jpeg"
        / "webp") and `output_compression` (0-100) control the encoding. Two combinations
        are contradictory and raise rather than producing something that looks right:
        transparency needs a format with an alpha channel, and a compression quality needs
        a lossy one. Both are judged against the format actually in effect -- the one named,
        or the service's own default when none is.

        None of this is ever silently substituted: a request this service cannot honour
        raises. Ask ahead with supported_image_sizes(), supported_aspect_ratios(),
        supported_image_qualities(), supported_input_fidelities(),
        supported_image_backgrounds() and supported_output_formats(), none of which need a
        client or a credential.

        Everything is validated before the provider is touched, and a call passing none of
        these behaves exactly as it did before the parameters existed."""
        self._require(IMAGE)
        pictures = validate_images(images)
        self._require_image_input(pictures)
        options = self._resolve_image_options(
            aspect_ratio=aspect_ratio, size=size, quality=quality,
            input_fidelity=input_fidelity, has_references=bool(pictures),
            background=background, output_format=output_format,
            output_compression=output_compression
        )
        request: dict[str, Any] = {"prompt": prompt}
        if pictures:
            # Only recorded when something was actually sent, matching complete(). The
            # trace writer already turns an `images` key into artifact files on disk, so
            # this needs nothing further to keep bytes out of the JSONL.
            request["images"] = list(pictures)
        if options:
            # Recorded as what was ASKED FOR. What came back is on the result, and keeping
            # the two apart in the record is what lets anyone notice they differ.
            asked: dict[str, Any] = {
                "size": options.size,
                "quality": options.quality,
                "input_fidelity": options.input_fidelity,
                "background": options.background,
                "output_format": options.output_format,
                "output_compression": options.output_compression,
            }
            if aspect_ratio is not None:
                asked["aspect_ratio"] = aspect_ratio
            request["requested"] = {k: v for k, v in asked.items() if v is not None}
        result = self._run(
            IMAGE, lambda: self._invoke_image(prompt, pictures, options), request
        )
        self._log_media(IMAGE, prompt, result)
        return result

    def _resolve_image_options(self, *, aspect_ratio: str | None, size: str | None,
                               quality: str | None, input_fidelity: str | None = None,
                               has_references: bool = False,
                               background: str | None = None,
                               output_format: str | None = None,
                               output_compression: int | None = None) -> ImageOptions:
        """Turn requested output settings into what this provider will be sent.

        Every rejection here happens before the provider is touched, and none of them
        substitutes. Refusing costs one error; substituting costs an application that
        believes it has something it does not."""
        if aspect_ratio is not None and size is not None:
            raise ValueError(
                "aspect_ratio and size are mutually exclusive -- pass one, not both "
                f"(got aspect_ratio={aspect_ratio!r}, size={size!r})."
            )

        resolved = size
        if aspect_ratio is not None:
            candidates = sizes_for_ratio(aspect_ratio, self.SPEC.image_sizes)
            if not candidates:
                raise UnsupportedSizeError(
                    self.SPEC.name, aspect_ratio, aspect_ratios_of(self.SPEC.image_sizes)
                )
            # Smallest first. Pixels drive cost, and the cheap end is the safe default to
            # pick on the caller's behalf: a storyboard thumbnail at 4K is money spent on
            # an image nobody will look at closely, while a caller who wants the large one
            # names a size and gets exactly it.
            resolved = candidates[0]

        if resolved is not None:
            self._require_size(resolved)
        if quality is not None:
            self._require_quality(quality)
        if input_fidelity is not None:
            # Order matters: the no-references case is refused BEFORE the value is checked
            # against the spec, so a caller who passed it on a plain generation is told the
            # real problem rather than being sent to look at the accepted values.
            if not has_references:
                raise ValueError(
                    "input_fidelity governs adherence to reference images, so it needs "
                    f"images= to act on (got input_fidelity={input_fidelity!r} with "
                    "none). Drop it, or supply the references it applies to."
                )
            self._require_fidelity(input_fidelity)

        if background is not None:
            self._require_background(background)
        if output_format is not None:
            self._require_output_format(output_format)
        if output_compression is not None:
            self._require_compression(output_compression)

        # Cross-parameter checks come last and reason about the format actually IN EFFECT:
        # the one named, or the service's declared default when none was. A caller who asks
        # for transparency and names no format is asking a perfectly sensible question, and
        # whether it works depends on what the endpoint produces by default.
        effective = output_format or self.SPEC.image_default_format
        if effective is not None:
            if background == "transparent" and effective not in ALPHA_FORMATS:
                raise ValueError(
                    f"background='transparent' needs a format with an alpha channel, and "
                    f"{effective!r} has none. Name an output_format from "
                    f"{sorted(ALPHA_FORMATS)}, or drop the transparent background -- the "
                    "alternative is an opaque image a caller believes is transparent."
                )
            if output_compression is not None and effective not in COMPRESSIBLE_FORMATS:
                raise ValueError(
                    f"output_compression has no effect on {effective!r}, which is "
                    f"lossless. Name an output_format from {sorted(COMPRESSIBLE_FORMATS)}, "
                    "or drop the compression. Passing it here would be a setting that "
                    "cannot do anything, which is indistinguishable from one that failed."
                )

        return ImageOptions(size=resolved, quality=quality,
                            input_fidelity=input_fidelity,
                            background=background, output_format=output_format,
                            output_compression=output_compression)

    def _require_size(self, size: str) -> None:
        spec = self.SPEC
        if size in spec.image_sizes:
            return
        # A well-formed resolution goes through on a service that documents custom sizes.
        # This is passthrough, not approval: if the provider refuses it, that refusal is
        # what the caller sees -- which is still better than this client guessing.
        if spec.image_custom_size and parse_size(size) is not None:
            return
        raise UnsupportedSizeError(spec.name, size, spec.image_sizes)

    def _require_quality(self, quality: str) -> None:
        if quality not in self.SPEC.image_qualities:
            raise UnsupportedQualityError(
                self.SPEC.name, quality, self.SPEC.image_qualities
            )

    def _require_fidelity(self, fidelity: str) -> None:
        if fidelity not in self.SPEC.image_fidelities:
            raise UnsupportedFidelityError(
                self.SPEC.name, fidelity, self.SPEC.image_fidelities
            )

    def _require_background(self, background: str) -> None:
        if background not in self.SPEC.image_backgrounds:
            raise UnsupportedBackgroundError(
                self.SPEC.name, background, self.SPEC.image_backgrounds
            )

    def _require_output_format(self, output_format: str) -> None:
        if output_format not in self.SPEC.image_output_formats:
            raise UnsupportedFormatError(
                self.SPEC.name, output_format, self.SPEC.image_output_formats
            )

    def _require_compression(self, compression: int) -> None:
        """Type and range before applicability. A bool is excluded explicitly because it
        IS an int in Python: output_compression=True would otherwise sail through as 1 and
        quietly mean maximum compression."""
        if isinstance(compression, bool) or not isinstance(compression, int):
            raise ValueError(
                "output_compression is a percentage from 0 to 100, as an int "
                f"(got {compression!r})."
            )
        if not 0 <= compression <= 100:
            raise ValueError(
                f"output_compression must be between 0 and 100 (got {compression})."
            )
        if not self.SPEC.image_compression:
            raise ValueError(
                f"Service {self.SPEC.name!r} has no output compression control, so "
                f"output_compression={compression} would be dropped. Omit it."
            )

    def _require_image_input(self, images: tuple[ImageInput, ...]) -> None:
        """Gate reference images on the PROVIDER's declared capability and cap.

        Not on the catalog's per-model accepts_images flag: the catalog is descriptive and
        gates nothing (DESIGN.md §6), so a model missing from it must still work. The cap
        is checked here rather than in validate_images() for the complementary reason --
        a provider's published API limit is a fact about that provider, while a count
        limit in shared validation would be policy, which belongs at the caller's own
        boundary."""
        if not images:
            return
        if not self.SPEC.image_input:
            raise UnsupportedImageInputError(self.SPEC.name)
        maximum = self.SPEC.max_reference_images
        if maximum is not None and len(images) > maximum:
            raise TooManyImagesError(self.SPEC.name, len(images), maximum)

    def _resolve_video_request(self, inputs: VideoInputs,
                               options: VideoOptions) -> VideoOptions:
        """Judge a video request for this client's service and model; return the options
        to send. Every video entry point goes through here, before the provider is
        touched.

        Reads the UNFILTERED catalog, never catalog_for(): a model a UI has filtered out
        of its dropdown must be judged exactly as it would be if listed. Narrowing what
        is shown must never change what a call is allowed to do (DESIGN.md section 6)."""
        entry = load_catalog(self._config.catalog_path).get(self._model)
        return resolve_request(self.SPEC, self._model, entry, inputs, options)

    def generate_speech(self, text: str) -> MediaResult:
        """Text-to-speech. Returns raw audio bytes plus the sniffed MIME type."""
        self._require(SOUND)
        result = self._run(SOUND, lambda: self._invoke_speech(text), {"prompt": text})
        self._log_media(SOUND, text, result)
        return result

    def _log_media(self, modality: str, prompt: str, result: MediaResult) -> None:
        self._logger.debug(
            "Model call %s/%s (%s) prompt:\n%s\n--- returned %d bytes of %s ---",
            self.SPEC.name, self._model, modality, prompt, len(result.data), result.mime_type,
        )
        if not result.data:
            self._logger.warning(
                "Model call %s/%s (%s) returned an EMPTY payload -- nothing to render.",
                self.SPEC.name, self._model, modality,
            )
