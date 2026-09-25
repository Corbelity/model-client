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

from .config import ModelConfig, get_default_config
from .errors import (
    MissingBaseUrlError,
    MissingCredentialsError,
    MissingDependencyError,
    TooManyImagesError,
    UnsupportedImageInputError,
    UnsupportedModalityError,
)
from .media import (
    IMAGE,
    SOUND,
    TEXT,
    ImageInput,
    LLMResult,
    MediaResult,
    ModelResult,
    validate_images,
)
from .messages import History, Message, validate_history
from .registry import ProviderSpec
from .trace import TraceSink

# finish_reason values that mean the model did NOT stop on its own -- the output is likely
# truncated or filtered, so complete() warns on them. The vocabulary is PROVIDER-SPECIFIC,
# which is why this is a union rather than a single equality check:
#   OpenAI / OpenRouter : clean = "stop";               bad = "length", "content_filter", "error"
#   Anthropic           : clean = "end_turn";           bad = "max_tokens", "refusal"
#   Ollama              : clean = "stop";               bad = "length"
#   HuggingFace / TGI   : clean = "stop" / "eos_token"; bad = "length"
# "tool_use" / "tool_calls" / "stop_sequence" are normal completions, intentionally absent.
# "error" is OpenRouter failing MID-STREAM: it still returns the partial text it had, with
# no usage block, so the only signal that the answer is cut short is this finish reason.
INCOMPLETE_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "content_filter", "refusal", "error"}
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

    def _invoke_image(self, prompt: str, images: tuple[ImageInput, ...]) -> MediaResult:
        """Overridden only by providers that generate images; the default keeps the
        contract honest for the rest.

        `images` are reference images to condition generation on, already validated and
        normalized to a tuple by generate_image() -- never None. Not defaulted, for the
        same reason `_invoke()`'s parameters are not: a provider that silently dropped a
        reference would return a confident generation that ignored it, and a TypeError at
        import is strictly better than that. A provider reaching this method with a
        non-empty tuple is a bug -- generate_image() rejects reference images for any
        service whose spec does not declare image_input."""
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
                       images: Sequence[ImageInput] | None = None) -> MediaResult:
        """Text-to-image. Returns raw bytes plus the MIME type needed to render them.

        `images` are REFERENCE images to condition the generation on -- a character sheet,
        a set, a prop -- so a face or a place stays the same between generations. They are
        input to the model, the mirror of the bytes coming back. Not every service can
        take them; ask its spec, or catch UnsupportedImageInputError.

        Validation runs before the provider is touched, and the no-images call is exactly
        what it was before this parameter existed."""
        self._require(IMAGE)
        pictures = validate_images(images)
        self._require_image_input(pictures)
        request: dict[str, Any] = {"prompt": prompt}
        if pictures:
            # Only recorded when something was actually sent, matching complete(). The
            # trace writer already turns an `images` key into artifact files on disk, so
            # this needs nothing further to keep bytes out of the JSONL.
            request["images"] = list(pictures)
        result = self._run(IMAGE, lambda: self._invoke_image(prompt, pictures), request)
        self._log_media(IMAGE, prompt, result)
        return result

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
