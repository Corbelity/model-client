"""Anthropic, direct API."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from ..catalog import load_catalog
from ..client import ModelClient, get_field, load_sdk
from ..media import ImageInput, LLMResult
from ..messages import Message
from ..registry import ProviderSpec, get_spec
from . import anthropic_user_content

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from anthropic import Anthropic


class AnthropicClient(ModelClient["Anthropic"]):
    SPEC: ClassVar[ProviderSpec] = get_spec("anthropic")

    # Some current Anthropic models REMOVED temperature/top_p -- sending either returns a
    # 400, not a warning. A UI's sliders always have a value, so the client drops them
    # rather than making every caller know which models care.
    #
    # This prefix list is a FALLBACK. The catalog is consulted first: an entry with
    # "supports_sampling": false (or true) is authoritative, which is how a model released
    # after this package shipped gets handled without a code change. Matched by prefix
    # because the families keep growing.
    _SAMPLING_UNSUPPORTED_PREFIXES: ClassVar[tuple[str, ...]] = (
        "claude-fable-", "claude-mythos-", "claude-opus-5", "claude-sonnet-5",
        "claude-opus-4-7", "claude-opus-4-8",
    )

    def _accepts_sampling_params(self) -> bool:
        entry = load_catalog(self._config.catalog_path).get(self._model)
        if entry is not None and entry.supports_sampling is not None:
            return entry.supports_sampling
        return not self._model.startswith(self._SAMPLING_UNSUPPORTED_PREFIXES)

    def _build_client(self) -> Anthropic:
        self._logger.info(
            "Initializing Anthropic client: %s (temperature=%s, max_tokens=%s)",
            self._model, self._temperature, self._max_tokens,
        )
        sdk = load_sdk("anthropic", self.SPEC.extra)
        # cast rather than `# type: ignore[no-any-return]`: with the SDK installed the call
        # returns a typed object and the ignore is REQUIRED; without it the expression is
        # already Any and the same ignore is reported as UNUSED. CI type-checks both
        # environments, so only a construct that is valid in both will pass.
        return cast("Anthropic", sdk.Anthropic(api_key=self._resolve_key()))

    def _invoke(self, system: str, user: str, history: tuple[Message, ...],
                images: tuple[ImageInput, ...]) -> LLMResult:
        # Anthropic takes the system prompt as its OWN top-level parameter, so `messages`
        # holds only the conversation turns -- unlike every other provider here.
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [
                *history,
                {"role": "user", "content": anthropic_user_content(user, images)},
            ],
        }
        if self._accepts_sampling_params():
            payload["temperature"] = self._temperature
            if self._top_p is not None:
                payload["top_p"] = self._top_p

        response = self._client.messages.create(**payload)

        # The Messages API can return content in multiple blocks; concatenate the text.
        parts: list[str] = [
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ]
        # Token usage lives on response.usage (input_tokens / output_tokens); absent on
        # some stubs and responses, so read defensively.
        usage = getattr(response, "usage", None)
        prompt = get_field(usage, "input_tokens") if usage is not None else None
        completion = get_field(usage, "output_tokens") if usage is not None else None
        total = (prompt + completion) if (prompt is not None and completion is not None) else None
        return LLMResult(
            text="".join(parts),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            finish_reason=getattr(response, "stop_reason", None),
        )
