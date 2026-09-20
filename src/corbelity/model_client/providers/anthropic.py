"""Anthropic, direct API.

Note on sampling: the Messages API no longer takes `temperature`, `top_p` or `top_k`. They
are absent from MessageCreateParams as of anthropic 1.7, so passing one is a TypeError from
the SDK rather than a 400 from the service -- the argument never leaves the machine. This
client therefore never sends them, and pyproject pins `anthropic>=1.7` so that behaviour
matches the SDK that is actually installed.

The consequence for callers: a temperature or top_p passed to an Anthropic client is
accepted and ignored. That is deliberate. A UI's sliders always hold a value, and failing
every Anthropic call because a slider exists would be worse than quietly not sending a
parameter the API has withdrawn. `supports_sampling` in the catalog is now meaningless for
this provider (it still governs the OpenAI-compatible ones, which do accept sampling).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from ..client import ModelClient, get_field, load_sdk
from ..media import ImageInput, LLMResult
from ..messages import Message
from ..registry import ProviderSpec, get_spec
from . import anthropic_user_content

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from anthropic import Anthropic


class AnthropicClient(ModelClient["Anthropic"]):
    SPEC: ClassVar[ProviderSpec] = get_spec("anthropic")

    def _build_client(self) -> Anthropic:
        self._logger.info(
            "Initializing Anthropic client: %s (max_tokens=%s)",
            self._model, self._max_tokens,
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
        # No temperature / top_p / top_k: the Messages API withdrew all three. See the
        # module docstring.

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
