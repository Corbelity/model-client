"""OpenRouter, via the OpenAI SDK's chat-completions interface."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from ..client import ModelClient, get_field, load_sdk
from ..media import ImageInput, LLMResult
from ..messages import Message
from ..registry import ProviderSpec, get_spec
from . import openai_user_content

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from openai import OpenAI


class OpenRouterClient(ModelClient["OpenAI"]):
    """Cloud OpenRouter. A local Ollama server also speaks this OpenAI-compatible
    protocol, but it has its own client here (OllamaLocalClient) so that no call site has
    to infer from a base URL which box it is talking to."""

    SPEC: ClassVar[ProviderSpec] = get_spec("openrouter")

    def _build_client(self) -> OpenAI:
        sdk = load_sdk("openai", self.SPEC.extra)
        return cast("OpenAI", sdk.OpenAI(
            base_url=self._resolve_base_url(),
            api_key=self._resolve_key(),
        ))

    def _invoke(self, system: str, user: str, history: tuple[Message, ...],
                images: tuple[ImageInput, ...]) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,    # bound completion on EVERY path
            "messages": [
                {"role": "system", "content": system},
                *history,
                {"role": "user", "content": openai_user_content(user, images)},
            ],
            "stream": False,
        }
        if self._top_p is not None:
            payload["top_p"] = self._top_p

        response = self._client.chat.completions.create(**payload)

        # The response carries a single content block; usage carries the token counts.
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        # OpenAI and OpenRouter return content=None (not "") when there is nothing to say;
        # coerce to "" so the base complete() empty-response check can see it (str(None)
        # would be the literal "None"). NOTE: finish_reason lives on the CHOICE, not the
        # response.
        content = choice.message.content
        return LLMResult(
            text="" if content is None else str(content),
            prompt_tokens=get_field(usage, "prompt_tokens") if usage is not None else None,
            completion_tokens=get_field(usage, "completion_tokens") if usage is not None else None,
            total_tokens=get_field(usage, "total_tokens") if usage is not None else None,
            finish_reason=getattr(choice, "finish_reason", None),
        )
