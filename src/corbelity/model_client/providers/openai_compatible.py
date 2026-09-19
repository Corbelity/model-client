"""The chat-completions dialect, shared by every provider that speaks it.

Three services here use the same `openai` SDK against different endpoints: OpenRouter,
OpenAI directly, and Gemini through Google's OpenAI-compatibility endpoint. They differ
only in how the client is built, so the request and response handling lives once, exactly
as the two Ollama clients share a base.

Two per-model quirks are read from the catalog rather than hardcoded, because both change
on the providers' schedule:

  * `supports_sampling` -- some models reject `temperature`/`top_p` with a 400.
  * `max_tokens_param`  -- newer OpenAI models reject `max_tokens` and require
    `max_completion_tokens` instead.

Both default to the permissive, historically-correct behaviour when a model is not in the
catalog, so an unlisted model still works.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ..catalog import ModelInfo, load_catalog
from ..client import ModelClient, get_field, load_sdk
from ..media import ImageInput, LLMResult
from ..messages import Message
from . import openai_user_content

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from openai import OpenAI

DEFAULT_MAX_TOKENS_PARAM = "max_tokens"


class OpenAICompatibleClient(ModelClient["OpenAI"]):
    """Shared chat-completions request and response handling.

    Public rather than underscore-private: a third party pointing this at another
    OpenAI-compatible gateway should subclass it, set a SPEC, and inherit everything.
    """

    def _catalog_entry(self) -> ModelInfo | None:
        return load_catalog(self._config.catalog_path).get(self._model)

    def _accepts_sampling_params(self) -> bool:
        """Whether this model tolerates temperature/top_p. Unlisted models are assumed to
        accept them, which is the long-standing behaviour of this dialect."""
        entry = self._catalog_entry()
        if entry is not None and entry.supports_sampling is not None:
            return entry.supports_sampling
        return True

    def _max_tokens_key(self) -> str:
        """The name this model wants for its completion cap.

        Newer OpenAI models reject `max_tokens` outright in favour of
        `max_completion_tokens`. Driven by catalog data rather than a model-name prefix
        list for the same reason the sampling flag is: the set of affected models grows
        on OpenAI's release schedule, not on this package's."""
        entry = self._catalog_entry()
        if entry is not None and entry.max_tokens_param:
            return entry.max_tokens_param
        return DEFAULT_MAX_TOKENS_PARAM

    def _build_client(self) -> OpenAI:
        # SPEC.extra differs per service (openrouter / openai / gemini) even though all
        # three resolve to the same SDK, so the install hint on a missing dependency names
        # the extra the caller actually asked for.
        sdk = load_sdk("openai", self.SPEC.extra)
        return cast("OpenAI", sdk.OpenAI(
            base_url=self._resolve_base_url(),
            api_key=self._resolve_key(),
        ))

    def _invoke(self, system: str, user: str, history: tuple[Message, ...],
                images: tuple[ImageInput, ...]) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self._model,
            # Bound completion on EVERY path, under whichever name this model accepts.
            self._max_tokens_key(): self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                *history,
                {"role": "user", "content": openai_user_content(user, images)},
            ],
            "stream": False,
        }
        if self._accepts_sampling_params():
            payload["temperature"] = self._temperature
            if self._top_p is not None:
                payload["top_p"] = self._top_p

        response = self._client.chat.completions.create(**payload)

        # The response carries a single content block; usage carries the token counts.
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        # OpenAI-shaped providers return content=None (not "") when there is nothing to
        # say; coerce to "" so the base complete() empty-response check can see it
        # (str(None) would be the literal "None"). NOTE: finish_reason lives on the
        # CHOICE, not the response.
        content = choice.message.content
        return LLMResult(
            text="" if content is None else str(content),
            prompt_tokens=get_field(usage, "prompt_tokens") if usage is not None else None,
            completion_tokens=get_field(usage, "completion_tokens") if usage is not None else None,
            total_tokens=get_field(usage, "total_tokens") if usage is not None else None,
            finish_reason=getattr(choice, "finish_reason", None),
        )
