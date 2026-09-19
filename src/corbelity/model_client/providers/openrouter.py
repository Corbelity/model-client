"""OpenRouter, via the OpenAI SDK's chat-completions interface."""
from __future__ import annotations

from typing import ClassVar

from ..registry import ProviderSpec, get_spec
from .openai_compatible import OpenAICompatibleClient


class OpenRouterClient(OpenAICompatibleClient):
    """Cloud OpenRouter.

    Request and response handling is inherited: OpenRouter, OpenAI and Gemini's
    compatibility endpoint all speak the same dialect through the same SDK, and differ
    only in endpoint and credential, which the spec supplies.

    A local Ollama server also speaks this protocol, but it has its own client
    (OllamaLocalClient) so that no call site has to infer from a base URL which box it is
    talking to.
    """

    SPEC: ClassVar[ProviderSpec] = get_spec("openrouter")
