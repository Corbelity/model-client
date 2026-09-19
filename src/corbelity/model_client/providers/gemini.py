"""Gemini, through Google's OpenAI-compatibility endpoint.

This is a shim, and Google says so: their guidance is that if you are not already using
the OpenAI libraries, you should call the Gemini API directly. It is here because it
gets Gemini text working through the existing dialect with no new dependency and no new
request handling, which is the right first step -- not the last one.

Text only, deliberately:

  * Image generation through the compatibility endpoint is unverified. Adding
    `gemini-3.1-flash-image` to the catalog before it is tested live would put a model in
    a dropdown that may not be reachable by this path.
  * Audio generation on Gemini runs over the Live API, which is a bidirectional streaming
    session rather than a request/response call. It is not exposed through this endpoint
    at all, and streaming is an explicit non-goal (DESIGN.md, "What is deliberately not
    here"), so `generate_speech()` could not honestly wrap it.

Both wait for a native provider built on `google-genai`, where image output and a
request/response TTS route both exist. When that lands, it becomes a second service
rather than a change to this one -- the same way `ollama-local` and `ollama` coexist --
so anyone depending on this shim is not broken by it.
"""
from __future__ import annotations

from typing import ClassVar

from ..registry import ProviderSpec, get_spec
from .openai_compatible import OpenAICompatibleClient


class GeminiClient(OpenAICompatibleClient):
    SPEC: ClassVar[ProviderSpec] = get_spec("gemini")
