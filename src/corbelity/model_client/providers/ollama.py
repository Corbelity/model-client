"""Ollama, local host and cloud.

The two differ ONLY in how the SDK client is built (host plus an auth header), so the
request and response handling lives once in the shared base below.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from ..client import ModelClient, get_field, load_sdk
from ..media import ImageInput, LLMResult
from ..messages import Message
from ..registry import ProviderSpec, get_spec

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from ollama import Client


def strip_v1(url: str) -> str:
    """Turn an OpenAI-compatible base URL into a bare Ollama host.

    LOCAL_OLLAMA_URL is commonly shared with an OpenAI-compatible path, which needs the
    '/v1' base; the NATIVE ollama client wants the bare host and appends its own '/api/...'
    routes, so a '/v1' suffix would produce '/v1/api/chat' -> '404 page not found'."""
    host = url.rstrip("/")
    if host.endswith("/v1"):
        host = host[: -len("/v1")]
    return host


class _OllamaClient(ModelClient["Client"]):
    """Shared Ollama request/response handling."""

    def _invoke(self, system: str, user: str, history: tuple[Message, ...],
                images: tuple[ImageInput, ...]) -> LLMResult:
        # Native Ollama: generation knobs go in `options`. num_predict caps the completion
        # length; num_ctx sizes the context window (its small default truncates long
        # generations). num_ctx is sent only when configured.
        options: dict[str, float] = {
            "num_predict": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._top_p is not None:
            options["top_p"] = self._top_p
        if self._num_ctx:
            options["num_ctx"] = self._num_ctx

        # Ollama has no content-blocks form: images ride in a sibling array of base64
        # strings. The SDK's Image type also accepts bytes or a path, so an explicit str
        # keeps the path-detection branch out of it. The key is omitted entirely when
        # empty -- an empty array is not the same request as no array.
        current: dict[str, Any] = {"role": "user", "content": user}
        if images:
            current["images"] = [image.base64() for image in images]

        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                *history,
                current,
            ],
            options=options,
        )
        # ollama returns a ChatResponse (mapping- AND attribute-accessible); read both ways.
        message = get_field(response, "message") or {}
        text = get_field(message, "content") or ""
        return LLMResult(
            text=str(text),
            prompt_tokens=get_field(response, "prompt_eval_count"),
            completion_tokens=get_field(response, "eval_count"),
            finish_reason=get_field(response, "done_reason"),
        )


class OllamaLocalClient(_OllamaClient):
    """Ollama on a local host. No credential -- the box is trusted."""

    SPEC: ClassVar[ProviderSpec] = get_spec("ollama-local")

    def _build_client(self) -> Client:
        # Configuration is resolved BEFORE the SDK is imported, and this ordering is
        # deliberate in both Ollama clients: it makes the failure for a misconfigured
        # client identical whether or not the optional SDK happens to be installed. With
        # the import first, the same missing LOCAL_OLLAMA_URL raises MissingBaseUrlError
        # on a developer's machine and MissingDependencyError in a bare CI job, which is
        # exactly the kind of environment-dependent error that is miserable to reproduce.
        host = self._resolve_base_url()
        sdk = load_sdk("ollama", self.SPEC.extra)
        return sdk.Client(host=strip_v1(host or ""))  # type: ignore[no-any-return]


class OllamaCloudClient(_OllamaClient):
    """Ollama Cloud. Same wire protocol as local, plus a bearer token. The ollama SDK
    forwards **kwargs to its underlying httpx client, which is how headers get through."""

    SPEC: ClassVar[ProviderSpec] = get_spec("ollama")

    def _build_client(self) -> Client:
        host = self._resolve_base_url()
        key = self._resolve_key()
        sdk = load_sdk("ollama", self.SPEC.extra)
        return sdk.Client(  # type: ignore[no-any-return]
            host=strip_v1(host or ""),
            headers={"Authorization": f"Bearer {key}"},
        )
