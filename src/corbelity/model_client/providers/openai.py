"""OpenAI, direct API: text, image and speech.

Text is inherited from the shared chat-completions base. Images and speech are NOT chat
completions -- they come from separate SDK surfaces (`images.generate` and
`audio.speech.create`) with their own response shapes, which is why they are implemented
here rather than shared.

Note on the module name: this file is `corbelity.model_client.providers.openai`, and it
imports the third-party `openai` package. Python 3 imports are absolute, so there is no
shadowing -- the same arrangement `providers/anthropic.py` already uses.
"""
from __future__ import annotations

import base64
from typing import TYPE_CHECKING, ClassVar, cast

from ..client import get_field, load_sdk
from ..errors import ModelClientError
from ..media import MediaResult, sniff_audio_mime, sniff_image_mime
from ..registry import ProviderSpec, get_spec
from .openai_compatible import OpenAICompatibleClient

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from openai import OpenAI


class OpenAIClient(OpenAICompatibleClient):
    SPEC: ClassVar[ProviderSpec] = get_spec("openai")

    # The speech endpoint requires a voice, and there is no provider default to fall back
    # on. A class attribute rather than a constructor argument, because the constructor is
    # uniform across every provider and a single provider's option does not belong in it.
    # Override per instance (`client.voice = "verse"`) or per subclass. If voice, format
    # and speed all end up needing configuration, they belong together in a provider
    # options object rather than accumulating here one at a time.
    voice: ClassVar[str] = "alloy"

    def _build_client(self) -> OpenAI:
        self._logger.info(
            "Initializing OpenAI client: %s (temperature=%s, max_tokens=%s)",
            self._model, self._temperature, self._max_tokens,
        )
        sdk = load_sdk("openai", self.SPEC.extra)
        return cast("OpenAI", sdk.OpenAI(
            base_url=self._resolve_base_url(),   # None -> the SDK's own default
            api_key=self._resolve_key(),
        ))

    def _invoke_image(self, prompt: str) -> MediaResult:
        response = self._client.images.generate(model=self._model, prompt=prompt, n=1)

        # `data` can be absent entirely when a request is filtered.
        entries = getattr(response, "data", None) or []
        if not entries:
            raise ModelClientError(
                f"{self.SPEC.name}/{self._model} returned no image data."
            )
        datum = entries[0]

        encoded = get_field(datum, "b64_json")
        if encoded is None:
            # Some image models return a URL instead of bytes. Fetching it would mean
            # this package making a network request to a host that is not the configured
            # provider endpoint, which SECURITY.md states it never does -- so this is a
            # clear error rather than a silent download.
            url = get_field(datum, "url")
            raise ModelClientError(
                f"{self.SPEC.name}/{self._model} returned an image URL rather than bytes"
                f"{f' ({url})' if url else ''}. This client does not fetch URLs; request "
                "base64 output from a model that supports it."
            )

        data = base64.b64decode(encoded)
        usage = getattr(response, "usage", None)
        return MediaResult(
            data=data,
            # The image endpoint's output format is a request option and models differ on
            # the default, so sniff rather than assume PNG.
            mime_type=sniff_image_mime(data, default="image/png"),
            prompt_tokens=get_field(usage, "input_tokens") if usage is not None else None,
            completion_tokens=get_field(usage, "output_tokens") if usage is not None else None,
            total_tokens=get_field(usage, "total_tokens") if usage is not None else None,
        )

    def _invoke_speech(self, text: str) -> MediaResult:
        response = self._client.audio.speech.create(
            model=self._model, voice=self.voice, input=text,
        )
        # The SDK wraps the binary body: `.content` on current versions, `.read()` on
        # others. Read whichever is there rather than pinning to one SDK generation.
        audio = getattr(response, "content", None)
        if audio is None and hasattr(response, "read"):
            audio = response.read()
        if not audio:
            raise ModelClientError(
                f"{self.SPEC.name}/{self._model} returned no audio data."
            )
        audio = bytes(audio)
        # Container varies with the requested format, so sniff it (see sniff_audio_mime).
        return MediaResult(data=audio, mime_type=sniff_audio_mime(audio))
