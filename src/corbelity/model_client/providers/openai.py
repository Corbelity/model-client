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
import io
from typing import TYPE_CHECKING, Any, ClassVar, cast

from ..client import get_field, load_sdk
from ..errors import ModelClientError
from ..media import (
    ImageInput,
    ImageOptions,
    MediaResult,
    mime_for_format,
    sniff_audio_mime,
    sniff_image_mime,
)
from ..registry import ProviderSpec, get_spec
from .openai_compatible import OpenAICompatibleClient

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from openai import OpenAI

# The edit endpoint is a multipart upload, so each reference arrives as a FILE, not as
# bytes or base64. The SDK reads the filename off the object's `.name`, and the API infers
# the part's content type from its extension -- a file-like object without one is rejected,
# which is an unobvious failure worth keeping in one place.
_MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def _as_upload(image: ImageInput, index: int) -> io.BytesIO:
    upload = io.BytesIO(image.data)
    suffix = _MIME_SUFFIXES.get(image.mime_type, ".png")
    # The caller's own filename when there is one, because it is what makes a failed
    # request legible in a provider dashboard; a positional fallback otherwise.
    name = image.name or f"reference-{index}{suffix}"
    upload.name = name if name.lower().endswith(suffix) else f"{name}{suffix}"
    return upload


# Response fields this package models with a field of its own. Everything else scalar goes
# to MediaResult.extra, so a field the provider adds later is observable without a release.
_MODELLED_RESPONSE_FIELDS = frozenset({
    "data", "usage", "size", "quality", "output_format", "background", "created",
})


def _unmodelled_fields(response: Any) -> dict[str, Any]:
    """Scalar response fields with no field of their own, carried through verbatim.

    Scalars only, and never `data`: an `extra` holding megabytes of base64 would defeat
    the artifact handling that keeps payloads out of the trace. A nested object is skipped
    rather than flattened, because guessing at a shape is how this would start lying."""
    if isinstance(response, dict):
        items = response.items()
    else:
        raw = getattr(response, "model_dump", None)
        if callable(raw):
            try:
                items = raw().items()
            except Exception:
                return {}
        else:
            return {}
    return {
        key: value
        for key, value in items
        if key not in _MODELLED_RESPONSE_FIELDS
        and isinstance(value, str | int | float | bool)
    }


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

    def _invoke_image(self, prompt: str, images: tuple[ImageInput, ...],
                      options: ImageOptions) -> MediaResult:
        # Omitted entirely when not requested, so the provider's own default applies
        # rather than a value invented here. Both endpoints take both settings.
        settings: dict[str, Any] = {}
        if options.size is not None:
            settings["size"] = options.size
        if options.quality is not None:
            settings["quality"] = options.quality
        if options.background is not None:
            settings["background"] = options.background
        if options.output_format is not None:
            settings["output_format"] = options.output_format
        if options.output_compression is not None:
            settings["output_compression"] = options.output_compression

        # Kept OUT of `settings` deliberately: input_fidelity exists on images.edit and
        # not on images.generate, so folding it in with the shared settings would hide an
        # asymmetry that matters. The client guarantees it arrives only with references,
        # and references are exactly what select the edit endpoint below -- but a reader
        # of this method should be able to see that without going to look.
        edit_only: dict[str, Any] = {}
        if options.input_fidelity is not None:
            edit_only["input_fidelity"] = options.input_fidelity

        # Reference images change the ENDPOINT, not just the payload: OpenAI's image
        # models take them through images.edit, while images.generate is text-only. The
        # response shape is identical either way, so only the call differs.
        if images:
            response = self._client.images.edit(
                model=self._model,
                prompt=prompt,
                # A list even for one image: the edit endpoint accepts several for the
                # gpt-image family, and a single-vs-list branch here would be one more
                # thing to get wrong.
                image=[_as_upload(image, index) for index, image in enumerate(images)],
                **settings,
                **edit_only,
            )
        else:
            response = self._client.images.generate(
                model=self._model, prompt=prompt, n=1, **settings
            )
        # The format in effect is the sniffer's FALLBACK, not its answer -- see
        # _parse_image_response. Resolved here because only this method knows what was
        # requested; the spec's default covers the case where nothing was.
        return self._parse_image_response(
            response,
            fallback_mime=mime_for_format(
                options.output_format or self.SPEC.image_default_format
            ),
        )

    def _parse_image_response(self, response: Any,
                              *, fallback_mime: str | None = None) -> MediaResult:
        # Every field is read through get_field rather than getattr, because the shape
        # varies: the SDK returns a pydantic model, a stub a SimpleNamespace, a raw HTTP
        # path a plain dict. Mixing the two accessors is how a parser works against one
        # shape and reports "no image data" against another.
        #
        # `data` can be absent entirely when a request is filtered.
        entries = get_field(response, "data") or []
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
        usage = get_field(response, "usage")
        # The component split of the input, when reported. Read through get_field twice
        # because the details arrive as a nested object on some SDK versions and a plain
        # dict on others; a provider that reports no details leaves all three as None.
        details = get_field(usage, "input_tokens_details") if usage is not None else None
        out_details = (
            get_field(usage, "output_tokens_details") if usage is not None else None
        )
        return MediaResult(
            data=data,
            # What the provider says it PRODUCED. The response states each of these, so
            # none of it is inferred -- which is what makes it trustworthy evidence that a
            # requested size or quality was actually honoured.
            size=get_field(response, "size"),
            quality=get_field(response, "quality"),
            output_format=get_field(response, "output_format"),
            background=get_field(response, "background"),
            created=get_field(response, "created"),
            extra=_unmodelled_fields(response),
            # The MIME type describes the BYTES IN HAND, always. Deliberately not taken
            # from the reported output_format above, nor from what was requested: if those
            # ever disagree with the bytes, the bytes are what a caller will render, and
            # the disagreement stays visible because output_format is on the result too.
            #
            # What the request contributes is the FALLBACK. An unrecognised container used
            # to be labelled PNG on no evidence at all; now it falls back to the format
            # actually asked for, or the service's default, and only then to PNG.
            mime_type=sniff_image_mime(data, default=fallback_mime or "image/png"),
            prompt_tokens=get_field(usage, "input_tokens") if usage is not None else None,
            completion_tokens=get_field(usage, "output_tokens") if usage is not None else None,
            total_tokens=get_field(usage, "total_tokens") if usage is not None else None,
            input_text_tokens=get_field(details, "text_tokens") if details is not None else None,
            input_image_tokens=get_field(details, "image_tokens") if details is not None else None,
            input_cached_tokens=(
                get_field(details, "cached_tokens") if details is not None else None
            ),
            output_text_tokens=(
                get_field(out_details, "text_tokens") if out_details is not None else None
            ),
            output_image_tokens=(
                get_field(out_details, "image_tokens") if out_details is not None else None
            ),
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
