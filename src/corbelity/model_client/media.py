"""Modality constants, normalized results, image input, and the MIME sniffers.

Everything here is provider-neutral on purpose: the results are the shape the base class
logs and traces, and the sniffers exist because providers disagree about content types
in ways that surface as opaque 400s if you guess.
"""
from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass

TEXT = "text"
IMAGE = "image"
SOUND = "sound"

ALL_MODALITIES = frozenset({TEXT, IMAGE, SOUND})


# --------------------------------------------------------------------------- #
# Results - normalized across providers so the base class can log every call
# identically. kw_only so no caller can slide content into a token field.
# --------------------------------------------------------------------------- #
@dataclass(kw_only=True)
class ModelResult:
    """Telemetry shared by every call, extracted best-effort -- any field a provider
    doesn't expose stays None and simply isn't counted."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None


@dataclass(kw_only=True)
class LLMResult(ModelResult):
    """A text completion. `text` is what complete() returns."""

    text: str = ""


@dataclass(kw_only=True)
class MediaResult(ModelResult):
    """Generated image or audio. `data` is the raw payload; `mime_type` is what the
    caller needs to render or store it (providers differ, so it is never assumed)."""

    data: bytes = b""
    mime_type: str = "application/octet-stream"


@dataclass(frozen=True, kw_only=True)
class ImageInput:
    """An image supplied WITH a prompt (vision input) -- the mirror of MediaResult, which
    is an image produced BY a model.

    Frozen because the same object is handed to a provider payload and to a trace record;
    it must be the same bytes at both ends. kw_only for the reason the result dataclasses
    are: positional construction would let a filename land in the mime_type field."""

    data: bytes
    mime_type: str
    name: str | None = None

    def base64(self) -> str:
        """Every provider wants base64; only the wrapper around it differs."""
        return base64.b64encode(self.data).decode("ascii")

    @classmethod
    def from_bytes(cls, data: bytes, *, name: str | None = None,
                   mime_type: str | None = None) -> ImageInput:
        """Build from raw bytes, sniffing the type when the caller does not know it --
        which is the common case for a browser upload or a fetched URL."""
        return cls(data=data, mime_type=mime_type or sniff_image_mime(data), name=name)


# Audio magic bytes -> MIME. TTS providers disagree on output container (Bark returns
# WAV, some routes return FLAC or MP3), and the browser <audio> element behaves better
# with an honest type, so sniff rather than assume.
_AUDIO_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"RIFF", "audio/wav"),
    (b"fLaC", "audio/flac"),
    (b"OggS", "audio/ogg"),
    (b"ID3", "audio/mpeg"),
    (b"\xff\xfb", "audio/mpeg"),
    (b"\xff\xf3", "audio/mpeg"),
)


def sniff_audio_mime(data: bytes, default: str = "audio/wav") -> str:
    for signature, mime in _AUDIO_SIGNATURES:
        if data.startswith(signature):
            return mime
    return default


# The intersection of what the supported providers document. Anthropic is the narrowest
# and accepts exactly these; sending anything else is a provider-side 400.
SUPPORTED_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_image_mime(data: bytes, default: str = "application/octet-stream") -> str:
    """Magic bytes -> MIME, for the same reason sniff_audio_mime exists: a fetched URL's
    Content-Type is frequently absent or 'application/octet-stream', and the provider
    needs an honest type to accept the image.

    Unlike the audio version the default is deliberately NOT an image type. A wrong guess
    would surface as an opaque provider-side 400; an unrecognised type instead falls
    through to validate_images(), which rejects it with a message naming the allowed
    set."""
    for signature, mime in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return mime
    # WEBP is 'RIFF' + a 4-byte size + 'WEBP', so the tag is at offset 8, not the start --
    # and a bare 'RIFF' prefix is just as likely to be a WAV file.
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return default


def validate_images(images: Sequence[ImageInput] | None) -> tuple[ImageInput, ...]:
    """Normalize attachments into an immutable, provider-safe tuple.

    Mirrors validate_history(): None becomes (), and every failure names the offending
    index. Raises ValueError, which web layers already map to a 400.

    Count and size caps are deliberately NOT enforced here. Those are policy and belong at
    the HTTP boundary, where policy already lives -- a script caller sending one 40 MB scan
    is doing something legitimate that a UI limit should not forbid."""
    if not images:
        return ()

    entries = tuple(images)
    for i, image in enumerate(entries):
        if not isinstance(image, ImageInput):
            raise ValueError(f"Image {i} is not an ImageInput: {type(image).__name__}.")
        if not isinstance(image.data, bytes) or not image.data:
            raise ValueError(f"Image {i} has no data.")
        if image.mime_type not in SUPPORTED_IMAGE_MIMES:
            raise ValueError(
                f"Image {i} has unsupported type {image.mime_type!r};"
                f" allowed: {', '.join(sorted(SUPPORTED_IMAGE_MIMES))}."
            )
    return entries
