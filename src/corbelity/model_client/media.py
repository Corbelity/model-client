"""Modality constants, normalized results, image input, and the MIME sniffers.

Everything here is provider-neutral on purpose: the results are the shape the base class
logs and traces, and the sniffers exist because providers disagree about content types
in ways that surface as opaque 400s if you guess.
"""
from __future__ import annotations

import base64
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import gcd
from typing import Any

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

    # Component breakdown of prompt_tokens, when the provider reports one. These are a
    # BILLING concern, not a curiosity: the components price at different rates (for
    # gpt-image-2.5-flare, $5/M text against $8/M image and $1.25/M cached text), so a
    # caller with only the flat figure charges everything at the text rate -- overstating
    # a cached call and understating one carrying reference images.
    #
    # They sit on the shared base rather than on MediaResult because the chat path reports
    # cached input too. prompt_tokens is left exactly as the provider reports it; these
    # are additional, never a replacement, because callers already depend on it.
    input_text_tokens: int | None = None
    input_image_tokens: int | None = None
    input_cached_tokens: int | None = None

    # The same split on the other side of the call. Output tokens dominate the cost of a
    # generated image ($30/M), and a caller pricing them as one undifferentiated figure
    # cannot tell whether that figure is exact or an approximation. Named to mirror the
    # input fields above so the two halves read symmetrically.
    output_text_tokens: int | None = None
    output_image_tokens: int | None = None
    # Tokens the model spent thinking before it answered. Billed at the output rate but
    # never seen in the text, so a caller reading only output_text_tokens undercounts the
    # cost of a reasoning model -- sometimes by a multiple. Included in completion_tokens
    # where the provider reports it separately (Gemini), so the flat figure stays the
    # billed figure; this field is the split, additive like the others.
    output_reasoning_tokens: int | None = None


@dataclass(kw_only=True)
class LLMResult(ModelResult):
    """A text completion. `text` is what complete() returns."""

    text: str = ""


@dataclass(kw_only=True)
class MediaResult(ModelResult):
    """Generated image or audio. `data` is the raw payload; `mime_type` is what the
    caller needs to render or store it (providers differ, so it is never assumed).

    The fields below report what the provider says it ACTUALLY PRODUCED -- never an echo
    of what was requested. That distinction is the point of carrying them: a caller that
    asked for 16:9 and silently got 1:1 has no other way to find out except by opening the
    file and looking at it. Something recording evidence about its own output needs to
    record what it got.

    All optional, so a provider that reports none of it yields the result it did before
    these existed.
    """

    data: bytes = b""
    mime_type: str = "application/octet-stream"

    # Dimensions as the provider states them, "WIDTHxHEIGHT" (e.g. "2048x1152").
    size: str | None = None
    quality: str | None = None
    output_format: str | None = None
    background: str | None = None
    # The provider's own timestamp for the generation -- better evidence than the local
    # clock for anything time-ordered, since it does not depend on this machine's.
    created: int | None = None

    # Response fields this package does not model, carried verbatim.
    #
    # The alternative is a package release every time a provider adds a field, which is
    # the staleness problem the model catalog exists to avoid -- and it is how a question
    # like "does this provider return a revised prompt?" ends up answered by reading a
    # schema rather than by looking at a response. Scalars only, and never the payload
    # itself: an `extra` carrying megabytes of base64 would defeat the artifact handling
    # that keeps bytes out of the trace.
    extra: Mapping[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True, kw_only=True)
class ImageOptions:
    """Output settings for one image generation.

    An object rather than more parameters on the provider seam: every image capability
    that lands wants another setting, and a field added here reaches every provider
    without breaking `_invoke_image()` again.

    None means NOT REQUESTED. The key is then omitted from the provider call entirely, so
    the provider's own default applies rather than a value this package invented.
    """

    size: str | None = None
    quality: str | None = None
    # Only meaningful alongside reference images. The client refuses it without them
    # rather than sending a setting that would have no subject to act on.
    input_fidelity: str | None = None
    # Output encoding. `background="transparent"` needs a format with an alpha channel,
    # and a compression quality needs a lossy one; both are checked before the provider is
    # touched, against the format actually in effect rather than only the one named.
    background: str | None = None
    output_format: str | None = None
    output_compression: int | None = None

    def __bool__(self) -> bool:
        """True when anything was actually requested, so callers can skip the whole
        settings branch on a plain generation."""
        return any(value is not None for value in (
            self.size, self.quality, self.input_fidelity,
            self.background, self.output_format, self.output_compression,
        ))


# Deliberately strict: a size is WIDTHxHEIGHT and a ratio is W:H. Anything else is not
# coerced into one, because a guess here becomes a silently wrong image later.
_SIZE_PATTERN = re.compile(r"^(\d+)\s*[x\u00d7]\s*(\d+)$", re.IGNORECASE)
_RATIO_PATTERN = re.compile(r"^(\d+)\s*[:/]\s*(\d+)$")


def parse_size(size: str) -> tuple[int, int] | None:
    """"WIDTHxHEIGHT" -> (width, height), or None when the string is not a resolution.

    None rather than a raise, because "auto" is a legitimate size for several providers:
    a string that is not a resolution is not necessarily an error, and only the caller
    knows whether this one is."""
    match = _SIZE_PATTERN.match(size.strip())
    if match is None:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    return (width, height) if width > 0 and height > 0 else None


def parse_aspect_ratio(ratio: str) -> tuple[int, int] | None:
    """"16:9" -> (16, 9), reduced. None when the string is not a ratio."""
    match = _RATIO_PATTERN.match(ratio.strip())
    if match is None:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    divisor = gcd(width, height)
    return width // divisor, height // divisor


def size_ratio(size: str) -> tuple[int, int] | None:
    """The reduced ratio of a resolution: "2048x1152" -> (16, 9)."""
    parsed = parse_size(size)
    if parsed is None:
        return None
    width, height = parsed
    divisor = gcd(width, height)
    return width // divisor, height // divisor


def sizes_for_ratio(ratio: str, sizes: Sequence[str]) -> tuple[str, ...]:
    """Every size in `sizes` at EXACTLY `ratio`, smallest first by pixel count.

    Exactly: 16:9 does not match 1920x1081. A near-match is precisely the silent
    substitution this whole path exists to avoid. Entries that are not resolutions
    ("auto") simply never match."""
    wanted = parse_aspect_ratio(ratio)
    if wanted is None:
        return ()

    def pixels(size: str) -> int:
        parsed = parse_size(size)
        return parsed[0] * parsed[1] if parsed else 0

    matches = [size for size in sizes if size_ratio(size) == wanted]
    return tuple(sorted(matches, key=pixels))


def aspect_ratios_of(sizes: Sequence[str]) -> tuple[str, ...]:
    """The distinct ratios reachable from a list of sizes, in the order they appear.

    Lets a caller ask what a service can frame before requesting it, rather than
    discovering the answer from an exception."""
    seen: dict[tuple[int, int], None] = {}
    for size in sizes:
        ratio = size_ratio(size)
        if ratio is not None:
            seen.setdefault(ratio, None)
    return tuple(f"{width}:{height}" for width, height in seen)


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


# Facts about the image FORMATS themselves, not about any provider, which is why they live
# here and not in ProviderSpec. JPEG has no alpha channel and never will; PNG is lossless
# and a compression quality means nothing to it. A provider can only choose which of these
# formats it offers -- it cannot change what they are.
FORMAT_MIMES: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}
ALPHA_FORMATS = frozenset({"png", "webp"})
COMPRESSIBLE_FORMATS = frozenset({"jpeg", "webp"})


def mime_for_format(output_format: str | None) -> str | None:
    """The MIME type a named output format produces, or None for an unknown/absent name.

    Used as the SNIFFER'S FALLBACK rather than as the answer: what was requested is the
    best guess available when magic bytes are unrecognised, and a far better one than a
    hardcoded default, but it is still second to the bytes in hand."""
    if output_format is None:
        return None
    return FORMAT_MIMES.get(output_format.lower())


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
