"""SA-361: background, output_format and output_compression.

Two threads run through this file.

The first is cross-parameter refusal. Transparency needs a format with an alpha channel and
a compression quality needs a lossy one, so two settings that are each individually valid
can be invalid together. Both are judged against the format actually IN EFFECT -- the one
named, or the service's declared default when none was -- which is what makes
`output_compression=80` with no format a decidable question rather than a guess.

The second is MIME resolution. The bytes in hand always win. What the request contributes
is the sniffer's FALLBACK: an unrecognised container used to be labelled PNG on no evidence
whatsoever, and now falls back to what was actually asked for.
"""
from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from corbelity.model_client import (
    ImageInput,
    ImageOptions,
    JsonlTraceLogger,
    UnsupportedBackgroundError,
    UnsupportedFormatError,
    get_spec,
    supported_image_backgrounds,
    supported_output_formats,
)
from corbelity.model_client.providers.openai import OpenAIClient

PNG = b"\x89PNG\r\n\x1a\n" + b"body"
WEBP = b"RIFF" + b"\x20\x00\x00\x00" + b"WEBP" + b"body"
JPEG = b"\xff\xd8\xff\xe0" + b"body"
GIBBERISH = b"\x00\x01\x02\x03 not a container this sniffer knows"
REF = ImageInput(data=PNG, mime_type="image/png", name="ref.png")


def _response(payload: bytes = PNG) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(payload).decode("ascii"))],
        size="2048x1152",
    )


class FakeOpenAIClient(OpenAIClient):
    """Records which endpoint was called and with exactly what."""

    payload: bytes = PNG

    def _build_client(self) -> Any:
        self.calls: list[tuple[str, dict[str, Any]]] = []

        def generate(**kwargs: Any) -> SimpleNamespace:
            self.calls.append(("generate", kwargs))
            return _response(type(self).payload)

        def edit(**kwargs: Any) -> SimpleNamespace:
            self.calls.append(("edit", kwargs))
            return _response(type(self).payload)

        return SimpleNamespace(images=SimpleNamespace(generate=generate, edit=edit))


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    FakeOpenAIClient.payload = PNG


def build() -> FakeOpenAIClient:
    return FakeOpenAIClient(model="gpt-image-2.5-flare")


# --- the settings reach the provider, on both endpoints ---------------------------------

def test_all_three_reach_a_plain_generation() -> None:
    client = build()
    client.generate_image("a corbel bracket", background="transparent",
                          output_format="webp", output_compression=80)

    endpoint, kwargs = client.calls[0]
    assert endpoint == "generate"
    assert kwargs["background"] == "transparent"
    assert kwargs["output_format"] == "webp"
    assert kwargs["output_compression"] == 80


def test_all_three_reach_the_edit_endpoint_too() -> None:
    """Unlike input_fidelity, these exist on both endpoints, so they travel in the shared
    settings rather than an edit-only payload."""
    client = build()
    client.generate_image("a corbel bracket", images=[REF], background="opaque",
                          output_format="jpeg", output_compression=50)

    endpoint, kwargs = client.calls[0]
    assert endpoint == "edit"
    assert kwargs["background"] == "opaque"
    assert kwargs["output_format"] == "jpeg"
    assert kwargs["output_compression"] == 50


def test_omitting_them_sends_no_keys() -> None:
    client = build()
    client.generate_image("a corbel bracket")

    _endpoint, kwargs = client.calls[0]
    for key in ("background", "output_format", "output_compression"):
        assert key not in kwargs


# --- each value on its own ---------------------------------------------------------------

def test_an_unaccepted_background_raises_before_the_provider() -> None:
    client = build()
    with pytest.raises(UnsupportedBackgroundError) as caught:
        client.generate_image("a corbel bracket", background="chroma-green")

    assert caught.value.requested == "chroma-green"
    assert caught.value.supported == ("transparent", "opaque", "auto")
    assert client.calls == []


def test_an_unproduced_format_raises_before_the_provider() -> None:
    client = build()
    with pytest.raises(UnsupportedFormatError) as caught:
        client.generate_image("a corbel bracket", output_format="tiff")

    assert caught.value.requested == "tiff"
    assert caught.value.supported == ("png", "jpeg", "webp")
    assert client.calls == []


@pytest.mark.parametrize("bad", [-1, 101, 1000])
def test_compression_outside_the_percentage_range_raises(bad: int) -> None:
    client = build()
    with pytest.raises(ValueError, match="between 0 and 100"):
        client.generate_image("x", output_format="webp", output_compression=bad)
    assert client.calls == []


@pytest.mark.parametrize("bad", ["80", 80.0, None.__class__])
def test_compression_that_is_not_an_int_raises(bad: Any) -> None:
    client = build()
    with pytest.raises(ValueError, match="percentage from 0 to 100"):
        client.generate_image("x", output_format="webp", output_compression=bad)


def test_a_bool_is_not_an_acceptable_compression() -> None:
    """bool IS an int in Python, so output_compression=True would otherwise sail through
    as 1 and silently mean near-total compression. The narrowest possible trap, and the
    one most likely to be hit by a caller wiring a checkbox to this parameter."""
    client = build()
    with pytest.raises(ValueError, match="percentage from 0 to 100"):
        client.generate_image("x", output_format="webp", output_compression=True)
    assert client.calls == []


def test_a_service_with_no_compression_control_refuses_it() -> None:
    class NoCompressionClient(FakeOpenAIClient):
        SPEC = replace(get_spec("openai"), image_compression=False)

    client = NoCompressionClient(model="gpt-image-2.5-flare")
    with pytest.raises(ValueError, match="no output compression control"):
        client.generate_image("x", output_format="webp", output_compression=80)
    assert client.calls == []


# --- the combinations that contradict each other -----------------------------------------

def test_transparency_with_a_format_that_has_no_alpha_raises() -> None:
    """The ticket's own example. The alternative is an opaque image the caller believes is
    transparent, which is discovered only by compositing it over something."""
    client = build()
    with pytest.raises(ValueError, match="alpha channel"):
        client.generate_image("a prop", background="transparent", output_format="jpeg")
    assert client.calls == []


@pytest.mark.parametrize("fmt", ["png", "webp"])
def test_transparency_is_fine_with_an_alpha_format(fmt: str) -> None:
    client = build()
    client.generate_image("a prop", background="transparent", output_format=fmt)
    assert client.calls[0][1]["background"] == "transparent"


def test_transparency_with_no_format_named_uses_the_service_default() -> None:
    """Asking for transparency and naming no format is a sensible request: whether it works
    depends on what the endpoint produces by default, which the spec records as png."""
    client = build()
    client.generate_image("a prop", background="transparent")
    assert client.calls[0][1]["background"] == "transparent"


def test_compression_on_an_explicitly_lossless_format_raises() -> None:
    client = build()
    with pytest.raises(ValueError, match="lossless"):
        client.generate_image("x", output_format="png", output_compression=80)
    assert client.calls == []


def test_compression_with_no_format_named_is_judged_against_the_default() -> None:
    """The reason image_default_format is worth recording. Without it this case would be
    undecidable and would have to be allowed -- passing a setting that the endpoint's own
    default format makes inert, which is exactly the silent no-op being refused."""
    client = build()
    with pytest.raises(ValueError, match="lossless"):
        client.generate_image("x", output_compression=80)
    assert client.calls == []


@pytest.mark.parametrize("fmt", ["jpeg", "webp"])
def test_compression_is_fine_with_a_lossy_format(fmt: str) -> None:
    client = build()
    client.generate_image("x", output_format=fmt, output_compression=80)
    assert client.calls[0][1]["output_compression"] == 80


def test_a_service_with_no_declared_default_skips_the_cross_checks() -> None:
    """None means unknown, and an unknown is not an excuse to guess. The provider gets to
    answer for itself rather than having this client invent a rule on its behalf."""
    class UnknownDefaultClient(FakeOpenAIClient):
        SPEC = replace(get_spec("openai"), image_default_format=None)

    client = UnknownDefaultClient(model="gpt-image-2.5-flare")
    client.generate_image("x", output_compression=80)
    assert client.calls[0][1]["output_compression"] == 80


# --- MIME resolution: the bytes win, the request is the fallback -------------------------

def test_recognised_bytes_beat_the_requested_format() -> None:
    """The SA-357 principle, kept. Ask for webp, receive PNG bytes, and the result says
    PNG -- because that is what a caller will render. The disagreement stays findable:
    output_format on the result carries what the API reported."""
    FakeOpenAIClient.payload = PNG
    client = build()
    result = client.generate_image("x", output_format="webp")
    assert result.mime_type == "image/png"


def test_real_webp_bytes_are_recognised() -> None:
    FakeOpenAIClient.payload = WEBP
    client = build()
    result = client.generate_image("x", output_format="webp")
    assert result.mime_type == "image/webp"


def test_unrecognised_bytes_fall_back_to_the_requested_format() -> None:
    """What SA-361 actually fixes. This case used to be labelled image/png on no evidence
    at all; the format that was asked for is a far better guess than a hardcoded one."""
    FakeOpenAIClient.payload = GIBBERISH
    client = build()
    result = client.generate_image("x", output_format="webp")
    assert result.mime_type == "image/webp"


def test_unrecognised_bytes_with_no_request_fall_back_to_the_service_default() -> None:
    FakeOpenAIClient.payload = GIBBERISH
    client = build()
    result = client.generate_image("x")
    assert result.mime_type == "image/png"


def test_jpeg_bytes_are_recognised_over_a_png_default() -> None:
    FakeOpenAIClient.payload = JPEG
    client = build()
    result = client.generate_image("x")
    assert result.mime_type == "image/jpeg"


# --- ask ahead, and the record ------------------------------------------------------------

def test_capabilities_answer_without_a_client() -> None:
    assert supported_image_backgrounds("openai") == ("transparent", "opaque", "auto")
    assert supported_output_formats("openai") == ("png", "jpeg", "webp")
    assert supported_image_backgrounds("anthropic") == ()
    assert supported_output_formats("anthropic") == ()


def test_image_options_is_truthy_on_any_encoding_setting_alone() -> None:
    assert bool(ImageOptions(background="transparent")) is True
    assert bool(ImageOptions(output_format="webp")) is True
    assert bool(ImageOptions(output_compression=80)) is True
    assert bool(ImageOptions()) is False


def test_the_trace_records_the_encoding_that_was_asked_for(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    client = FakeOpenAIClient(model="gpt-image-2.5-flare", trace=JsonlTraceLogger(path))
    client.generate_image("x", background="transparent", output_format="webp",
                          output_compression=60)

    asked = json.loads(path.read_text(encoding="utf-8").splitlines()[0])["request"]
    assert asked["requested"]["background"] == "transparent"
    assert asked["requested"]["output_format"] == "webp"
    assert asked["requested"]["output_compression"] == 60


def test_the_trace_omits_encoding_that_was_not_asked_for(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    client = FakeOpenAIClient(model="gpt-image-2.5-flare", trace=JsonlTraceLogger(path))
    client.generate_image("x", quality="medium")

    asked = json.loads(path.read_text(encoding="utf-8").splitlines()[0])["request"]
    for key in ("background", "output_format", "output_compression"):
        assert key not in asked["requested"]
