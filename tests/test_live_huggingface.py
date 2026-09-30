"""Live HuggingFace calls. Excluded by default; run with `pytest -m live`.

These exist because the rest of the suite cannot validate this provider at all. Every
other test overrides `_build_client`, so `huggingface_hub` is never imported and none of
`InferenceClient`, `chat_completion`, `text_to_image` or `text_to_speech` is ever called.
A fully green suite is compatible with the SDK being completely broken.

What the automated checks DO cover, for contrast:

  * mypy (in the all-extras CI job) verifies those three methods still exist on the typed
    `InferenceClient`, so a removal or rename fails the build.
  * Nothing checks the constructor: `sdk.InferenceClient(token=...)` goes through a
    ModuleType, so `sdk.InferenceClient` is `Any` and the argument is unchecked.

So this file covers the gap: the constructor, the three response shapes, and the two
post-processing steps that only exist here -- the PIL re-encode to PNG on the image path,
and MIME sniffing on the audio path.

HuggingFace is the only provider using three DISTINCT SDK surfaces, which is why there are
three tests rather than one. A passing text call says nothing about text_to_speech.

Run after any `huggingface-hub` major bump:

    HF_TOKEN=... uv run pytest -m live

Model ids are environment-overridable because serving availability moves independently of
this package, and the catalog deliberately carries only the image entry -- the text and
sound entries were dropped for being unreliable. Override when a default stops being
served:

    HF_LIVE_TEXT_MODEL, HF_LIVE_IMAGE_MODEL, HF_LIVE_SOUND_MODEL
"""
from __future__ import annotations

import os
from collections.abc import Callable

import pytest

from corbelity.model_client import (
    MediaResult,
    get_spec,
    make_model_client,
    sniff_audio_mime,
    sniff_image_mime,
)

pytestmark = pytest.mark.live

TEXT_MODEL = os.getenv("HF_LIVE_TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
IMAGE_MODEL = os.getenv("HF_LIVE_IMAGE_MODEL", "black-forest-labs/FLUX.1-dev")
SOUND_MODEL = os.getenv("HF_LIVE_SOUND_MODEL", "hexgrad/Kokoro-82M")

# The message HuggingFaceClient._call raises when the hub serves nothing for a model.
_NO_PROVIDER = "No HuggingFace inference provider currently serves"

@pytest.fixture(autouse=True)
def _requirements() -> None:
    """Skip rather than fail when this cannot run: a missing extra or a missing token is
    an environment state, not a defect in the code under test."""
    pytest.importorskip(
        "huggingface_hub",
        reason='huggingface extra not installed: pip install "corbelity-model-client[huggingface]"',
    )
    names = get_spec("huggingface").key_env          # read from the spec, not hardcoded
    if not any(os.getenv(name, "").strip() for name in names):
        pytest.skip(f"no HuggingFace credential: set one of {', '.join(names)}")


def unserved_is_a_skip[R](model: str, call: Callable[[], R]) -> R:
    """Run `call`, turning "nothing currently serves this model" into a skip.

    That condition is the hub's routing on the day, not a regression here, and letting it
    fail the suite is how a live test earns a reputation for flakiness and stops being
    trusted. Every other error propagates -- including a TypeError or AttributeError from
    an SDK change, which is the entire point of these tests."""
    try:
        return call()
    except ValueError as err:
        if _NO_PROVIDER in str(err):
            pytest.skip(f"hub serves nothing for {model!r} right now: {err}")
        raise


def test_text_completion_round_trip() -> None:
    """chat_completion: the constructor, the request, and usage extraction."""
    client = make_model_client(
        "huggingface", model=TEXT_MODEL, max_tokens=32, temperature=0.0
    )
    text = unserved_is_a_skip(
        TEXT_MODEL,
        lambda: client.complete("Answer in one word.", "Name a primary colour."),
    )

    assert text.strip(), "chat_completion returned an empty response"
    # Usage is best-effort per provider, so the assertion is on SHAPE not presence: a
    # value that is neither None nor an int means get_field() met a response it did not
    # understand, which is exactly what an SDK change looks like.
    assert client.last_result is not None
    for field in (client.prompt_tokens, client.completion_tokens, client.total_tokens):
        assert field is None or isinstance(field, int)


def test_image_generation_round_trip() -> None:
    """text_to_image, and the PIL re-encode that only this provider does."""
    client = make_model_client("huggingface", model=IMAGE_MODEL)
    result = unserved_is_a_skip(
        IMAGE_MODEL,
        lambda: client.generate_image("a single flat grey circle on a white background"),
    )

    assert isinstance(result, MediaResult)
    assert result.data, "text_to_image returned no bytes"
    # The provider re-encodes the returned PIL.Image so callers never need Pillow. These
    # two assertions are different claims: the first is what we LABELLED it, the second is
    # what the bytes ACTUALLY are. Only the pair proves the round trip worked.
    assert result.mime_type == "image/png"
    assert sniff_image_mime(result.data) == "image/png"


def test_speech_generation_round_trip() -> None:
    """text_to_speech, and the MIME sniffing that only this provider needs."""
    client = make_model_client("huggingface", model=SOUND_MODEL)
    result = unserved_is_a_skip(
        SOUND_MODEL, lambda: client.generate_speech("Testing, one two three.")
    )

    assert isinstance(result, MediaResult)
    assert result.data, "text_to_speech returned no bytes"
    assert result.mime_type.startswith("audio/")
    # Sniffing must have RECOGNISED the container rather than silently taking its default.
    # Passing a sentinel is what separates "this is a known audio format" from "we gave up
    # and called it WAV" -- the container varies by model, so a fallback here means the
    # signature table has fallen behind.
    assert sniff_audio_mime(result.data, default="unrecognised") != "unrecognised"
