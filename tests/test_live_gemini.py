"""Live calls to the native Gemini provider. Excluded by default; run with `pytest -m live`.

The hermetic suite proves the request matches the SDK's types and that the parser reads
responses the SDK builds. It cannot prove the API still answers that request the way the
parser expects. These do, cheaply: three short text calls.

    GEMINI_API_KEY=... uv run pytest -m live tests/test_live_gemini.py

Run after any `google-genai` bump. The model is overridable because availability moves on
Google's schedule, not this package's:

    GEMINI_LIVE_MODEL (default: the catalog's Gemini text model)

Skip vs fail follows the rule in DESIGN.md §14: a missing extra, a missing key, or a quota
or region refusal is an environment state and skips; anything else -- including a
TypeError or AttributeError from a moved SDK surface -- fails.
"""
from __future__ import annotations

import os
import struct
import zlib
from collections.abc import Callable

import pytest

from corbelity.model_client import (
    ImageInput,
    builtin_catalog,
    get_spec,
    make_model_client,
)

pytestmark = pytest.mark.live


def _default_model() -> str:
    for entry in builtin_catalog():
        if entry.service in ("gemini", "gemini-native") and entry.modality == "text":
            return entry.id
    return "gemini-3.8-flash"


MODEL = os.getenv("GEMINI_LIVE_MODEL") or _default_model()


@pytest.fixture(autouse=True)
def _requirements() -> None:
    pytest.importorskip(
        "google.genai",
        reason='gemini-native extra not installed: pip install '
               '"corbelity-model-client[gemini-native]"',
    )
    names = get_spec("gemini-native").key_env
    if not any(os.getenv(name, "").strip() for name in names):
        pytest.skip(f"no Gemini credential: set one of {', '.join(names)}")


def quota_is_a_skip[R](call: Callable[[], R]) -> R:
    """429 (quota) and 403 FAILED_PRECONDITION (region/billing) are account states, not
    defects. Keyed on the status code the SDK attaches, never on message wording."""
    from google.genai import errors

    try:
        return call()
    except errors.APIError as err:
        if err.code == 429 or (err.code == 403 and err.status == "FAILED_PRECONDITION"):
            pytest.skip(f"Gemini account refused the call ({err.code} {err.status})")
        raise


def red_png(size: int = 16) -> bytes:
    """A solid red PNG, built by hand so the test needs no imaging library."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    row = b"\x00" + b"\xff\x00\x00" * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def test_text_round_trip_and_usage() -> None:
    client = make_model_client("gemini-native", model=MODEL, max_tokens=1024)
    text = quota_is_a_skip(lambda: client.complete(
        "Answer with a single word.", "What is the capital of France?"
    ))
    assert "paris" in text.lower()
    assert client.finish_reason == "STOP"          # the enum's value, not its str()
    result = client.last_result
    assert result is not None
    assert isinstance(result.prompt_tokens, int) and result.prompt_tokens > 0
    assert isinstance(result.output_text_tokens, int)
    # completion includes thinking when the model reports it, so it is never smaller
    # than the answer alone.
    assert result.completion_tokens is not None
    assert result.completion_tokens >= result.output_text_tokens


def test_history_reaches_the_model() -> None:
    client = make_model_client("gemini-native", model=MODEL, max_tokens=1024)
    text = quota_is_a_skip(lambda: client.complete(
        "Answer with a single word.",
        "What was the codeword I gave you?",
        history=[
            {"role": "user", "content": "Remember this codeword: corbel."},
            {"role": "assistant", "content": "Understood."},
        ],
    ))
    assert "corbel" in text.lower()


def test_image_attachment_is_seen() -> None:
    client = make_model_client("gemini-native", model=MODEL, max_tokens=1024)
    text = quota_is_a_skip(lambda: client.complete(
        "Answer with a single lower-case colour name.",
        "What colour is this image?",
        images=[ImageInput.from_bytes(red_png())],
    ))
    assert "red" in text.lower()
    result = client.last_result
    assert result is not None
    # The image was counted as image input, so the modality split is being read.
    assert result.input_image_tokens is None or result.input_image_tokens > 0
