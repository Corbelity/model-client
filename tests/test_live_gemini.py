"""Live calls to both Gemini routes. Excluded by default; run with `pytest -m live`.

The hermetic suite proves the request matches the SDK's types and that the parser reads
responses the SDK builds. It cannot prove the API still answers that request the way the
parser expects. These do, cheaply: three short calls on `gemini-native`, and one on the
`gemini` compatibility shim so the catalog's example of that route is known to work.

    GEMINI_API_KEY=... uv run --extra gemini-native --extra gemini \
        pytest -m live tests/test_live_gemini.py -rs

Run after any `google-genai` or `openai` bump. Each route skips on its own if its extra is
missing. The models are overridable because availability moves on Google's schedule:

    GEMINI_LIVE_MODEL         (default: the catalog's gemini-native text model)
    GEMINI_LIVE_COMPAT_MODEL  (default: the catalog's gemini text model)

Video is opt-in on top of `-m live`, because each Veo clip is billed per second and takes
a minute or more: set GEMINI_LIVE_VIDEO=1. Two 4-second 720p clips on the cheapest model,
one from text (picked up again through a fresh client, as another process would) and one
from a start frame:

    GEMINI_LIVE_VIDEO=1 uv run --extra gemini-native pytest -m live \
        tests/test_live_gemini.py -k video -rs

    GEMINI_LIVE_VIDEO_MODEL   (default: veo-3.1-lite-generate-preview)

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
    MediaResult,
    builtin_catalog,
    get_spec,
    make_model_client,
)

pytestmark = pytest.mark.live


def _default_model(service: str, fallback: str) -> str:
    for entry in builtin_catalog():
        if entry.service == service and entry.modality == "text":
            return entry.id
    return fallback


MODEL = os.getenv("GEMINI_LIVE_MODEL") or _default_model("gemini-native", "gemini-3.8-flash")
COMPAT_MODEL = (
    os.getenv("GEMINI_LIVE_COMPAT_MODEL") or _default_model("gemini", "gemini-3.5-flash-lite")
)


def _require_key(service: str) -> None:
    names = get_spec(service).key_env
    if not any(os.getenv(name, "").strip() for name in names):
        pytest.skip(f"no Gemini credential: set one of {', '.join(names)}")


@pytest.fixture
def native() -> None:
    pytest.importorskip(
        "google.genai",
        reason='gemini-native extra not installed: pip install '
               '"corbelity-model-client[gemini-native]"',
    )
    _require_key("gemini-native")


VIDEO_MODEL = os.getenv("GEMINI_LIVE_VIDEO_MODEL") or "veo-3.1-lite-generate-preview"


@pytest.fixture
def video() -> None:
    if os.getenv("GEMINI_LIVE_VIDEO", "").strip() != "1":
        pytest.skip("video is billed per second: set GEMINI_LIVE_VIDEO=1 to run it")
    pytest.importorskip(
        "google.genai",
        reason='gemini-native extra not installed: pip install '
               '"corbelity-model-client[gemini-native]"',
    )
    _require_key("gemini-native")


@pytest.fixture
def compat() -> None:
    pytest.importorskip(
        "openai",
        reason='gemini extra not installed: pip install "corbelity-model-client[gemini]"',
    )
    _require_key("gemini")


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


def red_png(size: int = 16, height: int | None = None) -> bytes:
    """A solid red PNG, built by hand so the test needs no imaging library. Square unless
    a height is given."""
    height = size if height is None else height
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    row = b"\x00" + b"\xff\x00\x00" * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


@pytest.mark.usefixtures("native")
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


@pytest.mark.usefixtures("native")
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


@pytest.mark.usefixtures("native")
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


@pytest.mark.usefixtures("compat")
def test_compatibility_shim_round_trip() -> None:
    """The catalog's example of the compatibility route really works through it.

    The OpenAI SDK raises its own error types, so a quota refusal is recognised by the
    status code it carries rather than by importing the SDK's exception classes."""
    client = make_model_client("gemini", model=COMPAT_MODEL, max_tokens=1024)
    try:
        text = client.complete("Answer with a single word.", "What is the capital of France?")
    except Exception as err:
        if getattr(err, "status_code", None) == 429:
            pytest.skip(f"Gemini account refused the call (429): {type(err).__name__}")
        raise
    assert "paris" in text.lower()
    # The chat-completions dialect's vocabulary, lower-case: not the native enum.
    assert client.finish_reason == "stop"
    assert isinstance(client.prompt_tokens, int) and client.prompt_tokens > 0


def _assert_is_a_video(result: MediaResult) -> None:
    assert result.data, "empty video"
    assert result.mime_type == "video/mp4"           # sniffed from the bytes
    assert result.data[4:8] == b"ftyp"
    # The provider's handle, kept so the clip can be extended within two days.
    assert result.source_uri


@pytest.mark.usefixtures("video")
def test_video_from_text_resumed_in_a_fresh_client() -> None:
    submitter = make_model_client("gemini-native", model=VIDEO_MODEL)
    job = quota_is_a_skip(lambda: submitter.submit_video(
        "A slow pan across a calm harbour at sunrise, gentle waves.",
        duration_seconds=4, resolution="720p", aspect_ratio="16:9",
    ))
    stored = job.to_ref().to_dict()
    # Another process, knowing only the stored reference.
    poller = make_model_client("gemini-native", model=VIDEO_MODEL)
    resumed = poller.resume_video(stored)
    _assert_is_a_video(resumed.wait(timeout_s=600, poll_interval_s=10))
    assert resumed.status is not None and resumed.status.state == "succeeded"


@pytest.mark.usefixtures("video")
def test_video_from_a_start_frame() -> None:
    client = make_model_client("gemini-native", model=VIDEO_MODEL)
    result = quota_is_a_skip(lambda: client.generate_video(
        "The red square slowly turns blue.",
        # 16:9, matching the output, so nothing has to be cropped or padded.
        first_frame=ImageInput.from_bytes(red_png(1280, 720)),
        duration_seconds=4, resolution="720p", aspect_ratio="16:9",
        timeout_s=600,
    ))
    _assert_is_a_video(result)
