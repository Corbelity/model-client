"""SA-341: the component split of input tokens.

The flat prompt_tokens figure prices everything at the text rate. These tests defend the
three additions and, more importantly, the two things that must NOT change: prompt_tokens
itself, and the behaviour of a provider that reports no details at all.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from corbelity.model_client import (
    JsonlTraceLogger,
    LLMResult,
    MediaResult,
    ModelResult,
)
from corbelity.model_client.providers.openai import OpenAIClient

PNG = b"\x89PNG\r\n\x1a\n" + b"body"


def image_response(*, details: Any = None, usage: bool = True) -> SimpleNamespace:
    payload = SimpleNamespace(b64_json=base64.b64encode(PNG).decode("ascii"))
    if not usage:
        return SimpleNamespace(data=[payload])
    return SimpleNamespace(
        data=[payload],
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=200, total_tokens=300,
            input_tokens_details=details,
        ),
    )


class FakeOpenAIClient(OpenAIClient):
    """Overrides only _build_client, so no SDK is imported."""

    next_response: Any = None

    def _build_client(self) -> Any:
        return SimpleNamespace(
            images=SimpleNamespace(generate=lambda **_kw: type(self).next_response)
        )


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")


def generate(response: Any, model: str = "gpt-image-2.5-flare") -> MediaResult:
    FakeOpenAIClient.next_response = response
    return FakeOpenAIClient(model=model).generate_image("a corbel bracket")


class TestResultFields:
    def test_default_to_none_everywhere(self) -> None:
        for result in (ModelResult(), LLMResult(), MediaResult()):
            assert result.input_text_tokens is None
            assert result.input_image_tokens is None
            assert result.input_cached_tokens is None

    def test_they_live_on_the_shared_base(self) -> None:
        # On ModelResult rather than MediaResult, so the chat path can report cached
        # input through the same fields.
        assert "input_cached_tokens" in ModelResult.__dataclass_fields__


class TestImagePathExtraction:
    @pytest.mark.parametrize("as_dict", [False, True])
    def test_split_is_read_from_object_or_dict(self, as_dict: bool) -> None:
        # SDK versions differ on whether the details arrive as an object or a mapping.
        raw = {"text_tokens": 12, "image_tokens": 80, "cached_tokens": 8}
        details = raw if as_dict else SimpleNamespace(**raw)
        result = generate(image_response(details=details))
        assert (result.input_text_tokens, result.input_image_tokens,
                result.input_cached_tokens) == (12, 80, 8)

    def test_flat_figures_are_untouched(self) -> None:
        # Callers depend on these; the split is additive, never a replacement.
        result = generate(image_response(
            details={"text_tokens": 12, "image_tokens": 80, "cached_tokens": 8}
        ))
        assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (
            100, 200, 300
        )

    def test_partial_details_leave_the_rest_none(self) -> None:
        result = generate(image_response(details={"text_tokens": 12}))
        assert result.input_text_tokens == 12
        assert result.input_image_tokens is None
        assert result.input_cached_tokens is None

    def test_absent_details_do_not_break_the_call(self) -> None:
        result = generate(image_response(details=None))
        assert result.prompt_tokens == 100
        assert result.input_text_tokens is None

    def test_no_usage_block_at_all(self) -> None:
        result = generate(image_response(usage=False))
        assert result.data == PNG
        assert result.prompt_tokens is None
        assert result.input_image_tokens is None


class TestTraceRecord:
    def test_split_reaches_the_trace(self, tmp_path: Path) -> None:
        # A cost monitor reads usage back from the trace, so a split that reaches the
        # result but not the record is invisible to the thing that needs it.
        trace = JsonlTraceLogger(tmp_path / "traces.jsonl")
        FakeOpenAIClient.next_response = image_response(
            details={"text_tokens": 12, "image_tokens": 80, "cached_tokens": 8}
        )
        FakeOpenAIClient(model="gpt-image-2.5-flare", trace=trace).generate_image("x")

        record = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))
        assert record["usage"]["input_text_tokens"] == 12
        assert record["usage"]["input_image_tokens"] == 80
        assert record["usage"]["input_cached_tokens"] == 8
        assert record["usage"]["prompt_tokens"] == 100

    def test_absent_split_records_none_not_missing_keys(self, tmp_path: Path) -> None:
        trace = JsonlTraceLogger(tmp_path / "traces.jsonl")
        FakeOpenAIClient.next_response = image_response(details=None)
        FakeOpenAIClient(model="gpt-image-2.5-flare", trace=trace).generate_image("x")

        usage = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))["usage"]
        # Present-and-None rather than absent, so a reader can tell "not reported" from
        # "this trace predates the field".
        assert usage["input_image_tokens"] is None
        assert "input_text_tokens" in usage


class TestChatPathCachedTokens:
    def test_cached_tokens_are_read_from_prompt_tokens_details(self) -> None:
        from corbelity.model_client import ProviderSpec, register_provider
        from corbelity.model_client.providers.openai_compatible import (
            OpenAICompatibleClient,
        )

        spec = ProviderSpec(
            name="fake-cached",
            client_path=f"{__name__}:CachedClient",
            key_env=("FAKE_CACHED_KEY",),
            default_base_url="https://x.invalid/v1",
        )
        register_provider(spec, replace=True)

        class CachedClient(OpenAICompatibleClient):
            SPEC = spec

            def _build_client(self) -> Any:
                response = SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="hi"), finish_reason="stop"
                    )],
                    usage=SimpleNamespace(
                        prompt_tokens=50, completion_tokens=5, total_tokens=55,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
                    ),
                )
                return SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(create=lambda **_kw: response)
                    )
                )

        import os

        os.environ["FAKE_CACHED_KEY"] = "k"
        client = CachedClient(model="m")
        client.complete("sys", "usr")
        assert client.last_result is not None
        assert client.last_result.input_cached_tokens == 40
        assert client.last_result.prompt_tokens == 50
