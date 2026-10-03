"""The native Gemini provider (`gemini-native`), text.

Almost everything here runs with no SDK installed: the fake overrides `_build_client`, and
responses are SimpleNamespaces shaped like google-genai's. Two classes at the bottom DO
import the SDK, and skip without it. They exist because a fake can only prove the code
agrees with the fake. Building the request through the SDK's own types, and parsing a
response the SDK itself constructed, is what catches a drifted SDK in the all-extras job
-- the gap the HuggingFace 2.0.0 bump exposed (DESIGN.md §14).
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from corbelity.model_client import (
    INCOMPLETE_FINISH_REASONS,
    ImageInput,
    JsonlTraceLogger,
    MissingCredentialsError,
    ModelConfig,
    available_services,
    get_spec,
    known_services,
    resolve_service,
    supported_modalities,
)
from corbelity.model_client.providers import gemini_native
from corbelity.model_client.providers.gemini_native import GeminiNativeClient

PNG = b"\x89PNG\r\n\x1a\n" + b"body"


# Deliberately NOT StrEnum, which ruff suggests: StrEnum's str() returns the value, so it
# would hide the very bug this models. google-genai's enums are str + Enum mixins.
class FinishReason(str, Enum):  # noqa: UP042
    """Mirrors the SDK enum's shape: a str-mixin Enum. On Python 3.12, str() of a member
    is "FinishReason.STOP", which is exactly the trap the provider has to avoid."""

    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"


def part(text: str, *, thought: bool | None = None) -> SimpleNamespace:
    return SimpleNamespace(text=text, thought=thought)


def response(
    *parts: SimpleNamespace,
    finish: Any = FinishReason.STOP,
    usage: Any = None,
    feedback: Any = None,
    candidates: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=list(parts)), finish_reason=finish)
        ] if candidates else None,
        usage_metadata=usage,
        prompt_feedback=feedback,
    )


def usage(**fields: Any) -> SimpleNamespace:
    base = dict(prompt_token_count=10, candidates_token_count=5, total_token_count=15)
    base.update(fields)
    return SimpleNamespace(**base)


class _Models:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.next: Any = response(part("hi"))

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.next


class FakeGeminiNative(GeminiNativeClient):
    """Overrides only _build_client: no SDK is imported."""

    def _build_client(self) -> Any:
        self.models = _Models()
        return SimpleNamespace(models=self.models)


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_NATIVE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")


def build(**kwargs: Any) -> FakeGeminiNative:
    kwargs.setdefault("model", "unlisted-gemini")
    return FakeGeminiNative(**kwargs)


def last_call(client: FakeGeminiNative) -> dict[str, Any]:
    assert client.models.calls, "no call was made"
    return client.models.calls[-1]


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
class TestRegistration:
    @pytest.mark.parametrize("written", ["gemini-native", "google-genai", "Gemini-GenAI"])
    def test_names_and_aliases_resolve(self, written: str) -> None:
        assert resolve_service(written) == "gemini-native"

    def test_registered_and_text_only(self) -> None:
        assert "gemini-native" in known_services()
        assert supported_modalities("gemini-native") == frozenset({"text"})

    def test_shares_the_shims_credential_names_and_order(self) -> None:
        assert get_spec("gemini-native").key_env == get_spec("gemini").key_env

    def test_has_its_own_extra_and_no_hardcoded_endpoint(self) -> None:
        spec = get_spec("gemini-native")
        assert spec.extra == "gemini-native"
        assert spec.default_base_url is None
        assert spec.base_url_env == ("GEMINI_NATIVE_BASE_URL",)

    def test_the_shim_is_unchanged(self) -> None:
        # Option A: a new service beside the shim, never a change to it.
        spec = get_spec("gemini")
        assert spec.client_path.endswith("gemini:GeminiClient")
        assert spec.default_base_url.endswith("/v1beta/openai/")
        assert supported_modalities("gemini") == frozenset({"text"})

    def test_one_key_makes_both_available(self) -> None:
        assert available_services(env={"GEMINI_API_KEY": "k"}) == ("gemini", "gemini-native")


# --------------------------------------------------------------------------- #
# Building the SDK client
# --------------------------------------------------------------------------- #
class _FakeSdk:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def Client(self, **kwargs: Any) -> Any:  # noqa: N802 - mirrors the SDK's name
        self.kwargs = kwargs
        return SimpleNamespace(models=_Models())


class TestBuildClient:
    @pytest.fixture
    def sdk(self, monkeypatch: pytest.MonkeyPatch) -> _FakeSdk:
        fake = _FakeSdk()
        monkeypatch.setattr(gemini_native, "load_sdk", lambda module, extra: fake)
        return fake

    def test_passes_the_key_explicitly_and_pins_the_backend(self, sdk: _FakeSdk) -> None:
        GeminiNativeClient(model="m")
        assert sdk.kwargs == {"api_key": "g-key", "vertexai": False, "http_options": None}

    def test_spec_order_wins_over_the_sdks_own_preference(
        self, sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Left to itself, google-genai prefers GOOGLE_API_KEY. The spec says GEMINI first.
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        GeminiNativeClient(model="m")
        assert sdk.kwargs is not None and sdk.kwargs["api_key"] == "g-key"

    def test_falls_back_to_google_api_key(
        self, sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY")
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        GeminiNativeClient(model="m")
        assert sdk.kwargs is not None and sdk.kwargs["api_key"] == "google-key"

    def test_base_url_override_goes_through_http_options(
        self, sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_NATIVE_BASE_URL", "https://proxy.invalid")
        GeminiNativeClient(model="m")
        assert sdk.kwargs is not None
        assert sdk.kwargs["http_options"] == {"base_url": "https://proxy.invalid"}

    def test_missing_key_raises_before_the_sdk_is_imported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY")

        def must_not_import(module: str, extra: str) -> Any:
            raise AssertionError("SDK imported before the credential was checked")

        monkeypatch.setattr(gemini_native, "load_sdk", must_not_import)
        with pytest.raises(MissingCredentialsError):
            GeminiNativeClient(model="m")


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #
class TestRequest:
    def test_system_prompt_is_configuration_not_a_turn(self) -> None:
        client = build()
        client.complete("Be terse.", "Hello")
        call = last_call(client)
        assert call["config"]["system_instruction"] == "Be terse."
        assert call["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_blank_system_prompt_is_omitted(self) -> None:
        client = build()
        client.complete("   ", "Hello")
        assert "system_instruction" not in last_call(client)["config"]

    def test_history_maps_assistant_to_model(self) -> None:
        client = build()
        client.complete("s", "and now?", history=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ])
        roles = [turn["role"] for turn in last_call(client)["contents"]]
        assert roles == ["user", "model", "user"]

    def test_images_come_before_the_text(self) -> None:
        client = build()
        client.complete("s", "What is this?", images=[ImageInput.from_bytes(PNG)])
        parts = last_call(client)["contents"][-1]["parts"]
        assert parts[0] == {"inline_data": {"mime_type": "image/png", "data": PNG}}
        assert parts[-1] == {"text": "What is this?"}

    def test_cap_is_always_max_output_tokens(self, tmp_path: Path) -> None:
        # max_tokens_param is the chat-completions dialect's spelling problem, not this one.
        path = tmp_path / "models.json"
        path.write_text(json.dumps([{
            "id": "renamed", "service": "gemini-native",
            "max_tokens_param": "max_completion_tokens",
        }]), encoding="utf-8")
        client = build(model="renamed", max_tokens=321, config=ModelConfig(catalog_path=path))
        client.complete("s", "u")
        config = last_call(client)["config"]
        assert config["max_output_tokens"] == 321
        assert "max_completion_tokens" not in config

    def test_sampling_sent_by_default(self) -> None:
        client = build(temperature=0.3, top_p=0.8)
        client.complete("s", "u")
        config = last_call(client)["config"]
        assert (config["temperature"], config["top_p"]) == (0.3, 0.8)

    def test_top_p_omitted_when_unset(self) -> None:
        client = build(config=ModelConfig(default_top_p=None))
        client.complete("s", "u")
        assert "top_p" not in last_call(client)["config"]

    def test_catalog_can_suppress_sampling(self, tmp_path: Path) -> None:
        path = tmp_path / "models.json"
        path.write_text(json.dumps([{
            "id": "no-sampling", "service": "gemini-native", "supports_sampling": False,
        }]), encoding="utf-8")
        client = build(model="no-sampling", temperature=0.5,
                       config=ModelConfig(catalog_path=path))
        client.complete("s", "u")
        config = last_call(client)["config"]
        assert "temperature" not in config and "top_p" not in config


# --------------------------------------------------------------------------- #
# The response
# --------------------------------------------------------------------------- #
def complete_with(resp: Any, **kwargs: Any) -> tuple[str, FakeGeminiNative]:
    client = build(**kwargs)
    client.models.next = resp
    return client.complete("s", "u"), client


class TestAnswerText:
    def test_thought_parts_never_reach_the_text(self) -> None:
        text, _ = complete_with(response(
            part("Let me reason about this...", thought=True),
            part("The answer is 4."),
        ))
        assert text == "The answer is 4."

    def test_text_parts_are_concatenated_in_order(self) -> None:
        text, _ = complete_with(response(part("Hello, "), part("world.")))
        assert text == "Hello, world."

    def test_non_text_parts_are_skipped(self) -> None:
        text, _ = complete_with(response(SimpleNamespace(text=None, thought=None), part("ok")))
        assert text == "ok"


class TestFinishReason:
    def test_reports_the_enum_value_not_its_str(self) -> None:
        # The trap: str(FinishReason.STOP) == "FinishReason.STOP".
        assert str(FinishReason.STOP) != "STOP"
        _, client = complete_with(response(part("ok"), finish=FinishReason.STOP))
        assert client.finish_reason == "STOP"

    def test_a_plain_string_passes_through(self) -> None:
        _, client = complete_with(response(part("ok"), finish="STOP"))
        assert client.finish_reason == "STOP"

    def test_truncation_is_loud(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            complete_with(response(part("partial"), finish=FinishReason.MAX_TOKENS))
        assert "did not finish cleanly" in caplog.text

    def test_safety_stop_is_loud(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            complete_with(response(part("some"), finish=FinishReason.SAFETY))
        assert "did not finish cleanly" in caplog.text

    @pytest.mark.parametrize("reason", [
        "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
        "IMAGE_SAFETY", "LANGUAGE", "OTHER", "MAX_TOKENS", "prompt_blocked",
    ])
    def test_every_unclean_gemini_reason_is_recognised(self, reason: str) -> None:
        assert reason.lower() in INCOMPLETE_FINISH_REASONS

    def test_stop_is_clean(self) -> None:
        assert "stop" not in INCOMPLETE_FINISH_REASONS


class TestBlockedPrompt:
    def test_reported_as_prompt_blocked_with_the_reason_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        feedback = SimpleNamespace(block_reason=FinishReason.SAFETY,
                                   block_reason_message="unsafe request")
        with caplog.at_level(logging.WARNING):
            text, client = complete_with(response(candidates=False, feedback=feedback))
        assert text == ""
        assert client.finish_reason == "prompt_blocked"
        assert "block_reason=SAFETY: unsafe request" in caplog.text
        assert "EMPTY response (finish=prompt_blocked)" in caplog.text

    def test_no_candidate_and_no_reason_is_just_empty(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            text, client = complete_with(response(candidates=False))
        assert text == ""
        assert client.finish_reason is None
        assert "EMPTY response" in caplog.text


class TestUsage:
    def test_thinking_is_added_back_into_completion_tokens(self) -> None:
        _, client = complete_with(response(
            part("ok"), usage=usage(candidates_token_count=5, thoughts_token_count=40,
                                    total_token_count=55),
        ))
        result = client.last_result
        assert result is not None
        assert result.completion_tokens == 45          # billed output: answer + thinking
        assert result.output_text_tokens == 5
        assert result.output_reasoning_tokens == 40
        assert result.total_tokens == 55               # as reported, never recomputed
        assert result.prompt_tokens == 10

    def test_no_thinking_reported_means_none_not_zero(self) -> None:
        _, client = complete_with(response(part("ok"), usage=usage()))
        result = client.last_result
        assert result is not None
        assert result.completion_tokens == 5
        assert result.output_reasoning_tokens is None

    def test_cap_spent_entirely_on_thinking(self, caplog: pytest.LogCaptureFixture) -> None:
        # MAX_TOKENS with no visible text: the documented failure for a low cap.
        thinking_only = usage(candidates_token_count=None, thoughts_token_count=64)
        with caplog.at_level(logging.WARNING):
            text, client = complete_with(response(
                finish=FinishReason.MAX_TOKENS, usage=thinking_only,
            ))
        assert text == ""
        assert client.completion_tokens == 64
        assert "EMPTY response (finish=MAX_TOKENS)" in caplog.text

    def test_input_split_by_modality_and_cached(self) -> None:
        details = [
            SimpleNamespace(modality=SimpleNamespace(value="TEXT"), token_count=12),
            SimpleNamespace(modality="IMAGE", token_count=258),
        ]
        _, client = complete_with(response(part("ok"), usage=usage(
            prompt_token_count=270, prompt_tokens_details=details,
            cached_content_token_count=8,
        )))
        result = client.last_result
        assert result is not None
        assert (result.input_text_tokens, result.input_image_tokens) == (12, 258)
        assert result.input_cached_tokens == 8

    def test_missing_usage_leaves_everything_none(self) -> None:
        _, client = complete_with(response(part("ok"), usage=None))
        result = client.last_result
        assert result is not None
        assert result.prompt_tokens is None and result.completion_tokens is None

    def test_reasoning_tokens_reach_the_trace(self, tmp_path: Path) -> None:
        tracer = JsonlTraceLogger(tmp_path / "trace.jsonl", run_id="t")
        client = build(trace=tracer)
        client.models.next = response(part("ok"), usage=usage(thoughts_token_count=30))
        client.complete("s", "u")
        record = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
        assert record["usage"]["output_reasoning_tokens"] == 30
        assert record["usage"]["completion_tokens"] == 35
        assert record["finish_reason"] == "STOP"


# --------------------------------------------------------------------------- #
# Against the real SDK types (skipped without the gemini-native extra)
# --------------------------------------------------------------------------- #
class TestAgainstTheSdk:
    @pytest.fixture(autouse=True)
    def genai_types(self) -> Any:
        return pytest.importorskip("google.genai.types")

    def test_the_request_dicts_validate_as_sdk_types(self, genai_types: Any) -> None:
        client = build(temperature=0.2)
        client.complete("Be terse.", "What is this?",
                        history=[{"role": "user", "content": "hi"},
                                 {"role": "assistant", "content": "hello"}],
                        images=[ImageInput.from_bytes(PNG)])
        call = last_call(client)
        config = genai_types.GenerateContentConfig.model_validate(call["config"])
        assert config.max_output_tokens == client._max_tokens
        contents = [genai_types.Content.model_validate(c) for c in call["contents"]]
        assert [c.role for c in contents] == ["user", "model", "user"]
        assert contents[-1].parts[0].inline_data.data == PNG

    def test_parses_a_response_the_sdk_built(self, genai_types: Any) -> None:
        sdk_response = genai_types.GenerateContentResponse.model_validate({
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "thinking...", "thought": True},
                    {"text": "Four."},
                ]},
                "finish_reason": "STOP",
            }],
            "usage_metadata": {
                "prompt_token_count": 7, "candidates_token_count": 2,
                "thoughts_token_count": 20, "total_token_count": 29,
                "prompt_tokens_details": [{"modality": "TEXT", "token_count": 7}],
            },
        })
        text, client = complete_with(sdk_response)
        assert text == "Four."
        assert client.finish_reason == "STOP"
        result = client.last_result
        assert result is not None
        assert (result.completion_tokens, result.output_reasoning_tokens) == (22, 20)
        assert result.input_text_tokens == 7

    def test_parses_a_blocked_prompt_the_sdk_built(self, genai_types: Any) -> None:
        sdk_response = genai_types.GenerateContentResponse.model_validate({
            "prompt_feedback": {"block_reason": "SAFETY"},
        })
        text, client = complete_with(sdk_response)
        assert (text, client.finish_reason) == ("", "prompt_blocked")
