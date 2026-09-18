"""The base class contract, exercised through a fake provider.

No network, no SDKs, no credentials. A ProviderSpec plus a subclass is all it takes to
stand a provider up, which is itself the point: if these tests need anything else, the
seam between the base class and a provider has leaked.
"""
from __future__ import annotations

import logging
from typing import Any

import pytest

from corbelity.model_client import (
    LLMResult,
    MediaResult,
    ModelClient,
    ModelConfig,
    ProviderSpec,
    TraceSink,
    UnsupportedModalityError,
    make_model_client,
    register_provider,
    supported_modalities,
)
from corbelity.model_client.errors import MissingCredentialsError

# `__name__` rather than a hardcoded "tests.test_client": it resolves correctly whether
# pytest imports this module as a top-level module or as part of a package.
FAKE_SPEC = ProviderSpec(
    name="fake",
    client_path=f"{__name__}:FakeClient",
    key_env=("FAKE_API_KEY", "FAKE_LEGACY_KEY"),
    default_base_url="https://fake.invalid/v1",
    modalities=frozenset({"text", "image"}),
    aliases=("fake-service",),
)


class FakeClient(ModelClient[dict[str, Any]]):
    """Records what it was asked for and returns whatever the test told it to."""

    SPEC = FAKE_SPEC

    next_result: LLMResult = LLMResult(text="ok", finish_reason="stop")
    next_error: Exception | None = None

    def _build_client(self) -> dict[str, Any]:
        return {"api_key": self._resolve_key(), "base_url": self._resolve_base_url()}

    def _invoke(self, system: str, user: str, history: Any, images: Any) -> LLMResult:
        self.seen = {"system": system, "user": user, "history": history, "images": images}
        if type(self).next_error is not None:
            raise type(self).next_error
        return type(self).next_result

    def _invoke_image(self, prompt: str) -> MediaResult:
        return MediaResult(data=b"\x89PNG", mime_type="image/png")


class RecordingTrace:
    """A TraceSink that keeps every record instead of writing one."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def llm_call(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


class ExplodingTrace:
    def llm_call(self, **_kwargs: Any) -> None:
        raise RuntimeError("disk full")


@pytest.fixture(autouse=True)
def _register_fake() -> None:
    register_provider(FAKE_SPEC, replace=True)
    FakeClient.next_result = LLMResult(text="ok", finish_reason="stop")
    FakeClient.next_error = None


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "key-123")


def build(**kwargs: Any) -> FakeClient:
    client = make_model_client("fake", **kwargs)
    assert isinstance(client, FakeClient)
    return client


class TestConfiguration:
    def test_config_defaults_apply(self, env: None) -> None:
        config = ModelConfig(
            default_model="fake-1", default_temperature=0.1, default_max_tokens=99
        )
        client = build(config=config)
        assert client.model == "fake-1"
        assert client._temperature == 0.1
        assert client._max_tokens == 99

    def test_per_call_argument_beats_config(self, env: None) -> None:
        config = ModelConfig(default_model="fake-1")
        assert build(config=config, model="fake-2").model == "fake-2"

    def test_config_is_constructible_with_overrides(self) -> None:
        # The pre-package ModelConfig had no ANNOTATED fields, so @dataclass generated an
        # __init__ that accepted nothing and this raised TypeError.
        assert ModelConfig(default_temperature=0.0).default_temperature == 0.0

    def test_with_overrides_derives_a_new_config(self) -> None:
        base = ModelConfig(default_model="a")
        assert base.with_overrides(default_model="b").default_model == "b"
        assert base.default_model == "a"       # frozen: the original is untouched

    def test_from_env_reads_an_injected_mapping(self) -> None:
        config = ModelConfig.from_env(
            {"CORBELITY_MAX_TOKENS": "128", "CORBELITY_TOP_P": "0.9"}
        )
        assert config.default_max_tokens == 128
        assert config.default_top_p == 0.9

    def test_from_env_rejects_a_malformed_number(self) -> None:
        with pytest.raises(ValueError, match="CORBELITY_MAX_TOKENS"):
            ModelConfig.from_env({"CORBELITY_MAX_TOKENS": "lots"})


class TestCredentials:
    def test_explicit_key_beats_environment(self, env: None) -> None:
        assert build(api_key="override")._client["api_key"] == "override"

    def test_env_names_are_tried_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FAKE_API_KEY", raising=False)
        monkeypatch.setenv("FAKE_LEGACY_KEY", "legacy")
        assert build()._client["api_key"] == "legacy"

    def test_missing_key_names_every_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FAKE_API_KEY", raising=False)
        monkeypatch.delenv("FAKE_LEGACY_KEY", raising=False)
        with pytest.raises(MissingCredentialsError, match="FAKE_API_KEY or FAKE_LEGACY_KEY"):
            build()

    def test_missing_key_is_still_a_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Web layers map ValueError to 400; that must survive the move to a named type.
        monkeypatch.delenv("FAKE_API_KEY", raising=False)
        monkeypatch.delenv("FAKE_LEGACY_KEY", raising=False)
        with pytest.raises(ValueError):
            build()

    def test_base_url_falls_back_to_the_spec_default(self, env: None) -> None:
        assert build()._client["base_url"] == "https://fake.invalid/v1"


class TestModalities:
    def test_supported_modalities_needs_no_client(self) -> None:
        assert supported_modalities("fake") == frozenset({"text", "image"})

    def test_alias_folds_onto_the_canonical_name(self, env: None) -> None:
        assert build().service == "fake"
        assert make_model_client("fake-service").service == "fake"

    def test_unsupported_modality_raises_before_any_call(self, env: None) -> None:
        with pytest.raises(UnsupportedModalityError) as excinfo:
            build().generate_speech("hello")
        assert excinfo.value.modality == "sound"

    def test_supported_modality_runs(self, env: None) -> None:
        assert build().generate_image("a cat").mime_type == "image/png"


class TestObservability:
    def test_success_is_traced_with_prompts_and_usage(self, env: None) -> None:
        trace = RecordingTrace()
        FakeClient.next_result = LLMResult(text="hi", prompt_tokens=3, completion_tokens=1)
        build(trace=trace).complete("sys", "usr")
        (record,) = trace.records
        assert record["service"] == "fake"
        assert record["modality"] == "text"
        assert record["request"]["system"] == "sys"
        assert record["result"].prompt_tokens == 3
        assert record["error"] is None
        assert record["latency_ms"] >= 0

    def test_failure_is_traced_before_being_reraised(self, env: None) -> None:
        trace = RecordingTrace()
        FakeClient.next_error = RuntimeError("provider exploded")
        with pytest.raises(RuntimeError, match="provider exploded"):
            build(trace=trace).complete("sys", "usr")
        (record,) = trace.records
        assert "provider exploded" in record["error"]
        assert record["result"] is None

    def test_a_broken_tracer_does_not_break_the_call(self, env: None) -> None:
        # Observability must never be load-bearing.
        assert build(trace=ExplodingTrace()).complete("sys", "usr") == "ok"

    def test_last_result_exposes_usage(self, env: None) -> None:
        FakeClient.next_result = LLMResult(text="hi", total_tokens=7)
        client = build()
        client.complete("sys", "usr")
        assert client.total_tokens == 7

    def test_truncated_finish_reason_warns(
        self, env: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        FakeClient.next_result = LLMResult(text="cut off", finish_reason="length")
        with caplog.at_level(logging.WARNING):
            build().complete("sys", "usr")
        assert "did not finish cleanly" in caplog.text

    def test_empty_response_warns(
        self, env: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        FakeClient.next_result = LLMResult(text="   ", finish_reason="stop")
        with caplog.at_level(logging.WARNING):
            build().complete("sys", "usr")
        assert "EMPTY response" in caplog.text

    def test_clean_finish_does_not_warn(
        self, env: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            build().complete("sys", "usr")
        assert caplog.text == ""


class TestRequestShaping:
    def test_history_reaches_the_provider_as_a_tuple(self, env: None) -> None:
        client = build()
        client.complete(
            "sys", "usr",
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        )
        assert isinstance(client.seen["history"], tuple)
        assert len(client.seen["history"]) == 2

    def test_images_default_to_an_empty_tuple_not_none(self, env: None) -> None:
        client = build()
        client.complete("sys", "usr")
        assert client.seen["images"] == ()

    def test_invalid_history_fails_before_the_provider_is_touched(self, env: None) -> None:
        client = build()
        with pytest.raises(ValueError):
            client.complete("sys", "usr", [{"role": "assistant", "content": "a"}])
        assert not hasattr(client, "seen")

    def test_trace_omits_images_when_none_were_sent(self, env: None) -> None:
        trace = RecordingTrace()
        build(trace=trace).complete("sys", "usr")
        assert "images" not in trace.records[0]["request"]


def test_trace_sink_protocol_is_structural() -> None:
    # Nothing has to subclass or import TraceSink for the client to accept it.
    assert isinstance(RecordingTrace(), TraceSink)
