"""The shared chat-completions dialect, and the two catalog flags that vary by model.

No SDK is imported anywhere here: the fake below overrides `_build_client`, which is the
only place a provider touches its SDK. That the tests can do this at all is the seam
working as intended -- and it means these run in the no-extras CI job.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from corbelity.model_client import (
    ModelConfig,
    ProviderSpec,
    available_services,
    builtin_catalog,
    get_spec,
    known_services,
    provider_spec,
    resolve_service,
    supported_modalities,
)
from corbelity.model_client.providers.openai_compatible import OpenAICompatibleClient

FAKE_SPEC = ProviderSpec(
    name="fake-compat",
    client_path=f"{__name__}:FakeCompatClient",
    key_env=("FAKE_COMPAT_KEY",),
    default_base_url="https://compat.invalid/v1",
    modalities=frozenset({"text"}),
)


def _response(content: str = "hi") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )


class _Completions:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def create(self, **payload: Any) -> SimpleNamespace:
        self.payload = payload
        return _response()


class FakeCompatClient(OpenAICompatibleClient):
    """Stands in for any OpenAI-dialect provider, capturing the request payload."""

    SPEC = FAKE_SPEC

    def _build_client(self) -> Any:
        completions = _Completions()
        self.completions = completions
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_COMPAT_KEY", "k")


def build(**kwargs: Any) -> FakeCompatClient:
    return FakeCompatClient(**kwargs)


def catalog_file(tmp_path: Any, entry: dict[str, Any]) -> Any:
    import json

    path = tmp_path / "models.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    return path


class TestCompletionCapParameter:
    def test_defaults_to_max_tokens(self, env: None) -> None:
        client = build(model="unlisted-model", max_tokens=123)
        client.complete("sys", "usr")
        payload = client.completions.payload
        assert payload is not None
        assert payload["max_tokens"] == 123
        assert "max_completion_tokens" not in payload

    def test_catalog_can_rename_it(self, env: None, tmp_path: Any) -> None:
        # The motivating case: newer OpenAI models reject max_tokens outright.
        path = catalog_file(tmp_path, {
            "id": "picky-model", "service": "fake-compat",
            "max_tokens_param": "max_completion_tokens",
        })
        client = build(model="picky-model", max_tokens=77,
                       config=ModelConfig(catalog_path=path))
        client.complete("sys", "usr")
        payload = client.completions.payload
        assert payload is not None
        assert payload["max_completion_tokens"] == 77
        assert "max_tokens" not in payload

    def test_the_shipped_openai_text_entry_carries_the_flag(self) -> None:
        entry = builtin_catalog().get("gpt-5.6-terra")
        assert entry is not None
        assert entry.max_tokens_param == "max_completion_tokens"


class TestSamplingParameters:
    def test_sent_by_default(self, env: None) -> None:
        client = build(model="unlisted-model", temperature=0.25, top_p=0.9)
        client.complete("sys", "usr")
        payload = client.completions.payload
        assert payload is not None
        assert payload["temperature"] == 0.25
        assert payload["top_p"] == 0.9

    def test_top_p_omitted_when_unset(self, env: None) -> None:
        client = build(model="unlisted-model", config=ModelConfig(default_top_p=None))
        client.complete("sys", "usr")
        assert "top_p" not in (client.completions.payload or {})

    def test_catalog_can_suppress_them(self, env: None, tmp_path: Any) -> None:
        path = catalog_file(tmp_path, {
            "id": "no-sampling", "service": "fake-compat", "supports_sampling": False,
        })
        client = build(model="no-sampling", temperature=0.5,
                       config=ModelConfig(catalog_path=path))
        client.complete("sys", "usr")
        payload = client.completions.payload
        assert payload is not None
        assert "temperature" not in payload
        assert "top_p" not in payload


class TestResponseHandling:
    def test_usage_and_finish_reason_are_extracted(self, env: None) -> None:
        client = build(model="unlisted-model")
        assert client.complete("sys", "usr") == "hi"
        assert (client.prompt_tokens, client.completion_tokens, client.total_tokens) == (1, 2, 3)
        assert client.finish_reason == "stop"

    def test_system_prompt_is_the_first_message(self, env: None) -> None:
        # Unlike Anthropic, this dialect carries the system prompt inside `messages`.
        client = build(model="unlisted-model")
        client.complete("sys", "usr")
        messages = (client.completions.payload or {})["messages"]
        assert messages[0] == {"role": "system", "content": "sys"}
        assert messages[-1]["content"] == "usr"


class TestNewProviderRegistration:
    @pytest.mark.parametrize(
        ("written", "canonical"),
        [("openai", "openai"), ("open_ai", "openai"), ("OpenAI", "openai"),
         ("gemini", "gemini"), ("google", "gemini"), ("google-gemini", "gemini")],
    )
    def test_names_and_aliases_resolve(self, written: str, canonical: str) -> None:
        assert resolve_service(written) == canonical

    def test_both_are_registered(self) -> None:
        assert {"openai", "gemini"} <= set(known_services())

    def test_openai_declares_all_three_modalities(self) -> None:
        assert supported_modalities("openai") == frozenset({"text", "image", "sound"})

    def test_gemini_is_text_only_for_now(self) -> None:
        # Image is unverified through the compatibility shim, and audio runs over the
        # streaming Live API. Both wait for a native provider.
        assert supported_modalities("gemini") == frozenset({"text"})

    def test_gemini_defaults_to_the_compatibility_endpoint(self) -> None:
        assert get_spec("gemini").default_base_url.endswith("/v1beta/openai/")

    def test_openai_has_no_hardcoded_endpoint(self) -> None:
        # The SDK's own default is correct; pinning it here would mean tracking a value
        # this package does not own.
        assert provider_spec("openai").default_base_url is None

    def test_gemini_accepts_either_google_token_name(self) -> None:
        assert get_spec("gemini").key_env == ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    @pytest.mark.parametrize(
        ("env_name", "service"),
        [("OPENAI_API_KEY", "openai"), ("GEMINI_API_KEY", "gemini"),
         ("GOOGLE_API_KEY", "gemini")],
    )
    def test_availability_detection(self, env_name: str, service: str) -> None:
        assert available_services(env={env_name: "k"}) == (service,)

    def test_every_new_catalog_entry_is_reachable(self) -> None:
        for model_id in ("gpt-5.6-terra", "gpt-image-2.5-flare", "gpt-4o-mini-tts",
                         "gemini-3.8-flash"):
            entry = builtin_catalog().get(model_id)
            assert entry is not None, model_id
            assert entry.modality in supported_modalities(entry.service), model_id
