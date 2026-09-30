"""SA-356: input_fidelity, the dial for how strictly reference images are adhered to.

The parameter is small -- a two-value enum -- but it is the first setting in this client
whose validity depends on ANOTHER argument. Everything before it could be judged on its
own: a size is supported or it is not. `input_fidelity` is meaningless without `images`,
so these tests are mostly about the conditional refusal and its ordering, not the value.

The refusal is the point. A setting silently dropped because it had nothing to act on is
indistinguishable, from the caller's side, from a model that ignored it.
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
    UnsupportedFidelityError,
    get_spec,
    supported_input_fidelities,
)
from corbelity.model_client.providers.openai import OpenAIClient

PNG = b"\x89PNG\r\n\x1a\n" + b"body"
REF = ImageInput(data=PNG, mime_type="image/png", name="ref.png")


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(PNG).decode("ascii"))],
        size="2048x1152",
        quality="medium",
    )


class FakeOpenAIClient(OpenAIClient):
    """Records which endpoint was called and with exactly what."""

    def _build_client(self) -> Any:
        self.calls: list[tuple[str, dict[str, Any]]] = []

        def generate(**kwargs: Any) -> SimpleNamespace:
            self.calls.append(("generate", kwargs))
            return _response()

        def edit(**kwargs: Any) -> SimpleNamespace:
            self.calls.append(("edit", kwargs))
            return _response()

        return SimpleNamespace(images=SimpleNamespace(generate=generate, edit=edit))


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")


def build() -> FakeOpenAIClient:
    return FakeOpenAIClient(model="gpt-image-2.5-flare")


# --- it reaches the provider, on the right endpoint -------------------------------------

@pytest.mark.parametrize("fidelity", ["high", "low"])
def test_fidelity_reaches_the_edit_endpoint(fidelity: str) -> None:
    client = build()
    client.generate_image("a corbel bracket", images=[REF], input_fidelity=fidelity)

    endpoint, kwargs = client.calls[0]
    # images.edit, not images.generate: references are what select it, and input_fidelity
    # exists only there.
    assert endpoint == "edit"
    assert kwargs["input_fidelity"] == fidelity


def test_omitting_it_sends_no_key_at_all() -> None:
    """Absent, not None. A provider default is a decision this package does not get to
    make, and `input_fidelity=None` in the payload would be making it."""
    client = build()
    client.generate_image("a corbel bracket", images=[REF])

    _endpoint, kwargs = client.calls[0]
    assert "input_fidelity" not in kwargs


def test_a_plain_generation_is_untouched() -> None:
    client = build()
    client.generate_image("a corbel bracket")

    endpoint, kwargs = client.calls[0]
    assert endpoint == "generate"
    assert "input_fidelity" not in kwargs


# --- the conditional refusal -------------------------------------------------------------

def test_without_references_it_raises_rather_than_being_dropped() -> None:
    """The acceptance criterion. A caller who set this and saw no effect would have no way
    to tell whether the model ignored it or this client discarded it."""
    client = build()
    with pytest.raises(ValueError) as caught:
        client.generate_image("a corbel bracket", input_fidelity="high")

    message = str(caught.value)
    assert "input_fidelity" in message
    # Names the remedy, not just the rule.
    assert "images=" in message
    # And nothing was sent: the refusal is local, before any provider call.
    assert client.calls == []


def test_an_unaccepted_value_raises_before_the_provider_is_touched() -> None:
    client = build()
    with pytest.raises(UnsupportedFidelityError) as caught:
        client.generate_image("a corbel bracket", images=[REF], input_fidelity="medium")

    err = caught.value
    assert err.service == "openai"
    assert err.requested == "medium"
    assert err.supported == ("low", "high")
    assert client.calls == []


def test_an_unaccepted_value_is_still_a_value_error() -> None:
    """Per DESIGN §11: a web layer mapping ValueError to a 400 keeps working without
    importing this package's types."""
    client = build()
    with pytest.raises(ValueError):
        client.generate_image("a corbel bracket", images=[REF], input_fidelity="medium")


def test_the_missing_references_check_runs_first() -> None:
    """Both problems at once -- an unaccepted value AND no references -- must report the
    references. Checking the value first would send someone to study the accepted list
    when the accepted list was never the issue."""
    client = build()
    with pytest.raises(ValueError) as caught:
        client.generate_image("a corbel bracket", input_fidelity="nonsense")

    assert not isinstance(caught.value, UnsupportedFidelityError)
    assert "images=" in str(caught.value)


def test_a_provider_with_no_fidelity_control_refuses_every_value() -> None:
    """Empty `image_fidelities` means the service has no such dial -- which is a different
    statement from refusing references, and must not be confused with it. Built by
    replacing the spec on a subclass rather than registering a provider, so the global
    registry is untouched (a registered fake leaks into every other test's view of what
    services exist)."""
    class NoFidelityClient(FakeOpenAIClient):
        SPEC = replace(get_spec("openai"), image_fidelities=())

    client = NoFidelityClient(model="gpt-image-2.5-flare")
    with pytest.raises(UnsupportedFidelityError) as caught:
        client.generate_image("a corbel bracket", images=[REF], input_fidelity="high")

    assert caught.value.supported == ()
    # The message has to say the control is absent rather than list nothing and leave the
    # reader wondering whether they mistyped.
    assert "no reference-fidelity control" in str(caught.value)
    assert client.calls == []


# --- what a caller can ask before spending anything ------------------------------------

def test_supported_input_fidelities_answers_without_a_client() -> None:
    assert supported_input_fidelities("openai") == ("low", "high")
    # No credential was set for this one and none is needed: the answer is spec data.
    assert supported_input_fidelities("anthropic") == ()


# --- the plumbing it shares with size and quality ---------------------------------------

def test_image_options_is_truthy_on_fidelity_alone() -> None:
    """`if options:` is what gates the trace's `requested` block. Had __bool__ not learned
    about this field, a fidelity-only call would have been sent correctly and then recorded
    as though nothing had been asked for."""
    assert bool(ImageOptions(input_fidelity="high")) is True
    assert bool(ImageOptions()) is False


def test_the_trace_records_what_was_asked_for(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    client = FakeOpenAIClient(
        model="gpt-image-2.5-flare", trace=JsonlTraceLogger(path)
    )
    client.generate_image("a corbel bracket", images=[REF], input_fidelity="high")

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["request"]["requested"]["input_fidelity"] == "high"


def test_the_trace_omits_it_when_not_asked_for(tmp_path: Path) -> None:
    """Recorded only when set. An `input_fidelity: null` in every image record would make
    a reader of the log think the setting had been considered and declined."""
    path = tmp_path / "trace.jsonl"
    client = FakeOpenAIClient(
        model="gpt-image-2.5-flare", trace=JsonlTraceLogger(path)
    )
    client.generate_image("a corbel bracket", images=[REF], quality="medium")

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "input_fidelity" not in record["request"]["requested"]
