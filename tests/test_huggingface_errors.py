"""The two ways the hub declines to route a request, and why they get separate messages.

This file exists because of a hole these tests now fill. `_call` translated exactly one
failure shape -- a bare StopIteration from an empty provider mapping -- and huggingface-hub
2.0.0 stopped producing it for text: `conversational` short-circuits to an auto-router
before any mapping is fetched, so the router answers HTTP 400 with code
"model_not_supported" instead. The translation quietly became dead code on the most-used
modality, and only a live call revealed it.

Nothing here touches the network or imports the SDK; the fake overrides `_build_client`,
and `_call` is fed a callable that raises. So unlike the live suite, CI runs all of it.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from corbelity.model_client.media import IMAGE, SOUND, TEXT
from corbelity.model_client.providers.huggingface import HuggingFaceClient

MODEL = "acme/not-served"


class FakeHuggingFaceClient(HuggingFaceClient):
    """Overrides only _build_client, so no SDK is imported."""

    def _build_client(self) -> Any:
        return SimpleNamespace()


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-t")


def bad_request(body: str, status: int = 400) -> ValueError:
    """A stand-in for huggingface_hub's BadRequestError(HfHubHTTPError, ValueError).

    Built as a plain ValueError carrying a `response` rather than by importing the real
    class, for the same reason the production predicate reads the attribute instead of
    importing it: the extra is optional and must not be a test dependency. What is being
    checked is the SHAPE the predicate keys on -- a status code plus the error code in the
    message -- which is all it is entitled to rely on.
    """
    err = ValueError(f"\n\nBad request:\n{body}")
    err.response = SimpleNamespace(status_code=status, text=body)  # type: ignore[attr-defined]
    return err


def declining(modality: str, error: BaseException) -> ValueError:
    """Run _call with a callable that raises `error`; return the ValueError it produced."""
    def raises() -> Any:
        raise error

    with pytest.raises(ValueError) as caught:
        FakeHuggingFaceClient(model=MODEL)._call(modality, raises)
    return caught.value


NOT_SUPPORTED = (
    "{'message': \"The requested model 'acme/not-served' is not supported by any provider "
    "you have enabled.\", 'type': 'invalid_request_error', 'param': 'model', "
    "'code': 'model_not_supported'}"
)


# --- nothing serves it anywhere: StopIteration, image and speech ------------------------

@pytest.mark.parametrize(("modality", "task"), [(IMAGE, "text-to-image"),
                                               (SOUND, "text-to-speech")])
def test_empty_provider_mapping_becomes_an_actionable_error(modality: str, task: str) -> None:
    """A bare StopIteration carries no message at all and reaches a UI as "StopIteration: ".
    It must come back naming the model, the task, and where to look for a served one."""
    err = declining(modality, StopIteration())

    assert MODEL in str(err)
    assert task in str(err)
    assert "huggingface.co/models" in str(err)
    # The cause is preserved, so a traceback still shows where it actually came from.
    assert isinstance(err.__cause__, StopIteration)


# --- the account has no provider enabled: HTTP 400, text --------------------------------

def test_router_rejection_becomes_an_actionable_error() -> None:
    """The 2.0.0 path. A 400 carrying "model_not_supported" must be translated, not leak
    HfHubHTTPError's formatting to the caller."""
    err = declining(TEXT, bad_request(NOT_SUPPORTED))

    assert MODEL in str(err)
    assert "settings/inference-providers" in str(err)
    assert isinstance(err.__cause__, ValueError)


def test_the_two_messages_do_not_send_you_to_the_same_place() -> None:
    """The point of splitting them. One says "pick another model", the other says "change
    your account" -- collapsing them costs someone the time it takes to rule out every
    model they try before discovering the problem was never the model."""
    client = FakeHuggingFaceClient(model=MODEL)
    unserved = client._unserved_message(TEXT)
    not_enabled = client._not_enabled_message(TEXT)

    assert unserved != not_enabled
    # Each points at the remedy for ITS OWN cause, and not at the other one's.
    assert "huggingface.co/models" in unserved
    assert "settings/inference-providers" not in unserved
    assert "settings/inference-providers" in not_enabled
    assert "huggingface.co/models" not in not_enabled


# --- what must NOT be swallowed ---------------------------------------------------------

def test_a_400_without_the_error_code_is_left_alone() -> None:
    """Translation is keyed on the API's own error code, so an unrelated bad request -- a
    malformed payload, a bad parameter -- must pass through with its own message intact. A
    predicate that matched any 400 would bury real errors under a routing explanation."""
    original = bad_request("{'code': 'invalid_parameter', 'param': 'temperature'}")
    err = declining(TEXT, original)

    assert err is original
    assert "invalid_parameter" in str(err)


def test_a_plain_value_error_is_left_alone() -> None:
    """No response attribute at all: nothing to key on, nothing to claim about routing."""
    original = ValueError("something else entirely")
    err = declining(TEXT, original)

    assert err is original


def test_a_non_400_carrying_the_code_is_left_alone() -> None:
    """The status check is load-bearing. A 500 that happens to mention the code is a
    provider fault, not a routing decision, and must not be reported as the user's."""
    original = bad_request(NOT_SUPPORTED, status=500)
    err = declining(TEXT, original)

    assert err is original


def test_an_sdk_change_still_propagates() -> None:
    """The reason the live tests exist: a renamed or removed SDK surface raises TypeError
    or AttributeError, and `_call` must never dress that up as a routing problem."""
    def raises() -> Any:
        raise AttributeError("'InferenceClient' object has no attribute 'chat_completion'")

    with pytest.raises(AttributeError):
        FakeHuggingFaceClient(model=MODEL)._call(TEXT, raises)


# --- the coupling that actually broke ----------------------------------------------------

def test_live_suite_sentinels_still_match_these_messages() -> None:
    """The live suite decides skip-vs-fail by matching these strings. If a message is
    reworded and the sentinel is not, the live test stops skipping on a routing miss and
    starts failing on one -- which is exactly the trap 2.0.0 sprang. Assert the coupling
    here, where CI can see it, rather than discovering it on the next major bump."""
    from test_live_huggingface import _DECLINED

    client = FakeHuggingFaceClient(model=MODEL)
    for message in (client._unserved_message(TEXT), client._not_enabled_message(TEXT)):
        assert any(marker in message for marker in _DECLINED), message
