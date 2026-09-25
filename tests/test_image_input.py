"""SA-347: reference images as input to image generation.

The two things most worth defending: that reference images change the ENDPOINT on the
OpenAI path, and that a text-only `generate_image(prompt)` call is byte-identical to what
it was before this parameter existed.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from corbelity.model_client import (
    ImageInput,
    JsonlTraceLogger,
    MediaResult,
    ModelClient,
    ProviderSpec,
    TooManyImagesError,
    UnsupportedImageInputError,
    get_spec,
)
from corbelity.model_client.providers.openai import OpenAIClient

PNG = b"\x89PNG\r\n\x1a\n" + b"body"
JPEG = b"\xff\xd8\xff" + b"body"


def reference(name: str | None = None, data: bytes = PNG,
              mime: str = "image/png") -> ImageInput:
    return ImageInput(data=data, mime_type=mime, name=name)


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(PNG).decode("ascii"))],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
    )


class FakeOpenAIClient(OpenAIClient):
    """Records which endpoint was called and with what."""

    def _build_client(self) -> Any:
        self.calls: list[tuple[str, dict[str, Any]]] = []

        def generate(**kwargs: Any) -> SimpleNamespace:
            self.calls.append(("generate", kwargs))
            return _response()

        def edit(**kwargs: Any) -> SimpleNamespace:
            self.calls.append(("edit", kwargs))
            return _response()

        return SimpleNamespace(
            images=SimpleNamespace(generate=generate, edit=edit)
        )


class NoInputClient(ModelClient[object]):
    """An image provider whose spec does not declare image_input."""

    SPEC = ProviderSpec(
        name="fake-noinput",
        client_path=f"{__name__}:NoInputClient",
        modalities=frozenset({"image"}),
        image_input=False,
    )

    def __init__(self, **kwargs: Any) -> None:
        self.invoked = False
        super().__init__(**kwargs)

    def _build_client(self) -> object:
        return object()

    def _invoke(self, system: str, user: str, history: Any, images: Any) -> Any:
        raise NotImplementedError

    def _invoke_image(self, prompt: str, images: tuple[ImageInput, ...]) -> MediaResult:
        self.invoked = True
        return MediaResult(data=PNG, mime_type="image/png")


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")
    monkeypatch.setenv("HF_TOKEN", "hf-t")


def build(**kwargs: Any) -> FakeOpenAIClient:
    return FakeOpenAIClient(model="gpt-image-2.5-flare", **kwargs)


class TestEndpointSelection:
    def test_no_images_uses_generate(self) -> None:
        client = build()
        client.generate_image("a corbel bracket")
        endpoint, kwargs = client.calls[0]
        assert endpoint == "generate"
        # The pre-existing call, unchanged: same arguments, and no image key at all.
        assert kwargs == {"model": "gpt-image-2.5-flare", "prompt": "a corbel bracket", "n": 1}

    def test_images_switch_to_edit(self) -> None:
        client = build()
        client.generate_image("same character, side view", [reference("hero.png")])
        endpoint, kwargs = client.calls[0]
        assert endpoint == "edit"
        assert kwargs["prompt"] == "same character, side view"
        assert len(kwargs["image"]) == 1

    def test_empty_list_is_the_same_as_none(self) -> None:
        client = build()
        client.generate_image("x", [])
        assert client.calls[0][0] == "generate"

    def test_several_references_are_all_sent(self) -> None:
        client = build()
        client.generate_image("x", [reference("a.png"), reference("b.png"),
                                    reference("c.png")])
        assert len(client.calls[0][1]["image"]) == 3


class TestUploadShaping:
    def test_uploads_carry_the_callers_filename(self) -> None:
        client = build()
        client.generate_image("x", [reference("hero-sheet.png")])
        assert client.calls[0][1]["image"][0].name == "hero-sheet.png"

    def test_unnamed_references_get_a_positional_name(self) -> None:
        client = build()
        client.generate_image("x", [reference(None), reference(None)])
        names = [f.name for f in client.calls[0][1]["image"]]
        assert names == ["reference-0.png", "reference-1.png"]

    def test_extension_follows_the_mime_type_not_the_name(self) -> None:
        # The API infers the part's content type from the filename extension, so a
        # mismatched or missing one is a rejected upload.
        client = build()
        client.generate_image("x", [reference("photo", data=JPEG, mime="image/jpeg")])
        assert client.calls[0][1]["image"][0].name == "photo.jpg"

    def test_upload_contains_the_original_bytes(self) -> None:
        client = build()
        client.generate_image("x", [reference("a.png")])
        assert client.calls[0][1]["image"][0].read() == PNG


class TestCapability:
    def test_provider_without_image_input_is_rejected_early(self) -> None:
        # A fake rather than a real provider: this must hold without any SDK installed,
        # and the point under test is the spec flag, not any particular vendor. NOT
        # registered -- the client is constructed directly, and adding a nameless fake to
        # the shared registry would leak into every available_services() assertion.
        client = NoInputClient()
        with pytest.raises(UnsupportedImageInputError) as excinfo:
            client.generate_image("x", [reference()])
        assert excinfo.value.service == "fake-noinput"
        assert client.invoked is False

    def test_that_provider_still_generates_without_references(self) -> None:
        # The capability gate must not affect the text-only path.
        client = NoInputClient()
        assert client.generate_image("x").mime_type == "image/png"
        assert client.invoked is True

    def test_shipped_specs_declare_the_capability(self) -> None:
        assert get_spec("huggingface").image_input is False
        assert get_spec("openai").image_input is True

    def test_cap_comes_from_the_spec(self) -> None:
        maximum = get_spec("openai").max_reference_images
        assert maximum is not None
        client = build()
        with pytest.raises(TooManyImagesError) as excinfo:
            client.generate_image("x", [reference() for _ in range(maximum + 1)])
        assert excinfo.value.maximum == maximum
        # Rejected before the upload, which is the point of capping client-side.
        assert client.calls == []

    def test_at_the_cap_is_allowed(self) -> None:
        maximum = get_spec("openai").max_reference_images
        assert maximum is not None
        client = build()
        client.generate_image("x", [reference() for _ in range(maximum)])
        assert client.calls[0][0] == "edit"

    def test_invalid_reference_fails_validation_first(self) -> None:
        client = build()
        with pytest.raises(ValueError, match="unsupported type"):
            client.generate_image("x", [ImageInput(data=b"xx", mime_type="image/tiff")])
        assert client.calls == []


class TestTrace:
    def test_references_become_artifacts_not_bytes(self, tmp_path: Path) -> None:
        trace = JsonlTraceLogger(tmp_path / "traces.jsonl")
        build(trace=trace).generate_image("x", [reference("hero.png")])

        record = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))
        (descriptor,) = record["request"]["images"]
        assert descriptor["mime_type"] == "image/png"
        assert descriptor["bytes"] == len(PNG)
        assert descriptor["source_name"] == "hero.png"
        # The payload is on disk beside the trace, not inlined in the line.
        assert (tmp_path / "artifacts" / descriptor["name"]).read_bytes() == PNG

    def test_text_only_generation_records_no_images_key(self, tmp_path: Path) -> None:
        trace = JsonlTraceLogger(tmp_path / "traces.jsonl")
        build(trace=trace).generate_image("x")
        record = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))
        assert "images" not in record["request"]


class TestResultUnchanged:
    def test_result_shape_is_the_same_either_way(self) -> None:
        plain = build().generate_image("x")
        conditioned = build().generate_image("x", [reference()])
        for result in (plain, conditioned):
            assert isinstance(result, MediaResult)
            assert result.data == PNG
            assert result.mime_type == "image/png"
            assert result.prompt_tokens == 10
