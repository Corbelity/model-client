"""SA-357: reporting what the provider actually produced.

The distinction these tests defend is between REQUESTED and PRODUCED. Every field here is
read from the response, never echoed from a request, because the point is to catch the
case where the two differ.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from corbelity.model_client import JsonlTraceLogger, MediaResult, ModelResult
from corbelity.model_client.providers.openai import OpenAIClient

PNG = b"\x89PNG\r\n\x1a\n" + b"body"


def response(**overrides: Any) -> SimpleNamespace:
    """A full ImagesResponse-shaped object; pass None to drop a field's value."""
    fields: dict[str, Any] = {
        "data": [SimpleNamespace(b64_json=base64.b64encode(PNG).decode("ascii"))],
        "size": "2048x1152",
        "quality": "medium",
        "output_format": "png",
        "background": "opaque",
        "created": 1790000000,
        "usage": SimpleNamespace(
            input_tokens=100,
            output_tokens=4160,
            total_tokens=4260,
            input_tokens_details=SimpleNamespace(
                text_tokens=20, image_tokens=80, cached_tokens=0
            ),
            output_tokens_details=SimpleNamespace(text_tokens=60, image_tokens=4100),
        ),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class FakeOpenAIClient(OpenAIClient):
    next_response: Any = None

    def _build_client(self) -> Any:
        return SimpleNamespace(
            images=SimpleNamespace(
                generate=lambda **_kw: type(self).next_response,
                edit=lambda **_kw: type(self).next_response,
            )
        )


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-t")


def generate(resp: Any) -> MediaResult:
    FakeOpenAIClient.next_response = resp
    return FakeOpenAIClient(model="gpt-image-2.5-flare").generate_image("x")


class TestProducedFields:
    def test_all_reported_fields_are_read(self) -> None:
        result = generate(response())
        assert result.size == "2048x1152"
        assert result.quality == "medium"
        assert result.output_format == "png"
        assert result.background == "opaque"
        assert result.created == 1790000000

    def test_absent_fields_stay_none(self) -> None:
        # A provider reporting none of it yields the result it did before these existed.
        bare = SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(PNG).decode("ascii"))]
        )
        result = generate(bare)
        assert result.size is None
        assert result.quality is None
        assert result.created is None
        assert result.data == PNG

    def test_mime_type_describes_the_bytes_not_the_reported_format(self) -> None:
        # If the two ever disagree, the bytes are what a caller will render -- so
        # mime_type is sniffed and output_format is reported separately, on purpose.
        result = generate(response(output_format="webp"))
        assert result.output_format == "webp"
        assert result.mime_type == "image/png"


class TestOutputTokenSplit:
    def test_output_details_are_read(self) -> None:
        result = generate(response())
        assert result.output_text_tokens == 60
        assert result.output_image_tokens == 4100

    def test_flat_output_figure_is_untouched(self) -> None:
        assert generate(response()).completion_tokens == 4160

    def test_symmetry_with_the_input_split(self) -> None:
        # Both halves live on the shared base and are named to mirror one another.
        for name in ("input_text_tokens", "input_image_tokens",
                     "output_text_tokens", "output_image_tokens"):
            assert name in ModelResult.__dataclass_fields__

    def test_absent_output_details_leave_none(self) -> None:
        usage = SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3)
        result = generate(response(usage=usage))
        assert result.completion_tokens == 2
        assert result.output_image_tokens is None


class TestExtra:
    def test_unmodelled_scalar_fields_are_carried(self) -> None:
        payload = response().__dict__ | {"revised_prompt": "a wider shot of the bracket"}
        result = generate(payload)
        assert result.extra["revised_prompt"] == "a wider shot of the bracket"

    def test_modelled_fields_are_not_duplicated_into_extra(self) -> None:
        result = generate(response().__dict__)
        for name in ("size", "quality", "created", "output_format", "background"):
            assert name not in result.extra

    def test_payload_and_nested_objects_never_enter_extra(self) -> None:
        # An extra holding megabytes of base64 would defeat the artifact handling that
        # keeps payloads out of the trace.
        result = generate(response().__dict__)
        assert "data" not in result.extra
        assert "usage" not in result.extra

    def test_defaults_to_empty_rather_than_none(self) -> None:
        assert MediaResult().extra == {}


class TestTraceRecord:
    def _record(self, tmp_path: Path, resp: Any) -> dict[str, Any]:
        trace = JsonlTraceLogger(tmp_path / "traces.jsonl")
        FakeOpenAIClient.next_response = resp
        FakeOpenAIClient(model="gpt-image-2.5-flare", trace=trace).generate_image("x")
        return json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))

    def test_produced_settings_reach_the_trace(self, tmp_path: Path) -> None:
        # A calibration run reads its evidence back from the trace; a size that reached
        # the result but not the record could not be checked without measuring pixels.
        produced = self._record(tmp_path, response())["response"]["produced"]
        assert produced["size"] == "2048x1152"
        assert produced["quality"] == "medium"

    def test_output_split_reaches_the_trace(self, tmp_path: Path) -> None:
        usage = self._record(tmp_path, response())["usage"]
        assert usage["output_image_tokens"] == 4100
        assert usage["output_text_tokens"] == 60

    def test_artifact_still_written_and_bytes_stay_out(self, tmp_path: Path) -> None:
        record = self._record(tmp_path, response())
        descriptor = record["response"]["artifact"]
        assert descriptor["bytes"] == len(PNG)
        assert (tmp_path / "artifacts" / descriptor["name"]).read_bytes() == PNG

    def test_no_produced_key_when_nothing_was_reported(self, tmp_path: Path) -> None:
        bare = SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(PNG).decode("ascii"))]
        )
        assert "produced" not in self._record(tmp_path, bare)["response"]
