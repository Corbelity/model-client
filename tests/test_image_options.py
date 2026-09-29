"""SA-355: output size, aspect ratio and quality for image generation.

The rule these tests exist to defend is that nothing is ever silently substituted. A
request this client cannot honour raises; a request it can honour arrives at the provider
unchanged; and a call that asks for none of it is byte-identical to what it was before
these parameters existed.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from corbelity.model_client import (
    ImageOptions,
    JsonlTraceLogger,
    UnsupportedQualityError,
    UnsupportedSizeError,
    aspect_ratios_of,
    get_spec,
    parse_aspect_ratio,
    parse_size,
    sizes_for_ratio,
    supported_aspect_ratios,
    supported_image_qualities,
    supported_image_sizes,
)
from corbelity.model_client.providers.openai import OpenAIClient

PNG = b"\x89PNG\r\n\x1a\n" + b"body"


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(PNG).decode("ascii"))],
        size="2048x1152",
        quality="medium",
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=4160, total_tokens=4260,
            input_tokens_details=SimpleNamespace(text_tokens=20, image_tokens=80),
            output_tokens_details=SimpleNamespace(text_tokens=60, image_tokens=4100),
        ),
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


def sent(client: FakeOpenAIClient) -> dict[str, Any]:
    return client.calls[0][1]


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("2048x1152", (2048, 1152)), ("1024X1024", (1024, 1024)),
         (" 512 x 512 ", (512, 512))],
    )
    def test_sizes_parse(self, text: str, expected: tuple[int, int]) -> None:
        assert parse_size(text) == expected

    @pytest.mark.parametrize("text", ["auto", "", "big", "16:9", "-1x5", "0x0", "1024x"])
    def test_non_resolutions_return_none(self, text: str) -> None:
        # None rather than raising: "auto" is a legitimate size, so not-a-resolution is
        # not automatically an error.
        assert parse_size(text) is None

    def test_ratios_parse_and_reduce(self) -> None:
        assert parse_aspect_ratio("16:9") == (16, 9)
        assert parse_aspect_ratio("1920:1080") == (16, 9)
        assert parse_aspect_ratio("nonsense") is None

    def test_ratio_match_is_exact(self) -> None:
        # A near-match would be the silent substitution this all exists to prevent.
        assert sizes_for_ratio("16:9", ["1920x1081"]) == ()
        assert sizes_for_ratio("16:9", ["1920x1080"]) == ("1920x1080",)

    def test_ratio_matches_are_smallest_first(self) -> None:
        assert sizes_for_ratio("16:9", ["3840x2160", "1920x1080", "2048x1152"]) == (
            "1920x1080", "2048x1152", "3840x2160",
        )

    def test_unparseable_entries_never_match(self) -> None:
        assert sizes_for_ratio("1:1", ["auto", "1024x1024"]) == ("1024x1024",)

    def test_ratios_are_summarised_in_order(self) -> None:
        assert aspect_ratios_of(("auto", "1024x1024", "2048x1152", "2048x2048")) == (
            "1:1", "16:9",
        )


class TestCapabilityLookup:
    def test_answers_without_a_client_or_credential(self) -> None:
        assert "2048x1152" in supported_image_sizes("openai")
        assert "16:9" in supported_aspect_ratios("openai")
        assert "medium" in supported_image_qualities("openai")

    def test_a_service_without_the_controls_reports_empty(self) -> None:
        assert supported_image_sizes("huggingface") == ()
        assert supported_image_qualities("huggingface") == ()

    def test_openai_accepts_custom_sizes(self) -> None:
        # The narrow Literal on the edit endpoint looks like DALL-E-era residue, and
        # enforcing it would refuse 16:9 on the path that carries reference images.
        assert get_spec("openai").image_custom_size is True


class TestUnchangedWithoutSettings:
    def test_plain_generation_sends_exactly_what_it_did_before(self) -> None:
        client = build()
        client.generate_image("a corbel bracket")
        assert client.calls[0][0] == "generate"
        assert sent(client) == {
            "model": "gpt-image-2.5-flare", "prompt": "a corbel bracket", "n": 1,
        }

    def test_no_settings_means_no_keys_not_default_values(self) -> None:
        # Omitted, so the provider's own default applies rather than one invented here.
        client = build()
        client.generate_image("x")
        assert "size" not in sent(client)
        assert "quality" not in sent(client)

    def test_empty_options_is_falsey(self) -> None:
        assert not ImageOptions()
        assert ImageOptions(size="1024x1024")


class TestSize:
    def test_an_offered_size_is_passed_through(self) -> None:
        client = build()
        client.generate_image("x", size="2048x1152")
        assert sent(client)["size"] == "2048x1152"

    def test_a_custom_size_is_passed_through_unchanged(self) -> None:
        client = build()
        client.generate_image("x", size="1536x864")
        assert sent(client)["size"] == "1536x864"

    def test_a_malformed_size_is_refused(self) -> None:
        client = build()
        with pytest.raises(UnsupportedSizeError) as excinfo:
            client.generate_image("x", size="really big")
        assert excinfo.value.requested == "really big"
        assert client.calls == []

    def test_auto_is_accepted_because_the_provider_offers_it(self) -> None:
        client = build()
        client.generate_image("x", size="auto")
        assert sent(client)["size"] == "auto"


class TestAspectRatio:
    def test_resolves_to_a_concrete_size(self) -> None:
        # The motivating case: 16:9 without hand-typing it into the prompt.
        client = build()
        client.generate_image("a wide establishing shot", aspect_ratio="16:9")
        assert sent(client)["size"] == "2048x1152"

    def test_resolves_to_the_smallest_match(self) -> None:
        # Pixels drive cost, so the cheap end is the safe default to choose for someone;
        # 3840x2160 is also 16:9 and is reached by naming it.
        offered = supported_image_sizes("openai")
        assert "3840x2160" in offered and "2048x1152" in offered
        client = build()
        client.generate_image("x", aspect_ratio="16:9")
        assert sent(client)["size"] == "2048x1152"

    def test_equivalent_ratios_resolve_identically(self) -> None:
        client = build()
        client.generate_image("x", aspect_ratio="1920:1080")
        assert sent(client)["size"] == "2048x1152"

    def test_an_unreachable_ratio_raises_and_names_what_is_reachable(self) -> None:
        client = build()
        with pytest.raises(UnsupportedSizeError) as excinfo:
            client.generate_image("x", aspect_ratio="21:9")
        assert excinfo.value.requested == "21:9"
        assert "16:9" in excinfo.value.supported
        assert client.calls == []

    def test_with_size_is_an_error_not_a_precedence_puzzle(self) -> None:
        client = build()
        with pytest.raises(ValueError, match="mutually exclusive"):
            client.generate_image("x", aspect_ratio="16:9", size="1024x1024")
        assert client.calls == []


class TestQuality:
    @pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max", "auto"])
    def test_accepted_levels_pass_through(self, level: str) -> None:
        client = build()
        client.generate_image("x", quality=level)
        assert sent(client)["quality"] == level

    def test_an_unknown_level_is_refused(self) -> None:
        client = build()
        with pytest.raises(UnsupportedQualityError) as excinfo:
            client.generate_image("x", quality="ultra")
        assert "medium" in excinfo.value.supported
        assert client.calls == []

    def test_a_legacy_level_is_refused(self) -> None:
        # "hd" is DALL-E-era, absent from the edit endpoint and never reported back.
        with pytest.raises(UnsupportedQualityError):
            build().generate_image("x", quality="hd")

    def test_quality_is_independent_of_size(self) -> None:
        client = build()
        client.generate_image("x", aspect_ratio="16:9", quality="low")
        assert sent(client) | {"size": "2048x1152", "quality": "low"} == sent(client)


class TestWithReferenceImages:
    def test_settings_reach_the_edit_endpoint(self) -> None:
        from corbelity.model_client import ImageInput

        client = build()
        client.generate_image(
            "same character, wider",
            [ImageInput(data=PNG, mime_type="image/png", name="hero.png")],
            aspect_ratio="16:9",
            quality="medium",
        )
        endpoint, kwargs = client.calls[0]
        # The endpoint switch is the real risk: references force images.edit, and the size has
        # to survive that switch.
        assert endpoint == "edit"
        assert kwargs["size"] == "2048x1152"
        assert kwargs["quality"] == "medium"
        assert len(kwargs["image"]) == 1


class TestUsageAndReporting:
    def test_usage_parsing_is_unaffected_by_settings(self) -> None:
        # SA-341/SA-357 reporting must stay correct across sizes and quality levels.
        result = build().generate_image("x", aspect_ratio="16:9", quality="max")
        assert result.prompt_tokens == 100
        assert result.input_image_tokens == 80
        assert result.output_image_tokens == 4100
        assert result.total_tokens == 4260

    def test_the_result_reports_what_was_produced_not_requested(self) -> None:
        # Requested max; the stub reports medium. The result must say medium.
        result = build().generate_image("x", quality="max")
        assert result.quality == "medium"
        assert result.size == "2048x1152"


class TestTrace:
    def test_requested_settings_are_recorded_separately(self, tmp_path: Path) -> None:
        trace = JsonlTraceLogger(tmp_path / "traces.jsonl")
        FakeOpenAIClient(
            model="gpt-image-2.5-flare", trace=trace
        ).generate_image("x", aspect_ratio="16:9", quality="max")

        record = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))
        asked = record["request"]["requested"]
        # Both the director's framing and the resolution it became, because the mapping
        # between them is this client's decision and worth being able to audit.
        assert asked["aspect_ratio"] == "16:9"
        assert asked["size"] == "2048x1152"
        assert asked["quality"] == "max"
        # And what came back, which differs -- the whole reason they are kept apart.
        assert record["response"]["produced"]["quality"] == "medium"

    def test_no_requested_key_on_a_plain_generation(self, tmp_path: Path) -> None:
        trace = JsonlTraceLogger(tmp_path / "traces.jsonl")
        FakeOpenAIClient(model="gpt-image-2.5-flare", trace=trace).generate_image("x")
        record = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))
        assert "requested" not in record["request"]
