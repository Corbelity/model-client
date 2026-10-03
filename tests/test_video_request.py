"""The VIDEO modality's request types and their validation.

Provider-neutral, and no SDK anywhere: a fake ProviderSpec plus a catalog entry built in
the test is all a video request needs to be judged (DESIGN.md section 14). The catalog
block used for most tests is the one designed for Veo 3.1, including the
references-vs-frames exclusion confirmed by a live probe on 2026-10-03.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from corbelity.model_client import (
    VIDEO,
    VIDEO_ROLES,
    ImageInput,
    JsonlTraceLogger,
    MediaResult,
    ModelConfig,
    ModelInfo,
    ProviderSpec,
    TooManyImagesError,
    UnsupportedModalityError,
    UnsupportedVideoInputError,
    UnsupportedVideoSettingError,
    VideoInput,
    VideoInputs,
    VideoOptions,
    resolve_video_request,
    sniff_video_mime,
)
from corbelity.model_client.client import ModelClient
from corbelity.model_client.media import LLMResult
from corbelity.model_client.registry import DEFAULT_REGISTRY
from corbelity.model_client.video import resolve_request

PNG = b"\x89PNG\r\n\x1a\n" + b"body"
FRAME = ImageInput.from_bytes(PNG)
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16

SPEC = ProviderSpec(
    name="fake-video",
    client_path=f"{__name__}:FakeVideoClient",
    key_env=("FAKE_VIDEO_KEY",),
    modalities=frozenset({"text", VIDEO}),
    video_input_forms=("uri",),
)

# The Veo 3.1 block from the design (section 4.6), as the catalog will carry it.
VEO_BLOCK: dict[str, Any] = {
    "inputs": ["first_frame", "last_frame", "references", "extend"],
    "max_references": 3,
    "aspect_ratios": ["16:9", "9:16"],
    "resolutions": ["720p", "1080p", "4k"],
    "durations": [4, 6, 8],
    "audio": "always",
    "constraints": [
        {"when": "references", "require": {"duration_seconds": 8, "aspect_ratio": "16:9"}},
        {"when": "references", "exclude_input": ["first_frame", "last_frame"]},
        {"when": "resolution:1080p", "require": {"duration_seconds": 8}},
        {"when": "resolution:4k", "require": {"duration_seconds": 8}},
        {"when": "extend", "require": {"resolution": "720p"}},
        {"when": "last_frame", "require_input": "first_frame"},
    ],
}
LITE_BLOCK: dict[str, Any] = {
    "inputs": ["first_frame", "last_frame"],
    "aspect_ratios": ["16:9", "9:16"],
    "resolutions": ["720p", "1080p"],
    "durations": [4, 6, 8],
    "audio": "always",
    "constraints": [{"when": "last_frame", "require_input": "first_frame"}],
}


def entry(block: dict[str, Any] | None, model: str = "veo-test") -> ModelInfo:
    raw: dict[str, Any] = {"id": model, "service": SPEC.name, "modality": "video"}
    if block is not None:
        raw["video"] = block
    return ModelInfo.from_mapping(raw)


def resolve(block: dict[str, Any] | None = VEO_BLOCK, *, spec: ProviderSpec = SPEC,
            **kwargs: Any) -> VideoOptions:
    inputs = VideoInputs(**{k: v for k, v in kwargs.items() if k in VIDEO_ROLES})
    options = VideoOptions(**{k: v for k, v in kwargs.items() if k not in VIDEO_ROLES})
    model_entry = None if block is None else entry(block)
    return resolve_request(spec, "veo-test", model_entry, inputs, options)


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
class TestSniffVideoMime:
    def test_mp4(self) -> None:
        assert sniff_video_mime(MP4) == "video/mp4"

    def test_quicktime_by_brand(self) -> None:
        assert sniff_video_mime(b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 8) == "video/quicktime"

    def test_webm(self) -> None:
        assert sniff_video_mime(b"\x1a\x45\xdf\xa3" + b"\x00" * 8) == "video/webm"

    def test_ftyp_must_be_at_offset_four(self) -> None:
        # The trap: a prefix match finds nothing, a search anywhere finds false positives.
        assert sniff_video_mime(b"ftypisom" + b"\x00" * 8) == "application/octet-stream"

    def test_unknown_is_not_labelled_as_video(self) -> None:
        assert sniff_video_mime(b"RIFF....WAVE") == "application/octet-stream"


class TestInputTypes:
    def test_forms(self) -> None:
        assert VideoInput(uri="files/abc").form == "uri"
        assert VideoInput.from_bytes(MP4).form == "bytes"

    def test_from_bytes_sniffs_the_type(self) -> None:
        assert VideoInput.from_bytes(MP4).mime_type == "video/mp4"

    def test_from_result_uses_the_providers_handle(self) -> None:
        result = MediaResult(data=MP4, mime_type="video/mp4", source_uri="files/abc")
        assert VideoInput.from_result(result) == VideoInput(uri="files/abc")

    def test_from_result_without_a_handle_refuses(self) -> None:
        with pytest.raises(ValueError, match="no provider handle"):
            VideoInput.from_result(MediaResult(data=MP4, mime_type="video/mp4"))

    def test_roles_in_canonical_order(self) -> None:
        inputs = VideoInputs(extend=VideoInput(uri="u"), first_frame=FRAME)
        assert inputs.roles() == ("first_frame", "extend")
        assert VideoInputs().roles() == ()

    def test_options_truthiness_and_requested(self) -> None:
        assert not VideoOptions()
        options = VideoOptions(duration_seconds=8, seed=0)
        assert options
        assert options.requested() == {"duration_seconds": 8, "seed": 0}


# --------------------------------------------------------------------------- #
# Catalog parsing
# --------------------------------------------------------------------------- #
class TestCatalogBlock:
    def test_parses_the_designed_block(self) -> None:
        caps = entry(VEO_BLOCK).video
        assert caps is not None
        assert caps.inputs == ("first_frame", "last_frame", "references", "extend")
        assert caps.durations == (4, 6, 8)
        assert caps.audio == "always"
        assert len(caps.constraints) == 6
        exclusion = caps.constraints[1]
        assert exclusion.when == "references"
        assert exclusion.exclude_input == ("first_frame", "last_frame")

    def test_no_block_means_not_described(self) -> None:
        assert entry(None).video is None

    def test_missing_inputs_is_not_the_same_as_empty(self) -> None:
        assert entry({}).video is not None and entry({}).video.inputs is None  # type: ignore[union-attr]
        assert entry({"inputs": []}).video.inputs == ()  # type: ignore[union-attr]

    @pytest.mark.parametrize(("block", "message"), [
        ({"durations": [4, "8"]}, "durations"),
        ({"durations": [True]}, "durations"),
        ({"aspect_ratios": ["wide"]}, "not a W:H ratio"),
        ({"audio": "sometimes"}, "video.audio"),
        ({"max_references": -1}, "max_references"),
        ({"constraints": [{"require": {"seed": 1}}]}, "string 'when'"),
        ({"constraints": [{"when": "extend", "require": {}}]}, "map settings"),
    ])
    def test_malformed_data_raises_naming_the_model(self, block: dict[str, Any],
                                                     message: str) -> None:
        with pytest.raises(ValueError, match=message) as err:
            entry(block, model="broken-model")
        assert "broken-model" in str(err.value)

    @pytest.mark.parametrize("row", [
        {"when": "references", "forbid": ["first_frame"]},                 # unknown verb
        {"when": "references", "require_input": "first_frame",
         "exclude_input": ["last_frame"]},                                 # two verbs
        {"when": "start_frame", "require_input": "first_frame"},           # unknown trigger
        {"when": "references", "require": {"fps": 24}},                    # unknown setting
        {"when": "references", "exclude_input": ["start_frame"]},          # unknown role
    ])
    def test_unknown_vocabulary_is_warned_and_skipped(
        self, row: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        # Tolerated so a catalog from a newer version still loads -- but never silently,
        # because a skipped row is a rule not enforced.
        with caplog.at_level(logging.WARNING):
            caps = entry({"constraints": [row]}).video
        assert caps is not None and caps.constraints == ()
        assert "ignored" in caplog.text

    def test_unknown_input_role_is_dropped_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            caps = entry({"inputs": ["first_frame", "keyframes"]}).video
        assert caps is not None and caps.inputs == ("first_frame",)
        assert "keyframes" in caplog.text


# --------------------------------------------------------------------------- #
# Layer 1: structure
# --------------------------------------------------------------------------- #
class TestStructure:
    @pytest.mark.parametrize(("kwargs", "message"), [
        ({"duration_seconds": True}, "positive whole number"),
        ({"duration_seconds": 0}, "positive whole number"),
        ({"duration_seconds": 8.0}, "positive whole number"),
        ({"seed": "42"}, "seed must be an int"),
        ({"generate_audio": "yes"}, "True or False"),
        ({"aspect_ratio": "wide"}, "W:H"),
        ({"resolution": " "}, "non-empty string"),
    ])
    def test_bad_values_are_value_errors(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            resolve(None, **kwargs)

    def test_extend_must_be_exactly_one_form(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            resolve(None, extend=VideoInput(data=MP4, mime_type="video/mp4", uri="files/a"))
        with pytest.raises(ValueError, match="exactly one"):
            resolve(None, extend=VideoInput())

    def test_extend_bytes_must_be_a_known_container(self) -> None:
        spec = ProviderSpec(name="b", client_path="x:Y", modalities=frozenset({VIDEO}),
                            video_input_forms=("bytes",))
        with pytest.raises(ValueError, match="unsupported type"):
            resolve(None, spec=spec, extend=VideoInput(data=b"junk", mime_type="video/avi"))

    def test_frames_must_be_supported_images(self) -> None:
        with pytest.raises(ValueError, match="unsupported type"):
            resolve(None, first_frame=ImageInput(data=b"x", mime_type="image/tiff"))
        with pytest.raises(ValueError, match="must be an ImageInput"):
            resolve(None, first_frame=b"raw bytes")


# --------------------------------------------------------------------------- #
# Layer 2: the client's capability
# --------------------------------------------------------------------------- #
class TestClientCapability:
    def test_a_service_without_video_refuses_before_anything_else(self) -> None:
        spec = ProviderSpec(name="text-only", client_path="x:Y")
        with pytest.raises(UnsupportedModalityError):
            resolve(VEO_BLOCK, spec=spec, duration_seconds=True)   # even with a bad value

    def test_bytes_extend_on_a_handle_only_service_says_why(self) -> None:
        with pytest.raises(UnsupportedVideoInputError, match="handle to a video") as err:
            resolve(None, extend=VideoInput.from_bytes(MP4))
        assert "extend=result" in str(err.value)

    def test_a_service_with_no_video_input_refuses_any(self) -> None:
        spec = ProviderSpec(name="no-input", client_path="x:Y", modalities=frozenset({VIDEO}))
        with pytest.raises(UnsupportedVideoInputError, match="cannot send a video input"):
            resolve(None, spec=spec, extend=VideoInput(uri="files/a"))

    def test_a_handle_is_accepted(self) -> None:
        resolve(None, extend=VideoInput(uri="files/a"))


class TestServiceSettings:
    """ProviderSpec.video_settings: what the service's API can carry at all."""

    CLOSED = ProviderSpec(
        name="closed-video", client_path="x:Y", modalities=frozenset({VIDEO}),
        video_settings=("resolution", "duration_seconds"),
    )

    def test_none_means_everything_passes_through(self) -> None:
        assert SPEC.video_settings is None
        assert resolve(None, seed=3).seed == 3

    def test_a_setting_outside_the_list_is_refused_for_an_unlisted_model(self) -> None:
        with pytest.raises(UnsupportedVideoSettingError, match="has no seed parameter") as err:
            resolve(None, spec=self.CLOSED, seed=3)
        assert "Settings it takes: resolution, duration_seconds" in str(err.value)

    def test_it_comes_before_the_models_own_rules(self) -> None:
        # The service cannot carry it at all, whatever the model would say.
        with pytest.raises(UnsupportedVideoSettingError, match="has no generate_audio"):
            resolve(VEO_BLOCK, spec=self.CLOSED, generate_audio=False)

    def test_a_setting_the_catalog_forces_is_checked_too(self) -> None:
        # references force aspect_ratio 16:9, which this service cannot send: refused,
        # never silently dropped on the way out.
        with pytest.raises(UnsupportedVideoSettingError, match="has no aspect_ratio"):
            resolve(VEO_BLOCK, spec=self.CLOSED, references=(FRAME,))

    def test_listed_settings_pass(self) -> None:
        assert resolve(VEO_BLOCK, spec=self.CLOSED, resolution="4k").duration_seconds == 8


# --------------------------------------------------------------------------- #
# Layer 3: a model the catalog does not describe
# --------------------------------------------------------------------------- #
class TestUnlistedModel:
    def test_end_frame_without_start_frame_is_refused(self) -> None:
        with pytest.raises(UnsupportedVideoInputError, match="last_frame needs first_frame"):
            resolve(None, last_frame=FRAME)

    def test_everything_else_passes_through_unchanged(self) -> None:
        # A model released after this package must keep working: no capability is
        # assumed, so the provider gets to judge -- and nothing is filled in either.
        options = resolve(None, references=(FRAME,) * 5, first_frame=FRAME,
                          resolution="8k", duration_seconds=30, aspect_ratio="21:9")
        assert options == VideoOptions(resolution="8k", duration_seconds=30,
                                       aspect_ratio="21:9")


# --------------------------------------------------------------------------- #
# Layer 3: a model the catalog describes
# --------------------------------------------------------------------------- #
class TestStatedInputs:
    def test_a_role_the_model_does_not_take(self) -> None:
        with pytest.raises(UnsupportedVideoInputError, match="does not take references"):
            resolve(LITE_BLOCK, references=(FRAME,))

    def test_text_only_model_says_so(self) -> None:
        with pytest.raises(UnsupportedVideoInputError, match="a text prompt only"):
            resolve({"inputs": []}, first_frame=FRAME)

    def test_reference_cap(self) -> None:
        with pytest.raises(TooManyImagesError):
            resolve(references=(FRAME,) * 4)


class TestExclusion:
    """The probe result: Veo 3.1 rejects references with frames using a generic 400, so
    this refusal is the only place a caller learns what was wrong."""

    @pytest.mark.parametrize("frames", [
        {"first_frame": FRAME},
        {"first_frame": FRAME, "last_frame": FRAME},
    ])
    def test_references_with_frames_is_refused_with_both_ways_out(
        self, frames: dict[str, Any]
    ) -> None:
        with pytest.raises(UnsupportedVideoInputError) as err:
            resolve(references=(FRAME,), **frames)
        message = str(err.value)
        assert "cannot combine references with first_frame" in message
        assert "Use references alone" in message
        assert "without references" in message

    def test_references_alone_is_fine(self) -> None:
        resolve(references=(FRAME, FRAME))

    def test_a_stated_end_frame_rule(self) -> None:
        with pytest.raises(UnsupportedVideoInputError, match="last_frame needs first_frame"):
            resolve(last_frame=FRAME)


class TestForcedSettings:
    def test_unset_forced_settings_are_filled(self) -> None:
        options = resolve(references=(FRAME,))
        assert (options.duration_seconds, options.aspect_ratio) == (8, "16:9")

    def test_a_different_value_is_refused_not_overridden(self) -> None:
        with pytest.raises(UnsupportedVideoSettingError,
                           match="When references are supplied, duration_seconds must be 8"):
            resolve(references=(FRAME,), duration_seconds=4)

    def test_resolution_rules(self) -> None:
        assert resolve(resolution="1080p").duration_seconds == 8
        with pytest.raises(UnsupportedVideoSettingError, match="At resolution 4k"):
            resolve(resolution="4k", duration_seconds=6)

    def test_a_filled_value_can_switch_on_another_rule(self) -> None:
        block = {
            "constraints": [
                {"when": "extend", "require": {"resolution": "720p"}},
                {"when": "resolution:720p", "require": {"duration_seconds": 8}},
            ],
        }
        options = resolve(block, extend=VideoInput(uri="files/a"))
        assert (options.resolution, options.duration_seconds) == ("720p", 8)

    def test_two_rules_that_disagree_are_reported_as_such(self) -> None:
        block = {
            "constraints": [
                {"when": "references", "require": {"duration_seconds": 8}},
                {"when": "resolution:720p", "require": {"duration_seconds": 4}},
            ],
        }
        with pytest.raises(UnsupportedVideoSettingError,
                           match="another constraint already requires"):
            resolve(block, references=(FRAME,), resolution="720p")

    def test_nothing_is_filled_when_nothing_is_active(self) -> None:
        assert resolve() == VideoOptions()


class TestOfferedSettings:
    def test_resolution_not_offered_lists_what_is(self) -> None:
        with pytest.raises(UnsupportedVideoSettingError) as err:
            resolve(LITE_BLOCK, resolution="4k", duration_seconds=8)
        assert "It offers: 720p, 1080p" in str(err.value)
        assert err.value.supported == ("720p", "1080p")

    def test_aspect_ratio_matches_by_value_not_spelling(self) -> None:
        resolve(aspect_ratio="9 : 16")
        with pytest.raises(UnsupportedVideoSettingError, match="aspect_ratio"):
            resolve(aspect_ratio="1:1")

    def test_duration_not_offered(self) -> None:
        with pytest.raises(UnsupportedVideoSettingError, match="duration_seconds"):
            resolve(duration_seconds=5)

    def test_always_audio_refuses_silence(self) -> None:
        with pytest.raises(UnsupportedVideoSettingError, match="always produces audio"):
            resolve(generate_audio=False)
        resolve(generate_audio=True)

    def test_never_audio_refuses_sound(self) -> None:
        with pytest.raises(UnsupportedVideoSettingError, match="produces no audio"):
            resolve({"audio": "never"}, generate_audio=True)

    def test_errors_are_value_errors(self) -> None:
        # Values the caller could have chosen differently: the 400 side of an HTTP layer.
        assert issubclass(UnsupportedVideoSettingError, ValueError)
        assert not issubclass(UnsupportedVideoInputError, ValueError)


# --------------------------------------------------------------------------- #
# Through the client and the factory
# --------------------------------------------------------------------------- #
class FakeVideoClient(ModelClient[Any]):
    SPEC = SPEC

    def _build_client(self) -> Any:
        return object()

    def _invoke(self, system: str, user: str, history: Any, images: Any) -> LLMResult:
        raise AssertionError("no text calls in these tests")


def catalog_file(tmp_path: Path, block: dict[str, Any]) -> Path:
    path = tmp_path / "models.json"
    path.write_text(json.dumps([
        {"id": "veo-test", "service": SPEC.name, "modality": "video", "video": block},
    ]), encoding="utf-8")
    return path


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_VIDEO_KEY", "k")


class TestClientHook:
    @pytest.mark.usefixtures("key")
    @pytest.mark.parametrize("services", [None, ("fake-video",), ("some-other-service",)])
    def test_listing_filter_never_changes_enforcement(
        self, tmp_path: Path, services: tuple[str, ...] | None
    ) -> None:
        # A UI that filters this model out of its dropdown must not thereby let a call
        # through that the model would refuse (DESIGN.md section 6).
        config = ModelConfig(catalog_path=catalog_file(tmp_path, LITE_BLOCK),
                             services=services)
        client = FakeVideoClient(model="veo-test", config=config)
        with pytest.raises(UnsupportedVideoSettingError, match="resolution"):
            client._resolve_video_request(
                VideoInputs(), VideoOptions(resolution="4k", duration_seconds=8)
            )


class TestFactory:
    @pytest.fixture(autouse=True)
    def registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Registered for the test only; monkeypatch removes it afterwards.
        monkeypatch.setitem(DEFAULT_REGISTRY._specs, SPEC.name, SPEC)

    def test_judges_without_a_credential_or_a_client(self, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FAKE_VIDEO_KEY", raising=False)
        config = ModelConfig(catalog_path=catalog_file(tmp_path, VEO_BLOCK))
        options = resolve_video_request(
            "fake-video", "veo-test", VideoInputs(references=(FRAME,)), config=config
        )
        assert options.duration_seconds == 8
        with pytest.raises(UnsupportedVideoInputError, match="cannot combine"):
            resolve_video_request(
                "fake-video", "veo-test",
                VideoInputs(references=(FRAME,), first_frame=FRAME), config=config,
            )

    def test_a_text_only_service_refuses_video(self) -> None:
        # A service that does not implement the seam is refused before the network.
        with pytest.raises(UnsupportedModalityError):
            resolve_video_request("anthropic", "claude-sonnet-5")


def test_source_uri_reaches_the_trace(tmp_path: Path) -> None:
    tracer = JsonlTraceLogger(tmp_path / "trace.jsonl", run_id="t")
    tracer.llm_call(
        service="fake-video", model="veo-test", modality=VIDEO, latency_ms=1.0, request={},
        result=MediaResult(data=MP4, mime_type="video/mp4", source_uri="files/abc"),
        error=None,
    )
    record = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert record["response"]["produced"]["source_uri"] == "files/abc"
