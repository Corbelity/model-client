"""The native Gemini provider (`gemini-native`), video through Veo.

The first part runs with no SDK installed: the fake overrides `_build_client` and
`_sdk_types`, and operations are SimpleNamespaces shaped like google-genai's. The last
class DOES use the SDK, and skips without it: it drives the provider through a real
`google.genai.Client` with only the HTTP layer replaced, so the request body, the
operation parsing and the download path are the SDK's own -- the check that a fake cannot
make (DESIGN.md section 14).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from corbelity.model_client import (
    ContentFilteredError,
    ImageInput,
    JsonlTraceLogger,
    MediaResult,
    ModelClientError,
    UnsupportedVideoInputError,
    UnsupportedVideoSettingError,
    VideoInput,
    VideoInputs,
    VideoJobFailedError,
    VideoJobNotFoundError,
    VideoOptions,
    builtin_catalog,
    resolve_video_request,
)
from corbelity.model_client.providers.gemini_native import GeminiNativeClient

PNG = b"\x89PNG\r\n\x1a\n" + b"body"
FRAME = ImageInput.from_bytes(PNG)
END = ImageInput.from_bytes(PNG + b"end")
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16
VEO = "veo-3.1-generate-preview"
LITE = "veo-3.1-lite-generate-preview"
OPERATION = f"models/{VEO}/operations/op-1"
VIDEO_URI = "https://generativelanguage.googleapis.com/v1beta/files/abc123:download?alt=media"


def finished(*videos: Any, error: Any = None, reasons: list[str] | None = None,
             count: int | None = None) -> SimpleNamespace:
    """A done operation, shaped like GenerateVideosOperation."""
    return SimpleNamespace(
        name=OPERATION, done=True, error=error,
        response=SimpleNamespace(
            generated_videos=[SimpleNamespace(video=video) for video in videos],
            rai_media_filtered_reasons=reasons,
            rai_media_filtered_count=count,
        ),
    )


def video(uri: str | None = VIDEO_URI, data: bytes | None = None,
          mime_type: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(uri=uri, video_bytes=data, mime_type=mime_type)


class _Models:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.name: str | None = OPERATION

    def generate_videos(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(name=self.name, done=False)


class _Operations:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.next: Any = SimpleNamespace(name=OPERATION, done=False)

    def get(self, operation: Any) -> Any:
        self.asked.append(operation.name)
        if isinstance(self.next, Exception):
            raise self.next
        return self.next


class _Files:
    def __init__(self) -> None:
        self.downloaded: list[Any] = []
        self.data = MP4
        self.fail: Exception | None = None

    def download(self, *, file: Any) -> bytes:
        self.downloaded.append(file)
        if self.fail is not None:
            raise self.fail
        return self.data


class FakeVeo(GeminiNativeClient):
    """Overrides _build_client and _sdk_types: no SDK is imported."""

    def _build_client(self) -> Any:
        self.models = _Models()
        self.operations = _Operations()
        self.files = _Files()
        return SimpleNamespace(models=self.models, operations=self.operations,
                               files=self.files)

    def _sdk_types(self) -> Any:
        return SimpleNamespace(
            GenerateVideosOperation=lambda name: SimpleNamespace(name=name),
            VideoGenerationReferenceType=SimpleNamespace(ASSET="ASSET"),
        )


class NotFound(Exception):
    """The SDK's ClientError carries the HTTP status as `.code`; so does this."""

    code = 404


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_NATIVE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")


def build(model: str = VEO, **kwargs: Any) -> FakeVeo:
    return FakeVeo(model=model, **kwargs)


def submitted(client: FakeVeo) -> dict[str, Any]:
    assert client.models.calls, "nothing was submitted"
    return client.models.calls[-1]


# --------------------------------------------------------------------------- #
# Submit
# --------------------------------------------------------------------------- #
class TestSubmit:
    def test_text_to_video(self) -> None:
        client = build()
        job = client.submit_video("a lighthouse at dusk")
        call = submitted(client)
        assert call["model"] == VEO
        assert call["source"] == {"prompt": "a lighthouse at dusk"}
        # Nothing requested, so no config at all: the API's defaults apply.
        assert call["config"] is None
        assert job.operation == OPERATION

    def test_frames_map_to_image_and_last_frame(self) -> None:
        client = build()
        client.submit_video("pan left", first_frame=FRAME, last_frame=END)
        call = submitted(client)
        assert call["source"]["image"] == {"image_bytes": PNG, "mime_type": "image/png"}
        assert call["config"]["last_frame"] == {
            "image_bytes": END.data, "mime_type": "image/png",
        }

    def test_only_requested_settings_are_sent(self) -> None:
        client = build()
        client.submit_video("x", aspect_ratio="9:16", negative_prompt="text overlays",
                            person_generation="allow_adult")
        assert submitted(client)["config"] == {
            "aspect_ratio": "9:16", "negative_prompt": "text overlays",
            "person_generation": "allow_adult",
        }

    def test_a_setting_the_catalog_forces_is_sent(self) -> None:
        client = build()
        job = client.submit_video("x", resolution="1080p")
        assert submitted(client)["config"] == {"resolution": "1080p", "duration_seconds": 8}
        assert job.to_ref().requested == {"resolution": "1080p", "duration_seconds": 8}

    def test_a_blank_prompt_is_omitted_not_sent_empty(self) -> None:
        client = build()
        client.submit_video("  ", first_frame=FRAME)
        assert "prompt" not in submitted(client)["source"]

    def test_every_role_is_mapped_for_a_model_the_catalog_does_not_describe(self) -> None:
        # A role that passed validation must never be dropped on the way out.
        client = build(model="veo-next")
        client.submit_video("x", references=[FRAME, END])
        refs = submitted(client)["config"]["reference_images"]
        assert [ref["reference_type"] for ref in refs] == ["ASSET", "ASSET"]
        assert refs[1]["image"]["image_bytes"] == END.data

        client.submit_video("longer", extend=VideoInput(uri=VIDEO_URI))
        assert submitted(client)["source"]["video"] == {"uri": VIDEO_URI}

    def test_extend_accepts_an_earlier_result(self) -> None:
        client = build(model="veo-next")
        earlier = MediaResult(data=MP4, mime_type="video/mp4", source_uri=VIDEO_URI)
        client.submit_video("and then", extend=earlier)
        assert submitted(client)["source"]["video"] == {"uri": VIDEO_URI}

    @pytest.mark.parametrize("setting", [{"seed": 7}, {"generate_audio": True}])
    def test_settings_this_api_lacks_are_refused_before_the_network(
            self, setting: dict[str, Any]) -> None:
        client = build()
        with pytest.raises(UnsupportedVideoSettingError, match="has no .* parameter") as err:
            client.submit_video("x", **setting)
        assert "'gemini-native'" in str(err.value)
        assert client.models.calls == []

    def test_phase_one_catalog_refuses_references_on_veo(self) -> None:
        client = build()
        with pytest.raises(UnsupportedVideoInputError, match="does not take references"):
            client.submit_video("x", references=[FRAME])
        assert client.models.calls == []

    def test_no_operation_name_is_an_error_not_a_job(self) -> None:
        client = build()
        client.models.name = None
        with pytest.raises(ModelClientError, match="no operation name"):
            client.submit_video("x")

    def test_the_submit_record_carries_the_operation(self, tmp_path: Path) -> None:
        tracer = JsonlTraceLogger(tmp_path / "trace.jsonl", run_id="t")
        client = build(trace=tracer)
        client.submit_video("x", first_frame=FRAME)
        record = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
        assert record["request"]["job"]["operation"] == OPERATION
        assert record["finish_reason"] == "submitted"


# --------------------------------------------------------------------------- #
# Poll
# --------------------------------------------------------------------------- #
class TestPoll:
    def test_not_done_is_running_with_no_invented_progress(self) -> None:
        client = build()
        job = client.submit_video("x")
        status = job.poll()
        assert (status.state, status.progress, status.done) == ("running", None, False)
        assert client.operations.asked == [OPERATION]

    def test_success_downloads_once_and_keeps_the_handle(self) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = finished(video(mime_type="video/mp4"))
        assert job.poll().state == "succeeded"
        result = job.result()
        assert result.data == MP4
        assert result.mime_type == "video/mp4"
        assert result.source_uri == VIDEO_URI
        job.poll()
        assert len(client.files.downloaded) == 1
        assert client.operations.asked == [OPERATION]    # finished jobs stop asking

    def test_a_provider_error_is_a_failed_job(self) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = finished(error={"code": 3, "message": "bad request"})
        status = job.poll()
        assert status.state == "failed"
        assert status.error == "bad request (code 3)"
        with pytest.raises(VideoJobFailedError, match="bad request"):
            job.result()

    def test_a_safety_refusal_is_filtered_with_its_reasons(self) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = finished(reasons=["celebrity likeness"], count=1)
        status = job.poll()
        assert status.state == "filtered"
        assert status.filtered_reasons == ("celebrity likeness",)
        with pytest.raises(ContentFilteredError, match="celebrity likeness"):
            job.result()

    def test_a_count_without_reasons_is_still_filtered(self) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = finished(count=1)
        assert job.poll().state == "filtered"

    def test_done_with_nothing_is_a_failure_that_says_so(self) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = finished()
        status = job.poll()
        assert status.state == "failed"
        assert "without a video" in (status.error or "")

    def test_404_means_the_job_is_gone(self) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = NotFound("not found")
        with pytest.raises(VideoJobNotFoundError, match="expired or never existed"):
            job.poll()

    def test_any_other_error_propagates_and_the_job_stays_pollable(self) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = ConnectionError("reset")
        with pytest.raises(ConnectionError):
            job.poll()
        client.operations.next = finished(video())
        assert job.poll().state == "succeeded"

    def test_a_job_resumed_from_its_name_alone_is_polled_by_that_name(self) -> None:
        # A second process: never submitted anything, has only the stored id.
        client = build()
        client.operations.next = finished(video())
        job = client.resume_video(OPERATION)
        assert job.result().data == MP4
        assert client.operations.asked == [OPERATION]
        assert client.models.calls == []

    def test_more_videos_than_asked_for_keeps_the_first_and_says_so(
            self, caplog: pytest.LogCaptureFixture) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = finished(video(uri="files/first"), video(uri="files/second"))
        with caplog.at_level(logging.WARNING):
            result = job.result()
        assert result.source_uri == "files/first"
        assert "returned 2 videos" in caplog.text


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
class TestFetch:
    def test_inline_bytes_are_used_without_a_download(self) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = finished(video(uri=None, data=MP4))
        result = job.result()
        assert result.data == MP4
        assert result.source_uri is None      # nothing to extend from, and it says so
        assert client.files.downloaded == []

    def test_the_bytes_decide_the_type_over_the_label(self) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = finished(video(mime_type="video/webm"))
        assert job.result().mime_type == "video/mp4"

    def test_the_label_is_used_when_the_bytes_are_not_recognised(self) -> None:
        client = build()
        client.files.data = b"not a container"
        job = client.submit_video("x")
        client.operations.next = finished(video(mime_type="video/webm"))
        assert job.result().mime_type == "video/webm"

    def test_a_failed_download_leaves_the_job_to_try_again(self) -> None:
        client = build()
        job = client.submit_video("x")
        client.operations.next = finished(video())
        client.files.fail = ConnectionError("reset mid-download")
        with pytest.raises(ConnectionError):
            job.poll()
        assert job.status is None or not job.status.done
        client.files.fail = None
        assert job.result().data == MP4
        assert len(client.files.downloaded) == 2

    def test_the_terminal_record_carries_the_video(self, tmp_path: Path) -> None:
        tracer = JsonlTraceLogger(tmp_path / "trace.jsonl", run_id="t")
        client = build(trace=tracer)
        job = client.submit_video("x")
        client.operations.next = finished(video())
        job.result()
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        terminal = json.loads(lines[-1])
        assert terminal["request"]["job"]["phase"] == "terminal"
        assert terminal["response"]["produced"]["source_uri"] == VIDEO_URI


# --------------------------------------------------------------------------- #
# The built-in catalog's Veo entries
# --------------------------------------------------------------------------- #
class TestCatalogEntries:
    def test_both_models_are_listed_as_video_on_the_native_service(self) -> None:
        entries = {entry.id: entry for entry in builtin_catalog()}
        for model in (VEO, LITE):
            assert entries[model].service == "gemini-native"
            assert entries[model].modality == "video"
            assert entries[model].video is not None

    def test_judged_without_a_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        options = resolve_video_request(
            "gemini-native", VEO, VideoInputs(first_frame=FRAME),
            VideoOptions(resolution="4k"),
        )
        assert options.duration_seconds == 8

    def test_lite_has_no_4k(self) -> None:
        with pytest.raises(UnsupportedVideoSettingError, match="It offers: 720p, 1080p"):
            resolve_video_request("gemini-native", LITE, options=VideoOptions(resolution="4k"))

    def test_an_end_frame_needs_a_start_frame(self) -> None:
        with pytest.raises(UnsupportedVideoInputError, match="last_frame needs first_frame"):
            resolve_video_request("gemini-native", LITE, VideoInputs(last_frame=END))

    def test_veo_always_has_audio(self) -> None:
        # Refused by the service layer first: the Gemini API has no switch for it.
        with pytest.raises(UnsupportedVideoSettingError, match="generate_audio"):
            resolve_video_request("gemini-native", VEO,
                                  options=VideoOptions(generate_audio=False))


# --------------------------------------------------------------------------- #
# Through the real SDK, with only the HTTP layer replaced (skipped without the extra)
# --------------------------------------------------------------------------- #
class _Http:
    """Stands in for the SDK's API client transport: records every request and answers
    with the JSON the Gemini API would send."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.downloads: list[str] = []
        self.poll_body: dict[str, Any] = {"name": OPERATION, "done": False}
        self.poll_error: Exception | None = None

    def request(self, method: str, path: str, body: dict[str, Any],
                http_options: Any = None) -> Any:
        self.requests.append((method, path, body))
        if method == "post":
            return SimpleNamespace(body=json.dumps({"name": OPERATION}), headers={})
        if self.poll_error is not None:
            raise self.poll_error
        return SimpleNamespace(body=json.dumps(self.poll_body), headers={})

    def download_file(self, path: str, *, http_options: Any = None,
                      destination: Any = None) -> bytes:
        self.downloads.append(path)
        return MP4


class TestThroughTheSdk:
    @pytest.fixture(autouse=True)
    def genai(self) -> Any:
        return pytest.importorskip("google.genai")

    @pytest.fixture
    def http(self) -> _Http:
        return _Http()

    def sdk_client(self, http: _Http, model: str = VEO) -> GeminiNativeClient:
        client = GeminiNativeClient(model=model)
        api = client._client._api_client
        api.request = http.request
        api.download_file = http.download_file
        return client

    def test_the_request_body_the_sdk_sends(self, http: _Http) -> None:
        client = self.sdk_client(http)
        client.submit_video("a cat", first_frame=FRAME, last_frame=END,
                            resolution="1080p", negative_prompt="dogs")
        method, path, body = http.requests[0]
        assert (method, path) == ("post", f"models/{VEO}:predictLongRunning")
        instance = body["instances"][0]
        assert instance["prompt"] == "a cat"
        assert instance["image"]["mimeType"] == "image/png"
        assert "lastFrame" in instance
        assert body["parameters"] == {
            "durationSeconds": 8, "resolution": "1080p", "negativePrompt": "dogs",
        }

    def test_references_and_extend_reach_the_body(self, http: _Http) -> None:
        client = self.sdk_client(http, model="veo-next")
        client.submit_video("x", references=[FRAME])
        instance = http.requests[-1][2]["instances"][0]
        assert instance["referenceImages"][0]["referenceType"] == "ASSET"
        client.submit_video("x", extend=VideoInput(uri=VIDEO_URI))
        assert http.requests[-1][2]["instances"][0]["video"]["uri"] == VIDEO_URI

    def test_a_finished_operation_parses_and_downloads(self, http: _Http) -> None:
        client = self.sdk_client(http)
        job = client.submit_video("x")
        http.poll_body = {
            "name": OPERATION, "done": True,
            "response": {"generateVideoResponse": {
                "generatedSamples": [{"video": {"uri": VIDEO_URI}}],
            }},
        }
        result = job.result()
        assert (result.data, result.mime_type, result.source_uri) == (
            MP4, "video/mp4", VIDEO_URI,
        )
        assert http.requests[-1][:2] == ("get", OPERATION)

    def test_the_download_never_goes_to_the_host_in_the_response(self, http: _Http) -> None:
        # Only the file id is taken from the URI; the request goes through the client's
        # own transport, base URL and key. A tampered URI cannot steer the key.
        client = self.sdk_client(http)
        job = client.submit_video("x")
        http.poll_body = {
            "name": OPERATION, "done": True,
            "response": {"generateVideoResponse": {"generatedSamples": [
                {"video": {"uri": "https://attacker.example/v1beta/files/abc123:download"}},
            ]}},
        }
        job.result()
        assert http.downloads == ["files/abc123:download?alt=media"]

    def test_a_filtered_operation(self, http: _Http) -> None:
        client = self.sdk_client(http)
        job = client.submit_video("x")
        http.poll_body = {
            "name": OPERATION, "done": True,
            "response": {"generateVideoResponse": {
                "raiMediaFilteredCount": 1, "raiMediaFilteredReasons": ["blocked"],
            }},
        }
        assert job.poll().filtered_reasons == ("blocked",)

    def test_the_sdks_404_is_job_not_found(self, http: _Http, genai: Any) -> None:
        client = self.sdk_client(http)
        job = client.submit_video("x")
        http.poll_error = genai.errors.ClientError(
            404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}}
        )
        with pytest.raises(VideoJobNotFoundError):
            job.poll()

    @pytest.mark.parametrize("setting", [{"seed": 1}, {"generate_audio": True}])
    def test_the_sdk_still_refuses_what_video_settings_leaves_out(
            self, http: _Http, setting: dict[str, Any]) -> None:
        # The reason those two are missing from the spec's video_settings. If this starts
        # passing, the API has gained the parameter: add it to the spec.
        client = self.sdk_client(http)
        with pytest.raises(ValueError, match="not in Gemini Developer API"):
            client._client.models.generate_videos(
                model=VEO, source={"prompt": "x"}, config=setting,
            )
        assert http.requests == []
