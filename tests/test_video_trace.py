"""Trace artifacts for video: inputs by role with the submit record, the video with the
terminal record, the video_artifacts switch, and no payload ever inline in the JSONL.

End to end through a fake provider and a real JsonlTraceLogger, so what is checked is the
record and the files actually on disk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from corbelity.model_client import (
    VIDEO,
    ImageInput,
    JsonlTraceLogger,
    MediaResult,
    ProviderSpec,
    VideoInput,
    VideoInputs,
    VideoOptions,
    VideoPoll,
)
from corbelity.model_client.client import ModelClient
from corbelity.model_client.media import LLMResult
from corbelity.model_client.trace import _suffix_for

PNG_A = b"\x89PNG\r\n\x1a\n" + b"first-frame"
PNG_B = b"\x89PNG\r\n\x1a\n" + b"last-frame"
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
BIG_MP4 = b"\x00\x00\x00\x18ftypisom" + b"\xab" * 200_000


def spec(forms: tuple[str, ...]) -> ProviderSpec:
    return ProviderSpec(name="fake-video", client_path="x:Y", key_env=("FAKE_VIDEO_KEY",),
                        modalities=frozenset({VIDEO}), video_input_forms=forms)


class FakeVideoClient(ModelClient[Any]):
    SPEC = spec(("uri",))

    def _build_client(self) -> Any:
        return object()

    def _invoke(self, system: str, user: str, history: Any, images: Any) -> LLMResult:
        raise AssertionError("no text calls")

    def _submit_video(self, prompt: str, inputs: VideoInputs, options: VideoOptions) -> str:
        return "operations/7"

    def _poll_video(self, operation: str) -> VideoPoll:
        return VideoPoll(state="succeeded", output="h")

    def _fetch_video(self, poll: VideoPoll) -> MediaResult:
        return MediaResult(data=MP4, mime_type="video/mp4", source_uri="files/h")


class BytesInputClient(FakeVideoClient):
    SPEC = spec(("bytes", "uri"))


def run(tmp_path: Path, client_class: type[FakeVideoClient] = FakeVideoClient,
        *, video_artifacts: bool = True, **request: Any) -> tuple[list[dict[str, Any]], Path]:
    tracer = JsonlTraceLogger(tmp_path / "trace.jsonl", run_id="r",
                              video_artifacts=video_artifacts)
    client = client_class(model="veo-test", trace=tracer)
    client.submit_video("p", **request).result()
    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines], tmp_path / "artifacts"


def files(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


class TestArtifacts:
    def test_text_to_video_writes_one_video_with_the_terminal_record(self,
                                                                     tmp_path: Path) -> None:
        (submit, terminal), artifacts = run(tmp_path)
        assert submit["request"]["job"]["operation"] == terminal["request"]["job"]["operation"]
        assert submit["response"] is None              # the submission carries no payload
        artifact = terminal["response"]["artifact"]
        assert artifact == {"name": "r-002.mp4", "mime_type": "video/mp4", "bytes": len(MP4)}
        assert files(artifacts) == ["r-002.mp4"]
        assert (artifacts / "r-002.mp4").read_bytes() == MP4

    def test_frames_are_written_by_role_with_the_submit_record(self, tmp_path: Path) -> None:
        (submit, _), artifacts = run(
            tmp_path,
            first_frame=ImageInput(data=PNG_A, mime_type="image/png", name="start.png"),
            last_frame=ImageInput.from_bytes(PNG_B),
        )
        inputs = submit["request"]["inputs"]
        assert inputs["first_frame"]["name"] == "r-001-first.png"
        assert inputs["first_frame"]["source_name"] == "start.png"
        assert inputs["last_frame"]["name"] == "r-001-last.png"
        assert (artifacts / "r-001-first.png").read_bytes() == PNG_A
        assert (artifacts / "r-001-last.png").read_bytes() == PNG_B

    def test_references_are_numbered(self, tmp_path: Path) -> None:
        (submit, _), artifacts = run(
            tmp_path, references=[ImageInput.from_bytes(PNG_A), ImageInput.from_bytes(PNG_B)],
        )
        names = [d["name"] for d in submit["request"]["inputs"]["references"]]
        assert names == ["r-001-ref0.png", "r-001-ref1.png"]
        assert {"r-001-ref0.png", "r-001-ref1.png"} <= set(files(artifacts))

    def test_a_handle_is_recorded_as_the_handle_never_downloaded(self,
                                                                 tmp_path: Path) -> None:
        (submit, _), artifacts = run(tmp_path, extend=VideoInput(uri="files/earlier"))
        assert submit["request"]["inputs"]["extend"] == {"uri": "files/earlier"}
        assert files(artifacts) == ["r-002.mp4"]        # only the output

    def test_extend_bytes_become_an_artifact(self, tmp_path: Path) -> None:
        (submit, _), artifacts = run(tmp_path, BytesInputClient,
                                     extend=VideoInput.from_bytes(MP4))
        assert submit["request"]["inputs"]["extend"]["name"] == "r-001-extend.mp4"
        assert (artifacts / "r-001-extend.mp4").read_bytes() == MP4

    def test_suffixes(self) -> None:
        assert (_suffix_for("video/mp4"), _suffix_for("video/quicktime"),
                _suffix_for("video/webm")) == (".mp4", ".mov", ".webm")


class TestNoPayloadInline:
    def test_nothing_large_or_binary_reaches_the_jsonl(self, tmp_path: Path) -> None:
        # default=str would write an unrendered VideoInput as its repr, bytes and all.
        run(tmp_path, BytesInputClient, extend=VideoInput.from_bytes(BIG_MP4),
            first_frame=ImageInput.from_bytes(PNG_A))
        raw = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")
        assert "\\xab" not in raw and "b'" not in raw and "VideoInput(" not in raw
        assert max(len(line) for line in raw.splitlines()) < 5_000


class TestVideoArtifactsOff:
    def test_records_are_complete_but_no_files_are_written(self, tmp_path: Path) -> None:
        (submit, terminal), artifacts = run(
            tmp_path, video_artifacts=False,
            first_frame=ImageInput.from_bytes(PNG_A), duration_seconds=8,
        )
        assert files(artifacts) == []
        frame = submit["request"]["inputs"]["first_frame"]
        assert frame == {"name": None, "mime_type": "image/png", "bytes": len(PNG_A),
                         "not_written": "video_artifacts is off"}
        video = terminal["response"]["artifact"]
        assert video["name"] is None and video["bytes"] == len(MP4)
        # Everything else is still there.
        assert terminal["request"]["job"]["state"] == "succeeded"
        assert terminal["response"]["produced"]["source_uri"] == "files/h"
        assert submit["request"]["requested"] == {"duration_seconds": 8}

    def test_other_modalities_are_unaffected(self, tmp_path: Path) -> None:
        tracer = JsonlTraceLogger(tmp_path / "trace.jsonl", run_id="r", video_artifacts=False)
        record = tracer.llm_call(
            service="s", model="m", modality="image", latency_ms=1.0, request={},
            result=MediaResult(data=PNG_A, mime_type="image/png"), error=None,
        )
        assert record["response"]["artifact"]["name"] == "r-001.png"
        assert files(tmp_path / "artifacts") == ["r-001.png"]


class TestWriteFailure:
    def test_a_failed_write_keeps_the_record_and_the_video(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def disk_full(self: Path, data: bytes) -> int:
            raise OSError("No space left on device")

        monkeypatch.setattr(Path, "write_bytes", disk_full)
        tracer = JsonlTraceLogger(tmp_path / "trace.jsonl", run_id="r")
        client = FakeVideoClient(model="veo-test", trace=tracer)
        with caplog.at_level(logging.WARNING):
            result = client.submit_video("p", first_frame=ImageInput.from_bytes(PNG_A)).result()
        assert result.data == MP4                      # the call is unaffected
        submit, terminal = (
            json.loads(line)
            for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        )
        assert "No space left" in submit["request"]["inputs"]["first_frame"]["write_failed"]
        assert "No space left" in terminal["response"]["artifact"]["write_failed"]
        assert "Could not write trace artifact" in caplog.text
