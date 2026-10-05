"""The video job handle: submit, poll, wait, resume, and the two trace records.

A fake provider implements the three seam methods from a script of poll results, and a
fake clock replaces the job module's time functions, so nothing here sleeps or touches a
network (DESIGN.md section 14). The one exception is the concurrency test, which uses
real threads -- the race it guards against only exists with them.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from corbelity import model_client
from corbelity.model_client import (
    VIDEO,
    ContentFilteredError,
    ImageInput,
    JsonlTraceLogger,
    MediaResult,
    ModelConfig,
    ProviderSpec,
    VideoInput,
    VideoInputs,
    VideoJobFailedError,
    VideoJobNotFoundError,
    VideoJobRef,
    VideoNotReadyError,
    VideoOptions,
    VideoPoll,
    VideoStatus,
    VideoTimeoutError,
)
from corbelity.model_client import jobs as jobs_module
from corbelity.model_client.client import ModelClient
from corbelity.model_client.media import LLMResult

MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16
PNG = b"\x89PNG\r\n\x1a\n" + b"body"

SPEC = ProviderSpec(
    name="fake-video",
    client_path=f"{__name__}:FakeVideoClient",
    key_env=("FAKE_VIDEO_KEY",),
    modalities=frozenset({"text", VIDEO}),
    video_input_forms=("uri",),
)

RUNNING = VideoPoll(state="running")
DONE = VideoPoll(state="succeeded", output="handle-1")


class FakeVideoClient(ModelClient[Any]):
    """Implements the three seam methods from a script; records every call."""

    SPEC = SPEC

    def _build_client(self) -> Any:
        self.script: list[VideoPoll | Exception] = []
        self.submitted: list[tuple[str, VideoInputs, VideoOptions]] = []
        self.polls = 0
        self.fetches = 0
        self.fetch_error: Exception | None = None
        self.fetch_delay = 0.0
        self.submit_error: Exception | None = None
        return object()

    def _invoke(self, system: str, user: str, history: Any, images: Any) -> LLMResult:
        raise AssertionError("no text calls in these tests")

    def _submit_video(self, prompt: str, inputs: VideoInputs, options: VideoOptions) -> str:
        if self.submit_error is not None:
            raise self.submit_error
        self.submitted.append((prompt, inputs, options))
        return f"operations/{len(self.submitted)}"

    def _poll_video(self, operation: str) -> VideoPoll:
        self.polls += 1
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        return step

    def _fetch_video(self, poll: VideoPoll) -> MediaResult:
        self.fetches += 1
        if self.fetch_delay:
            time.sleep(self.fetch_delay)
        if self.fetch_error is not None:
            error, self.fetch_error = self.fetch_error, None
            raise error
        return MediaResult(data=MP4, mime_type="video/mp4", source_uri=f"files/{poll.output}")


class FakeClock:
    """Stands in for the job module's _now / _monotonic / _sleep."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.wall = start
        self.mono = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.wall += seconds
        self.mono += seconds

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(jobs_module, "_now", lambda: fake.wall)
    monkeypatch.setattr(jobs_module, "_monotonic", lambda: fake.mono)
    monkeypatch.setattr(jobs_module, "_sleep", fake.sleep)
    return fake


@pytest.fixture
def trace_path(tmp_path: Path) -> Path:
    return tmp_path / "trace.jsonl"


def records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def build(trace_path: Path | None = None, **kwargs: Any) -> FakeVideoClient:
    tracer = JsonlTraceLogger(trace_path, run_id="t") if trace_path else None
    kwargs.setdefault("model", "veo-test")
    return FakeVideoClient(trace=tracer, **kwargs)


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #
class TestSubmit:
    def test_returns_a_job_at_once(self, clock: FakeClock) -> None:
        client = build()
        job = client.submit_video("a corbel", duration_seconds=8)
        assert job.operation == "operations/1"
        assert job.status is None                      # nothing polled yet
        assert client.polls == 0
        prompt, inputs, options = client.submitted[0]
        assert prompt == "a corbel" and options == VideoOptions(duration_seconds=8)

    def test_a_refused_request_never_reaches_the_provider(self, trace_path: Path) -> None:
        client = build(trace_path)
        with pytest.raises(ValueError):
            client.submit_video("x", duration_seconds=True)
        assert client.submitted == []
        assert records(trace_path) == []               # validated before the call, as always

    def test_extend_accepts_an_earlier_result(self, clock: FakeClock) -> None:
        client = build()
        earlier = MediaResult(data=MP4, mime_type="video/mp4", source_uri="files/abc")
        client.submit_video("and on into town", extend=earlier)
        assert client.submitted[0][1].extend == VideoInput(uri="files/abc")

    def test_roles_reach_the_provider(self, clock: FakeClock) -> None:
        client = build()
        frame = ImageInput.from_bytes(PNG)
        client.submit_video("p", first_frame=frame, references=None)
        inputs = client.submitted[0][1]
        assert inputs.first_frame is frame and inputs.references == ()

    def test_the_submit_record(self, clock: FakeClock, trace_path: Path,
                               tmp_path: Path) -> None:
        catalog = tmp_path / "models.json"
        catalog.write_text(json.dumps([{
            "id": "veo-test", "service": "fake-video", "modality": "video",
            "video": {"constraints": [
                {"when": "references", "require": {"duration_seconds": 8}},
            ]},
        }]), encoding="utf-8")
        client = build(trace_path, config=ModelConfig(catalog_path=catalog))
        client.submit_video("p", references=[ImageInput.from_bytes(PNG)], seed=7)
        (record,) = records(trace_path)
        assert record["modality"] == "video"
        assert record["finish_reason"] == "submitted"
        job = record["request"]["job"]
        assert job == {"phase": "submit", "latency_kind": "round_trip",
                       "operation": "operations/1"}
        assert list(record["request"]["inputs"]) == ["references"]
        assert record["request"]["inputs"]["references"][0]["name"].endswith("-ref0.png")
        # What was asked for and what a constraint filled in, kept apart.
        assert record["request"]["requested"] == {"seed": 7}
        assert record["request"]["resolved"] == {"duration_seconds": 8, "seed": 7}

    def test_a_failed_submission_is_traced_and_raised(self, trace_path: Path) -> None:
        client = build(trace_path)
        client.submit_error = RuntimeError("quota")
        with pytest.raises(RuntimeError, match="quota"):
            client.submit_video("p")
        (record,) = records(trace_path)
        assert record["request"]["job"]["phase"] == "submit"
        assert "operation" not in record["request"]["job"]
        assert "quota" in record["error"]


# --------------------------------------------------------------------------- #
# Polling and the terminal record
# --------------------------------------------------------------------------- #
class TestLifecycle:
    def test_running_then_succeeded(self, clock: FakeClock, trace_path: Path) -> None:
        client = build(trace_path)
        client.script = [RUNNING, RUNNING, DONE]
        job = client.submit_video("p")
        clock.advance(12)
        status = job.poll()
        assert (status.state, status.done, status.elapsed_s) == ("running", False, 12)
        with pytest.raises(VideoNotReadyError, match="still running"):
            job.result()                               # polls once more, still running
        clock.advance(30)
        assert job.poll().state == "succeeded"
        result = job.result()
        assert result.data == MP4 and result.source_uri == "files/handle-1"
        assert client.last_result is result

    def test_a_finished_job_stops_contacting_the_provider(self, clock: FakeClock) -> None:
        client = build()
        client.script = [DONE]
        job = client.submit_video("p")
        job.poll()
        polls = client.polls
        for _ in range(3):
            job.poll()
            job.result()
        assert (client.polls, client.fetches) == (polls, 1)

    def test_polls_are_not_traced_and_the_terminal_record_is(
        self, clock: FakeClock, trace_path: Path
    ) -> None:
        client = build(trace_path)
        client.script = [RUNNING] * 5 + [DONE]
        job = client.submit_video("p", duration_seconds=6)
        clock.advance(42)
        job.wait(poll_interval_s=1)
        submit, terminal = records(trace_path)
        assert terminal["request"]["job"] == {
            "phase": "terminal", "operation": "operations/1", "state": "succeeded",
            "latency_kind": "observed",
        }
        assert terminal["request"]["requested"] == {"duration_seconds": 6}
        assert terminal["latency_ms"] == pytest.approx((42 + 5) * 1000)
        assert terminal["response"]["artifact"]                # the video, as a file
        assert terminal["response"]["produced"]["source_uri"] == "files/handle-1"
        assert submit["request"]["job"]["operation"] == terminal["request"]["job"]["operation"]

    def test_failed(self, clock: FakeClock, trace_path: Path) -> None:
        client = build(trace_path)
        client.script = [VideoPoll(state="failed", error="internal error 13")]
        job = client.submit_video("p")
        assert job.poll().state == "failed"
        with pytest.raises(VideoJobFailedError, match="internal error 13") as err:
            job.result()
        assert err.value.ref == job.to_ref()
        terminal = records(trace_path)[-1]
        assert terminal["request"]["job"]["state"] == "failed"
        assert "internal error 13" in terminal["error"]
        assert client.fetches == 0

    def test_filtered(self, clock: FakeClock) -> None:
        client = build()
        client.script = [VideoPoll(state="filtered", filtered_reasons=("person", "violence"))]
        job = client.submit_video("p")
        assert job.poll().filtered_reasons == ("person", "violence")
        with pytest.raises(ContentFilteredError) as err:
            job.result()
        assert err.value.reasons == ("person", "violence")
        assert not isinstance(err.value, ValueError)  # a well-formed request

    def test_a_failed_download_leaves_the_job_retryable(
        self, clock: FakeClock, trace_path: Path
    ) -> None:
        client = build(trace_path)
        client.script = [DONE]
        client.fetch_error = ConnectionError("reset")
        job = client.submit_video("p")
        with pytest.raises(ConnectionError):
            job.poll()
        assert len(records(trace_path)) == 1           # no terminal record yet
        assert job.result().data == MP4                # the next attempt downloads
        assert client.fetches == 2
        assert len(records(trace_path)) == 2

    def test_a_transient_poll_error_leaves_the_job_pollable(self, clock: FakeClock) -> None:
        client = build()
        client.script = [TimeoutError("read timed out"), DONE]
        job = client.submit_video("p")
        with pytest.raises(TimeoutError):
            job.poll()
        assert job.poll().state == "succeeded"

    def test_an_expired_job(self, clock: FakeClock) -> None:
        client = build()
        client.script = [VideoJobNotFoundError("fake-video", "operations/1")]
        job = client.submit_video("p")
        with pytest.raises(VideoJobNotFoundError) as err:
            job.poll()
        assert isinstance(err.value, LookupError)

    def test_concurrent_pollers_finalize_exactly_once(self, trace_path: Path) -> None:
        client = build(trace_path)
        client.script = [DONE]
        client.fetch_delay = 0.05                       # widen the race window
        job = client.submit_video("p")
        start = threading.Barrier(4)

        def poller() -> None:
            start.wait()
            job.poll()

        threads = [threading.Thread(target=poller) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert client.fetches == 1
        terminals = [r for r in records(trace_path) if r["request"]["job"]["phase"] == "terminal"]
        assert len(terminals) == 1


# --------------------------------------------------------------------------- #
# Waiting
# --------------------------------------------------------------------------- #
class TestWait:
    def test_polls_at_the_interval_until_done(self, clock: FakeClock) -> None:
        client = build()
        client.script = [RUNNING, RUNNING, DONE]
        result = client.submit_video("p").wait(timeout_s=100, poll_interval_s=10)
        assert result.data == MP4
        assert clock.sleeps == [10, 10]

    def test_defaults_come_from_config(self, clock: FakeClock) -> None:
        client = build(config=ModelConfig(video_poll_interval_s=3, video_wait_timeout_s=7))
        client.script = [RUNNING]
        with pytest.raises(VideoTimeoutError):
            client.submit_video("p").wait()
        assert clock.sleeps == [3, 3, 1]                # the last sleep stops at the deadline

    def test_timeout_is_not_a_failure(self, clock: FakeClock, trace_path: Path) -> None:
        client = build(trace_path)
        client.script = [RUNNING]
        job = client.submit_video("p")
        with pytest.raises(VideoTimeoutError) as err:
            job.wait(timeout_s=25, poll_interval_s=10)
        assert isinstance(err.value, TimeoutError)
        assert err.value.ref == job.to_ref()           # the paid-for job stays reachable
        assert clock.sleeps == [10, 10, 5]
        assert len(records(trace_path)) == 1           # no terminal record: not ended
        client.script = [DONE]
        assert job.poll().state == "succeeded"         # and it finishes later
        assert len(records(trace_path)) == 2

    def test_generate_video_is_submit_then_wait(self, clock: FakeClock) -> None:
        client = build()
        client.script = [RUNNING, DONE]
        assert client.generate_video("p", timeout_s=60).data == MP4
        assert len(client.submitted) == 1

    def test_generate_video_timeout_carries_the_ref(self, clock: FakeClock) -> None:
        client = build()
        client.script = [RUNNING]
        with pytest.raises(VideoTimeoutError) as err:
            client.generate_video("p", timeout_s=5)
        assert isinstance(err.value.ref, VideoJobRef)


# --------------------------------------------------------------------------- #
# References and resuming
# --------------------------------------------------------------------------- #
class TestResume:
    def test_round_trips_through_json_and_keeps_its_age(self, clock: FakeClock,
                                                       trace_path: Path) -> None:
        first = build()
        first.script = [RUNNING]
        stored = json.dumps(first.submit_video("p", duration_seconds=4).to_ref().to_dict())

        clock.advance(300)                             # "after a restart"
        second = build(trace_path)
        second.script = [DONE]
        job = second.resume_video(json.loads(stored))
        assert job.poll().elapsed_s == 300
        assert job.result().data == MP4
        (terminal,) = records(trace_path)
        assert terminal["request"]["job"]["latency_kind"] == "observed"
        assert terminal["request"]["requested"] == {"duration_seconds": 4}

    def test_a_bare_operation_id_has_no_known_age(self, clock: FakeClock,
                                                  trace_path: Path) -> None:
        client = build(trace_path)
        client.script = [DONE]
        job = client.resume_video("operations/99")
        assert job.poll().elapsed_s is None
        (terminal,) = records(trace_path)
        assert terminal["request"]["job"]["latency_kind"] == "unknown"

    def test_a_job_from_another_model_is_refused(self, clock: FakeClock) -> None:
        ref = VideoJobRef(service="fake-video", model="other-model", operation="operations/1")
        with pytest.raises(ValueError, match="belongs to fake-video/other-model"):
            build().resume_video(ref)

    @pytest.mark.parametrize("raw", [
        {"service": "fake-video", "model": "m"},
        {"service": "fake-video", "model": "m", "operation": ""},
        {"service": "fake-video", "model": "m", "operation": "o", "submitted_at": "yesterday"},
        {"service": "fake-video", "model": "m", "operation": "o", "requested": [1]},
    ])
    def test_a_malformed_reference_is_refused(self, raw: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            VideoJobRef.from_dict(raw)


# --------------------------------------------------------------------------- #
# Observability never load-bearing; the TraceSink protocol unchanged
# --------------------------------------------------------------------------- #
class TestTracing:
    def test_a_tracer_that_raises_cannot_lose_the_video(self, clock: FakeClock) -> None:
        class Exploding:
            def llm_call(self, **_kwargs: Any) -> None:
                raise OSError("disk full")

        client = FakeVideoClient(model="veo-test", trace=Exploding())
        client.script = [DONE]
        assert client.submit_video("p").result().data == MP4

    def test_a_sink_written_against_the_exact_signature_still_works(
        self, clock: FakeClock
    ) -> None:
        # No **kwargs: if the job had added keyword arguments to llm_call(), this sink
        # would raise TypeError -- swallowed, so the assertion is on what it received.
        received: list[dict[str, Any]] = []

        class StrictSink:
            def llm_call(self, *, service: str, model: str, modality: str,
                         latency_ms: float, request: Any, result: Any,
                         error: str | None) -> None:
                received.append({"modality": modality, "job": request.get("job")})

        client = FakeVideoClient(model="veo-test", trace=StrictSink())
        client.script = [DONE]
        client.submit_video("p").result()
        assert [r["job"]["phase"] for r in received] == ["submit", "terminal"]


def test_config_reads_the_video_settings_from_the_environment() -> None:
    config = ModelConfig.from_env({
        "CORBELITY_VIDEO_POLL_INTERVAL": "2.5", "CORBELITY_VIDEO_WAIT_TIMEOUT": "90",
    })
    assert (config.video_poll_interval_s, config.video_wait_timeout_s) == (2.5, 90)
    with pytest.raises(ValueError, match="positive"):
        ModelConfig(video_poll_interval_s=0)


# --------------------------------------------------------------------------- #
# The names the package publishes
# --------------------------------------------------------------------------- #
# These exist so an application can branch on a job's state, or name a video input role,
# without writing the string literal and hoping it still matches. That only holds if the
# exported name, the module it comes from and the behaviour all agree -- which is what
# this asserts. Reaching into corbelity.model_client.jobs instead would be worse than a
# literal, so the package root is the surface under test.
class TestThePublishedNames:

    ROLE_NAMES = ("FIRST_FRAME", "LAST_FRAME", "REFERENCES", "EXTEND")
    STATE_NAMES = ("VIDEO_RUNNING", "VIDEO_SUCCEEDED", "VIDEO_FAILED", "VIDEO_FILTERED")

    @pytest.mark.parametrize("name", [*ROLE_NAMES, *STATE_NAMES,
                                      "VIDEO_TERMINAL_STATES", "VideoState"])
    def test_each_name_is_exported(self, name: str) -> None:
        assert hasattr(model_client, name), f"{name} is not importable from the package"
        assert name in model_client.__all__, f"{name} is importable but not in __all__"

    def test_every_name_in_all_actually_exists(self) -> None:
        """A name in __all__ that nothing defines breaks `import *`, and tells a reader
        the package offers something it does not."""
        assert [n for n in model_client.__all__ if not hasattr(model_client, n)] == []

    def test_the_roles_are_exactly_what_video_roles_holds(self) -> None:
        """VIDEO_ROLES was already exported and its members were not, so the tuple and the
        individual names could drift apart. In order, because a catalog constraint's
        `when` is matched against it."""
        named = tuple(getattr(model_client, name) for name in self.ROLE_NAMES)
        assert named == model_client.VIDEO_ROLES

    def test_the_states_agree_with_the_status_that_reports_them(self) -> None:
        """The behavioural anchor, and the reason the constants are worth exporting at
        all. A state re-spelled without VideoStatus.done following would make
        `state in VIDEO_TERMINAL_STATES` and `status.done` disagree -- and an application
        polling on the first would wait for ever."""
        assert VideoStatus(state=model_client.VIDEO_RUNNING, elapsed_s=None).done is False
        for state in model_client.VIDEO_TERMINAL_STATES:
            assert VideoStatus(state=state, elapsed_s=None).done is True

    def test_terminal_states_is_every_state_except_running(self) -> None:
        every = {getattr(model_client, name) for name in self.STATE_NAMES}
        assert every - {model_client.VIDEO_RUNNING} == model_client.VIDEO_TERMINAL_STATES

    def test_the_exported_names_are_the_module_s_own(self) -> None:
        """Re-exported, not copied -- a copy is what falls out of step."""
        for name in (*self.STATE_NAMES, "VIDEO_TERMINAL_STATES"):
            assert getattr(model_client, name) is getattr(jobs_module, name)

    def test_a_state_name_is_accepted_where_a_literal_was(self) -> None:
        """What the exports are for: the published names are the values a VideoPoll
        actually carries, so a provider or an application can use them in place of a
        string."""
        carried = {VideoPoll(state=getattr(model_client, name)).state
                   for name in self.STATE_NAMES}
        assert carried == {model_client.VIDEO_RUNNING, *model_client.VIDEO_TERMINAL_STATES}
