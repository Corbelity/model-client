"""Video jobs: the handle a submitted generation returns.

Video generation takes seconds to minutes, and every provider this package targets is
submit-then-poll. So `submit_video()` returns at once with a `VideoJob`, and the caller
decides how to wait: poll from a UI, block in a script with `wait()`, or store the job's
reference and resume it from another process after a restart.

    job = client.submit_video("a corbel bracket, slowly rotating")
    job.poll().state          # "running" | "succeeded" | "failed" | "filtered"
    ref = job.to_ref()        # VideoJobRef -- plain, JSON-able (ref.to_dict())
    job = client.resume_video(ref)
    result = job.wait()       # or job.result() once done

The base class owns the lifecycle -- the clock, finalization, both trace records -- so a
provider implements only three seam methods and arrives fully instrumented, which is the
reason `_run()` exists for every other modality.

Two records per job, not one, because a job really has two observable moments separated
by an unknown gap: submission (traced by `_run()`, as every call is) and the first time
anything observes it finished. Both carry the operation id, so they join trivially, and a
submission with no terminal record is, by itself, a list of jobs that were paid for and
never collected. Polls are not traced: a five-minute job polled every ten seconds would
write thirty identical records.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .errors import (
    ContentFilteredError,
    VideoJobFailedError,
    VideoNotReadyError,
    VideoTimeoutError,
)
from .media import MediaResult, ModelResult

if TYPE_CHECKING:
    from .client import ModelClient

_logger = logging.getLogger(__name__)

# The job reads time only through these three names, so tests can replace them and never
# sleep. Wall clock for job age, because a job outlives the process that submitted it and
# a monotonic clock means nothing after a restart; monotonic for one wait() call.
_now = time.time
_monotonic = time.monotonic
_sleep = time.sleep

type VideoState = Literal["running", "succeeded", "failed", "filtered"]

# Prefixed because these are part of the package's public surface, where a bare
# SUCCEEDED or TERMINAL_STATES says nothing about what it is a state OF -- the same
# reason VIDEO_ROLES and SUPPORTED_VIDEO_MIMES are prefixed. The video INPUT ROLES are
# not, because FIRST_FRAME names itself and VIDEO_ROLES already groups them.
VIDEO_RUNNING: VideoState = "running"
VIDEO_SUCCEEDED: VideoState = "succeeded"
VIDEO_FAILED: VideoState = "failed"
VIDEO_FILTERED: VideoState = "filtered"
VIDEO_TERMINAL_STATES = frozenset({VIDEO_SUCCEEDED, VIDEO_FAILED, VIDEO_FILTERED})


@dataclass(frozen=True, kw_only=True)
class VideoPoll:
    """What a provider's `_poll_video()` reports about a job.

    `output` is the provider's own handle to the finished video (for Veo, the object
    `files.download` needs). It is private to the provider/base boundary: it never reaches
    a caller, a status, or a trace record."""

    state: VideoState
    progress: float | None = None
    error: str | None = None
    filtered_reasons: tuple[str, ...] = ()
    output: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, kw_only=True)
class VideoStatus:
    """A job's state as a caller sees it."""

    state: VideoState
    # Wall-clock seconds since submission, or None when the job was resumed from a bare
    # operation id and nobody knows when it was submitted.
    elapsed_s: float | None
    # Only when the provider reports it. Veo does not.
    progress: float | None = None
    error: str | None = None
    filtered_reasons: tuple[str, ...] = ()

    @property
    def done(self) -> bool:
        return self.state in VIDEO_TERMINAL_STATES


@dataclass(frozen=True, kw_only=True)
class VideoJobRef:
    """Everything needed to find a job again, in any process. Store `to_dict()`; rebuild
    with `from_dict()`; hand either to `client.resume_video()`.

    The library serializes; the application stores. Owning persistence would mean owning
    a database, threads and shared state, none of which belong in a client library."""

    service: str
    model: str
    # The provider-side id: the one field that must never be lost, because the job keeps
    # running -- and is billed -- whether or not anyone still knows it exists.
    operation: str
    # Epoch seconds, wall clock. None for a job resumed from a bare operation id.
    submitted_at: float | None = None
    # The settings sent with the job, after any constraint fills. Recorded again on the
    # terminal trace record, so that record stands alone.
    requested: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "model": self.model,
            "operation": self.operation,
            "submitted_at": self.submitted_at,
            "requested": dict(self.requested),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> VideoJobRef:
        """Rebuild a stored reference. Raises ValueError naming the bad field, because a
        reference that half-loads resumes the wrong job or none."""
        for name in ("service", "model", "operation"):
            if not isinstance(raw.get(name), str) or not raw[name]:
                raise ValueError(f"A video job reference needs a non-empty {name!r}.")
        submitted = raw.get("submitted_at")
        if submitted is not None and (
            isinstance(submitted, bool) or not isinstance(submitted, int | float)
        ):
            raise ValueError(f"submitted_at must be epoch seconds (got {submitted!r}).")
        requested = raw.get("requested") or {}
        if not isinstance(requested, Mapping):
            raise ValueError("requested must be an object of settings.")
        return cls(
            service=raw["service"], model=raw["model"], operation=raw["operation"],
            submitted_at=float(submitted) if submitted is not None else None,
            requested=dict(requested),
        )


@dataclass(kw_only=True)
class VideoSubmission(ModelResult):
    """The result of the submit call itself, so it can go through `_run()` and get the
    same timing, logging and trace record as every other call. Its finish_reason is
    "submitted"; the video arrives later, through the job."""

    operation: str = ""


class VideoJob:
    """A submitted video generation.

    `poll()` is one status request and never blocks for long. The first poll that sees
    the job finished also downloads the video (so the terminal trace record can carry
    it) -- that one poll is slower; every other is a single GET. Once finished, the job
    answers from its cache and stops contacting the provider.

    Safe to poll from several threads: each poll is a read, and finalization (download,
    cache, terminal record) happens exactly once, under a lock. Not safe to share across
    processes -- resume from the reference there instead. A job resumed in a second
    process after the first already finalized it will record a second terminal record;
    both carry the operation id, so they deduplicate downstream.

    No cancel(): the Gemini API documents no cancellation for video jobs."""

    def __init__(self, client: ModelClient[Any], ref: VideoJobRef) -> None:
        self._client = client
        self._ref = ref
        self._lock = threading.Lock()
        self._status: VideoStatus | None = None
        self._result: MediaResult | None = None
        self._failure: Exception | None = None
        self._finalized = False

    def __repr__(self) -> str:
        state = self._status.state if self._status else "unpolled"
        return f"VideoJob({self._ref.service}/{self._ref.model}, {self._ref.operation!r}, {state})"

    @property
    def operation(self) -> str:
        return self._ref.operation

    @property
    def status(self) -> VideoStatus | None:
        """The last status seen, without contacting the provider. None before the first
        poll."""
        return self._status

    def to_ref(self) -> VideoJobRef:
        return self._ref

    def poll(self) -> VideoStatus:
        """Ask the provider once. A transient failure is raised and the job stays
        pollable; a job the provider no longer knows raises VideoJobNotFoundError."""
        if self._finalized:
            assert self._status is not None
            return self._status
        report = self._client._poll_video(self._ref.operation)
        status = VideoStatus(
            state=report.state, elapsed_s=self._elapsed(), progress=report.progress,
            error=report.error, filtered_reasons=tuple(report.filtered_reasons),
        )
        if status.done:
            self._finalize(report, status)
        else:
            self._client._logger.debug(
                "Video job %s still %s (%s)", self._ref.operation, status.state,
                f"{status.elapsed_s:.0f}s" if status.elapsed_s is not None else "age unknown",
            )
            self._status = status
        assert self._status is not None
        return self._status

    def result(self) -> MediaResult:
        """The finished video. Raises VideoNotReadyError if the job is still running
        (it never blocks -- that is wait()), VideoJobFailedError or ContentFilteredError
        if it ended without one."""
        status = self._status if self._finalized else self.poll()
        assert status is not None
        if not status.done:
            raise VideoNotReadyError(self._ref.operation, status.state)
        if self._result is not None:
            return self._result
        assert self._failure is not None
        raise self._failure

    def wait(self, timeout_s: float | None = None,
             poll_interval_s: float | None = None) -> MediaResult:
        """Poll until the job finishes, then return result().

        Running out of time raises VideoTimeoutError carrying this job's reference, and
        writes NO terminal record: the job has not ended. It is still running, still
        billed, and can be resumed."""
        config = self._client.config
        timeout = timeout_s if timeout_s is not None else config.video_wait_timeout_s
        interval = (
            poll_interval_s if poll_interval_s is not None else config.video_poll_interval_s
        )
        started = _monotonic()
        while True:
            if self.poll().done:
                return self.result()
            waited = _monotonic() - started
            if waited >= timeout:
                self._client._logger.warning(
                    "Stopped waiting for video job %s after %.0fs; it is still running. "
                    "Resume it with resume_video().", self._ref.operation, waited,
                )
                raise VideoTimeoutError(self._ref, waited)
            # Never sleep past the deadline: the last poll happens AT the timeout rather
            # than up to one interval after it.
            _sleep(min(interval, timeout - waited))

    def _elapsed(self) -> float | None:
        if self._ref.submitted_at is None:
            return None
        return max(0.0, _now() - self._ref.submitted_at)

    def _finalize(self, report: VideoPoll, status: VideoStatus) -> None:
        with self._lock:
            if self._finalized:
                return
            result: MediaResult | None = None
            failure: Exception | None = None
            if status.state == VIDEO_SUCCEEDED:
                # A failed download is raised and leaves the job unfinalized, so the next
                # poll tries again: the video is still on the provider's side.
                result = self._client._fetch_video(report)
            elif status.state == VIDEO_FILTERED:
                failure = ContentFilteredError(self._ref, status.filtered_reasons)
            else:
                failure = VideoJobFailedError(self._ref, status.error or "no detail given")
            self._result, self._failure, self._status = result, failure, status
            self._finalized = True
            self._client._record_video_terminal(self._ref, status, result, failure)
