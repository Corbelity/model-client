"""Structured capture of every model call: the full prompts sent, the response back,
timing, and token usage. This is the observability seam ModelClient was built around --
the base class funnels every provider and every modality through one place, so tracing
plugs in there and covers all of them at once.

A trace is the record you need when a run produced a surprising answer and the INFO log
line ("23 tokens, 842 ms") is not enough to reconstruct why. Nothing is truncated; a
trimmed trace cannot reproduce the call.

Binary payloads (generated images and audio, and attached input images) are NOT inlined:
each is written beside the trace as a named file and the record keeps
{name, mime_type, bytes}. A single generated PNG can run to megabytes, and base64 in a
JSONL line makes the trace unreadable, while a filename means you can still open the
thing the model actually produced.

PRIVACY: trace records contain full prompt and response text. Treat the trace file and
its artifacts directory as sensitive, keep them out of version control, and think before
enabling tracing on anything handling third-party data.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import threading
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Relative to the working directory, not to this file: a library has no business writing
# next to its own installed source, and an absolute default would be wrong on every
# machine but one. Override with TRACE_FILE or by passing a path.
DEFAULT_TRACE_FILE = Path("logs") / "traces.jsonl"

_logger = logging.getLogger(__name__)


@runtime_checkable
class TraceSink(Protocol):
    """What ModelClient needs from a tracer.

    Structural, not nominal: anything with a matching llm_call() works, and nothing has
    to subclass or import this. The client never imports a concrete tracer, which is what
    keeps observability from becoming a dependency of the request path.

    Implementations must NOT raise. The client guards against it anyway, but a tracer
    that throws turns a working model call into a failed one on some other code path
    sooner or later."""

    def llm_call(
        self,
        *,
        service: str,
        model: str,
        modality: str,
        latency_ms: float,
        request: Mapping[str, Any] | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> Any: ...


class NullTrace:
    """A sink that records nothing.

    Provided for callers who want a non-None sink unconditionally. Passing trace=None is
    still marginally cheaper, because the client's `if self._trace is not None` guard
    skips the call entirely."""

    def llm_call(self, **_kwargs: Any) -> None:
        return None


# Extensions we care about, because mimetypes guesses poorly for audio (it maps audio/wav
# to .wav on some platforms and nothing on others, and audio/mpeg to .mp2 before .mp3).
# Anything unlisted falls back to mimetypes, then to .bin.
_MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3",
}


def _suffix_for(mime_type: str) -> str:
    if mime_type in _MIME_SUFFIXES:
        return _MIME_SUFFIXES[mime_type]
    return mimetypes.guess_extension(mime_type or "") or ".bin"


class JsonlTraceLogger:
    """Append-only JSONL trace, one object per model call.

    One instance per run; `run_id` groups the calls that belong together and `seq` orders
    them. Appends are locked because a sync request handler run in a threadpool (FastAPI
    does exactly this for `def` endpoints) produces genuinely overlapping calls, and
    unlocked writes interleave into unparseable lines."""

    def __init__(self, path: Path | str, *, run_id: str | None = None,
                 artifacts_dir: Path | str | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir else self.path.parent / "artifacts"
        )
        self._lock = threading.Lock()
        self._seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def llm_call(self, *, service: str, model: str, modality: str, latency_ms: float,
                 request: Mapping[str, Any] | None = None,
                 result: Any = None,
                 error: str | None = None) -> dict[str, Any]:
        """Record one call. `result` is whatever _invoke returned (duck-typed rather than
        imported, so tracing does not drag the provider SDKs in); `error` is set instead
        when the call raised. Returns the record written, for callers that want it.

        Never raises: a broken trace must not take down a working model call."""
        with self._lock:
            self._seq += 1
            seq = self._seq
            record = {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace(
                    "+00:00", "Z"
                ),
                "run_id": self.run_id,
                "seq": seq,
                "service": service,
                "model": model,
                "modality": modality,
                "latency_ms": latency_ms,
                "request": self._render_request(request, seq),
                "response": self._render_response(result, seq),
                "usage": {
                    "prompt_tokens": getattr(result, "prompt_tokens", None),
                    "completion_tokens": getattr(result, "completion_tokens", None),
                    "total_tokens": getattr(result, "total_tokens", None),
                    # The component split, when the provider reports one. Recorded here
                    # because the trace is where a cost monitor reads usage back from --
                    # a split that reaches the result but not the record is invisible to
                    # the thing that needs it. Absent components stay None.
                    "input_text_tokens": getattr(result, "input_text_tokens", None),
                    "input_image_tokens": getattr(result, "input_image_tokens", None),
                    "input_cached_tokens": getattr(result, "input_cached_tokens", None),
                },
                "finish_reason": getattr(result, "finish_reason", None),
                "error": error,
            }
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError as err:
                _logger.warning("Could not write trace record: %s", err)
            return record

    def _render_request(self, request: Mapping[str, Any] | None, seq: int) -> dict[str, Any]:
        """Input images get exactly the treatment generated ones already get: the payload
        becomes a file beside the trace and the record keeps a descriptor.

        The "no bytes in the JSONL" rule has to cover the request as well as the response
        -- a single attached photo inlined as base64 would make the line unreadable, which
        is the whole reason the rule exists. Keeping both halves here means there is one
        place that decides it, rather than one in the writer and one in the client."""
        rendered = dict(request) if request else {}
        images = rendered.get("images")
        if not images:
            return rendered
        descriptors = []
        for n, image in enumerate(images):
            descriptor = self._write_artifact(
                bytes(getattr(image, "data", b"") or b""),
                getattr(image, "mime_type", ""), seq, suffix=f"-in{n}",
            )
            # The artifact is named for the run; the user's own filename is what makes a
            # trace legible six weeks later, so keep it alongside.
            source_name = getattr(image, "name", None)
            if source_name:
                descriptor["source_name"] = source_name
            descriptors.append(descriptor)
        rendered["images"] = descriptors
        return rendered

    def _render_response(self, result: Any, seq: int) -> dict[str, Any] | None:
        if result is None:
            return None

        data = getattr(result, "data", None)
        if isinstance(data, bytes | bytearray):
            return {
                "artifact": self._write_artifact(
                    bytes(data), getattr(result, "mime_type", ""), seq
                )
            }

        text = getattr(result, "text", None)
        if text is not None:
            return {"text": text}
        return None

    def _write_artifact(self, data: bytes, mime_type: str, seq: int,
                        suffix: str = "") -> dict[str, Any]:
        """Payload to its own file; the record keeps the name that points at it. A write
        failure degrades to metadata rather than losing the whole trace record.

        `suffix` discriminates several artifacts from one call -- a vision request's inputs
        and the call's own output otherwise collide on <run_id>-<seq>."""
        name = f"{self.run_id}-{seq:03d}{suffix}{_suffix_for(mime_type)}"
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            (self.artifacts_dir / name).write_bytes(data)
        except OSError as err:
            _logger.warning("Could not write trace artifact %s: %s", name, err)
        return {"name": name, "mime_type": mime_type, "bytes": len(data)}


def trace_file_path() -> Path:
    """Where traces are written, and therefore where they are read back from.

    Deliberately independent of TRACE_ENABLED: reviewing a trace captured yesterday must
    not require turning tracing on today. Kept separate from make_trace_logger() so the
    reader and the writer cannot disagree about the location."""
    return Path(os.getenv("TRACE_FILE", "").strip() or DEFAULT_TRACE_FILE)


def make_trace_logger(*, run_id: str | None = None,
                      path: Path | str | None = None) -> JsonlTraceLogger | None:
    """Build the configured trace logger, or None when tracing is off.

    TRACE_ENABLED turns it on; TRACE_FILE (or an explicit `path`) sets the destination.
    Off by default because traces hold full prompt and response content, so writing them
    to disk should be a deliberate choice. Returning None rather than a no-op object keeps
    the disabled path genuinely free."""
    if os.getenv("TRACE_ENABLED", "").strip().lower() not in _TRUTHY:
        return None
    return JsonlTraceLogger(path or trace_file_path(), run_id=run_id)
