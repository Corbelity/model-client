"""Exception types.

Two rules govern the hierarchy, and both exist so that callers written against the old
flat module keep working:

  * everything derives from ModelClientError, so an application can catch one type;
  * errors that used to be raised as a built-in ALSO derive from that built-in, because
    web layers commonly map ValueError to a 400 and letting that break would be a silent
    behaviour change at the HTTP boundary.

Validation failures in messages.py and media.py deliberately raise plain ValueError
rather than a subclass: they describe a caller mistake, not a library condition, and
nothing is gained by making callers import a type to catch them.
"""
from __future__ import annotations

from collections.abc import Sequence


class ModelClientError(RuntimeError):
    """Base class for every error this package raises on its own behalf."""


class UnsupportedModalityError(ModelClientError):
    """A provider was asked for a modality it cannot produce. Raised BEFORE any network
    call so the caller gets a precise message instead of a provider-specific 400."""

    def __init__(self, service: str, modality: str, supported: frozenset[str]) -> None:
        super().__init__(
            f"Service {service!r} does not support the {modality!r} modality "
            f"(supports: {', '.join(sorted(supported))})."
        )
        self.service = service
        self.modality = modality
        self.supported = supported


class UnsupportedImageInputError(ModelClientError):
    """A provider was asked to condition image generation on reference images, and this
    client cannot route them to that service.

    Mirrors UnsupportedModalityError: raised BEFORE any network call, naming the service,
    so the caller gets a precise message instead of a provider-specific 400 -- or worse,
    a successful generation that silently ignored the references."""

    def __init__(self, service: str) -> None:
        super().__init__(
            f"Service {service!r} cannot take reference images for image generation. "
            "Call generate_image() without images, or use a service that supports them."
        )
        self.service = service


class TooManyImagesError(ModelClientError, ValueError):
    """More reference images than the provider accepts.

    Checked here rather than left to the provider because the alternative is uploading
    several megabytes and then being rejected for it."""

    def __init__(self, service: str, count: int, maximum: int) -> None:
        super().__init__(
            f"Service {service!r} accepts at most {maximum} reference image(s); "
            f"{count} were supplied."
        )
        self.service = service
        self.count = count
        self.maximum = maximum


class UnsupportedSizeError(ModelClientError, ValueError):
    """A size or aspect ratio this client will not send to that service.

    Raised, never substituted. A silent nearest-match means the application believes it
    has 16:9 frames when it does not, and finds that out much later by looking at them --
    the caller's explicit instruction overridden by something it cannot see."""

    def __init__(self, service: str, requested: str, supported: Sequence[str]) -> None:
        offered = ", ".join(supported) if supported else "(none)"
        super().__init__(
            f"Service {service!r} cannot produce {requested!r}. It offers: {offered}."
        )
        self.service = service
        self.requested = requested
        self.supported = tuple(supported)


class UnsupportedQualityError(ModelClientError, ValueError):
    """A quality level the service does not accept, or a quality request to a service
    that has no such control at all."""

    def __init__(self, service: str, requested: str, supported: Sequence[str]) -> None:
        offered = (
            ", ".join(supported) if supported
            else "(none -- this service has no quality control)"
        )
        super().__init__(
            f"Service {service!r} does not accept quality {requested!r}. "
            f"It accepts: {offered}."
        )
        self.service = service
        self.requested = requested
        self.supported = tuple(supported)


class UnsupportedFidelityError(ModelClientError, ValueError):
    """A reference-adherence level the service does not accept, or a fidelity request to a
    service that has no such control at all."""

    def __init__(self, service: str, requested: str, supported: Sequence[str]) -> None:
        offered = (
            ", ".join(supported) if supported
            else "(none -- this service has no reference-fidelity control)"
        )
        super().__init__(
            f"Service {service!r} does not accept input_fidelity {requested!r}. "
            f"It accepts: {offered}."
        )
        self.service = service
        self.requested = requested
        self.supported = tuple(supported)


class UnsupportedBackgroundError(ModelClientError, ValueError):
    """A background treatment the service does not accept, or a background request to a
    service that has no such control at all."""

    def __init__(self, service: str, requested: str, supported: Sequence[str]) -> None:
        offered = (
            ", ".join(supported) if supported
            else "(none -- this service has no background control)"
        )
        super().__init__(
            f"Service {service!r} does not accept background {requested!r}. "
            f"It accepts: {offered}."
        )
        self.service = service
        self.requested = requested
        self.supported = tuple(supported)


class UnsupportedFormatError(ModelClientError, ValueError):
    """An output format the service does not produce, or a format request to a service
    that offers no choice of format."""

    def __init__(self, service: str, requested: str, supported: Sequence[str]) -> None:
        offered = (
            ", ".join(supported) if supported
            else "(none -- this service does not offer a choice of output format)"
        )
        super().__init__(
            f"Service {service!r} does not produce output format {requested!r}. "
            f"It offers: {offered}."
        )
        self.service = service
        self.requested = requested
        self.supported = tuple(supported)


class UnsupportedVideoSettingError(ModelClientError, ValueError):
    """A video setting this model does not offer, or one a model constraint rules out.

    A ValueError, like the image setting errors: it reports a value the caller could have
    chosen differently. Raised, never substituted -- a model that cannot do 4K is not
    quietly given 1080p."""

    def __init__(self, service: str, model: str, setting: str, requested: object,
                 supported: Sequence[object] = (), *, reason: str | None = None) -> None:
        if reason is None:
            offered = ", ".join(str(value) for value in supported) if supported else "(none)"
            reason = f"It offers: {offered}."
        super().__init__(
            f"{model!r} on {service!r} cannot take {setting}={requested!r}. {reason}"
        )
        self.service = service
        self.model = model
        self.setting = setting
        self.requested = requested
        self.supported = tuple(supported)


class UnsupportedVideoInputError(ModelClientError):
    """A video input role this model or client cannot route, or a combination of roles
    the model refuses.

    Not a ValueError, matching UnsupportedImageInputError: it reports a capability, not a
    value the caller mistyped. The message names the remedy, because the provider's own
    rejection of these cases is often generic -- Veo answers an unsupported combination
    with "Unsupported video generation request" and a link to its docs index."""

    def __init__(self, service: str, model: str, message: str) -> None:
        super().__init__(f"{model!r} on {service!r}: {message}")
        self.service = service
        self.model = model


class VideoNotReadyError(ModelClientError):
    """`result()` was asked for before the video job reached a terminal state.

    Raised rather than blocking: a method that sometimes returns at once and sometimes
    takes six minutes is the ambiguity the job handle exists to remove. Waiting is what
    `wait()` is for."""

    def __init__(self, operation: str, state: str) -> None:
        super().__init__(
            f"Video job {operation!r} is still {state}. Poll again later, or call wait()."
        )
        self.operation = operation
        self.state = state


class VideoTimeoutError(ModelClientError, TimeoutError):
    """`wait()` gave up. The job has NOT failed -- it is still running on the provider's
    side, and is billed whether or not anyone waits for it.

    Carries the job's reference so the paid-for video stays reachable: resume it with
    `client.resume_video(err.ref)`. Also a TimeoutError, so code that already catches
    that keeps working (DESIGN.md section 11)."""

    def __init__(self, ref: object, waited_s: float) -> None:
        operation = getattr(ref, "operation", ref)
        super().__init__(
            f"Video job {operation!r} was still running after {waited_s:.0f}s of waiting. "
            "It has not failed: resume it later with client.resume_video(err.ref)."
        )
        self.ref = ref
        self.waited_s = waited_s


class VideoJobFailedError(ModelClientError):
    """The provider reported that the video job failed."""

    def __init__(self, ref: object, provider_error: object) -> None:
        operation = getattr(ref, "operation", ref)
        super().__init__(f"Video job {operation!r} failed: {provider_error}")
        self.ref = ref
        self.provider_error = provider_error


class ContentFilteredError(ModelClientError):
    """The job finished but the provider withheld the output for safety reasons.

    Not a ValueError: the request was well-formed, so it does not belong on the 400 side
    of an HTTP boundary the way a caller's mistake does."""

    def __init__(self, ref: object, reasons: Sequence[str]) -> None:
        operation = getattr(ref, "operation", ref)
        listed = "; ".join(reasons) if reasons else "no reason given"
        super().__init__(f"Video job {operation!r} produced no video: filtered ({listed}).")
        self.ref = ref
        self.reasons = tuple(reasons)


class VideoJobNotFoundError(ModelClientError, LookupError):
    """The provider no longer knows the job: it expired, or the id is wrong.

    Distinct from a transient polling failure because the remedy differs -- retrying is
    pointless. Providers keep finished jobs for a limited time (Veo: 2 days)."""

    def __init__(self, service: str, operation: str) -> None:
        super().__init__(
            f"{service!r} has no video job {operation!r}: it has expired or never existed."
        )
        self.service = service
        self.operation = operation


class MissingCredentialsError(ModelClientError, ValueError):
    """No API key was found for a provider that requires one.

    Names every environment variable that was consulted, in order, so the fix is in the
    message rather than in the source."""

    def __init__(self, service: str, env_names: Sequence[str]) -> None:
        names = " or ".join(env_names) if env_names else "(none configured)"
        super().__init__(
            f"No API key for {service!r}: set {names}, or pass api_key= when building "
            "the client."
        )
        self.service = service
        self.env_names = tuple(env_names)


class MissingBaseUrlError(ModelClientError, ValueError):
    """A provider needs an endpoint that is machine-specific and has no sensible default
    (the local Ollama host is the only one today)."""

    def __init__(self, service: str, env_names: Sequence[str]) -> None:
        names = " or ".join(env_names) if env_names else "(none configured)"
        super().__init__(
            f"No base URL for {service!r}: set {names}, or pass base_url= when building "
            "the client."
        )
        self.service = service
        self.env_names = tuple(env_names)


class MissingDependencyError(ModelClientError, ImportError):
    """A provider SDK is an optional install and is not present.

    The message carries the exact pip command, because the alternative -- a bare
    ModuleNotFoundError naming a package the user never asked for -- makes the extras
    system look like a bug."""

    def __init__(self, module: str, extra: str) -> None:
        super().__init__(
            f"The {module!r} package is required for this provider but is not installed. "
            f'Install it with: pip install "corbelity-model-client[{extra}]"'
        )
        self.module = module
        self.extra = extra


class UnknownServiceError(ModelClientError, ValueError):
    """A service name that no registered provider claims."""

    def __init__(self, service: str | None, known: Sequence[str]) -> None:
        super().__init__(
            f"Unsupported model service: {service!r}. "
            f"Known services: {', '.join(known)}"
        )
        self.service = service
        self.known = tuple(known)
