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
