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
