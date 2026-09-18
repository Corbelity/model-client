"""The provider registry: what each service is called, where its credential lives, what
it can produce, and which class implements it.

This is the file that turns provider identity from CODE into DATA. Nothing here imports a
provider SDK, and nothing here imports the provider classes -- each spec names its class
as an import path that is resolved on first use. That indirection buys two things:

  * no circular import (providers import the base client, which imports this module);
  * `supported_modalities("huggingface")` answers without importing anything at all.

Environment variable names are a TUPLE per provider, tried in order. That is how a token
stored under an older or an ecosystem-standard name is honoured without mutating
os.environ to bridge one name onto another.

Registering your own provider:

    from corbelity.model_client import ProviderSpec, register_provider
    register_provider(ProviderSpec(
        name="my-gateway",
        key_env=("MY_GATEWAY_KEY",),
        default_base_url="https://gateway.internal/v1",
        client_path="my_package.clients:MyGatewayClient",
        extra="",
    ))
"""
from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .errors import UnknownServiceError
from .media import IMAGE, SOUND, TEXT

if TYPE_CHECKING:
    from .client import ModelClient


@dataclass(frozen=True, kw_only=True)
class ProviderSpec:
    """Everything about a provider that is not its request/response handling."""

    name: str
    # Import path of the implementing class, "module:ClassName". Resolved lazily.
    client_path: str
    # The pip extra that provides the SDK, used to build the install hint on ImportError.
    extra: str = ""
    # Credential environment variables, tried in order. Empty means no credential needed.
    key_env: tuple[str, ...] = ()
    # Endpoint overrides, tried in order, before default_base_url.
    base_url_env: tuple[str, ...] = ()
    default_base_url: str | None = None
    # When True, a client cannot be built without a base URL from somewhere. Used for
    # endpoints that are machine-specific and have no sensible default.
    requires_base_url: bool = False
    modalities: frozenset[str] = field(default_factory=lambda: frozenset({TEXT}))
    # Alternative spellings folded onto `name`, so a stale config or a hand-edited
    # catalog entry does not fail with "unsupported service".
    aliases: tuple[str, ...] = ()

    @property
    def requires_key(self) -> bool:
        return bool(self.key_env)

    def load_client_class(self) -> type[ModelClient[Any]]:
        module_name, _, class_name = self.client_path.partition(":")
        if not class_name:
            raise ValueError(
                f"client_path for {self.name!r} must be 'module:ClassName', "
                f"got {self.client_path!r}."
            )
        module = importlib.import_module(module_name)
        return getattr(module, class_name)  # type: ignore[no-any-return]


_PROVIDERS_MODULE = "corbelity.model_client.providers"

BUILTIN_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="anthropic",
        client_path=f"{_PROVIDERS_MODULE}.anthropic:AnthropicClient",
        extra="anthropic",
        key_env=("ANTHROPIC_API_KEY",),
        modalities=frozenset({TEXT}),
    ),
    ProviderSpec(
        name="openrouter",
        client_path=f"{_PROVIDERS_MODULE}.openrouter:OpenRouterClient",
        extra="openrouter",
        key_env=("OPENROUTER_API_KEY",),
        base_url_env=("OPENROUTER_BASE_URL",),
        default_base_url="https://openrouter.ai/api/v1",
        modalities=frozenset({TEXT}),
        aliases=("open_router", "open-router"),
    ),
    ProviderSpec(
        name="ollama-local",
        client_path=f"{_PROVIDERS_MODULE}.ollama:OllamaLocalClient",
        extra="ollama",
        # No credential: the local box is trusted. The host, however, is machine-specific
        # and has no default worth guessing, so it is required.
        base_url_env=("LOCAL_OLLAMA_URL", "OLLAMA_HOST"),
        requires_base_url=True,
        modalities=frozenset({TEXT}),
        aliases=("ollama_local", "ollama-host"),
    ),
    ProviderSpec(
        name="ollama",
        client_path=f"{_PROVIDERS_MODULE}.ollama:OllamaCloudClient",
        extra="ollama",
        key_env=("OLLAMA_API_KEY",),
        base_url_env=("OLLAMA_CLOUD_URL",),
        default_base_url="https://ollama.com",
        modalities=frozenset({TEXT}),
        aliases=("ollama-cloud",),
    ),
    ProviderSpec(
        name="huggingface",
        client_path=f"{_PROVIDERS_MODULE}.huggingface:HuggingFaceClient",
        extra="huggingface",
        # HF_TOKEN is what the HF libraries themselves read; the other two are names this
        # code has historically used. Tried in order rather than bridged via os.environ.
        key_env=("HF_TOKEN", "HUGGINGFACE_HUB_KEY", "HUGGINGFACEHUB_API_TOKEN"),
        modalities=frozenset({TEXT, IMAGE, SOUND}),
        aliases=("hugging_face", "hugging-face", "hf"),
    ),
)


class ProviderRegistry:
    """Name -> ProviderSpec, with alias folding. One module-level instance is the default;
    build your own if you want an isolated set (tests do)."""

    def __init__(self, specs: Iterable[ProviderSpec] = ()) -> None:
        self._specs: dict[str, ProviderSpec] = {}
        self._aliases: dict[str, str] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ProviderSpec, *, replace: bool = False) -> None:
        if spec.name in self._specs and not replace:
            raise ValueError(
                f"Provider {spec.name!r} is already registered; pass replace=True to "
                "override it."
            )
        self._specs[spec.name] = spec
        for alias in spec.aliases:
            self._aliases[alias] = spec.name

    def resolve(self, service: str | None, *, default: str) -> str:
        """Canonical service name: trimmed, lower-cased, aliases folded."""
        name = (service or default).strip().lower()
        return self._aliases.get(name, name)

    def get(self, service: str) -> ProviderSpec:
        try:
            return self._specs[service]
        except KeyError:
            raise UnknownServiceError(service, self.known()) from None

    def known(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


DEFAULT_REGISTRY = ProviderRegistry(BUILTIN_SPECS)


def get_spec(service: str) -> ProviderSpec:
    """The spec for an already-canonical service name."""
    return DEFAULT_REGISTRY.get(service)


def register_provider(spec: ProviderSpec, *, replace: bool = False) -> None:
    """Add a provider to the default registry. Call before building any client."""
    DEFAULT_REGISTRY.register(spec, replace=replace)
