"""Building a client by service name.

Every function here takes the service name a call site actually wrote, folds aliases, and
falls back to the configured default when nothing is named. Keeping that in one place is
what lets client_class() and supported_modalities() answer questions about a provider
without constructing one -- neither needs a credential, and supported_modalities() does
not even import the provider module.
"""
from __future__ import annotations

from typing import Any

from .client import ModelClient
from .config import ModelConfig, get_default_config
from .registry import DEFAULT_REGISTRY, ProviderSpec


def resolve_service(service: str | None = None, *, config: ModelConfig | None = None) -> str:
    """Canonical service name: trimmed, lower-cased, aliases folded. Falls back to the
    configured default when nothing is named."""
    settings = config if config is not None else get_default_config()
    return DEFAULT_REGISTRY.resolve(service, default=settings.default_service)


def known_services() -> tuple[str, ...]:
    return DEFAULT_REGISTRY.known()


def provider_spec(service: str | None = None, *,
                  config: ModelConfig | None = None) -> ProviderSpec:
    """The registry entry for `service` -- its credential variables, endpoint and
    modalities -- without importing or building anything."""
    return DEFAULT_REGISTRY.get(resolve_service(service, config=config))


def client_class(service: str | None = None, *,
                 config: ModelConfig | None = None) -> type[ModelClient[Any]]:
    """The class that would handle `service`, without building it (no credentials
    needed). This DOES import the provider module, which is why supported_modalities()
    below reads the spec instead of calling through here."""
    return provider_spec(service, config=config).load_client_class()


def supported_modalities(service: str | None = None, *,
                         config: ModelConfig | None = None) -> frozenset[str]:
    """What `service` can produce -- lets a caller reject an impossible request before
    building a client, installing an SDK, or spending a network round trip."""
    return provider_spec(service, config=config).modalities


def make_model_client(service: str | None = None, **kwargs: Any) -> ModelClient[Any]:
    """Build the provider client for `service`.

    Extra kwargs (model / temperature / top_p / max_tokens / num_ctx / api_key / base_url /
    trace / config) flow straight through to the subclass constructor:

        make_model_client("anthropic", model="claude-sonnet-5")
        make_model_client("ollama-local", model="gemma3:12b")
    """
    config = kwargs.get("config")
    return client_class(service, config=config)(**kwargs)
