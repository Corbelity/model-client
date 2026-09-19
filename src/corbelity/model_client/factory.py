"""Building a client by service name.

Every function here takes the service name a call site actually wrote, folds aliases, and
falls back to the configured default when nothing is named. Keeping that in one place is
what lets client_class() and supported_modalities() answer questions about a provider
without constructing one -- neither needs a credential, and supported_modalities() does
not even import the provider module.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .catalog import ModelCatalog, load_catalog
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


def available_services(*, env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The services whose credentials and endpoints are actually present in the
    environment -- what you can call right now, as opposed to what exists.

    Intended for building a `ModelConfig(services=...)` allow-list without hand-listing
    providers:

        config = ModelConfig(services=available_services())

    This is NOT applied automatically. Silently hiding a provider because an environment
    variable is missing is the kind of behaviour that produces "why is my model gone?" --
    so filtering stays an explicit choice, and this function only supplies the answer.

    Presence, not validity: it checks that a variable is set, never that the credential
    works. Verifying would mean a network call per provider, and a provider that is down
    at startup is not a provider you should stop offering.

    `env` is injectable for tests.
    """
    source = os.environ if env is None else env

    def _has(names: tuple[str, ...]) -> bool:
        return any(source.get(name, "").strip() for name in names)

    found = []
    for name in DEFAULT_REGISTRY.known():
        spec = DEFAULT_REGISTRY.get(name)
        if spec.requires_key and not _has(spec.key_env):
            continue
        # A provider with no usable default endpoint (local Ollama) is not available
        # until its host is configured, credential or no credential.
        if spec.requires_base_url and not _has(spec.base_url_env):
            continue
        found.append(name)
    return tuple(found)


def catalog_for(config: ModelConfig | None = None) -> ModelCatalog:
    """The model catalog as this configuration wants it shown: the built-in catalog, with
    any user catalog merged over it, narrowed to `config.services` when that is set.

    This is the listing path -- what a dropdown should offer. It is deliberately NOT what
    provider code reads: `_accepts_sampling_params()` loads the catalog unfiltered,
    because a capability flag has to be found for a model whatever a UI happens to be
    listing. Narrowing what you show must never change how a call behaves.
    """
    settings = config if config is not None else get_default_config()
    catalog = load_catalog(settings.catalog_path)
    if settings.services is None:
        return catalog
    # Fold aliases so ("hf",) matches entries whose service is "huggingface". An unknown
    # name simply matches nothing rather than raising: a stale entry in an allow-list
    # should not take down a UI that is merely listing models.
    wanted = {DEFAULT_REGISTRY.resolve(name, default=name) for name in settings.services}
    return catalog.for_services(wanted)
