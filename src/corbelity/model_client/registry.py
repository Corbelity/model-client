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
    # Whether this provider can carry REFERENCE IMAGES into image generation -- a fact
    # about this client's implementation, not about the vendor: it is True once the
    # provider knows how to route them. Gating on this rather than on the catalog's
    # per-model accepts_images flag is deliberate. The catalog is descriptive and gates
    # nothing (DESIGN.md §6); this is capability.
    image_input: bool = False
    # Provider cap on how many reference images one call may carry. None means no cap
    # this client knows of. Lives here rather than on a client class because three
    # services share OpenAICompatibleClient and their caps differ -- and because a
    # published API limit is data that changes on the vendor's schedule.
    max_reference_images: int | None = None
    # Image OUTPUT controls. Per-provider rather than per-model because they describe an
    # endpoint; if models within one provider ever diverge, these move to the catalog the
    # way supports_sampling did.
    #
    # Sizes this service offers. They drive aspect_ratio resolution and answer "what can
    # this service frame?". Non-resolution values ("auto") are allowed and never match a
    # ratio. Empty plus image_custom_size False means size cannot be requested at all.
    image_sizes: tuple[str, ...] = ()
    # True when the provider documents an arbitrary WIDTHxHEIGHT, so a well-formed size
    # outside image_sizes is passed through rather than refused.
    image_custom_size: bool = False
    # Quality values the provider accepts. Empty means quality cannot be requested.
    image_qualities: tuple[str, ...] = ()

    # How strictly reference images are adhered to. Empty means the service has no such
    # control, which is a different statement from "it takes no references" -- a provider
    # can accept reference images and give no dial for how closely they are followed.
    image_fidelities: tuple[str, ...] = ()

    # Output encoding. image_default_format is the format the endpoint produces when none
    # is requested: a published fact, and the thing that makes "does compression apply
    # here?" answerable when the caller named no format. None means unknown, and the
    # cross-parameter checks that depend on it are then skipped rather than guessed.
    image_backgrounds: tuple[str, ...] = ()
    image_output_formats: tuple[str, ...] = ()
    image_default_format: str | None = None
    image_compression: bool = False
    # Which forms of VideoInput this client can route ("bytes", "uri"). A fact about the
    # client and the provider together, like image_input. Veo only extends videos it made
    # and still holds, so its service accepts ("uri",). Empty means no video input at all.
    video_input_forms: tuple[str, ...] = ()
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
        name="openai",
        client_path=f"{_PROVIDERS_MODULE}.openai:OpenAIClient",
        extra="openai",
        key_env=("OPENAI_API_KEY",),
        # No default: the SDK's own base URL is correct, and hardcoding it here would
        # mean tracking a value we do not own. An override is still honoured.
        base_url_env=("OPENAI_BASE_URL",),
        # The only provider here that does all three natively -- but images and speech
        # come from separate SDK surfaces, not from chat completions.
        modalities=frozenset({TEXT, IMAGE, SOUND}),
        image_input=True,
        # OpenAI's documented ceiling for reference images on an edit request. In data so
        # a change on their side is an edit here rather than a release.
        max_reference_images=16,
        # Documented sizes, NOT a closed set -- see image_custom_size below. Largest
        # 16:9 and 9:16 entries included because true 16:9 is natively supported, so a
        # caller wanting it never has to crop or pad.
        image_sizes=(
            "auto",
            "1024x1024", "1536x1024", "1024x1536",
            "2048x2048", "2048x1152", "3840x2160", "2160x3840",
        ),
        # `size` also accepts an arbitrary WIDTHxHEIGHT, and the narrower enumeration on
        # the EDIT endpoint looks like DALL-E-era residue rather than a live constraint.
        # Enforcing that narrow list would refuse 16:9 on exactly the path that carries
        # reference images, so an explicit well-formed size is passed through and the
        # provider gets to be the one that rejects it, if it does.
        image_custom_size=True,
        # The edit endpoint omits the DALL-E-era "hd", and the response only ever reports
        # these, so the legacy values are not worth exposing.
        image_qualities=("auto", "low", "medium", "high", "xhigh", "max"),
        # Edit endpoint only, which is the same condition as "references were supplied":
        # references are what route a call to images.edit in the first place.
        image_fidelities=("low", "high"),
        image_backgrounds=("transparent", "opaque", "auto"),
        image_output_formats=("png", "jpeg", "webp"),
        image_default_format="png",
        image_compression=True,
        aliases=("open_ai",),
    ),
    ProviderSpec(
        name="gemini",
        client_path=f"{_PROVIDERS_MODULE}.gemini:GeminiClient",
        extra="gemini",
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        base_url_env=("GEMINI_BASE_URL",),
        # Google's OpenAI-compatibility endpoint. Text only, and it stays that way: image
        # generation is unverified through this shim, and audio generation runs over the
        # bidirectional Live API, which does not fit a request/response interface. The
        # native provider (`gemini-native`, below) is where new Gemini capability lands;
        # this service is kept unchanged so nothing written against it breaks.
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        modalities=frozenset({TEXT}),
        aliases=("google", "google-gemini"),
    ),
    ProviderSpec(
        name="gemini-native",
        client_path=f"{_PROVIDERS_MODULE}.gemini_native:GeminiNativeClient",
        extra="gemini-native",
        # The same credential names as the compatibility shim, in the same order, so one
        # key serves both. Resolved here and passed to the SDK explicitly: left to itself,
        # google-genai prefers GOOGLE_API_KEY over GEMINI_API_KEY -- the opposite order.
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        # A separate variable from the shim's GEMINI_BASE_URL: that one points at the
        # OpenAI-compatibility path, and pointing the native SDK at it would fail.
        base_url_env=("GEMINI_NATIVE_BASE_URL",),
        # No default: the SDK's own endpoint is correct, the same reasoning as `openai`.
        # Text only for now. Video, image and speech land on this same service in later
        # releases, as the routes behind them are verified.
        modalities=frozenset({TEXT}),
        aliases=("google-genai", "gemini-genai"),
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
