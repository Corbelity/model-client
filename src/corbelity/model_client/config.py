"""Defaults for calls that do not name their own model or sampling parameters.

The resolution order is fixed and is the only thing a caller needs to memorise:

    per-call argument  ->  injected ModelConfig  ->  environment  ->  package default

Two things this module deliberately does NOT do, both of which are correct in an
application and disqualifying in a library:

  * it does not call load_dotenv(). Reading a .env file is the application's decision
    about its own process; a library that does it silently reaches outside its own
    boundary and surprises anyone who imports it inside a larger system.
  * it does not mutate os.environ. Bridging one provider's token name onto another's is
    handled as a *lookup* over several names (see ProviderSpec.key_env), which gets the
    same result without writing to global state that other libraries also read.

If you want .env loading, do it in your application entry point before building a client:

    from dotenv import load_dotenv
    load_dotenv()
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

# Every field below is ANNOTATED on purpose. @dataclass only turns annotated class
# attributes into fields; an unannotated one stays an ordinary class attribute, which
# means it never reaches __init__ and cannot be overridden per instance.
DEFAULT_SERVICE = "openrouter"
DEFAULT_MODEL = "anthropic/claude-opus-4.8"


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    """Immutable defaults. Build one, pass it to clients, or register it globally with
    set_default_config(). Use dataclasses.replace() (or .with_overrides()) to derive a
    variant rather than mutating one."""

    # Used ONLY when a caller names no service / model of its own. Routing is otherwise
    # per-call: the service name passed to make_model_client() picks the provider, and
    # "ollama-local" is what selects a local box ("ollama" is Ollama Cloud).
    default_service: str = DEFAULT_SERVICE
    default_model: str = DEFAULT_MODEL

    # A max, not a target. Well-behaved models stop well short of it. Raising it costs
    # nothing until a model actually runs long; lowering it truncates long structured
    # output (a JSON array of candidates is the usual casualty) with finish="length".
    default_max_tokens: int = 4096
    default_temperature: float = 0.7
    # None leaves top_p to the provider's own default rather than pinning it alongside
    # temperature; most providers advise tuning one or the other, not both.
    default_top_p: float | None = None

    # Ollama context window (num_ctx), local and cloud. Providers that manage their own
    # context ignore it. Ollama's default is small enough that long generations run into
    # the context wall and come back empty, which is why this is set rather than left off.
    num_ctx: int | None = 16384

    # Optional path to a JSON model catalog that merges over the built-in one, so adding
    # a model released after this package shipped needs no new release.
    catalog_path: Path | None = None

    # Optional allow-list of services worth offering, e.g. ("openrouter",). None means no
    # filter. This is PRESENTATION ONLY -- it narrows what catalog_for() lists, and
    # nothing more. It does not prevent a call to a filtered-out service, because the
    # catalog is descriptive rather than enforcing, and because the capability flags for
    # every service must stay loadable however the list is narrowed. Aliases are folded,
    # so ("hf",) and ("huggingface",) mean the same thing.
    services: tuple[str, ...] | None = None

    def with_overrides(self, **changes: object) -> ModelConfig:
        """A derived config. Thin wrapper over dataclasses.replace, kept because it reads
        better at call sites and keeps `replace` out of application imports."""
        return replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ModelConfig:
        """Read defaults from the environment, falling back to the package defaults.

        `env` is injectable so tests never have to touch os.environ. Malformed numeric
        values raise here, at construction, rather than at the first model call -- a
        typo'd CORBELITY_MAX_TOKENS should fail while you are still looking at the config,
        not three seconds into a provider round trip."""
        source = os.environ if env is None else env

        def _text(name: str, fallback: str) -> str:
            value = source.get(name, "").strip()
            return value or fallback

        def _int(name: str, fallback: int | None) -> int | None:
            raw = source.get(name, "").strip()
            if not raw:
                return fallback
            try:
                return int(raw)
            except ValueError as err:
                raise ValueError(f"{name} must be an integer, got {raw!r}.") from err

        def _float(name: str, fallback: float | None) -> float | None:
            raw = source.get(name, "").strip()
            if not raw:
                return fallback
            try:
                return float(raw)
            except ValueError as err:
                raise ValueError(f"{name} must be a number, got {raw!r}.") from err

        catalog = source.get("CORBELITY_MODEL_CATALOG", "").strip()
        # Comma-separated, e.g. CORBELITY_SERVICES=openrouter,ollama-local. An empty or
        # absent value means no filter, which is not the same as an empty allow-list.
        services = tuple(
            name.strip()
            for name in source.get("CORBELITY_SERVICES", "").split(",")
            if name.strip()
        ) or None
        max_tokens = _int("CORBELITY_MAX_TOKENS", cls.default_max_tokens)
        temperature = _float("CORBELITY_TEMPERATURE", cls.default_temperature)

        return cls(
            default_service=_text("CORBELITY_DEFAULT_SERVICE", cls.default_service),
            default_model=_text("CORBELITY_DEFAULT_MODEL", cls.default_model),
            # The two asserts are for the type checker only: the fallbacks are non-None
            # class defaults, so neither can actually be None here.
            default_max_tokens=max_tokens if max_tokens is not None else cls.default_max_tokens,
            default_temperature=(
                temperature if temperature is not None else cls.default_temperature
            ),
            default_top_p=_float("CORBELITY_TOP_P", cls.default_top_p),
            num_ctx=_int("CORBELITY_NUM_CTX", cls.num_ctx),
            catalog_path=Path(catalog) if catalog else None,
            services=services,
        )


_default_config: ModelConfig | None = None


def get_default_config() -> ModelConfig:
    """The process-wide config, built from the environment on first use.

    Lazy rather than module-level so that importing this package reads nothing: an import
    with side effects is what makes a library hard to embed."""
    global _default_config
    if _default_config is None:
        _default_config = ModelConfig.from_env()
    return _default_config


def set_default_config(config: ModelConfig | None) -> None:
    """Install (or, with None, clear) the process-wide config. Passing an explicit
    `config=` to a client always wins over this."""
    global _default_config
    _default_config = config
