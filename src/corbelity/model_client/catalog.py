"""The model catalog: which models exist, what they cost, and what they can do.

Why this is data and not code: model facts churn on the providers' schedule, not on this
package's release schedule. A user who wants to call a model released yesterday should
edit JSON, not wait for a version bump. The built-in catalog is therefore a starting set,
and a user catalog merges OVER it by model id.

    CORBELITY_MODEL_CATALOG=/etc/models.json     # or ModelConfig(catalog_path=...)

The catalog is descriptive, not enforcing: calling a model that is not listed works fine.
It exists so a UI can populate a dropdown and so provider code can look up a capability
flag instead of hardcoding a list of model-name prefixes. The one place a STATED fact is
enforced is a model's `video` block: what it says a model cannot do is refused before the
network, and a model without one is passed through for the provider to judge (DESIGN.md
section 6, "Enforced when stated").
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from .media import VIDEO_ROLES, VIDEO_SETTINGS, parse_aspect_ratio

CATALOG_RESOURCE = "models.json"

_logger = logging.getLogger(__name__)

# How a model treats audio in its video output. "always": audio is produced whatever is
# asked, so generate_audio=False is refused rather than silently ignored.
AUDIO_POLICIES = ("always", "optional", "never")

# The constraint verbs. Exactly three, each readable as one sentence in the error it
# produces: a setting must take a value, a role needs another role, a role forbids others.
CONSTRAINT_VERBS = ("require", "require_input", "exclude_input")


@dataclass(frozen=True, kw_only=True)
class VideoConstraint:
    """One row of a model's constraint table: when `when` is active, one rule applies.

    `when` is a role name ("references") -- active when that role is supplied -- or
    "resolution:<value>", active when that resolution is requested. A lookup table, not a
    rule engine: if a vendor limit ever needs logic this cannot express, that is a signal
    for code, not for growing the vocabulary."""

    when: str
    require: Mapping[str, Any] | None = None
    require_input: str | None = None
    exclude_input: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class VideoCapabilities:
    """What a model can do for video, as the catalog states it.

    Every field is "as stated": None means the catalog says nothing about that axis, and
    the provider decides. An empty `inputs` is a statement (text-to-video only); a missing
    one is not."""

    inputs: tuple[str, ...] | None = None
    max_references: int | None = None
    aspect_ratios: tuple[str, ...] | None = None
    resolutions: tuple[str, ...] | None = None
    durations: tuple[int, ...] | None = None
    audio: str = "optional"
    constraints: tuple[VideoConstraint, ...] = ()

    @classmethod
    def from_mapping(cls, model_id: str, raw: object) -> VideoCapabilities:
        """Parse a catalog `video` block.

        Malformed data RAISES, naming the model: a wrong type is an authoring error, and a
        catalog that loads but enforces the wrong thing is worse than one that fails to
        load. An unknown verb or role is WARNED and SKIPPED instead, because the catalog
        deliberately tolerates fields from a newer version of this package -- but a
        skipped rule is a rule not enforced, which should never be silent."""
        if not isinstance(raw, Mapping):
            raise ValueError(f"Catalog entry {model_id!r}: 'video' must be an object.")

        def strings(key: str) -> tuple[str, ...] | None:
            value = raw.get(key)
            if value is None:
                return None
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(
                    f"Catalog entry {model_id!r}: video.{key} must be a list of strings."
                )
            return tuple(value)

        inputs = strings("inputs")
        if inputs is not None:
            inputs = tuple(_known_roles(model_id, "video.inputs", inputs))

        aspect_ratios = strings("aspect_ratios")
        for ratio in aspect_ratios or ():
            if parse_aspect_ratio(ratio) is None:
                raise ValueError(
                    f"Catalog entry {model_id!r}: video.aspect_ratios has {ratio!r}, "
                    "which is not a W:H ratio."
                )

        durations_raw = raw.get("durations")
        durations: tuple[int, ...] | None = None
        if durations_raw is not None:
            if not isinstance(durations_raw, list) or not all(
                isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in durations_raw
            ):
                raise ValueError(
                    f"Catalog entry {model_id!r}: video.durations must be a list of "
                    "positive whole seconds."
                )
            durations = tuple(durations_raw)

        max_refs = raw.get("max_references")
        if max_refs is not None and (
            not isinstance(max_refs, int) or isinstance(max_refs, bool) or max_refs < 0
        ):
            raise ValueError(
                f"Catalog entry {model_id!r}: video.max_references must be a whole number."
            )

        audio = raw.get("audio", "optional")
        if audio not in AUDIO_POLICIES:
            raise ValueError(
                f"Catalog entry {model_id!r}: video.audio must be one of "
                f"{', '.join(AUDIO_POLICIES)} (got {audio!r})."
            )

        constraints_raw = raw.get("constraints", [])
        if not isinstance(constraints_raw, list):
            raise ValueError(f"Catalog entry {model_id!r}: video.constraints must be a list.")
        constraints = tuple(
            constraint
            for row in constraints_raw
            if (constraint := _parse_constraint(model_id, row)) is not None
        )

        return cls(
            inputs=inputs,
            max_references=max_refs,
            aspect_ratios=aspect_ratios,
            resolutions=strings("resolutions"),
            durations=durations,
            audio=audio,
            constraints=constraints,
        )


def _known_roles(model_id: str, where: str, roles: Iterable[str]) -> Iterator[str]:
    for role in roles:
        if role in VIDEO_ROLES:
            yield role
        else:
            _logger.warning(
                "Catalog entry %r: %s names unknown role %r (known: %s); ignored.",
                model_id, where, role, ", ".join(VIDEO_ROLES),
            )


def _parse_constraint(model_id: str, row: object) -> VideoConstraint | None:
    """One constraint row, or None (with a warning) when it uses vocabulary this version
    does not know."""
    if not isinstance(row, Mapping) or not isinstance(row.get("when"), str):
        raise ValueError(
            f"Catalog entry {model_id!r}: each video constraint must be an object with a "
            f"string 'when' (got {row!r})."
        )
    when: str = row["when"]
    verbs = [verb for verb in CONSTRAINT_VERBS if verb in row]
    unknown = sorted(set(row) - {"when", *CONSTRAINT_VERBS})
    if unknown or len(verbs) != 1:
        _logger.warning(
            "Catalog entry %r: video constraint %r has %s; ignored. Each row takes exactly "
            "one of: %s.", model_id, row,
            f"unknown key(s) {unknown}" if unknown else f"{len(verbs)} verbs",
            ", ".join(CONSTRAINT_VERBS),
        )
        return None
    trigger_role = when.split(":", 1)[0] if when.startswith("resolution:") else when
    if trigger_role != "resolution" and trigger_role not in VIDEO_ROLES:
        _logger.warning(
            "Catalog entry %r: video constraint 'when' %r is not a role or "
            "'resolution:<value>'; ignored.", model_id, when,
        )
        return None

    verb = verbs[0]
    value = row[verb]
    if verb == "require":
        if not isinstance(value, Mapping) or not value:
            raise ValueError(
                f"Catalog entry {model_id!r}: 'require' must map settings to values."
            )
        unknown_settings = sorted(set(value) - set(VIDEO_SETTINGS))
        if unknown_settings:
            _logger.warning(
                "Catalog entry %r: constraint on %r requires unknown setting(s) %s; "
                "ignored.", model_id, when, unknown_settings,
            )
            return None
        return VideoConstraint(when=when, require=dict(value))
    if verb == "require_input":
        if not isinstance(value, str):
            raise ValueError(
                f"Catalog entry {model_id!r}: 'require_input' must name one role."
            )
        roles = list(_known_roles(model_id, f"constraint on {when!r}", [value]))
        return VideoConstraint(when=when, require_input=roles[0]) if roles else None
    # exclude_input
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(
            f"Catalog entry {model_id!r}: 'exclude_input' must be a list of roles."
        )
    excluded = tuple(_known_roles(model_id, f"constraint on {when!r}", value))
    return VideoConstraint(when=when, exclude_input=excluded) if excluded else None


@dataclass(frozen=True, kw_only=True)
class ModelInfo:
    """One catalog entry. Unknown keys in the JSON are ignored rather than rejected, so a
    catalog written for a newer version of this package still loads."""

    id: str
    service: str
    name: str = ""
    modality: str = "text"
    description: str = ""
    accepts_images: bool = False
    # None means "not stated" -- the provider decides. Only set this when a model is known
    # to reject sampling parameters, which some do with a 400 rather than a warning.
    supports_sampling: bool | None = None
    # The name this model wants for its completion cap. Newer OpenAI models reject
    # `max_tokens` and require `max_completion_tokens`. None means the dialect's default.
    max_tokens_param: str | None = None
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    # Video capabilities, when the catalog states them. Enforced before the network when
    # present; absent means the provider judges (see VideoCapabilities).
    video: VideoCapabilities | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ModelInfo:
        model_id = raw.get("id")
        # `model_service` is accepted for catalogs written against the pre-package schema.
        service = raw.get("service") or raw.get("model_service")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"Catalog entry is missing a string 'id': {raw!r}")
        if not isinstance(service, str) or not service:
            raise ValueError(f"Catalog entry {model_id!r} is missing a string 'service'.")

        def _opt_float(key: str) -> float | None:
            value = raw.get(key)
            return float(value) if isinstance(value, int | float) else None

        supports = raw.get("supports_sampling")
        max_tokens_param = raw.get("max_tokens_param")
        video = raw.get("video")
        return cls(
            id=model_id,
            service=service,
            name=str(raw.get("name") or model_id),
            modality=str(raw.get("modality") or "text"),
            description=str(raw.get("description") or ""),
            accepts_images=bool(raw.get("accepts_images", False)),
            supports_sampling=supports if isinstance(supports, bool) else None,
            max_tokens_param=(
                max_tokens_param if isinstance(max_tokens_param, str) and max_tokens_param
                else None
            ),
            cost_per_1k_input=_opt_float("cost_per_1k_input"),
            cost_per_1k_output=_opt_float("cost_per_1k_output"),
            video=VideoCapabilities.from_mapping(model_id, video) if video is not None else None,
        )


class ModelCatalog:
    """An ordered, id-indexed collection of ModelInfo."""

    def __init__(self, models: Sequence[ModelInfo] = ()) -> None:
        self._by_id: dict[str, ModelInfo] = {model.id: model for model in models}

    def __iter__(self) -> Iterator[ModelInfo]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._by_id

    def get(self, model_id: str) -> ModelInfo | None:
        return self._by_id.get(model_id)

    def for_service(self, service: str) -> tuple[ModelInfo, ...]:
        return tuple(m for m in self if m.service == service)

    def for_modality(self, modality: str) -> tuple[ModelInfo, ...]:
        return tuple(m for m in self if m.modality == modality)

    def for_services(self, services: Iterable[str]) -> ModelCatalog:
        """A catalog narrowed to the given services.

        Returns a ModelCatalog rather than a tuple, unlike the two filters above, because
        this is the one a caller is likely to narrow again or hand onward -- a workbench
        filters to the services it has credentials for, then asks that result for a
        modality. Order is preserved, so a UI's list stays stable."""
        wanted = frozenset(services)
        return ModelCatalog(tuple(m for m in self if m.service in wanted))

    def merged_with(self, other: ModelCatalog) -> ModelCatalog:
        """`other` wins on id collision -- a user catalog overrides the built-in entry
        rather than duplicating it."""
        combined = {**self._by_id, **other._by_id}
        return ModelCatalog(tuple(combined.values()))

    @classmethod
    def from_json(cls, text: str, *, source: str = "") -> ModelCatalog:
        """Parse a catalog. `source` names where it came from, for error messages.

        A model id may appear only ONCE in a catalog. The id is both the catalog's key and
        the name sent to the API, and this class keys by id -- so a duplicate would load
        without complaint, the later entry would replace the earlier, and the earlier would
        simply vanish. That is a silent substitution (DESIGN.md section 7), so it fails the
        load instead, naming the id and where both entries are. (A user catalog that reuses
        a built-in id is different: that is one file overriding another, by design.)"""
        raw = json.loads(text)
        where = f" {source}" if source else ""
        if not isinstance(raw, list):
            raise ValueError(f"Model catalog{where} must be a JSON array of objects.")
        models = tuple(ModelInfo.from_mapping(entry) for entry in raw)
        positions: dict[str, list[int]] = {}
        for index, model in enumerate(models, start=1):
            positions.setdefault(model.id, []).append(index)
        duplicates = {model_id: at for model_id, at in positions.items() if len(at) > 1}
        if duplicates:
            listed = "; ".join(
                f"{model_id!r} (entries {', '.join(map(str, at))})"
                for model_id, at in duplicates.items()
            )
            raise ValueError(
                f"Model catalog{where} lists the same model id more than once: {listed}. "
                "An id is the catalog's key and the name sent to the API, so it can appear "
                "only once; loading this would silently keep the later entry and drop the "
                "earlier. Give each entry its own id."
            )
        return cls(models)

    @classmethod
    def from_file(cls, path: Path | str) -> ModelCatalog:
        return cls.from_json(Path(path).read_text(encoding="utf-8"), source=str(path))

    @classmethod
    def builtin(cls) -> ModelCatalog:
        # importlib.resources rather than __file__ arithmetic, so this keeps working from
        # a wheel, a zipapp, or anywhere else the package is not a loose directory.
        text = (
            resources.files(__package__)
            .joinpath(CATALOG_RESOURCE)
            .read_text(encoding="utf-8")
        )
        return cls.from_json(text, source=f"(built-in {CATALOG_RESOURCE})")


@lru_cache(maxsize=1)
def builtin_catalog() -> ModelCatalog:
    """The shipped catalog, parsed once."""
    return ModelCatalog.builtin()


def load_catalog(path: Path | str | None = None, *, merge_builtin: bool = True) -> ModelCatalog:
    """The catalog to use: built-in, a user file, or the user file merged over the
    built-in (the default, so a user file only has to carry what it changes)."""
    if path is None:
        return builtin_catalog()
    user = ModelCatalog.from_file(path)
    return builtin_catalog().merged_with(user) if merge_builtin else user
