"""The model catalog: which models exist, what they cost, and what they can do.

Why this is data and not code: model facts churn on the providers' schedule, not on this
package's release schedule. A user who wants to call a model released yesterday should
edit JSON, not wait for a version bump. The built-in catalog is therefore a starting set,
and a user catalog merges OVER it by model id.

    CORBELITY_MODEL_CATALOG=/etc/models.json     # or ModelConfig(catalog_path=...)

The catalog is descriptive, not enforcing: calling a model that is not listed works fine.
Nothing here gates a request. It exists so a UI can populate a dropdown and so provider
code can look up a capability flag instead of hardcoding a list of model-name prefixes.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

CATALOG_RESOURCE = "models.json"


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
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None

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
        return cls(
            id=model_id,
            service=service,
            name=str(raw.get("name") or model_id),
            modality=str(raw.get("modality") or "text"),
            description=str(raw.get("description") or ""),
            accepts_images=bool(raw.get("accepts_images", False)),
            supports_sampling=supports if isinstance(supports, bool) else None,
            cost_per_1k_input=_opt_float("cost_per_1k_input"),
            cost_per_1k_output=_opt_float("cost_per_1k_output"),
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

    def merged_with(self, other: ModelCatalog) -> ModelCatalog:
        """`other` wins on id collision -- a user catalog overrides the built-in entry
        rather than duplicating it."""
        combined = {**self._by_id, **other._by_id}
        return ModelCatalog(tuple(combined.values()))

    @classmethod
    def from_json(cls, text: str) -> ModelCatalog:
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError("A model catalog must be a JSON array of objects.")
        return cls(tuple(ModelInfo.from_mapping(entry) for entry in raw))

    @classmethod
    def from_file(cls, path: Path | str) -> ModelCatalog:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def builtin(cls) -> ModelCatalog:
        # importlib.resources rather than __file__ arithmetic, so this keeps working from
        # a wheel, a zipapp, or anywhere else the package is not a loose directory.
        text = (
            resources.files(__package__)
            .joinpath(CATALOG_RESOURCE)
            .read_text(encoding="utf-8")
        )
        return cls.from_json(text)


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
