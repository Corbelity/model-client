"""The catalog (model facts as data) and the registry (provider identity as data)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corbelity.model_client import (
    ModelCatalog,
    builtin_catalog,
    get_spec,
    known_services,
    load_catalog,
    provider_spec,
    resolve_service,
    supported_modalities,
)
from corbelity.model_client.errors import UnknownServiceError


class TestCatalog:
    def test_builtin_catalog_loads_from_package_data(self) -> None:
        catalog = builtin_catalog()
        assert len(catalog) > 0
        assert all(entry.id and entry.service for entry in catalog)

    def test_every_builtin_entry_names_a_registered_service(self) -> None:
        # Catches a typo'd service in models.json at test time rather than at call time.
        for entry in builtin_catalog():
            assert entry.service in known_services(), entry.id

    def test_user_catalog_overrides_by_id(self, tmp_path: Path) -> None:
        existing = next(iter(builtin_catalog()))
        override = tmp_path / "models.json"
        override.write_text(
            json.dumps(
                [{"id": existing.id, "service": existing.service, "name": "Renamed"}]
            ),
            encoding="utf-8",
        )
        merged = load_catalog(override)
        entry = merged.get(existing.id)
        assert entry is not None and entry.name == "Renamed"
        # Merging, not replacing: the rest of the built-in catalog survives.
        assert len(merged) == len(builtin_catalog())

    def test_legacy_model_service_key_is_accepted(self) -> None:
        catalog = ModelCatalog.from_json(
            json.dumps([{"id": "x", "model_service": "openrouter"}])
        )
        entry = catalog.get("x")
        assert entry is not None and entry.service == "openrouter"

    def test_unknown_keys_are_ignored(self) -> None:
        # A catalog written for a newer version must still load.
        catalog = ModelCatalog.from_json(
            json.dumps([{"id": "x", "service": "openrouter", "invented_later": True}])
        )
        assert "x" in catalog

    def test_missing_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing a string 'id'"):
            ModelCatalog.from_json(json.dumps([{"service": "openrouter"}]))

    def test_filtering(self) -> None:
        catalog = builtin_catalog()
        assert all(e.service == "huggingface" for e in catalog.for_service("huggingface"))
        assert all(e.modality == "image" for e in catalog.for_modality("image"))


class TestRegistry:
    @pytest.mark.parametrize(
        ("written", "canonical"),
        [
            ("open_router", "openrouter"),
            ("OpenRouter", "openrouter"),
            ("  ollama_local ", "ollama-local"),
            ("ollama-cloud", "ollama"),
            ("hugging_face", "huggingface"),
            ("hf", "huggingface"),
        ],
    )
    def test_aliases_and_casing_fold(self, written: str, canonical: str) -> None:
        assert resolve_service(written) == canonical

    def test_unknown_service_lists_the_known_ones(self) -> None:
        with pytest.raises(UnknownServiceError, match="Known services"):
            provider_spec("nope")

    def test_modalities_resolve_without_importing_a_provider(self) -> None:
        # No SDK installed, no credential, no provider module imported.
        assert supported_modalities("huggingface") == frozenset({"text", "image", "sound"})
        assert supported_modalities("anthropic") == frozenset({"text"})

    def test_local_ollama_requires_a_host_and_has_no_credential(self) -> None:
        spec = get_spec("ollama-local")
        assert spec.requires_base_url is True
        assert spec.requires_key is False

    def test_huggingface_tries_the_ecosystem_token_name_first(self) -> None:
        # The bridge that used to be done by writing to os.environ is now a lookup order.
        assert get_spec("huggingface").key_env[0] == "HF_TOKEN"
