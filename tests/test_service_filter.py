"""Service filtering: what a UI should offer, as distinct from what can be called.

The distinction these tests defend: `catalog_for()` narrows a *listing*, and nothing
about narrowing it may change how a call behaves.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corbelity.model_client import (
    ModelConfig,
    available_services,
    builtin_catalog,
    catalog_for,
    load_catalog,
)

FULL_ENV = {
    "ANTHROPIC_API_KEY": "k",
    "OPENROUTER_API_KEY": "k",
    "OLLAMA_API_KEY": "k",
    "HF_TOKEN": "k",
    "LOCAL_OLLAMA_URL": "http://localhost:11434",
}


class TestCatalogFor:
    def test_no_filter_returns_everything(self) -> None:
        assert len(catalog_for(ModelConfig())) == len(builtin_catalog())

    def test_filter_narrows_to_one_service(self) -> None:
        catalog = catalog_for(ModelConfig(services=("openrouter",)))
        assert len(catalog) > 0
        assert {m.service for m in catalog} == {"openrouter"}

    def test_filter_folds_aliases(self) -> None:
        by_alias = catalog_for(ModelConfig(services=("hf",)))
        by_name = catalog_for(ModelConfig(services=("huggingface",)))
        assert len(by_alias) == len(by_name) > 0

    def test_unknown_service_matches_nothing_and_does_not_raise(self) -> None:
        # A stale name in an allow-list should not take down a UI that is only listing.
        assert len(catalog_for(ModelConfig(services=("nope",)))) == 0

    def test_empty_tuple_is_not_the_same_as_none(self) -> None:
        assert len(catalog_for(ModelConfig(services=()))) == 0
        assert len(catalog_for(ModelConfig(services=None))) == len(builtin_catalog())

    def test_user_catalog_still_merges_under_a_filter(self, tmp_path: Path) -> None:
        extra = tmp_path / "mine.json"
        extra.write_text(
            json.dumps([{"id": "mine/model-1", "service": "openrouter", "name": "Mine"}]),
            encoding="utf-8",
        )
        catalog = catalog_for(
            ModelConfig(catalog_path=extra, services=("openrouter",))
        )
        assert "mine/model-1" in catalog
        assert {m.service for m in catalog} == {"openrouter"}

    def test_filtering_does_not_affect_capability_lookups(self) -> None:
        # The contract that matters: a flag must still be findable for a model whose
        # service a UI has filtered out. Provider code reads load_catalog(), not
        # catalog_for(), and this test fails if that ever gets "tidied up".
        config = ModelConfig(services=("openrouter",))
        assert catalog_for(config).get("claude-sonnet-5") is None
        assert load_catalog(config.catalog_path).get("claude-sonnet-5") is not None

    def test_chaining_onto_a_modality(self) -> None:
        catalog = catalog_for(ModelConfig(services=("huggingface",)))
        assert all(m.service == "huggingface" for m in catalog.for_modality("image"))


class TestAvailableServices:
    def test_full_environment_reports_everything(self) -> None:
        assert set(available_services(env=FULL_ENV)) == {
            "anthropic", "openrouter", "ollama", "ollama-local", "huggingface",
        }

    def test_only_openrouter(self) -> None:
        # The motivating case: one account, and a dropdown that shows only what it can call.
        assert available_services(env={"OPENROUTER_API_KEY": "k"}) == ("openrouter",)

    def test_empty_environment_reports_nothing(self) -> None:
        assert available_services(env={}) == ()

    def test_local_ollama_needs_a_host_not_a_key(self) -> None:
        assert available_services(env={"LOCAL_OLLAMA_URL": "http://x:11434"}) == (
            "ollama-local",
        )

    def test_blank_value_does_not_count(self) -> None:
        assert available_services(env={"OPENROUTER_API_KEY": "   "}) == ()

    @pytest.mark.parametrize(
        "name", ["HF_TOKEN", "HUGGINGFACE_HUB_KEY", "HUGGINGFACEHUB_API_TOKEN"]
    )
    def test_any_accepted_token_name_counts(self, name: str) -> None:
        assert available_services(env={name: "k"}) == ("huggingface",)

    def test_composes_into_a_config(self) -> None:
        # The documented one-liner, end to end.
        config = ModelConfig(services=available_services(env={"OPENROUTER_API_KEY": "k"}))
        assert {m.service for m in catalog_for(config)} == {"openrouter"}
