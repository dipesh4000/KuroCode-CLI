"""
Unit tests for kurocode.model_registry.

Happy-path tests are pure (no I/O, no network).
Disk-cache tests use ``tmp_path`` — real in-memory filesystem, no mocking.
Network-failure tests inject a lightweight test-double to avoid real I/O.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from kurocode.core.model_registry import (
    ModelInfo,
    ModelRegistry,
    _model_from_dict,
    _model_to_dict,
    _parse_model,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FREE_RAW: dict[str, Any] = {
    "id": "meta-llama/llama-3.1-8b-instruct:free",
    "name": "Llama 3.1 8B Instruct (free)",
    "context_length": 131072,
    "pricing": {"prompt": "0", "completion": "0"},
    "description": "A great free model.",
}

_PAID_RAW: dict[str, Any] = {
    "id": "openai/gpt-4o",
    "name": "GPT-4o",
    "context_length": 128000,
    "pricing": {"prompt": "0.000005", "completion": "0.000015"},
    "description": "A paid model.",
}


def _make_model(idx: int = 0) -> ModelInfo:
    return ModelInfo(
        id=f"vendor/model-{idx}:free",
        name=f"Model {idx}",
        context_length=4096,
        pricing={"prompt": "0", "completion": "0"},
    )


def _disk_payload(models: list[ModelInfo] | None = None) -> str:
    models = models or [_make_model(0)]
    return json.dumps([_model_to_dict(m) for m in models])


# ---------------------------------------------------------------------------
# ModelInfo — property & immutability
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_is_free_true(self) -> None:
        m = ModelInfo(
            id="a/b", name="B", context_length=4096,
            pricing={"prompt": "0", "completion": "0"},
        )
        assert m.is_free is True

    def test_is_free_false_prompt_nonzero(self) -> None:
        m = ModelInfo(
            id="a/b", name="B", context_length=4096,
            pricing={"prompt": "0.001", "completion": "0"},
        )
        assert m.is_free is False

    def test_is_free_false_completion_nonzero(self) -> None:
        m = ModelInfo(
            id="a/b", name="B", context_length=4096,
            pricing={"prompt": "0", "completion": "0.002"},
        )
        assert m.is_free is False

    def test_is_free_false_missing_keys(self) -> None:
        m = ModelInfo(id="a/b", name="B", context_length=4096, pricing={})
        assert m.is_free is False

    def test_frozen_rejects_mutation(self) -> None:
        m = ModelInfo(id="a/b", name="B", context_length=8192)
        with pytest.raises((AttributeError, TypeError)):
            m.id = "x"  # type: ignore[misc]

    def test_default_pricing_empty(self) -> None:
        m = ModelInfo(id="x", name="X", context_length=0)
        assert m.pricing == {}
        assert m.is_free is False


# ---------------------------------------------------------------------------
# _parse_model — pure function, zero I/O
# ---------------------------------------------------------------------------


class TestParseModel:
    def test_parses_all_fields(self) -> None:
        m = _parse_model(_FREE_RAW)
        assert m.id == "meta-llama/llama-3.1-8b-instruct:free"
        assert m.name == "Llama 3.1 8B Instruct (free)"
        assert m.context_length == 131072
        assert m.pricing == {"prompt": "0", "completion": "0"}
        assert m.description == "A great free model."

    def test_name_falls_back_to_id_when_absent(self) -> None:
        raw = {k: v for k, v in _FREE_RAW.items() if k != "name"}
        m = _parse_model(raw)
        assert m.name == _FREE_RAW["id"]

    def test_pricing_values_cast_to_str(self) -> None:
        raw = {**_FREE_RAW, "pricing": {"prompt": 0, "completion": 0}}
        m = _parse_model(raw)
        assert m.pricing == {"prompt": "0", "completion": "0"}

    def test_missing_context_length_defaults_to_zero(self) -> None:
        raw = {k: v for k, v in _FREE_RAW.items() if k != "context_length"}
        m = _parse_model(raw)
        assert m.context_length == 0

    def test_missing_description_defaults_to_empty(self) -> None:
        raw = {k: v for k, v in _FREE_RAW.items() if k != "description"}
        m = _parse_model(raw)
        assert m.description == ""

    def test_paid_model_is_not_free(self) -> None:
        m = _parse_model(_PAID_RAW)
        assert m.is_free is False


# ---------------------------------------------------------------------------
# _model_to_dict / _model_from_dict — round-trip, zero I/O
# ---------------------------------------------------------------------------


class TestModelSerialisation:
    def test_round_trip_preserves_all_fields(self) -> None:
        original = ModelInfo(
            id="a/b:free",
            name="A B",
            context_length=8192,
            pricing={"prompt": "0", "completion": "0"},
            description="desc",
        )
        restored = _model_from_dict(_model_to_dict(original))
        assert restored == original

    def test_round_trip_default_description(self) -> None:
        original = ModelInfo(id="x", name="X", context_length=0)
        restored = _model_from_dict(_model_to_dict(original))
        assert restored.description == ""


# ---------------------------------------------------------------------------
# ModelRegistry — in-memory cache (pure, zero I/O)
# ---------------------------------------------------------------------------


class TestModelRegistryMemoryCache:
    async def test_returns_memory_cache_same_object(self, tmp_path: Path) -> None:
        registry = ModelRegistry(cache_path=tmp_path / "cache.json")
        models = [_make_model(0), _make_model(1)]
        registry._memory = models

        result = await registry.fetch()
        assert result is models  # exact same list — no copy, no I/O

    async def test_memory_cache_skipped_on_force_refresh(
        self, tmp_path: Path
    ) -> None:
        """``force_refresh=True`` must bypass memory cache and attempt network."""

        class _FailingClient:
            async def get(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
                raise httpx.ConnectError("simulated failure")

        registry = ModelRegistry(
            cache_path=tmp_path / "cache.json",
            http_client=_FailingClient(),  # type: ignore[arg-type]
        )
        registry._memory = [_make_model(99)]  # stale — should be ignored

        # No disk cache either → exception must propagate
        with pytest.raises(httpx.ConnectError):
            await registry.fetch(force_refresh=True)

    async def test_second_fetch_uses_memory_without_network(
        self, tmp_path: Path
    ) -> None:
        """After a successful fetch the registry must not hit the network again."""
        call_count = 0

        class _CountingClient:
            async def get(self, *_: object, **__: object) -> MagicMock:
                nonlocal call_count
                call_count += 1
                mock = MagicMock()
                mock.raise_for_status = lambda: None
                mock.json.return_value = {"data": [_FREE_RAW]}
                return mock

        registry = ModelRegistry(
            cache_path=tmp_path / "cache.json",
            http_client=_CountingClient(),  # type: ignore[arg-type]
        )

        await registry.fetch()
        await registry.fetch()
        assert call_count == 1  # second call served from memory


# ---------------------------------------------------------------------------
# ModelRegistry — disk cache (tmp_path, real filesystem, no mocking)
# ---------------------------------------------------------------------------


class TestModelRegistryDiskCache:
    def test_load_returns_none_when_absent(self, tmp_path: Path) -> None:
        registry = ModelRegistry(cache_path=tmp_path / "nonexistent.json")
        assert registry._load_disk_cache() is None

    def test_load_returns_models_when_present(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(_disk_payload([_make_model(0)]), encoding="utf-8")

        registry = ModelRegistry(cache_path=cache_file)
        models = registry._load_disk_cache()
        assert models is not None
        assert len(models) == 1
        assert models[0].id == "vendor/model-0:free"

    def test_load_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("not valid json", encoding="utf-8")

        registry = ModelRegistry(cache_path=cache_file)
        assert registry._load_disk_cache() is None

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "sub" / "cache.json"  # parent doesn't exist yet
        registry = ModelRegistry(cache_path=cache_file)
        models = [
            ModelInfo(
                id="vendor/x:free",
                name="X",
                context_length=2048,
                pricing={"prompt": "0", "completion": "0"},
            )
        ]
        registry._save_disk_cache(models)

        assert cache_file.exists()
        loaded = json.loads(cache_file.read_text(encoding="utf-8"))
        assert loaded[0]["id"] == "vendor/x:free"
        assert loaded[0]["context_length"] == 2048

    def test_save_is_idempotent(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.json"
        registry = ModelRegistry(cache_path=cache_file)
        models = [_make_model(0)]
        registry._save_disk_cache(models)
        registry._save_disk_cache(models)  # second save must not raise

        loaded = json.loads(cache_file.read_text(encoding="utf-8"))
        assert len(loaded) == 1

    async def test_offline_fallback_returns_disk_cache(
        self, tmp_path: Path
    ) -> None:
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(_disk_payload([_make_model(7)]), encoding="utf-8")

        class _FailingClient:
            async def get(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
                raise httpx.ConnectError("offline")

        registry = ModelRegistry(
            cache_path=cache_file,
            http_client=_FailingClient(),  # type: ignore[arg-type]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            models = await registry.fetch()

        assert len(models) == 1
        assert models[0].id == "vendor/model-7:free"
        assert any("stale model cache" in str(w.message) for w in caught)

    async def test_offline_no_cache_raises(self, tmp_path: Path) -> None:
        class _FailingClient:
            async def get(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
                raise httpx.ConnectError("offline")

        registry = ModelRegistry(
            cache_path=tmp_path / "nonexistent.json",
            http_client=_FailingClient(),  # type: ignore[arg-type]
        )
        with pytest.raises(httpx.ConnectError):
            await registry.fetch()


# ---------------------------------------------------------------------------
# ModelRegistry — network happy path (injected AsyncClient)
# ---------------------------------------------------------------------------


class TestModelRegistryNetworkFetch:
    async def test_filters_to_free_models_only(self, tmp_path: Path) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {"data": [_FREE_RAW, _PAID_RAW]}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        registry = ModelRegistry(
            cache_path=tmp_path / "cache.json",
            http_client=mock_client,
        )
        models = await registry.fetch()

        assert len(models) == 1
        assert models[0].id == _FREE_RAW["id"]

    async def test_result_is_written_to_disk(self, tmp_path: Path) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {"data": [_FREE_RAW]}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        cache_file = tmp_path / "cache.json"
        registry = ModelRegistry(cache_path=cache_file, http_client=mock_client)
        await registry.fetch()

        assert cache_file.exists()
        loaded = json.loads(cache_file.read_text("utf-8"))
        assert loaded[0]["id"] == _FREE_RAW["id"]

    async def test_empty_data_returns_empty_list(self, tmp_path: Path) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        registry = ModelRegistry(
            cache_path=tmp_path / "cache.json", http_client=mock_client
        )
        models = await registry.fetch()
        assert models == []
