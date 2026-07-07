"""Unit tests for the cached_loader/clear_all model cache registry."""

import pytest

from Jabberjay.Utilities import model_cache


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Give each test its own registry so loaders defined here don't leak
    into (or get cleared by) the real per-test cache-clearing fixture."""
    monkeypatch.setattr(model_cache, "_registry", [])


class TestCachedLoader:
    def test_repeated_call_with_same_arguments_hits_cache(self):
        calls = []

        @model_cache.cached_loader(maxsize=4)
        def load(x):
            calls.append(x)
            return x * 2

        assert load(1) == 2
        assert load(1) == 2
        assert calls == [1]

    def test_different_arguments_are_not_cached_together(self):
        calls = []

        @model_cache.cached_loader(maxsize=4)
        def load(x):
            calls.append(x)
            return x

        load(1)
        load(2)
        load(1)
        assert calls == [1, 2]

    def test_maxsize_evicts_least_recently_used(self):
        calls = []

        @model_cache.cached_loader(maxsize=2)
        def load(x):
            calls.append(x)
            return x

        load(1)
        load(2)
        load(3)  # evicts 1 (least recently used)
        load(1)  # cache miss again
        assert calls == [1, 2, 3, 1]


class TestClearAll:
    def test_clear_all_forces_a_reload(self):
        calls = []

        @model_cache.cached_loader(maxsize=4)
        def load(x):
            calls.append(x)
            return x

        load(1)
        model_cache.clear_all()
        load(1)
        assert calls == [1, 1]

    def test_clear_all_clears_every_registered_loader(self):
        calls_a: list[int] = []
        calls_b: list[int] = []

        @model_cache.cached_loader(maxsize=4)
        def load_a(x):
            calls_a.append(x)
            return x

        @model_cache.cached_loader(maxsize=4)
        def load_b(x):
            calls_b.append(x)
            return x

        load_a(1)
        load_b(1)
        model_cache.clear_all()
        load_a(1)
        load_b(1)
        assert calls_a == [1, 1]
        assert calls_b == [1, 1]
