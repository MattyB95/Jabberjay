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


class TestThreadSafety:
    def test_concurrent_misses_load_only_once(self):
        import threading
        import time

        calls = []
        barrier = threading.Barrier(4)

        @model_cache.cached_loader(maxsize=4)
        def load(x):
            calls.append(x)
            time.sleep(0.05)  # a slow load, so all threads miss together
            return object()

        results = []

        def worker():
            barrier.wait()
            results.append(load(1))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert calls == [1]
        assert all(r is results[0] for r in results)

    def test_cache_info_is_exposed(self):
        @model_cache.cached_loader(maxsize=4)
        def load(x):
            return x

        load(1)
        load(1)
        info = load.cache_info()  # ty: ignore[missing-attribute]
        assert (info.hits, info.misses) == (1, 1)


class TestReleaseMemory:
    def test_collects_garbage_and_empties_accelerator_caches(self):
        from unittest.mock import patch

        with (
            patch("gc.collect") as mock_gc,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_cuda,
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.mps.empty_cache") as mock_mps,
        ):
            model_cache.release_memory()
        mock_gc.assert_called_once()
        mock_cuda.assert_called_once()
        mock_mps.assert_called_once()

    def test_skips_unavailable_accelerators(self):
        from unittest.mock import patch

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.empty_cache") as mock_cuda,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.mps.empty_cache") as mock_mps,
        ):
            model_cache.release_memory()
        mock_cuda.assert_not_called()
        mock_mps.assert_not_called()
