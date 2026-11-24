"""Shared caching helper for expensive model/pipeline loads.

Wraps functools.lru_cache and keeps a registry of every cached loader so
callers (and tests) can reset all of them at once via clear_all().
"""

import functools
import gc
import threading
from collections.abc import Callable

_registry: list[Callable[[], None]] = []


def cached_loader[F: Callable](maxsize: int = 8) -> Callable[[F], F]:
    """functools.lru_cache that also registers itself for clear_all().

    Calls are serialised per loader: lru_cache alone lets two threads that
    miss at the same moment both run the (multi-GB, multi-second) load, so a
    lock makes the second caller wait for, then reuse, the first one's result.
    """

    def decorator(func: F) -> F:
        cached = functools.lru_cache(maxsize=maxsize)(func)
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return cached(*args, **kwargs)

        def cache_clear() -> None:
            with lock:
                cached.cache_clear()

        wrapper.cache_clear = cache_clear  # ty: ignore[missing-attribute]
        wrapper.cache_info = cached.cache_info  # ty: ignore[missing-attribute]
        _registry.append(cache_clear)
        return wrapper  # ty: ignore[invalid-explicit-return]

    return decorator


def clear_all() -> None:
    """Clear every cache created via cached_loader()."""
    for clear in _registry:
        clear()


def release_memory() -> None:
    """Return freed model memory to the OS and the accelerator.

    Dropping cache references (clear_all) leaves the tensors to the garbage
    collector and to torch's caching allocator, so the memory isn't actually
    handed back until later; collect and empty both now. Kept separate from
    clear_all() because a full collection is slow to repeat (tests clear the
    caches around every case).
    """
    gc.collect()
    # Imported lazily: this module is imported by tests that never touch torch
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
