"""Shared caching helper for expensive model/pipeline loads.

Wraps functools.lru_cache and keeps a registry of every cached loader so
callers (and tests) can reset all of them at once via clear_all().
"""

import functools
from collections.abc import Callable
from typing import TypeVar

_F = TypeVar("_F", bound=Callable)

_registry: list[Callable[[], None]] = []


def cached_loader(maxsize: int = 8) -> Callable[[_F], _F]:
    """functools.lru_cache that also registers itself for clear_all()."""

    def decorator(func: _F) -> _F:
        wrapped = functools.lru_cache(maxsize=maxsize)(func)
        _registry.append(wrapped.cache_clear)
        return wrapped  # ty: ignore[invalid-return-type]

    return decorator


def clear_all() -> None:
    """Clear every cache created via cached_loader(). Mainly useful in tests."""
    for clear in _registry:
        clear()
