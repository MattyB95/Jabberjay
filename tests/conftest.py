import matplotlib
import pytest

# Force the non-interactive Agg backend before any test imports matplotlib.
# The uv-managed Python distribution does not bundle Tk, so the default
# TkAgg backend raises TclError when plt.subplots() is called in tests.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _clear_model_caches():
    """Reset every cached_loader() cache before each test.

    Model/pipeline loaders are cached by (model, args) for performance, but
    tests reuse the same model id with different mocked return values across
    cases — without clearing, a later test would see an earlier test's
    cached (and now stale) mock instead of exercising its own patch.
    """
    from Jabberjay.Utilities import model_cache

    model_cache.clear_all()
    yield
