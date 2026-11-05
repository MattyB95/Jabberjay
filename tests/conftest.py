import matplotlib
import pytest

# Force the non-interactive Agg backend before any test imports matplotlib.
# The uv-managed Python distribution does not bundle Tk, so the default
# TkAgg backend raises TclError when plt.subplots() is called in tests.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _clear_model_caches(request):
    """Reset every cached_loader() cache around each unit test.

    Model/pipeline loaders are cached by (model, args) for performance, but
    tests reuse the same model id with different mocked return values across
    cases — without clearing, a later test would see an earlier test's
    cached (and now stale) mock instead of exercising its own patch.

    Integration tests are exempt: they load real multi-GB weights, so they
    share one cache across cases. Clearing after each unit test as well keeps
    a mock from ever leaking into an integration test in a mixed run.
    """
    from Jabberjay.Utilities import model_cache

    if request.node.get_closest_marker("integration"):
        yield
        return
    model_cache.clear_all()
    yield
    model_cache.clear_all()
