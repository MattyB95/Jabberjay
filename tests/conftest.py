import matplotlib

# Force the non-interactive Agg backend before any test imports matplotlib.
# The uv-managed Python distribution does not bundle Tk, so the default
# TkAgg backend raises TclError when plt.subplots() is called in tests.
matplotlib.use("Agg")
