import torch
from loguru import logger


def get_device() -> str:
    """Return the best available torch device: 'cuda', 'mps', or 'cpu'.

    Apple Silicon (``mps``) is preferred over CPU when present. If a model
    hits an operation MPS does not yet implement, set the environment
    variable ``PYTORCH_ENABLE_MPS_FALLBACK=1`` to fall back to CPU for
    those ops.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.debug(f"Using device: {device}")
    return device
