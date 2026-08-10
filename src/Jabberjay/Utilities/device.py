import torch
from loguru import logger


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")
    return device
