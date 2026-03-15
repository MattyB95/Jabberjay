import numpy as np
from loguru import logger
from transformers import pipeline

from Jabberjay.Utilities.enum_handler import Dataset


def predict(y: np.ndarray, sr: float, dataset: Dataset) -> list[dict[str, float]]:
    model = f"MattyB95/AST-{dataset.value}-Synthetic-Voice-Detection"
    logger.info(f"Loading AST model: {model}")
    pipe = pipeline("audio-classification", model=model)
    logger.debug(f"Running AST inference on {len(y)} samples at {int(sr)}Hz")
    return pipe({"raw": y, "sampling_rate": int(sr)})
