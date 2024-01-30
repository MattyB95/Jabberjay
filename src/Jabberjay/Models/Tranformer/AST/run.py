import logging

import numpy as np
from transformers import pipeline

from Jabberjay.Utilities.enum_handler import Dataset


def predict(y: np.ndarray, dataset: Dataset) -> list[dict[str, float]]:
    model = f"MattyB95/AST-{dataset.value}-Synthetic-Voice-Detection"
    logging.info(f"Using model: {model}")
    pipe = pipeline("audio-classification", model=model)
    return pipe(y)
