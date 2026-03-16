from typing import cast

import numpy as np
from loguru import logger
from transformers import pipeline

from Jabberjay.Utilities.enum_handler import Dataset
from Jabberjay.Utilities.label_normalizer import normalize_pipeline_scores
from Jabberjay.Utilities.types import PredictionScore


def predict(y: np.ndarray, sr: float, dataset: Dataset) -> list[PredictionScore]:
    model = f"MattyB95/AST-{dataset.value}-Synthetic-Voice-Detection"
    logger.info(f"Loading AST model: {model}")
    pipe = pipeline("audio-classification", model=model)
    logger.debug(f"Running AST inference on {len(y)} samples at {int(sr)}Hz")
    raw = cast(list[dict[str, object]], pipe({"raw": y, "sampling_rate": int(sr)}))
    return normalize_pipeline_scores(raw)
