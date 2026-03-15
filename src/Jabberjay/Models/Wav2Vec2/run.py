from typing import cast

import numpy as np
from loguru import logger
from transformers import pipeline

from Jabberjay.Utilities.label_normalizer import normalize_pipeline_scores
from Jabberjay.Utilities.types import PredictionScore

_MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
_TARGET_SR = 16_000


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    logger.info(f"Loading Wav2Vec2 model: {_MODEL_ID}")
    pipe = pipeline("audio-classification", model=_MODEL_ID, sampling_rate=_TARGET_SR)
    logger.debug(f"Running Wav2Vec2 inference on {len(y)} samples at {int(sr)}Hz")
    raw = cast(list[dict[str, object]], pipe({"raw": y, "sampling_rate": int(sr)}))
    return normalize_pipeline_scores(raw)
