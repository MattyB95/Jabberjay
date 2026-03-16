import numpy as np
from loguru import logger
from transformers import pipeline

from Jabberjay.Utilities.label_normalizer import normalize_pipeline_scores
from Jabberjay.Utilities.types import PredictionScore

_MODEL_ID = "DavidCombei/wavLM-base-Deepfake_V2"
_TARGET_SR = 16_000


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    logger.info(f"Loading WavLM model: {_MODEL_ID}")
    pipe = pipeline("audio-classification", model=_MODEL_ID, sampling_rate=_TARGET_SR)
    logger.debug(f"Running WavLM inference on {len(y)} samples at {int(sr)}Hz")
    raw = pipe({"raw": y, "sampling_rate": int(sr)})
    return normalize_pipeline_scores(raw)
