import numpy as np

from Jabberjay.Utilities.pipeline import run_pipeline
from Jabberjay.Utilities.types import PredictionScore

_MODEL_ID = "DavidCombei/wavLM-base-Deepfake_V2"
_TARGET_SR = 16_000


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    return run_pipeline(_MODEL_ID, y, sr, "WavLM", sampling_rate=_TARGET_SR)
