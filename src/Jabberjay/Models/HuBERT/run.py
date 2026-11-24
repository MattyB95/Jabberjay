import numpy as np

from Jabberjay.Utilities.pipeline import run_pipeline
from Jabberjay.Utilities.types import PredictionScore

MODEL_ID = "abhishtagatya/hubert-base-960h-itw-deepfake"


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    return run_pipeline(MODEL_ID, y, sr, "HuBERT")
