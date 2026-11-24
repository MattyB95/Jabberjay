import numpy as np

from Jabberjay.Utilities.pipeline import run_pipeline
from Jabberjay.Utilities.types import PredictionScore

MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    return run_pipeline(MODEL_ID, y, sr, "Wav2Vec2")
