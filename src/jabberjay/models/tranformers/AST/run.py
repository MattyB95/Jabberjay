import numpy as np
from transformers import pipeline


def predict(y: np.ndarray, sr: float) -> list[dict[str, float]]:
    pipe = pipeline("audio-classification", model="MattyB95/AST-Synthetic-Voice-Detection")
    return pipe(y)
