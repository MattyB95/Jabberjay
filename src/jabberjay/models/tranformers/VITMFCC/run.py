import librosa
import numpy as np
from transformers import pipeline

from ..utility import get_image


def predict(y: np.ndarray, sr: float) -> list[dict[str, float]]:
    pipe = pipeline("image-classification", model="MattyB95/VIT-MFCC-Synthetic-Voice-Detection")
    M = librosa.feature.mfcc(y=y, sr=sr)
    image = get_image(data=M, sr=sr)
    return pipe(image)
