import librosa
import numpy as np
from transformers import pipeline

from ..utility import get_image


def predict(y: np.ndarray, sr: float) -> list[dict[str, float]]:
    pipe = pipeline(task="image-classification", model="MattyB95/VIT-ConstantQ-Synthetic-Voice-Detection")
    CQT = np.abs(librosa.cqt(y=y, sr=sr))
    S_db = librosa.amplitude_to_db(S=CQT, ref=np.max)
    image = get_image(data=S_db, sr=sr)
    return pipe(image)
