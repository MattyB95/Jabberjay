import librosa
import numpy as np
from transformers import pipeline

from ..utility import get_image


def predict(y: np.ndarray, sr: float) -> list[dict[str, float]]:
    pipe = pipeline(task="image-classification", model="MattyB95/VIT-Mel_Spectrogram-Synthetic-Voice-Detection")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S=S, ref=np.max)
    image = get_image(data=S_db, sr=sr)
    return pipe(image)
