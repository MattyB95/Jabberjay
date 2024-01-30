import logging

import librosa
import numpy as np
from transformers import pipeline

from Jabberjay.Models.Tranformer.VIT.utility import get_image
from Jabberjay.Utilities.enum_handler import Dataset


def predict(
    audio: tuple[np.ndarray, float], dataset: Dataset
) -> list[dict[str, float]]:
    y, sr = audio
    model = f"MattyB95/VIT-{dataset.value}-ConstantQ-Synthetic-Voice-Detection"
    logging.info(f"Using Model: {model}")
    pipe = pipeline(task="image-classification", model=model)
    CQT = np.abs(librosa.cqt(y=y, sr=sr))
    S_db = librosa.amplitude_to_db(S=CQT, ref=np.max)
    image = get_image(data=S_db, sr=sr)
    return pipe(image)
