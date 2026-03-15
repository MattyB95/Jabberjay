from typing import cast

import librosa
import numpy as np
from loguru import logger
from transformers import pipeline

from Jabberjay.Models.Transformer.VIT.utility import get_image
from Jabberjay.Utilities.enum_handler import Dataset
from Jabberjay.Utilities.types import PredictionScore


def predict(audio: tuple[np.ndarray, float], dataset: Dataset) -> list[PredictionScore]:
    y, sr = audio
    model = f"MattyB95/VIT-{dataset.value}-Mel_Spectrogram-Synthetic-Voice-Detection"
    logger.info(f"Loading VIT model: {model}")
    pipe = pipeline(task="image-classification", model=model)
    logger.debug("Computing Mel spectrogram")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S=S, ref=np.max)
    image = get_image(data=S_db, sr=sr)
    return cast(list[PredictionScore], pipe(image))
