import librosa
import numpy as np
from loguru import logger
from transformers import pipeline

from Jabberjay.Models.Transformer.VIT.utility import get_image
from Jabberjay.Utilities.enum_handler import Dataset


def predict(
    audio: tuple[np.ndarray, float], dataset: Dataset
) -> list[dict[str, float]]:
    y, sr = audio
    model = f"MattyB95/VIT-{dataset.value}-MFCC-Synthetic-Voice-Detection"
    logger.info(f"Loading VIT model: {model}")
    pipe = pipeline("image-classification", model=model)
    logger.debug("Computing MFCC")
    M = librosa.feature.mfcc(y=y, sr=sr)
    image = get_image(data=M, sr=sr)
    return pipe(image)
