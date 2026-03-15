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
    model = f"MattyB95/VIT-{dataset.value}-ConstantQ-Synthetic-Voice-Detection"
    logger.info(f"Loading VIT model: {model}")
    pipe = pipeline(task="image-classification", model=model)
    logger.debug("Computing Constant-Q Transform")
    CQT = np.abs(librosa.cqt(y=y, sr=sr))
    S_db = librosa.amplitude_to_db(S=CQT, ref=np.max)
    image = get_image(data=S_db, sr=sr)
    return pipe(image)
