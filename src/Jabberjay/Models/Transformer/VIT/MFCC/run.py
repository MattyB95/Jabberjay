from typing import cast

import librosa
import numpy as np
from loguru import logger
from transformers import pipeline

from Jabberjay.Models.Transformer.VIT.utility import get_image
from Jabberjay.Utilities.enum_handler import Dataset
from Jabberjay.Utilities.label_normalizer import normalize_pipeline_scores
from Jabberjay.Utilities.types import PredictionScore


def predict(audio: tuple[np.ndarray, float], dataset: Dataset) -> list[PredictionScore]:
    y, sr = audio
    model = f"MattyB95/VIT-{dataset.value}-MFCC-Synthetic-Voice-Detection"
    logger.info(f"Loading VIT model: {model}")
    pipe = pipeline(task="image-classification", model=model)
    logger.debug("Computing MFCC")
    M = librosa.feature.mfcc(y=y, sr=sr)
    image = get_image(data=M, sr=sr)
    raw = cast(list[dict[str, object]], pipe(image))
    return normalize_pipeline_scores(raw)
