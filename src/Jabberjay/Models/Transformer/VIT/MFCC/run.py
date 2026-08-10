import librosa
import numpy as np
from loguru import logger

from Jabberjay.Models.Transformer.VIT.utility import get_image, load_pipeline
from Jabberjay.Utilities.enum_handler import Dataset
from Jabberjay.Utilities.label_normalizer import normalize_pipeline_scores
from Jabberjay.Utilities.types import PredictionScore


def predict(audio: tuple[np.ndarray, float], dataset: Dataset) -> list[PredictionScore]:
    y, sr = audio
    model = f"MattyB95/VIT-{dataset.value}-MFCC-Synthetic-Voice-Detection"
    pipe = load_pipeline(model)
    logger.debug("Computing MFCC")
    M = librosa.feature.mfcc(y=y, sr=sr)
    image = get_image(data=M, sr=sr)
    raw = pipe(image)
    return normalize_pipeline_scores(raw)
