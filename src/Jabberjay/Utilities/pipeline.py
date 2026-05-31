import numpy as np
from loguru import logger
from transformers import pipeline

from Jabberjay.Utilities.label_normalizer import normalize_pipeline_scores
from Jabberjay.Utilities.types import PredictionScore


def run_pipeline(
    model_id: str,
    y: np.ndarray,
    sr: float,
    model_name: str,
    sampling_rate: int | None = None,
) -> list[PredictionScore]:
    """Load a transformers audio-classification pipeline and run inference."""
    logger.info(f"Loading {model_name} model: {model_id}")
    kwargs: dict = {"sampling_rate": sampling_rate} if sampling_rate is not None else {}
    pipe = pipeline("audio-classification", model=model_id, **kwargs)
    logger.debug(f"Running {model_name} inference on {len(y)} samples at {int(sr)}Hz")
    raw = pipe({"raw": y, "sampling_rate": int(sr)})
    return normalize_pipeline_scores(raw)
