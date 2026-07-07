import numpy as np
from loguru import logger
from transformers import Pipeline, pipeline

from Jabberjay.Utilities.label_normalizer import normalize_pipeline_scores
from Jabberjay.Utilities.model_cache import cached_loader
from Jabberjay.Utilities.types import PredictionScore


@cached_loader(maxsize=8)
def _load_pipeline(model_id: str, sampling_rate: int | None) -> Pipeline:
    """Load and cache a transformers audio-classification pipeline by model id."""
    logger.info(f"Loading model: {model_id}")
    kwargs: dict = {"sampling_rate": sampling_rate} if sampling_rate is not None else {}
    return pipeline("audio-classification", model=model_id, **kwargs)


def run_pipeline(
    model_id: str,
    y: np.ndarray,
    sr: float,
    model_name: str,
    sampling_rate: int | None = None,
) -> list[PredictionScore]:
    """Load a transformers audio-classification pipeline and run inference.

    The pipeline is cached per (model_id, sampling_rate) so repeated calls for
    the same model reuse already-loaded weights instead of reloading them.
    """
    pipe = _load_pipeline(model_id, sampling_rate)
    logger.debug(f"Running {model_name} inference on {len(y)} samples at {int(sr)}Hz")
    raw = pipe({"raw": y, "sampling_rate": int(sr)})
    return normalize_pipeline_scores(raw)
