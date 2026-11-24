from collections.abc import Mapping
from typing import Literal

import numpy as np
import torch
from loguru import logger
from transformers import Pipeline, pipeline

from Jabberjay.Utilities.device import get_device
from Jabberjay.Utilities.hugging_face import revision_for
from Jabberjay.Utilities.label_normalizer import normalize_pipeline_scores
from Jabberjay.Utilities.model_cache import cached_loader
from Jabberjay.Utilities.types import Label, PredictionScore


@cached_loader(maxsize=8)
def load_pipeline(
    task: Literal["audio-classification", "image-classification"], model_id: str
) -> Pipeline:
    """Load and cache a transformers pipeline at the model's pinned revision."""
    device = get_device()
    revision = revision_for(model_id)
    logger.info(f"Loading model: {model_id}@{revision[:7]} on {device}")
    return pipeline(
        task,
        model=model_id,
        revision=revision,
        device=torch.device(device),
    )


def run_pipeline(
    model_id: str,
    y: np.ndarray,
    sr: float,
    model_name: str,
    label_map: Mapping[str, Label] | None = None,
) -> list[PredictionScore]:
    """Run a cached transformers audio-classification pipeline on ``y``.

    The pipeline resamples ``y`` from ``sr`` to the rate its feature
    extractor expects, so callers can pass audio at any sample rate.
    """
    pipe = load_pipeline("audio-classification", model_id)
    logger.debug(f"Running {model_name} inference on {len(y)} samples at {int(sr)}Hz")
    raw = pipe({"raw": y, "sampling_rate": int(sr)})
    return normalize_pipeline_scores(raw, label_map=label_map)
