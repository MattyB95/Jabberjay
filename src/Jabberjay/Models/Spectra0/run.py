import numpy as np
import torch
from loguru import logger

from Jabberjay.Models.Spectra.shared import _TARGET_SR, load_pretrained, preprocess
from Jabberjay.Models.Spectra0.model import Spectra0Model
from Jabberjay.Utilities.device import get_device
from Jabberjay.Utilities.types import PredictionScore

_MODEL_ID = "lab260/spectra_0"
# lab260's own default in Spectra0Model.classify() — documented on the model
# card as EER-tuned on their eval set: https://huggingface.co/lab260/spectra_0
_THRESHOLD = -1.0625009


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    device = get_device()
    model = load_pretrained(Spectra0Model, _MODEL_ID, device)
    audio = preprocess(y, sr).to(device)
    logger.debug(
        f"Running Spectra-0 inference on {audio.shape[1]} samples at {_TARGET_SR}Hz"
    )
    with torch.inference_mode():
        bonafide_logit = model(audio)[0, 1]
        bonafide_prob = torch.sigmoid(bonafide_logit - _THRESHOLD)
    return [
        {"label": "Spoof", "score": float(1 - bonafide_prob)},
        {"label": "Bonafide", "score": float(bonafide_prob)},
    ]
