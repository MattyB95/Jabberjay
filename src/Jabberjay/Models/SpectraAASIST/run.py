import numpy as np
import torch
from loguru import logger

from Jabberjay.Models.Spectra.shared import _TARGET_SR, load_pretrained, preprocess
from Jabberjay.Models.SpectraAASIST.model import SpectraAASIST
from Jabberjay.Utilities.device import get_device
from Jabberjay.Utilities.types import PredictionScore

_MODEL_ID = "lab260/Spectra-AASIST"
# The model card documents -1.140625 as the EER-tuned threshold, but
# SpectraAASIST.classify()'s own default parameter is -1.0625009 — a stale
# value carried over from Spectra0's model.py. We use the documented value
# rather than the buggy code default: https://huggingface.co/lab260/Spectra-AASIST
_THRESHOLD = -1.140625


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    device = get_device()
    model = load_pretrained(SpectraAASIST, _MODEL_ID, device)
    audio = preprocess(y, sr).to(device)
    logger.debug(
        f"Running Spectra-AASIST inference on {audio.shape[1]} samples at {_TARGET_SR}Hz"
    )
    with torch.inference_mode():
        bonafide_logit = model(audio)[0, 1]
        bonafide_prob = torch.sigmoid(bonafide_logit - _THRESHOLD)
    return [
        {"label": "Spoof", "score": float(1 - bonafide_prob)},
        {"label": "Bonafide", "score": float(bonafide_prob)},
    ]
