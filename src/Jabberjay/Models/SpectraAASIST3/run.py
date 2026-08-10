import numpy as np
import torch
from loguru import logger

from Jabberjay.Models.Spectra.shared import _TARGET_SR, load_pretrained, preprocess
from Jabberjay.Models.SpectraAASIST3.model import SpectraAASIST3
from Jabberjay.Utilities.device import get_device
from Jabberjay.Utilities.types import PredictionScore

_MODEL_ID = "lab260/Spectra-AASIST3"


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    device = get_device()
    model = load_pretrained(SpectraAASIST3, _MODEL_ID, device)
    audio = preprocess(y, sr).to(device)
    logger.debug(
        f"Running Spectra-AASIST3 inference on {audio.shape[1]} samples at {_TARGET_SR}Hz"
    )
    with torch.inference_mode():
        probs = torch.softmax(model(audio), dim=1)[0]
    return [
        {"label": "Spoof", "score": float(probs[0])},
        {"label": "Bonafide", "score": float(probs[1])},
    ]
