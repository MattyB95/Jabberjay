import numpy as np
import torch
from loguru import logger

from Jabberjay.Models.Spectra.shared import _TARGET_SR, preprocess
from Jabberjay.Models.SpectraAASIST.model import SpectraAASIST
from Jabberjay.Utilities.types import PredictionScore

_MODEL_ID = "lab260/Spectra-AASIST"


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Spectra-AASIST model: {_MODEL_ID}")
    logger.debug(f"Using device: {device}")
    model = SpectraAASIST.from_pretrained(_MODEL_ID).eval().to(device)
    audio = preprocess(y, sr).to(device)
    logger.debug(
        f"Running Spectra-AASIST inference on {audio.shape[1]} samples at {_TARGET_SR}Hz"
    )
    with torch.inference_mode():
        probs = torch.softmax(model(audio), dim=1)[0]
    return [
        {"label": "Spoof", "score": float(probs[0])},
        {"label": "Bonafide", "score": float(probs[1])},
    ]
