import numpy as np
import torch
import torchaudio
from loguru import logger

from Jabberjay.Models.Spectra0.model import Spectra0Model
from Jabberjay.Utilities.types import PredictionScore

_MODEL_ID = "lab260/spectra_0"
_TARGET_SR = 16_000
_MAX_LEN = 64600


def _preprocess(y: np.ndarray, sr: float) -> torch.Tensor:
    audio = torch.from_numpy(y).float()
    if sr != _TARGET_SR:
        audio = torchaudio.functional.resample(audio, int(sr), _TARGET_SR)
    audio = torchaudio.functional.preemphasis(audio.unsqueeze(0)).squeeze(0)
    x_len = audio.shape[0]
    if x_len >= _MAX_LEN:
        audio = audio[:_MAX_LEN]
    else:
        audio = audio.repeat(int(_MAX_LEN / x_len) + 1)[:_MAX_LEN]
    return audio.unsqueeze(0)  # (1, _MAX_LEN)


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Spectra-0 model: {_MODEL_ID}")
    logger.debug(f"Using device: {device}")
    model = Spectra0Model.from_pretrained(_MODEL_ID).eval().to(device)
    audio = _preprocess(y, sr).to(device)
    logger.debug(
        f"Running Spectra-0 inference on {audio.shape[1]} samples at {_TARGET_SR}Hz"
    )
    with torch.inference_mode():
        probs = torch.softmax(model(audio), dim=1)[0]
    return [
        {"label": "Spoof", "score": float(probs[0])},
        {"label": "Bonafide", "score": float(probs[1])},
    ]
