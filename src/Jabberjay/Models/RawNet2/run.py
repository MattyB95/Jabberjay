import os

import numpy as np
import torch
import torchaudio
import yaml
from loguru import logger
from torch import Tensor

from Jabberjay.Models.RawNet2.model import RawNet
from Jabberjay.Utilities.device import get_device
from Jabberjay.Utilities.hugging_face import download_pretrained_model
from Jabberjay.Utilities.model_cache import cached_loader

_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_DIR, "model_config_RawNet.yaml")
_CONFIG: dict | None = None
_TARGET_SR = 16_000


@cached_loader(maxsize=4)
def _load_model(device: str) -> RawNet:
    global _CONFIG
    if _CONFIG is None:
        try:
            with open(_CONFIG_PATH) as f_yaml:
                _CONFIG = yaml.safe_load(f_yaml)
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(
                f"Failed to load RawNet2 config: {_CONFIG_PATH}"
            ) from exc
    model = RawNet(_CONFIG["model"], device)
    model.to(device)
    repo_id = "MattyB95/pre_trained_DF_RawNet2"
    logger.info(f"Downloading model weights from {repo_id}")
    model_file = download_pretrained_model(
        repo_id=repo_id, filename="pre_trained_DF_RawNet2.pth"
    )
    model.load_state_dict(
        torch.load(model_file, map_location=torch.device(device), weights_only=True)
    )
    model.eval()
    return model


def predict(y: np.ndarray, sr: float) -> tuple[Tensor, float]:
    device = get_device()
    model = _load_model(device)
    if _CONFIG is None:  # pragma: no cover — invariant guaranteed by _load_model()
        raise RuntimeError(
            "Model configuration was not loaded; _load_model() must set _CONFIG."
        )
    max_len = _CONFIG["model"]["nb_samp"]
    audio = torch.from_numpy(y).float()
    if sr != _TARGET_SR:
        audio = torchaudio.functional.resample(audio, int(sr), _TARGET_SR)
    audio_len = audio.shape[0]
    if audio_len == 0:
        raise ValueError(
            "Input audio array is empty; cannot run inference on zero samples."
        )
    if audio_len >= max_len:
        audio = audio[:max_len]
    else:
        audio = audio.repeat(max_len // audio_len + 1)[:max_len]
    logger.debug(
        f"Running RawNet2 inference on {audio.shape[0]} samples at {_TARGET_SR}Hz"
    )
    audio_tensor = audio.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(audio_tensor)
        probs = out.exp()  # log_softmax → probabilities
        _, predicted = out.max(dim=1)
    confidence = float(probs[0][predicted.item()])
    return predicted, confidence
