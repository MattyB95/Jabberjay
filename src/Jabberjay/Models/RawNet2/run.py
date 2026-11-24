from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

from Jabberjay.Models.RawNet2.model import RawNet
from Jabberjay.Utilities.audio import fixed_length, resample
from Jabberjay.Utilities.device import get_device
from Jabberjay.Utilities.hugging_face import download_pretrained_model
from Jabberjay.Utilities.model_cache import cached_loader
from Jabberjay.Utilities.types import PredictionScore

REPO_ID = "MattyB95/pre_trained_DF_RawNet2"
FILENAME = "pre_trained_DF_RawNet2.pth"
_CONFIG_PATH = Path(__file__).with_name("model_config_RawNet.yaml")
_TARGET_SR = 16_000


@cached_loader(maxsize=1)
def _load_config() -> dict:
    try:
        with _CONFIG_PATH.open() as f_yaml:
            return yaml.safe_load(f_yaml)["model"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Failed to load RawNet2 config: {_CONFIG_PATH}") from exc


@cached_loader(maxsize=4)
def _load_model(device: str) -> RawNet:
    model = RawNet(_load_config(), device)
    model.to(device)
    logger.info(f"Downloading model weights from {REPO_ID}")
    model_file = download_pretrained_model(repo_id=REPO_ID, filename=FILENAME)
    state_dict = torch.load(
        model_file, map_location=torch.device(device), weights_only=True
    )
    # Legacy checkpoints persisted the sinc filter bank, which the model now
    # recomputes on the fly
    state_dict.pop("Sinc_conv.filters", None)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    device = get_device()
    model = _load_model(device)
    audio = fixed_length(resample(y, sr, _TARGET_SR), _load_config()["nb_samp"])
    logger.debug(
        f"Running RawNet2 inference on {audio.shape[0]} samples at {_TARGET_SR}Hz"
    )
    with torch.inference_mode():
        # The network ends in log_softmax over (spoof, bonafide)
        spoof, bonafide = model(audio.unsqueeze(0).to(device)).exp()[0]
    return [
        {"label": "Spoof", "score": float(spoof)},
        {"label": "Bonafide", "score": float(bonafide)},
    ]
