import os

import numpy as np
import torch
import yaml
from loguru import logger
from torch import Tensor

from Jabberjay.Models.RawNet2.model import RawNet
from Jabberjay.Utilities.hugging_face import download_pretrained_model

_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_DIR, "model_config_RawNet.yaml")
_CONFIG: dict | None = None


def predict(y: np.ndarray) -> tuple[Tensor, float]:
    global _CONFIG
    if _CONFIG is None:
        try:
            with open(_CONFIG_PATH) as f_yaml:
                _CONFIG = yaml.safe_load(f_yaml)
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(
                f"Failed to load RawNet2 config: {_CONFIG_PATH}"
            ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")
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
    logger.debug(f"Running RawNet2 inference on {len(y)} samples")
    audio_tensor = Tensor(y).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(audio_tensor)
        probs = out.exp()  # log_softmax → probabilities
        _, predicted = out.max(dim=1)
    confidence = float(probs[0][predicted.item()])
    return predicted, confidence
