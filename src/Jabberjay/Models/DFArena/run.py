import numpy as np
import torch
from loguru import logger

from Jabberjay.Models.DFArena.model import DFArena1B
from Jabberjay.Models.Spectra.shared import _TARGET_SR, preprocess
from Jabberjay.Utilities.device import get_device
from Jabberjay.Utilities.hugging_face import download_pretrained_model, revision_for
from Jabberjay.Utilities.model_cache import cached_loader
from Jabberjay.Utilities.types import PredictionScore

MODEL_ID = "Speech-Arena-2025/DF_Arena_1B_V_1"
# The checkpoint is a pickled .bin: download_pretrained_model() loads only the
# reviewed, pinned revision, and torch.load always uses weights_only=True
WEIGHTS_FILE = "pytorch_model.bin"
# The checkpoint is a transformers PreTrainedModel wrapper around the network
_STATE_DICT_PREFIX = "backbone."


@cached_loader(maxsize=2)
def _load_model(device: str) -> DFArena1B:
    logger.info(f"Loading model: {MODEL_ID}@{revision_for(MODEL_ID)[:7]}")
    weights_path = download_pretrained_model(repo_id=MODEL_ID, filename=WEIGHTS_FILE)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = {k.removeprefix(_STATE_DICT_PREFIX): v for k, v in state_dict.items()}
    # Build on the meta device and adopt the checkpoint tensors directly
    # (assign=True): randomly initialising 1B parameters only to overwrite them
    # costs ~15s. strict=True: any checkpoint/architecture drift must raise,
    # not silently leave layers uninitialised
    with torch.device("meta"):
        model = DFArena1B()
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval().to(device)


def predict(y: np.ndarray, sr: float) -> list[PredictionScore]:
    device = get_device()
    model = _load_model(device)
    # DF Arena's own feature extractor applies no preemphasis, unlike Spectra
    audio = preprocess(y, sr, preemphasis=False).to(device)
    logger.debug(
        f"Running DF-Arena-1B inference on {audio.shape[1]} samples at {_TARGET_SR}Hz"
    )
    with torch.inference_mode():
        # No calibrated threshold is published, so use the softmax directly
        spoof, bonafide = torch.softmax(model(audio), dim=-1)[0]
    return [
        {"label": "Spoof", "score": float(spoof)},
        {"label": "Bonafide", "score": float(bonafide)},
    ]
