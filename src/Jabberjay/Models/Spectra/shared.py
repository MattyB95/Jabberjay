import numpy as np
import torch
import torch.nn as nn
import torchaudio
from loguru import logger
from safetensors.torch import load_file
from transformers import Wav2Vec2Config, Wav2Vec2Model

from Jabberjay.Utilities.audio import fixed_length, resample
from Jabberjay.Utilities.hugging_face import download_pretrained_model, revision_for
from Jabberjay.Utilities.model_cache import cached_loader

_TARGET_SR: int = 16_000
_MAX_LEN: int = 64600
_WEIGHTS_FILE = "model.safetensors"


@cached_loader(maxsize=4)
def load_pretrained(
    model_cls: type[nn.Module], model_id: str, device: str
) -> nn.Module:
    """Load, cache, and move a HuggingFace-hosted Spectra model to device.

    Built on the meta device and adopting the checkpoint tensors directly
    (assign=True), rather than via PyTorchModelHubMixin.from_pretrained():
    that randomly initialises all ~300M parameters first only to overwrite
    them, which is ~5s of a ~7s load. strict=True: any key drift between the
    checkpoint and the ported architecture must raise, not silently leave
    layers randomly initialised.
    """
    logger.info(f"Loading model: {model_id}@{revision_for(model_id)[:7]}")
    weights_path = download_pretrained_model(repo_id=model_id, filename=_WEIGHTS_FILE)
    state_dict = load_file(weights_path)
    with torch.device("meta"):
        model = model_cls()
    model.load_state_dict(state_dict, strict=True, assign=True)
    leftover = [
        name
        for name, t in [*model.named_parameters(), *model.named_buffers()]
        if t.is_meta
    ]
    if leftover:
        raise RuntimeError(
            f"{model_id}: tensors not provided by the checkpoint: {leftover}"
        )
    return model.eval().to(device)


def preprocess(y: np.ndarray, sr: float, preemphasis: bool = True) -> torch.Tensor:
    """Resample, optionally apply preemphasis, and pad/trim to _MAX_LEN samples."""
    audio = resample(y, sr, _TARGET_SR)
    if preemphasis:
        audio = torchaudio.functional.preemphasis(audio.unsqueeze(0)).squeeze(0)
    return fixed_length(audio, _MAX_LEN).unsqueeze(0)  # (1, _MAX_LEN)


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, model_name_or_path: str = "facebook/wav2vec2-xls-r-300m"):
        super().__init__()
        # Build from config only: every Spectra checkpoint ships its own
        # fine-tuned copy of these weights (enforced by strict loading in
        # load_pretrained), so fetching the 1.27 GB base checkpoint here would
        # be downloaded and loaded only to be overwritten immediately
        config = Wav2Vec2Config.from_pretrained(
            model_name_or_path, revision=revision_for(model_name_or_path)
        )
        config.apply_spec_augment = False
        self.model = Wav2Vec2Model(config)
        self.model.masked_spec_embed = None  # type: ignore

    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(-1)
        return self.model(x, return_dict=True).last_hidden_state


class MLPBridge(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        activation=None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.SELU()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)
