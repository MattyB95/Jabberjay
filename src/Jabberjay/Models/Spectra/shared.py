import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model

_TARGET_SR: int = 16_000
_MAX_LEN: int = 64600


def preprocess(y: np.ndarray, sr: float) -> torch.Tensor:
    """Resample, apply preemphasis, and pad/trim to _MAX_LEN samples."""
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


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, model_name_or_path: str = "facebook/wav2vec2-xls-r-300m"):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(
            model_name_or_path, gradient_checkpointing=False
        )
        self.model.config.apply_spec_augment = False
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
