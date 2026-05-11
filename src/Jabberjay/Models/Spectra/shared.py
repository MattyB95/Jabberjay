import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, model_name_or_path: str = "facebook/wav2vec2-xls-r-300m"):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(
            model_name_or_path, gradient_checkpointing=False
        )
        self.model.config.apply_spec_augment = False
        self.model.masked_spec_embed = None

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
