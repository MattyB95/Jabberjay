"""DF Arena 1B (RAPTOR) architecture.

A port of Speech-Arena-2025/DF_Arena_1B_V_1 (Kulkarni et al., 2026,
arXiv:2603.06164): a Wav2Vec2-XLS-R-1B encoder whose per-layer hidden states
are fused by attention pooling and a sigmoid layer gate, then classified by a
Conformer head with a class token. The Conformer follows TCM
(github.com/ductuantruong/tcm_add, MIT), itself derived from
github.com/lucidrains/conformer (MIT).

Module and attribute names mirror the upstream checkpoint so its state dict
loads with strict=True — including the parameter-free placeholders
(``_Transpose``, ``nn.Dropout``) that keep ``nn.Sequential`` indices aligned.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model


class _Transpose(nn.Module):
    """Swap the channel and time axes: (B, N, C) <-> (B, C, N)."""

    def forward(self, x):
        return x.transpose(1, 2)


class Scale(nn.Module):
    def __init__(self, scale: float, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x):
        return self.fn(x) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class DepthWiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        # "Same" padding, asymmetric for even kernel sizes
        pad = kernel_size // 2
        self.padding = (pad, pad - (kernel_size + 1) % 2)
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels)

    def forward(self, x):
        return self.conv(F.pad(x, self.padding))


class HeadTokenAttention(nn.Module):
    """Multi-head self-attention with head tokens (arXiv:2210.05958).

    Each head's mean feature is projected into an extra "head token" that
    joins the sequence for attention, then is folded back into the class
    token (position 0) afterwards.
    """

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.num_heads = heads
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.proj = nn.Linear(inner_dim, dim)
        self.act = nn.GELU()
        self.ht_proj = nn.Linear(dim_head, dim, bias=True)
        self.ht_norm = nn.LayerNorm(dim_head)
        self.pos_embed = nn.Parameter(torch.zeros(1, heads, dim))

    def forward(self, x):
        b, n, c = x.shape
        h = self.num_heads

        # Head tokens: per-head mean over time, projected back to model width
        head_tokens = x.reshape(b, -1, h, c // h).permute(0, 2, 1, 3).mean(dim=2)
        head_tokens = self.ht_proj(head_tokens).reshape(b, -1, h, c // h)
        head_tokens = self.act(self.ht_norm(head_tokens)).flatten(2)
        head_tokens = head_tokens + self.pos_embed.expand(b, -1, -1)
        x = torch.cat([x, head_tokens], dim=1)

        qkv = self.qkv(x).reshape(b, n + h, 3, h, c // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = self.proj((attn @ v).transpose(1, 2).reshape(b, n + h, c))

        # Fold the head tokens and the frame mean into the class token
        cls, frames, head_tokens = torch.split(x, [1, n - 1, h], dim=1)
        cls = (
            cls
            + torch.mean(head_tokens, dim=1, keepdim=True)
            + torch.mean(frames, dim=1, keepdim=True)
        )
        return torch.cat([cls, frames], dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(dim * mult, dim),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, expansion_factor: int, kernel_size: int):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            _Transpose(),
            nn.Conv1d(dim, inner_dim * 2, 1),
            nn.GLU(dim=1),
            DepthWiseConv1d(inner_dim, kernel_size),
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            _Transpose(),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ff_mult: int,
        conv_expansion_factor: int,
        conv_kernel_size: int,
    ):
        super().__init__()
        self.ff1 = Scale(0.5, PreNorm(dim, FeedForward(dim, ff_mult)))
        self.attn = PreNorm(dim, HeadTokenAttention(dim, heads, dim // heads))
        self.conv = ConformerConvModule(dim, conv_expansion_factor, conv_kernel_size)
        self.ff2 = Scale(0.5, PreNorm(dim, FeedForward(dim, ff_mult)))
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ff1(x) + x
        x = self.attn(x) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        return self.post_norm(x)


class ConformerClassifier(nn.Module):
    """Conformer encoder over a prepended class token, with a 2-way head."""

    def __init__(
        self,
        emb_size: int,
        heads: int = 4,
        ff_mult: int = 4,
        exp_fac: int = 2,
        kernel_size: int = 31,
        n_encoders: int = 4,
        max_len: int = 10_000,
    ):
        super().__init__()
        # Sinusoidal in the original, but frozen and stored in the checkpoint —
        # zeros here are overwritten on load (and skip a slow 10k x 1280 build)
        self.positional_emb = nn.Parameter(
            torch.zeros(1, max_len, emb_size), requires_grad=False
        )
        self.encoder_blocks = nn.ModuleList(
            ConformerBlock(emb_size, heads, ff_mult, exp_fac, kernel_size)
            for _ in range(n_encoders)
        )
        self.class_token = nn.Parameter(torch.zeros(1, emb_size))
        self.fc5 = nn.Linear(emb_size, 2)

    def forward(self, x):
        x = x + self.positional_emb[:, : x.size(1), :]
        cls = self.class_token.expand(x.size(0), 1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.encoder_blocks:
            x = block(x)
        return self.fc5(x[:, 0, :])


class DFArena1B(nn.Module):
    def __init__(self, ssl_model_name: str = "facebook/wav2vec2-xls-r-1b"):
        super().__init__()
        # Config only: the checkpoint carries its own fine-tuned encoder weights
        self.ssl_model = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(ssl_model_name))
        self.ssl_model.config.output_hidden_states = True
        emb_size = self.ssl_model.config.hidden_size
        self.attn_scores = nn.Linear(emb_size, 1, bias=False)
        self.fc0 = nn.Linear(emb_size, 1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU()
        self.conformer = ConformerClassifier(emb_size)

    def forward(self, x):
        """Map a (B, samples) 16kHz waveform to (B, 2) [spoof, bonafide] logits."""
        hidden = torch.stack(self.ssl_model(x).hidden_states, dim=1)  # (B, L, T, D)

        # Gate each layer by a sigmoid of its attention-pooled summary, then sum
        weights = torch.softmax(self.attn_scores(hidden), dim=2)  # (B, L, T, 1)
        pooled = torch.sum(weights * hidden, dim=2)  # (B, L, D)
        gate = torch.sigmoid(self.fc0(pooled)).unsqueeze(-1)  # (B, L, 1, 1)
        fused = torch.sum(hidden * gate, dim=1)  # (B, T, D)

        fused = self.selu(self.first_bn(fused.unsqueeze(1))).squeeze(1)
        return self.conformer(fused)
