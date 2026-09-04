"""Pool each modality to a fixed token budget and project into the backbone.

The contract with the language model is deliberately small: every window
becomes exactly ``n_modalities * tokens_per_modality`` tokens - 16 by default
- regardless of how many grid points survived masking. A fixed budget keeps
the sequence length static, which is what makes ``torch.compile`` and a
pre-allocated KV cache usable in Phase 3.

Three details here are load-bearing:

* Masked grid points are excluded from pooling, not merely zeroed. A zero
  after normalisation is the channel mean, which is a plausible value; letting
  it into the attention average teaches the model that a dead sensor reads
  average.
* A modality with no valid points at all emits a learned *absent* token rather
  than NaN or zeros, so "the radar was down" is a thing the model can read.
* Modality and grid-time embeddings are added before projection, so the
  backbone can tell a camera token from a CAN token and early from late.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .encoders import build_encoders

__all__ = ["DaystormFusion", "MaskedAttentionPool"]


class MaskedAttentionPool(nn.Module):
    """Pool (B, G, H) to (B, Q, H) with learned queries, honouring a mask."""

    def __init__(self, hidden: int, n_queries: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, hidden) * 0.02)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.absent = nn.Parameter(torch.randn(n_queries, hidden) * 0.02)
        self.scale = 1.0 / math.sqrt(hidden)
        self.n_queries = n_queries

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """h: (B, G, H); valid: (B, G) bool -> (B, Q, H)"""
        k, v = self.key(h), self.value(h)
        scores = torch.einsum("qh,bgh->bqg", self.queries, k) * self.scale
        scores = scores.masked_fill(~valid[:, None, :], torch.finfo(scores.dtype).min)

        weights = scores.softmax(dim=-1)
        any_valid = valid.any(dim=-1)  # (B,)
        # Rows with nothing valid produced a uniform softmax over -inf; drop it
        # and substitute the learned absent token instead.
        weights = weights * any_valid[:, None, None]
        pooled = torch.einsum("bqg,bgh->bqh", weights, v)
        absent = self.absent.unsqueeze(0).expand(h.shape[0], -1, -1)
        return torch.where(any_valid[:, None, None], pooled, absent)


class DaystormFusion(nn.Module):
    """Four aligned streams -> a short sequence of backbone-width tokens.

    Parameters
    ----------
    dims : per-modality input width, e.g. ``{"can": 9, "radar": 6,
        "camera": 1152, "audio": 768}``.
    d_model : the backbone's hidden size (2048 for Qwen2.5-VL-3B).
    """

    MODALITIES = ("camera", "can", "radar", "audio")

    def __init__(
        self,
        dims: dict[str, int],
        d_model: int = 2048,
        hidden: int = 256,
        tokens_per_modality: int = 4,
        n_grid: int = 5,
    ):
        super().__init__()
        # +1 input channel per modality: the validity bit itself. Rule 3 of the
        # aligner says missing data is an input the model consumes, and that has
        # to be true at the encoder, not only at the pooling stage.
        self.encoders = build_encoders({k: v + 1 for k, v in dims.items()}, hidden)
        self.pools = nn.ModuleDict(
            {m: MaskedAttentionPool(hidden, tokens_per_modality) for m in self.MODALITIES}
        )
        self.modality_emb = nn.Parameter(torch.randn(len(self.MODALITIES), hidden) * 0.02)
        self.register_buffer("time_emb", _sinusoidal(n_grid, hidden), persistent=False)
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.tokens_per_modality = tokens_per_modality
        self.n_tokens = tokens_per_modality * len(self.MODALITIES)
        self.d_model = d_model

    def forward(
        self, features: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """features: name -> (B, G, D_in); mask: (B, G, n_modalities) -> (B, T, d_model)"""
        tokens = []
        for i, name in enumerate(self.MODALITIES):
            bit = mask[:, :, i : i + 1].to(features[name].dtype)
            h = self.encoders[name](torch.cat([features[name], bit], dim=-1))
            h = h + self.time_emb[None, : h.shape[1], :]     # when in the window
            valid = mask[:, :, i] > 0.5
            pooled = self.pools[name](h, valid)              # (B, Q, hidden)
            tokens.append(pooled + self.modality_emb[i][None, None, :])
        return self.projector(torch.cat(tokens, dim=1))


def _sinusoidal(n: int, dim: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(0, dim, 2, dtype=torch.float32)[None, :]
    ang = pos / torch.pow(10_000.0, i / dim)
    out = torch.zeros(n, dim)
    out[:, 0::2], out[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return out
