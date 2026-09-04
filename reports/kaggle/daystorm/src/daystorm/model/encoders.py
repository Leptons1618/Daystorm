"""Per-modality encoders. Each maps one aligned stream to (B, G, H).

Two of these are new and trained from scratch (telemetry, radar tracks); two
are thin adapters over frozen foundation encoders whose outputs are computed
once and cached (see ``scripts/precompute_reference.py``). That split is
deliberate: on a 16 GB T4 there is no room to run SigLIP and Whisper inside
the training loop, and there is no reason to - they are frozen, so their
outputs are a function of the data alone.

Causality is preserved inside the telemetry encoder too: the dilated
convolutions are left-padded, so the representation at grid point t is a
function of grid points <= t only. Rule 1 from the aligner does not stop
holding at the tensor boundary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["CachedEmbeddingAdapter", "ChannelNorm", "TelemetryTCN", "TrackMLP", "build_encoders"]


class CausalConv1d(nn.Conv1d):
    """Conv1d that cannot see the future: left padding only."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        self._left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, G)
        return super().forward(F.pad(x, (self._left_pad, 0)))


class ChannelNorm(nn.Module):
    """LayerNorm across channels at each timestep.

    Deliberately not GroupNorm: GroupNorm on a (B, C, G) tensor normalises
    over channels *and* time, so a value at the last grid point shifts the
    statistics of every earlier one and the encoder quietly stops being
    causal. ``test_telemetry_encoder_is_causal`` catches exactly that.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, G)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TelemetryTCN(nn.Module):
    """Dilated causal TCN over the CAN channels.

    The grid is short (5 points for a 2 s window), so two residual blocks with
    dilations 1 and 2 already reach the whole window. Depth beyond that buys
    nothing and costs T4 memory.
    """

    def __init__(self, in_dim: int = 9, hidden: int = 256, dilations: tuple[int, ...] = (1, 2)):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "conv": CausalConv1d(hidden, hidden, kernel_size=3, dilation=d),
                    "norm": ChannelNorm(hidden),
                }
            )
            for d in dilations
        )
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, G, in_dim) -> (B, G, hidden)"""
        h = self.inp(x.transpose(1, 2))
        for blk in self.blocks:
            h = h + F.gelu(blk["norm"](blk["conv"](h)))
        return h.transpose(1, 2)


class TrackMLP(nn.Module):
    """Per-grid-point MLP over radar track features.

    Radar rows are already an ordered summary (lead range, range rate, TTC),
    so there is nothing sequential left to model here - the time axis is
    handled by the pooling stage.
    """

    def __init__(self, in_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CachedEmbeddingAdapter(nn.Module):
    """Projects a frozen encoder's cached output into the fusion width.

    ``in_dim`` is whatever the frozen encoder emits - 1152 for SigLIP-so400m,
    768 for Whisper-small. The frozen encoder itself never enters the training
    graph.
    """

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


def build_encoders(
    dims: dict[str, int], hidden: int = 256
) -> nn.ModuleDict:
    """Assemble the four encoders from the per-modality input widths."""
    return nn.ModuleDict(
        {
            "can": TelemetryTCN(dims["can"], hidden),
            "radar": TrackMLP(dims["radar"], hidden),
            "camera": CachedEmbeddingAdapter(dims["camera"], hidden),
            "audio": CachedEmbeddingAdapter(dims["audio"], hidden),
        }
    )
