"""CNN encoder for the (32, 17, 17) Chinese Checkers state tensor.

Replaces the toy MLPEncoder when running CD-MCTS on the actual
tournament game. Architecture: ResNet-style trunk with global-average
pooling. Produces a fixed-length feature vector ingested by the four
heads (policy, PL, coalition, scalar value).

This is real code - not a stub - and matches the style of v4 RL
network, so the trained weights would in principle be transferable
between the v4 RL agent and CD-MCTS.

Why ResNet-style trunk
----------------------
The state tensor's 32 channels carry per-player and game-state
information; spatial structure on the 17x17 grid is significant. CNNs
with residual connections handle this efficiently. We deliberately
keep this smaller than v4's 16-block NBT trunk (this is an
architecture for the CD-MCTS *experiments*, not the tournament agent).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.n1 = nn.BatchNorm2d(channels)
        self.n2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.relu(x + h)


class CCCNNEncoder(nn.Module):
    """Compact ResNet trunk for Chinese Checkers.

    Args:
        in_channels: number of input planes (32 in v4).
        channels: width of the trunk.
        num_blocks: number of residual blocks.
        out_dim: dimensionality of the produced feature vector.
    """

    def __init__(
        self,
        in_channels: int = 32,
        channels: int = 64,
        num_blocks: int = 4,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.stem_norm = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([_ResBlock(channels) for _ in range(num_blocks)])
        self.head = nn.Linear(channels, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, 17, 17)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h = F.relu(self.stem_norm(self.stem(x)))
        for blk in self.blocks:
            h = blk(h)
        # Global average pool over spatial dims
        h = h.mean(dim=(-2, -1))
        h = self.head(h)
        return h


class CCEncoderForCMAZ(nn.Module):
    """Same backbone as CCCNNEncoder, exposed as a module compatible
    with CMAZ's expectation that .out_dim attribute exists."""

    def __init__(self, **kw):
        super().__init__()
        self.inner = CCCNNEncoder(**kw)
        self.out_dim = self.inner.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


def feature_to_tensor(feats_2d) -> torch.Tensor:
    """Convert numpy (32, 17, 17) feature ndarray to torch (1, 32, 17, 17)."""
    import numpy as np
    if isinstance(feats_2d, np.ndarray):
        return torch.from_numpy(feats_2d).float().unsqueeze(0)
    if feats_2d.dim() == 3:
        return feats_2d.unsqueeze(0)
    return feats_2d
