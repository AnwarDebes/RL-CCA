"""ResTNet-v4 backbone with KataGo-style nested-bottleneck (NBT) residual blocks.

KataGo's NBT block (Wu, KataGoMethods.md): each "outer" residual block
contains an "inner" bottleneck: 1x1 down-channel -> 3x3 -> 1x1 up-channel,
with SE on the inner conv. Reported +200-300 Elo over plain ResBlock at
equal compute (KataGo b18c384nbt vs b15c192).

Architecture:
    Conv(C) -> [NBTBlock x 6] -> Tokens -> Transformer -> Spatial -> [NBTBlock x 6] -> GAP
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class NBTBlock(nn.Module):
    """Nested-bottleneck residual block - canonical KataGo b18c384nbt structure.

    Outer flow (with outer skip from input):
        x -> 1x1 down (C -> Cb) -> [TWO inner 3x3 convs with INNER skip] -> 1x1 up (Cb -> C) -> add outer skip -> ReLU

    Inner residual: bn -> 3x3 -> bn -> 3x3, with skip across both 3x3.
    Verified against KataGo issue #793 + KataGoMethods.md (no SE inside the
    NBT block; SE is applied separately to selected blocks elsewhere).
    """

    def __init__(self, channels: int, bottleneck: int):
        super().__init__()
        # outer 1x1 down + up
        self.proj_down = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.bn_down = nn.BatchNorm2d(bottleneck)
        # inner residual pair (two 3x3 at bottleneck width with inner skip)
        self.bn_inner1 = nn.BatchNorm2d(bottleneck)
        self.conv_inner1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1, bias=False)
        self.bn_inner2 = nn.BatchNorm2d(bottleneck)
        self.conv_inner2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1, bias=False)
        self.proj_up = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.bn_up = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outer_skip = x
        h = F.relu(self.bn_down(self.proj_down(x)))
        # Inner residual: two 3x3 convs with skip
        inner_skip = h
        h = F.relu(self.bn_inner1(self.conv_inner1(h)))
        h = self.bn_inner2(self.conv_inner2(h))
        h = F.relu(h + inner_skip)
        h = self.bn_up(self.proj_up(h))
        return F.relu(h + outer_skip)


class _TokensFromSpatial(nn.Module):
    def __init__(self, channels: int, num_valid_cells: int = Config.NUM_CELLS):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_valid_cells, channels) * 0.02)
        self._gx: List[int] = []
        self._gy: List[int] = []

    def register_valid_positions(self, positions: List[Tuple[int, int]]):
        self._gx = [p[0] for p in positions]
        self._gy = [p[1] for p in positions]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x[:, :, self._gx, self._gy].permute(0, 2, 1)
        return tokens + self.pos_embed


class _SpatialFromTokens(nn.Module):
    def __init__(self, channels: int, grid_size: int = Config.GRID_SIZE):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self._gx: List[int] = []
        self._gy: List[int] = []

    def register_valid_positions(self, positions: List[Tuple[int, int]]):
        self._gx = [p[0] for p in positions]
        self._gy = [p[1] for p in positions]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, _, C = tokens.shape
        out = torch.zeros(B, C, self.grid_size, self.grid_size,
                          device=tokens.device, dtype=tokens.dtype)
        out[:, :, self._gx, self._gy] = tokens.permute(0, 2, 1)
        return out


class TransformerBlockV4(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ResTNetBackboneV4(nn.Module):
    """v4 backbone: NBT blocks + transformer in the middle.

    [B,32,17,17] -> Conv(192) -> 6× NBTBlock -> Tokens -> Transformer ->
                    Spatial(+residual) -> 6× NBTBlock -> GAP over valid cells -> [B,192]
    """

    def __init__(self, board_positions: List[Tuple[int, int]]):
        super().__init__()
        C = Config.HIDDEN_DIM_V4
        Cb = Config.NBT_BOTTLENECK_V4

        self.init_conv = nn.Sequential(
            nn.Conv2d(Config.NUM_CHANNELS, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(),
        )

        half = Config.NUM_RES_BLOCKS_V4 // 2
        self.blocks_a = nn.ModuleList([NBTBlock(C, Cb) for _ in range(half)])
        self.blocks_b = nn.ModuleList([NBTBlock(C, Cb) for _ in range(half)])

        self.to_tokens = _TokensFromSpatial(C)
        self.to_spatial = _SpatialFromTokens(C)
        self.to_tokens.register_valid_positions(board_positions)
        self.to_spatial.register_valid_positions(board_positions)
        self.transformer = TransformerBlockV4(
            d_model=C,
            n_heads=Config.TRANSFORMER_HEADS_V4,
            ffn_dim=Config.TRANSFORMER_FFN_DIM_V4,
        )

        self._gx = [p[0] for p in board_positions]
        self._gy = [p[1] for p in board_positions]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(x)
        for blk in self.blocks_a:
            h = blk(h)
        tokens = self.to_tokens(h)
        tokens = self.transformer(tokens)
        h = self.to_spatial(tokens) + h        # residual around transformer
        for blk in self.blocks_b:
            h = blk(h)
        valid = h[:, :, self._gx, self._gy]    # [B, C, 121]
        return valid.mean(dim=2)               # [B, C]
