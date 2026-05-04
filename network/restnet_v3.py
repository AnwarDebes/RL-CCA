"""ResTNet-v3 backbone: deeper, with Squeeze-Excitation, single transformer block.

Design rationale (informed by KataGoMethods.md, Wu 2019):
- Depth-before-width: 8 residual blocks @ HIDDEN_DIM_V3 channels (vs v2: 4 blocks).
- SE blocks on every res block (~30-60 Elo per KataGo).
- One transformer block (kept for global hex board context, vs v2: 2 blocks).
- Same input shape [B, 22, 17, 17] and same backbone output dim → drop-in compatible
  for callers that only need a board representation.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: global avg pool -> 2 FC -> sigmoid -> channel scale."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        s = x.mean(dim=(2, 3))               # [B, C]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))       # [B, C]
        return x * s.unsqueeze(-1).unsqueeze(-1)


class ResBlockSE(nn.Module):
    """Residual block + SE."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


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


class TransformerBlock(nn.Module):
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


class ResTNetBackboneV3(nn.Module):
    """v3 backbone: stronger features per FLOP than v2.

    Architecture:
        Conv -> [ResSE x 4] -> Tokens -> Trans -> Spatial -> [ResSE x 4] -> GAP -> [B, C]
    """

    def __init__(self, board_positions: List[Tuple[int, int]]):
        super().__init__()
        C = Config.HIDDEN_DIM_V3

        self.init_conv = nn.Sequential(
            nn.Conv2d(Config.NUM_CHANNELS, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(),
        )

        # 8 residual+SE blocks, transformer in the middle
        self.res_blocks_a = nn.ModuleList([
            ResBlockSE(C) for _ in range(Config.NUM_RES_BLOCKS_V3 // 2)
        ])
        self.res_blocks_b = nn.ModuleList([
            ResBlockSE(C) for _ in range(Config.NUM_RES_BLOCKS_V3 // 2)
        ])

        self.to_tokens = _TokensFromSpatial(C)
        self.to_spatial = _SpatialFromTokens(C)
        self.to_tokens.register_valid_positions(board_positions)
        self.to_spatial.register_valid_positions(board_positions)
        self.transformer = TransformerBlock(
            d_model=C,
            n_heads=Config.TRANSFORMER_HEADS_V3,
            ffn_dim=Config.TRANSFORMER_FFN_DIM_V3,
        )

        self._gx = [p[0] for p in board_positions]
        self._gy = [p[1] for p in board_positions]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(x)
        for blk in self.res_blocks_a:
            h = blk(h)
        tokens = self.to_tokens(h)
        tokens = self.transformer(tokens)
        h = self.to_spatial(tokens) + h  # residual around the transformer
        for blk in self.res_blocks_b:
            h = blk(h)
        valid = h[:, :, self._gx, self._gy]   # [B, C, 121]
        return valid.mean(dim=2)              # [B, C]
