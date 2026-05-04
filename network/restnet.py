"""ResTNet backbone: interleaved Residual blocks + Transformer blocks.

Architecture (RRTRRTR pattern):
  Conv3x3 -> ResBlock1 -> ResBlock2 -> FeatureConvert -> Transformer1 ->
  FeatureConvert -> ResBlock3 -> ResBlock4 -> FeatureConvert -> Transformer2 ->
  FeatureConvert -> GlobalAvgPool -> [B, 128]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from config import Config


class ResBlock(nn.Module):
    """Residual block: two Conv 3x3 with BatchNorm + ReLU + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class FeatureConvertToTokens(nn.Module):
    """Convert CNN spatial features to Transformer tokens.

    Flattens only VALID board cells (121 tokens), adds learnable positional
    embeddings.
    """

    def __init__(self, channels: int, num_valid_cells: int = Config.NUM_CELLS):
        super().__init__()
        self.channels = channels
        self.num_valid_cells = num_valid_cells
        self.pos_embed = nn.Parameter(torch.randn(1, num_valid_cells, channels) * 0.02)

        # Precompute valid cell grid positions (set in register_valid_positions)
        self._valid_gx: List[int] = []
        self._valid_gy: List[int] = []

    def register_valid_positions(self, positions: List[Tuple[int, int]]):
        """Register the (gx, gy) grid positions of valid cells."""
        self._valid_gx = [p[0] for p in positions]
        self._valid_gy = [p[1] for p in positions]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, 17, 17] -> [B, 121, C]"""
        B = x.shape[0]
        # Extract valid cells: x[:, :, gx, gy] for each valid cell
        tokens = x[:, :, self._valid_gx, self._valid_gy]  # [B, C, 121]
        tokens = tokens.permute(0, 2, 1)  # [B, 121, C]
        tokens = tokens + self.pos_embed
        return tokens


class FeatureConvertToSpatial(nn.Module):
    """Convert Transformer tokens back to CNN spatial features."""

    def __init__(self, channels: int, grid_size: int = Config.GRID_SIZE):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self._valid_gx: List[int] = []
        self._valid_gy: List[int] = []

    def register_valid_positions(self, positions: List[Tuple[int, int]]):
        self._valid_gx = [p[0] for p in positions]
        self._valid_gy = [p[1] for p in positions]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, 121, C] -> [B, C, 17, 17]"""
        B, N, C = tokens.shape
        spatial = torch.zeros(B, C, self.grid_size, self.grid_size,
                              device=tokens.device, dtype=tokens.dtype)
        tokens_perm = tokens.permute(0, 2, 1)  # [B, C, 121]
        spatial[:, :, self._valid_gx, self._valid_gy] = tokens_perm
        return spatial


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention + FFN + LayerNorm."""

    def __init__(
        self,
        d_model: int = Config.HIDDEN_DIM,
        n_heads: int = Config.TRANSFORMER_HEADS,
        ffn_dim: int = Config.TRANSFORMER_FFN_DIM,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, d_model] -> [B, N, d_model]"""
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class ResTNetBackbone(nn.Module):
    """ResTNet backbone producing a [B, 128] global representation.

    Pattern: Conv -> Res1 -> Res2 -> FC_to_tokens -> Trans1 -> FC_to_spatial ->
             Res3 -> Res4 -> FC_to_tokens -> Trans2 -> FC_to_spatial -> GAP
    """

    def __init__(self, board_positions: List[Tuple[int, int]]):
        super().__init__()
        C = Config.HIDDEN_DIM

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(Config.NUM_CHANNELS, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(),
        )

        # Residual blocks
        self.res1 = ResBlock(C)
        self.res2 = ResBlock(C)
        self.res3 = ResBlock(C)
        self.res4 = ResBlock(C)

        # Feature converters
        self.to_tokens1 = FeatureConvertToTokens(C)
        self.to_spatial1 = FeatureConvertToSpatial(C)
        self.to_tokens2 = FeatureConvertToTokens(C)
        self.to_spatial2 = FeatureConvertToSpatial(C)

        # Register valid positions in all converters
        for conv in [self.to_tokens1, self.to_spatial1, self.to_tokens2, self.to_spatial2]:
            conv.register_valid_positions(board_positions)

        # Transformer blocks
        self.trans1 = TransformerBlock()
        self.trans2 = TransformerBlock()

        # Global average pool over valid cells only
        self._valid_gx = [p[0] for p in board_positions]
        self._valid_gy = [p[1] for p in board_positions]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 22, 17, 17] -> [B, 128]"""
        # Initial conv
        h = self.init_conv(x)

        # Res blocks 1-2
        h = self.res1(h)
        h = self.res2(h)

        # Transformer 1
        tokens = self.to_tokens1(h)
        tokens = self.trans1(tokens)
        h = self.to_spatial1(tokens)

        # Res blocks 3-4
        h = self.res3(h)
        h = self.res4(h)

        # Transformer 2
        tokens = self.to_tokens2(h)
        tokens = self.trans2(tokens)
        h = self.to_spatial2(tokens)

        # Global average pool over valid cells
        B, C = h.shape[0], h.shape[1]
        valid_features = h[:, :, self._valid_gx, self._valid_gy]  # [B, C, 121]
        pooled = valid_features.mean(dim=2)  # [B, C]

        return pooled
