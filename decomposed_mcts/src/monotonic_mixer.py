"""State-conditional monotonic mixer for CMAZ.

Given K per-component value estimates v ∈ ℝ^K and state features φ(s), the
mixer outputs a scalar
    Q(s, v) = Σ_k w_k(s) · v_k + b(s),
where w(s) ∈ Δ^K (non-negative, sum to 1) is produced by a hyper-network
conditioned on φ(s). Monotonicity in each v_k is guaranteed by the
non-negativity of w.

This is a faithful adaptation of QMIX's mixing network (Rashid et al.
2018) from per-AGENT decomposition to per-OBJECTIVE decomposition. The
key novel property for CMAZ is **inference-time override**: the same
trained mixer can be evaluated with a user-supplied w' instead of w(s),
allowing one network to serve any user-tunable utility function over the
known semantic components of the game's reward (e.g., Chinese Checkers'
4-component score formula).

Tests in tests/test_monotonic_mixer.py verify:
  - softmax output sums to 1
  - increasing v_k => Q increases (monotonicity)
  - inference-time override returns the override-weighted sum exactly
  - bias term has no monotonicity constraint
  - gradients are finite and have the expected signs
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicMixer(nn.Module):
    """QMIX-style state-conditional monotonic mixer.

    Args:
        feature_dim: dimensionality of φ(s).
        num_components: K, the number of value components.
        hidden_dim: hypernetwork width.
    """

    def __init__(self, feature_dim: int, num_components: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_components = num_components
        # Hypernetwork producing (logits for w(s), bias b(s))
        self.hyper = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_components + 1),
        )

    def get_weights(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (w, b) where w ∈ Δ^K and b is the scalar bias."""
        out = self.hyper(features)
        w_logits = out[..., :self.num_components]
        b = out[..., self.num_components]
        w = F.softmax(w_logits, dim=-1)
        return w, b

    def forward(
        self,
        v: torch.Tensor,                       # (..., K)
        features: torch.Tensor,                # (..., feature_dim)
        override_weights: Optional[torch.Tensor] = None,  # (..., K) or (K,)
    ) -> torch.Tensor:
        """Mix the per-component values into a scalar.

        If override_weights is given, use it instead of the hypernetwork
        output (still under the constraint that it sums to 1; we
        renormalise to be safe). The bias term is taken from the
        hypernetwork in either case.
        """
        if v.shape[-1] != self.num_components:
            raise ValueError(f"v last dim {v.shape[-1]} != K={self.num_components}")
        w, b = self.get_weights(features)
        if override_weights is not None:
            ow = override_weights.to(v.dtype)
            if ow.dim() == 1:
                # broadcast over batch
                ow = ow.unsqueeze(0).expand_as(w)
            ow = ow / ow.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            w = ow
        Q = (w * v).sum(dim=-1) + b
        return Q


class ComponentValueHead(nn.Module):
    """Outputs K per-component value estimates from feature vector.

    Each component is bounded to [-1, 1] via tanh - matching the
    AlphaZero-style normalised-utility convention.
    """

    def __init__(self, feature_dim: int, num_components: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, num_components)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(features))
