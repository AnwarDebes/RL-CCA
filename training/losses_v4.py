"""v4 losses: v3 losses + score_margin + pin_final.

Inherits NaN-guard logic from v3 (losses_v3.py): _safe_log_softmax,
opp_policy_loss with illegal-target row drop, etc.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from config import Config
from training.losses_v3 import (
    _safe_log_softmax, policy_ce_loss, policy_entropy_bonus,
    value_vec_loss, opp_policy_loss, plies_loss,
)


MAX_PLAYERS = Config.MAX_PLAYERS


def score_margin_loss(
    pred: torch.Tensor,            # [B, 6]
    target: torch.Tensor,          # [B, 6]
    n_players: torch.Tensor,       # [B] long
) -> torch.Tensor:
    """MSE on first n_players slots, masked rest."""
    K = pred.size(-1)
    seat_idx = torch.arange(K, device=pred.device).unsqueeze(0)
    mask = (seat_idx < n_players.unsqueeze(1).long()).float()
    sq = (pred - target) ** 2
    denom = mask.sum().clamp_min(1.0)
    return (sq * mask).sum() / denom


def pin_final_loss(
    pred_logits: torch.Tensor,     # [B, NUM_PIECES, K]
    target: torch.Tensor,          # [B, NUM_PIECES] long (bucket indices)
    valid: torch.Tensor,           # [B] bool
) -> torch.Tensor:
    if not valid.any():
        return pred_logits.sum() * 0.0
    P = pred_logits[valid]                     # [V, NUM_PIECES, K]
    T = target[valid]                          # [V, NUM_PIECES] long
    return F.cross_entropy(
        P.reshape(-1, P.size(-1)),
        T.reshape(-1),
    )


def nexus_loss_v4(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    legal_mask: torch.Tensor,
    old_logits: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Combined v4 loss. Returns dict with `total` and components.

    Note: `target_policy` here is the MCTS-improved policy (visit-count
    distribution from MCTS), not network argmax. This is the canonical
    AlphaZero policy target.
    """
    logits = out["logits"]
    target_policy = batch["policies"]

    l_policy = policy_ce_loss(logits, target_policy)

    l_value_vec = value_vec_loss(
        out["value_vec"], batch["value_vec"], batch["n_players"]
    )

    l_opp = opp_policy_loss(
        out["opp_logits"],
        batch["opp_action"],
        opp_legal_mask=batch.get("opp_legal_mask"),
    )

    l_plies = plies_loss(
        out["plies"], batch["plies_target"].float(), batch["plies_valid"]
    )

    l_score_margin = score_margin_loss(
        out["score_margin"], batch["score_margin_target"], batch["n_players"]
    )

    l_pin_final = pin_final_loss(
        out["pin_final"], batch["pin_final_target"], batch["pin_final_valid"]
    )

    h_policy = policy_entropy_bonus(logits, legal_mask)

    total = (
        l_policy
        + Config.VALUE_VEC_LOSS_WEIGHT * l_value_vec
        + Config.OPP_POLICY_LOSS_WEIGHT * l_opp
        + Config.PLIES_LOSS_WEIGHT * l_plies
        + Config.SCORE_MARGIN_LOSS_WEIGHT * l_score_margin
        + Config.PIN_FINAL_LOSS_WEIGHT * l_pin_final
        - Config.ENTROPY_BONUS_WEIGHT * h_policy
    )

    l_kl = torch.tensor(0.0, device=logits.device)
    if old_logits is not None:
        from training.losses import kl_loss as _kl
        l_kl = Config.KL_LOSS_WEIGHT * _kl(logits, old_logits, legal_mask)
        total = total + l_kl

    return {
        "total": total,
        "policy": l_policy.detach(),
        "value_vec": l_value_vec.detach(),
        "opp_policy": l_opp.detach() if hasattr(l_opp, 'detach') else l_opp,
        "plies": l_plies.detach() if hasattr(l_plies, 'detach') else l_plies,
        "score_margin": l_score_margin.detach(),
        "pin_final": l_pin_final.detach() if hasattr(l_pin_final, 'detach') else l_pin_final,
        "entropy": h_policy.detach(),
        "kl": l_kl.detach(),
    }
