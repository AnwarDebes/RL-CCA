"""NEXUS v3 loss: policy + value vector + opp policy + plies + entropy bonus.

Policy is supervised by the MCTS-improved policy.
Value vector is supervised by per-player normalized teacher final_score in [-1, 1],
  with a per-batch length-N mask (entries beyond N are zero-grad).
Opp-policy is supervised by the actual next-opponent action (one-hot).
  When the next opponent move is unknown (terminal state), the loss is masked out.
Plies is supervised by `plies_remaining / 200.0` regression.
Entropy bonus = -beta * H(pi_legal) - added to policy loss to fight collapse.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from config import Config


def _safe_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """log_softmax that's robust to all-`-inf` rows (which produce NaN).

    For all-illegal rows, returns zeros - those rows then contribute zero
    loss, which is the only sensible thing.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.nan_to_num(log_probs, nan=0.0, posinf=0.0, neginf=0.0)


def policy_ce_loss(logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    log_probs = _safe_log_softmax(logits)
    return -(target_policy * log_probs).sum(dim=-1).mean()


def policy_entropy_bonus(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Negative entropy of the network's *current* policy (over legal moves).
    Returned as a positive scalar; trainer subtracts it (× beta) from total loss
    to *encourage* high entropy. Critical for preventing collapse on stale data.

    Robust to all-`-inf` rows via _safe_log_softmax.
    """
    log_probs = _safe_log_softmax(logits)
    probs = log_probs.exp()
    H = -(probs * log_probs * legal_mask.float()).sum(dim=-1)
    return H.mean()


def value_vec_loss(
    pred_vec: torch.Tensor,        # [B, 6]
    target_vec: torch.Tensor,      # [B, 6]  (zeros for unused slots)
    n_players: torch.Tensor,       # [B]   long, in [2, 6]
) -> torch.Tensor:
    """MSE on the first n_players entries per row; rest masked out."""
    B, K = pred_vec.shape
    seat_idx = torch.arange(K, device=pred_vec.device).unsqueeze(0)   # [1, K]
    mask = (seat_idx < n_players.unsqueeze(1).long()).float()         # [B, K]
    sq = (pred_vec - target_vec) ** 2
    denom = mask.sum().clamp_min(1.0)
    return (sq * mask).sum() / denom


def opp_policy_loss(
    opp_logits: torch.Tensor,           # [B, 1210]
    opp_action: torch.Tensor,           # [B]   long; -1 = no target
    opp_legal_mask: Optional[torch.Tensor] = None,    # [B, 1210] bool or None
) -> torch.Tensor:
    """Cross-entropy from opp_logits to one-hot at opp_action.
    Rows with opp_action == -1 are excluded.
    """
    valid = opp_action >= 0
    if not valid.any():
        # Preserve graph connection so opp_policy_head still gets gradient
        # contribution structure even when all rows are masked out.
        return opp_logits.sum() * 0.0
    L = opp_logits[valid]
    a = opp_action[valid]
    if opp_legal_mask is not None:
        # If the recorded action is itself illegal under the recorded mask
        # (e.g. off-by-one between turn-of-record and next-state mask),
        # F.cross_entropy(masked_logits, a) returns +inf → NaN gradients.
        # Drop those rows defensively.
        legal_for_target = opp_legal_mask[valid].gather(1, a.unsqueeze(1)).squeeze(1)
        if not legal_for_target.all():
            keep = legal_for_target
            if not keep.any():
                return opp_logits.sum() * 0.0
            L = L[keep]
            a = a[keep]
            row_mask = opp_legal_mask[valid][keep]
        else:
            row_mask = opp_legal_mask[valid]
        L = L.masked_fill(~row_mask, float("-inf"))
    return F.cross_entropy(L, a)


def plies_loss(pred: torch.Tensor, target: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
    """MSE regression on (plies_remaining / 200.0). `valid` masks entries
    where the target is unknown."""
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[valid], target[valid])


def nexus_loss_v3(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    legal_mask: torch.Tensor,
    old_logits: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute v3 combined loss. Returns dict with `total` and components."""
    logits = out["logits"]
    target_policy = batch["policies"]

    l_policy = policy_ce_loss(logits, target_policy)

    # Value vector - N-player aware
    l_value_vec = value_vec_loss(
        out["value_vec"], batch["value_vec"], batch["n_players"]
    )

    # Aux: opp policy
    l_opp = opp_policy_loss(
        out["opp_logits"],
        batch["opp_action"],
        opp_legal_mask=batch.get("opp_legal_mask"),
    )

    # Aux: plies remaining
    plies_target = batch["plies_target"].float()       # already normalized
    plies_valid = batch["plies_valid"]
    l_plies = plies_loss(out["plies"], plies_target, plies_valid)

    # Entropy bonus (we *subtract* it from total to encourage entropy)
    h_policy = policy_entropy_bonus(logits, legal_mask)

    total = (
        l_policy
        + Config.VALUE_VEC_LOSS_WEIGHT * l_value_vec
        + Config.OPP_POLICY_LOSS_WEIGHT * l_opp
        + Config.PLIES_LOSS_WEIGHT * l_plies
        - Config.ENTROPY_BONUS_WEIGHT * h_policy
    )

    # Optional KL regularizer to old network (unchanged from v2)
    l_kl = torch.tensor(0.0, device=logits.device)
    if old_logits is not None:
        from training.losses import kl_loss as _kl
        l_kl = Config.KL_LOSS_WEIGHT * _kl(logits, old_logits, legal_mask)
        total = total + l_kl

    return {
        "total": total,
        "policy": l_policy.detach(),
        "value_vec": l_value_vec.detach(),
        "opp_policy": l_opp.detach(),
        "plies": l_plies.detach(),
        "entropy": h_policy.detach(),
        "kl": l_kl.detach(),
    }
