"""NEXUS training losses (v3 - single-scalar value).

L = L_policy + Config.VALUE_LOSS_WEIGHT * L_value + Config.KL_LOSS_WEIGHT * L_kl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def policy_loss(logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    """Cross-entropy from MCTS-improved policy target to network logits.

    logits:        [B, 1210] raw masked logits (illegal = -inf).
    target_policy: [B, 1210] probability distribution over legal moves.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.where(
        torch.isinf(log_probs), torch.zeros_like(log_probs), log_probs
    )
    return -torch.sum(target_policy * log_probs, dim=-1).mean()


def value_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between predicted scalar value and target scalar value.

    pred:   [B] from network's value head (already in [-1, 1]).
    target: [B] normalized teacher final_score in [-1, 1].
    """
    if pred.dim() > 1:
        pred = pred.squeeze(-1)
    if target.dim() > 1:
        target = target.squeeze(-1)
    return F.mse_loss(pred, target)


def kl_loss(
    logits: torch.Tensor,
    old_logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """KL(old || current) - stability regularizer.

    Robust to all-`-inf` rows: nan_to_num rescues both log_p and q from
    NaN propagation when a row has zero legal moves.
    """
    masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
    masked_old = old_logits.masked_fill(~legal_mask, float("-inf"))

    log_p = F.log_softmax(masked_logits, dim=-1)
    q = F.softmax(masked_old, dim=-1)

    log_p = torch.nan_to_num(log_p, nan=0.0, posinf=0.0, neginf=0.0)
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    return F.kl_div(log_p, q, reduction="batchmean")


def consistency_loss(pred_repr: torch.Tensor, target_repr: torch.Tensor) -> torch.Tensor:
    """SimSiam-style consistency on representations."""
    return F.mse_loss(
        F.normalize(pred_repr, dim=-1),
        F.normalize(target_repr, dim=-1),
    )


def nexus_loss(
    logits: torch.Tensor,
    target_policy: torch.Tensor,
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
    old_logits: torch.Tensor = None,
    legal_mask: torch.Tensor = None,
    return_components: bool = False,
):
    """Combined NEXUS primary loss."""
    l_policy = policy_loss(logits, target_policy)
    l_value = Config.VALUE_LOSS_WEIGHT * value_loss(value_pred, value_target)

    total = l_policy + l_value
    l_kl_val = torch.tensor(0.0, device=logits.device)

    if old_logits is not None and legal_mask is not None:
        l_kl_val = Config.KL_LOSS_WEIGHT * kl_loss(logits, old_logits, legal_mask)
        total = total + l_kl_val

    if return_components:
        return total, {
            "policy": l_policy.item(),
            "value": l_value.item(),
            "kl": l_kl_val.item(),
        }
    return total
