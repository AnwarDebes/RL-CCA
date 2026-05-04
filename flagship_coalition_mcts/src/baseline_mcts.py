"""Scalar-PUCT baseline: Multiplayer AlphaZero (Petosa & Balch 2019 style).

This is ablation **A0** in the paper. It uses:

  * The same encoder + policy head + scalar value head as CD-MCTS.
  * NO Plackett-Luce head, NO coalition head.
  * Standard PUCT selection (no EXP-IX, no CCE).
  * Scalar value backup (not vector placement-marginal).

The point is to provide a head-to-head comparison where the only
difference is the *MCTS-side novelty*. Encoder + policy + scalar value
are kept architecturally identical so any performance gap is
attributable to the new MCTS components, not to a fairer architecture.

API mirrors mcts.run_mcts so experiment scripts can swap freely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np


@dataclass
class ScalarNode:
    state: Any
    current_player: int
    num_players: int
    legal_actions: List[int]
    prior_policy: np.ndarray         # restricted to legal actions
    leaf_value: np.ndarray           # per-player scalar value at this node
    children: dict = field(default_factory=dict)
    child_visits: np.ndarray = field(default=None)
    child_value_sum: np.ndarray = field(default=None)  # per-player Q sum

    def __post_init__(self) -> None:
        K = len(self.legal_actions)
        N = self.num_players
        if self.child_visits is None:
            self.child_visits = np.zeros(K, dtype=np.int64)
        if self.child_value_sum is None:
            self.child_value_sum = np.zeros((K, N), dtype=np.float64)

    @property
    def num_actions(self) -> int:
        return len(self.legal_actions)


def _restrict_prior(full_prior: np.ndarray, legal: List[int]) -> np.ndarray:
    sub = full_prior[legal]
    s = sub.sum()
    if s <= 0:
        return np.full(len(legal), 1.0 / len(legal))
    return sub / s


def _puct_select(node: ScalarNode, c_puct: float = 1.5) -> int:
    """Standard PUCT selection."""
    K = node.num_actions
    total_visits = max(1, node.child_visits.sum())
    sqrt_T = math.sqrt(total_visits)
    cp = node.current_player
    # Q from current player's perspective
    q_per_action = np.zeros(K)
    for a in range(K):
        if node.child_visits[a] > 0:
            q_per_action[a] = node.child_value_sum[a, cp] / node.child_visits[a]
        else:
            # First-play urgency: small negative initial value (prevents
            # exploration collapse).
            q_per_action[a] = node.leaf_value[cp]
    u = c_puct * node.prior_policy * sqrt_T / (1 + node.child_visits)
    score = q_per_action + u
    return int(score.argmax())


@dataclass
class ScalarNetworkOutput:
    prior_policy: np.ndarray   # full action space
    per_player_value: np.ndarray  # (N,) scalar per player from current state


def _build_root_scalar(state: Any, network: Any, game: Any) -> ScalarNode:
    out = network.evaluate_scalar(state)
    legal = game.legal_actions(state)
    return ScalarNode(
        state=state,
        current_player=game.current_player(state),
        num_players=game.num_players(state),
        legal_actions=legal,
        prior_policy=_restrict_prior(out.prior_policy, legal),
        leaf_value=out.per_player_value,
    )


def run_simulation_scalar(
    root: ScalarNode, network: Any, game: Any, c_puct: float = 1.5
) -> None:
    path: List[tuple[ScalarNode, int]] = []
    node = root
    while True:
        if game.is_terminal(node.state):
            ranks = _terminal_ranks(node.state, game)
            N = node.num_players
            leaf_v = np.array(
                [(N - r) / (N - 1) for r in ranks], dtype=np.float64
            )
            break
        action_idx = _puct_select(node, c_puct=c_puct)
        action = node.legal_actions[action_idx]
        path.append((node, action_idx))
        if action_idx not in node.children:
            next_state, _ = game.step(node.state, action)
            if game.is_terminal(next_state):
                ranks = _terminal_ranks(next_state, game)
                N = node.num_players
                leaf_v = np.array(
                    [(N - r) / (N - 1) for r in ranks], dtype=np.float64
                )
                terminal_child = ScalarNode(
                    state=next_state,
                    current_player=game.current_player(next_state),
                    num_players=node.num_players,
                    legal_actions=[],
                    prior_policy=np.zeros(0),
                    leaf_value=leaf_v,
                )
                node.children[action_idx] = terminal_child
            else:
                out = network.evaluate_scalar(next_state)
                legal = game.legal_actions(next_state)
                child = ScalarNode(
                    state=next_state,
                    current_player=game.current_player(next_state),
                    num_players=game.num_players(next_state),
                    legal_actions=legal,
                    prior_policy=_restrict_prior(out.prior_policy, legal),
                    leaf_value=out.per_player_value,
                )
                node.children[action_idx] = child
                leaf_v = out.per_player_value
            break
        node = node.children[action_idx]
    # Backup
    for parent, action_idx in path:
        parent.child_visits[action_idx] += 1
        parent.child_value_sum[action_idx] += leaf_v


def _terminal_ranks(state: Any, game: Any) -> tuple:
    M = game.terminal_marginal(state)
    return tuple(int(M[p].argmax()) + 1 for p in range(M.shape[0]))


def run_mcts_scalar(
    state: Any,
    network: Any,
    game: Any,
    num_simulations: int,
    c_puct: float = 1.5,
) -> tuple[ScalarNode, np.ndarray]:
    root = _build_root_scalar(state, network, game)
    for _ in range(num_simulations):
        run_simulation_scalar(root, network, game, c_puct=c_puct)
    counts = root.child_visits.astype(np.float64)
    if counts.sum() == 0:
        return root, root.prior_policy.copy()
    return root, counts / counts.sum()


# ----------------------------------------------------------------------
# A scalar evaluator that wraps the same CDMCTSNetwork - for fair
# comparison, we share architecture and only differ in MCTS.
# ----------------------------------------------------------------------


import torch
import torch.nn.functional as F


class ScalarEvaluator:
    """Wraps a CDMCTSNetwork to expose `evaluate_scalar` as expected by
    run_mcts_scalar. The per-player value is:

      v_self = network's scalar value head output (current player)
      v_others = -v_self / (N-1)  (zero-sum-like default)

    This is the simplest baseline; alternative: use the PL head's expected
    rank as the per-player value (intermediate ablation A1). See A1
    variant in baseline_mcts_a1.py if needed.
    """

    def __init__(
        self,
        network: Any,
        state_to_features: Callable[[Any], np.ndarray],
        current_player_fn: Callable[[Any], int],
        num_players_fn: Callable[[Any], int],
    ) -> None:
        self.network = network
        self.state_to_features = state_to_features
        self.current_player_fn = current_player_fn
        self.num_players_fn = num_players_fn

    @torch.no_grad()
    def evaluate_scalar(self, state: Any) -> ScalarNetworkOutput:
        feats = self.state_to_features(state)
        x = torch.from_numpy(feats).float().unsqueeze(0)
        policy_logits, _theta, _A, _beta, scalar_v = self.network(x)
        prior = F.softmax(policy_logits[0], dim=-1).cpu().numpy().astype(np.float64)
        N = self.num_players_fn(state)
        cp = self.current_player_fn(state)
        v_cp = float(scalar_v[0].item())
        per_player = np.full(N, -v_cp / max(1, N - 1), dtype=np.float64)
        per_player[cp] = v_cp
        return ScalarNetworkOutput(prior_policy=prior, per_player_value=per_player)
