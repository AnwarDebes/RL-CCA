"""CMAZ MCTS: PUCT with vector Q backup and monotonic-mixer aggregation.

This is the workshop-tier sister of the flagship CD-MCTS. The novelty
slice is:

  * Per-component Q backups in MCTS (vector instead of scalar).
  * State-conditional monotonic mixer (QMIX-style) collapses the vector
    to a scalar at every PUCT selection step.
  * Inference-time override: the mixer's weights can be swapped for a
    user-specified utility without retraining.

What makes this different from CD-MCTS (the flagship):
  * No Plackett-Luce rank head - the components are *score elements*
    (e.g., Chinese Checkers' pin_goal_score, distance_score,
    time_score, move_score), not players.
  * No coalition-belief head.
  * No EXP-IX selector - uses standard PUCT.
  * The monotonic mixer is the only departure from vanilla AlphaZero.

What makes this not a reinvention of KataGo:
  * KataGo uses a *fixed* utility λ_winrate · winrate + λ_score ·
    tanh(score). Coefficients are hand-tuned, state-independent, and the
    combiner is linear, not a hypernetwork.
  * Our combiner is a learned, state-dependent QMIX-style hypernetwork.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np


@dataclass
class CMAZNode:
    state: Any
    current_player: int
    legal_actions: List[int]
    prior_policy: np.ndarray              # over legal actions
    component_value_leaf: np.ndarray      # (K,) per-component value at leaf
    encoder_features: np.ndarray          # (feature_dim,) - for mixer
    children: dict = field(default_factory=dict)
    child_visits: np.ndarray = field(default=None)
    # Per-action vector Q sums: shape (num_actions, K)
    child_value_sum: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        K = self.component_value_leaf.shape[0]
        L = len(self.legal_actions)
        if self.child_visits is None:
            self.child_visits = np.zeros(L, dtype=np.int64)
        if self.child_value_sum is None:
            self.child_value_sum = np.zeros((L, K), dtype=np.float64)

    @property
    def num_actions(self) -> int:
        return len(self.legal_actions)


@dataclass
class CMAZNetworkOutput:
    prior_policy: np.ndarray         # over full action space
    component_values: np.ndarray      # (K,) per-component values
    encoder_features: np.ndarray      # (feature_dim,)


def _restrict_prior(full_prior: np.ndarray, legal: List[int]) -> np.ndarray:
    sub = full_prior[legal]
    s = sub.sum()
    if s <= 0:
        return np.full(len(legal), 1.0 / len(legal))
    return sub / s


def _build_root(state: Any, network: Any, game: Any) -> CMAZNode:
    out = network.evaluate_cmaz(state)
    legal = game.legal_actions(state)
    return CMAZNode(
        state=state,
        current_player=game.current_player(state),
        legal_actions=legal,
        prior_policy=_restrict_prior(out.prior_policy, legal),
        component_value_leaf=out.component_values,
        encoder_features=out.encoder_features,
    )


def _puct_select(
    node: CMAZNode,
    mixer_apply: Callable[[np.ndarray, np.ndarray], float],
    c_puct: float = 1.5,
) -> int:
    """PUCT but Q is computed via the monotonic mixer applied to the
    accumulated component-value vector.

    mixer_apply(q_vec, features) -> scalar Q value.
    """
    K = node.num_actions
    total_visits = max(1, node.child_visits.sum())
    sqrt_T = math.sqrt(total_visits)
    q_per_action = np.zeros(K)
    for a in range(K):
        if node.child_visits[a] > 0:
            v_avg = node.child_value_sum[a] / node.child_visits[a]
        else:
            v_avg = node.component_value_leaf
        q_per_action[a] = mixer_apply(v_avg, node.encoder_features)
    u = c_puct * node.prior_policy * sqrt_T / (1 + node.child_visits)
    return int(np.argmax(q_per_action + u))


def run_simulation_cmaz(
    root: CMAZNode,
    network: Any,
    game: Any,
    mixer_apply: Callable[[np.ndarray, np.ndarray], float],
    c_puct: float = 1.5,
) -> None:
    path: List[tuple[CMAZNode, int]] = []
    node = root
    while True:
        if game.is_terminal(node.state):
            leaf_v = network.terminal_components(node.state)
            break
        action_idx = _puct_select(node, mixer_apply, c_puct=c_puct)
        action = node.legal_actions[action_idx]
        path.append((node, action_idx))
        if action_idx not in node.children:
            next_state, _ = game.step(node.state, action)
            if game.is_terminal(next_state):
                leaf_v = network.terminal_components(next_state)
                terminal_child = CMAZNode(
                    state=next_state,
                    current_player=game.current_player(next_state),
                    legal_actions=[],
                    prior_policy=np.zeros(0),
                    component_value_leaf=leaf_v,
                    encoder_features=np.zeros_like(node.encoder_features),
                )
                node.children[action_idx] = terminal_child
            else:
                out = network.evaluate_cmaz(next_state)
                legal = game.legal_actions(next_state)
                child = CMAZNode(
                    state=next_state,
                    current_player=game.current_player(next_state),
                    legal_actions=legal,
                    prior_policy=_restrict_prior(out.prior_policy, legal),
                    component_value_leaf=out.component_values,
                    encoder_features=out.encoder_features,
                )
                node.children[action_idx] = child
                leaf_v = out.component_values
            break
        node = node.children[action_idx]

    # Backup vector
    for parent, action_idx in path:
        parent.child_visits[action_idx] += 1
        parent.child_value_sum[action_idx] += leaf_v


def run_mcts_cmaz(
    state: Any,
    network: Any,
    game: Any,
    mixer_apply: Callable[[np.ndarray, np.ndarray], float],
    num_simulations: int,
    c_puct: float = 1.5,
) -> tuple[CMAZNode, np.ndarray]:
    root = _build_root(state, network, game)
    for _ in range(num_simulations):
        run_simulation_cmaz(root, network, game, mixer_apply, c_puct=c_puct)
    counts = root.child_visits.astype(np.float64)
    if counts.sum() == 0:
        return root, root.prior_policy.copy()
    return root, counts / counts.sum()
