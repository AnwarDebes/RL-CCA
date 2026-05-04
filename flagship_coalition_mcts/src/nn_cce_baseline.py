"""NN-CCE baseline (Yu et al., NeurIPS 2024) extended to N-player sequential games.

This is the **strongest external baseline** for the flagship paper. The
adversarial novelty-verification round identified NN-CCE (arxiv
2406.10411) as the closest published threat. This file implements an
extension of NN-CCE to N-player sequential perfect-information games so
we can compare CD-MCTS against it head-to-head.

NN-CCE in original form (Yu et al. 2024)
----------------------------------------
For *simultaneous-move 2-player matrix games at each tree node*, NN-CCE
maintains per-action regrets and selects via a no-regret learner whose
average distribution converges to a CCE of the per-node matrix game.
The value head of the network gives expected payoffs.

Extension to N-player sequential
--------------------------------
In a perfect-info sequential N-player game, only one player moves at
each node. The "matrix game" at the node degenerates: the moving player
picks an action; the resulting value is the network's predicted
per-player utility vector at the child state.

The natural extension of the NN-CCE *spirit* - no-regret selection
with CCE convergence guarantees - to this setting is:

  1. Network has a per-player **scalar** value head V[i](s) ∈ ℝ^N (one
     scalar per player). NO Plackett-Luce, NO coalition head. This is
     the key difference from CD-MCTS.
  2. At each node, the moving player selects via an EXP-IX no-regret
     learner over its legal actions, with regret updates driven by
     V[mover](child) - V[mover](current).
  3. Backup: per-player scalar value averaged up the tree.

By Lemma 2 of our theorem document, the resulting empirical root play
converges to a CCE of the induced root meta-game. This is exactly the
NN-CCE convergence claim, applied to sequential games via the standard
extensive-form-as-meta-game reduction.

What this baseline lacks vs CD-MCTS
-----------------------------------
  * No structured rank-distribution value (it has scalar V[i] per
    player; rank correlations between opponents' outcomes are not
    captured).
  * No coalition belief.

Any improvement of CD-MCTS over this baseline is therefore directly
attributable to the PL head + coalition head, which is the load-bearing
ablation argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np

from .cce_selector import (
    SelectorState,
    policy_at_root,
    select_action,
    update_regrets,
)


@dataclass
class NNCCENode:
    state: Any
    current_player: int
    num_players: int
    legal_actions: List[int]
    prior_policy: np.ndarray             # restricted to legal actions
    leaf_value_per_player: np.ndarray    # (N,) per-player scalar value at leaf
    children: dict = field(default_factory=dict)
    selector_state: Optional[SelectorState] = None
    child_visits: np.ndarray = field(default=None)
    child_value_sum: np.ndarray = field(default=None)  # (num_actions, N) per-player

    def __post_init__(self) -> None:
        K = len(self.legal_actions)
        N = self.num_players
        if self.selector_state is None:
            self.selector_state = SelectorState(
                num_actions=K,
                prior=self.prior_policy.copy(),
                # NN-CCE does NOT use the coalition penalty.
                coalition_weight=0.0,
            )
        if self.child_visits is None:
            self.child_visits = np.zeros(K, dtype=np.int64)
        if self.child_value_sum is None:
            self.child_value_sum = np.zeros((K, N), dtype=np.float64)


@dataclass
class NNCCENetworkOutput:
    prior_policy: np.ndarray              # full action space
    per_player_value: np.ndarray          # (N,)


def _restrict_prior(full_prior: np.ndarray, legal: List[int]) -> np.ndarray:
    sub = full_prior[legal]
    s = sub.sum()
    if s <= 0:
        return np.full(len(legal), 1.0 / len(legal))
    return sub / s


def _build_root(state: Any, network: Any, game: Any) -> NNCCENode:
    out = network.evaluate_nncce(state)
    legal = game.legal_actions(state)
    return NNCCENode(
        state=state,
        current_player=game.current_player(state),
        num_players=game.num_players(state),
        legal_actions=legal,
        prior_policy=_restrict_prior(out.prior_policy, legal),
        leaf_value_per_player=out.per_player_value,
    )


def _terminal_per_player_value(state: Any, game: Any) -> np.ndarray:
    """For a terminal state, derive per-player utility (N - rank)/(N-1) ∈ [0,1]."""
    M = game.terminal_marginal(state)
    N = M.shape[0]
    ranks = np.array([int(M[p].argmax()) + 1 for p in range(N)])
    return (N - ranks) / (N - 1)


def run_simulation_nncce(
    root: NNCCENode,
    network: Any,
    game: Any,
    rng: np.random.Generator,
) -> None:
    """One NN-CCE rollout (no-regret selection per node, scalar backup)."""
    path: List[tuple[NNCCENode, int]] = []
    node = root
    while True:
        if game.is_terminal(node.state):
            leaf_v = _terminal_per_player_value(node.state, game)
            break
        action_idx = select_action(node.selector_state, rng=rng)
        action = node.legal_actions[action_idx]
        path.append((node, action_idx))
        if action_idx not in node.children:
            next_state, _ = game.step(node.state, action)
            if game.is_terminal(next_state):
                leaf_v = _terminal_per_player_value(next_state, game)
                terminal_child = NNCCENode(
                    state=next_state,
                    current_player=game.current_player(next_state),
                    num_players=node.num_players,
                    legal_actions=[],
                    prior_policy=np.zeros(0),
                    leaf_value_per_player=leaf_v,
                )
                node.children[action_idx] = terminal_child
            else:
                out = network.evaluate_nncce(next_state)
                legal = game.legal_actions(next_state)
                child = NNCCENode(
                    state=next_state,
                    current_player=game.current_player(next_state),
                    num_players=game.num_players(next_state),
                    legal_actions=legal,
                    prior_policy=_restrict_prior(out.prior_policy, legal),
                    leaf_value_per_player=out.per_player_value,
                )
                node.children[action_idx] = child
                leaf_v = out.per_player_value
            break
        node = node.children[action_idx]

    # Backup
    for parent, action_idx in path:
        parent.child_visits[action_idx] += 1
        parent.child_value_sum[action_idx] += leaf_v
        # All-action regret update from current player's perspective.
        K = len(parent.legal_actions)
        cp = parent.current_player
        q_vec = np.zeros(K)
        for i in range(K):
            if i == action_idx:
                q_vec[i] = leaf_v[cp]
            elif parent.child_visits[i] > 0:
                q_vec[i] = parent.child_value_sum[i, cp] / parent.child_visits[i]
            else:
                q_vec[i] = parent.leaf_value_per_player[cp]
        update_regrets(parent.selector_state, action_idx, q_vec)


def run_mcts_nncce(
    state: Any,
    network: Any,
    game: Any,
    num_simulations: int,
    seed: Optional[int] = None,
) -> tuple[NNCCENode, np.ndarray]:
    rng = np.random.default_rng(seed)
    root = _build_root(state, network, game)
    for _ in range(num_simulations):
        run_simulation_nncce(root, network, game, rng)
    pi = policy_at_root(root.selector_state, temperature=1.0)
    return root, pi


# ----------------------------------------------------------------------
# Evaluator wrapping CDMCTSNetwork (so NN-CCE shares architecture with
# CD-MCTS for fair comparison - only differs in MCTS-side machinery).
# ----------------------------------------------------------------------


import torch
import torch.nn.functional as F


class NNCCEEvaluator:
    """Returns per-player scalar values from the CDMCTSNetwork's scalar
    value head, distributed evenly to opponents (zero-sum-like default).

    For a fair comparison, we use the SAME architecture as CD-MCTS - only
    the MCTS-side machinery differs.
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
    def evaluate_nncce(self, state: Any) -> NNCCENetworkOutput:
        feats = self.state_to_features(state)
        x = torch.from_numpy(feats).float().unsqueeze(0)
        policy_logits, _theta, _A, _beta, scalar_v = self.network(x)
        prior = F.softmax(policy_logits[0], dim=-1).cpu().numpy().astype(np.float64)
        N = self.num_players_fn(state)
        cp = self.current_player_fn(state)
        v_cp = float(scalar_v[0].item())
        per_player = np.full(N, -v_cp / max(1, N - 1), dtype=np.float64)
        per_player[cp] = v_cp
        return NNCCENetworkOutput(prior_policy=prior, per_player_value=per_player)
