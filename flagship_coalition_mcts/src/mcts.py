"""CD-MCTS tree: per-component placement-marginal backup + CCE-regret selection.

This module ties together the three pillars:

  1. PlackettLuceHead -> placement marginal M ∈ [0,1]^{N×N} where M[p, k] is
     P(player p finishes in position k+1) at the leaf state.
  2. CoalitionHead -> per-action coalition penalty score used by the selector.
  3. CCE EXP-IX selector -> action choice.

The KEY DEPARTURE from standard AlphaZero MCTS is the **vector backup**.
Rather than averaging a scalar V(s) up the tree, we average the *full
placement marginal matrix* M(s_leaf) up the tree. The current player's
expected rank - derived from the relevant row of M - is the scalar passed
to the regret update; but storing the full matrix lets us:

    * Re-aggregate at any node via any utility function (the "inference-time
      re-weighting" experiment from CMAZ that survives even at flagship
      scale).
    * Compute interpretable per-state mixer weights for the paper.
    * Plug in alternative aggregation operators in ablations without
      re-running self-play.

Network interface
-----------------
The MCTS is generic over a network. We define a minimal Protocol-style
interface so the test stub and the real network can both plug in:

    network.evaluate(state) -> NetworkOutput
        prior_policy: ndarray (num_actions,)
        placement_marginals: ndarray (num_players, num_players)
        coalition_alignment: ndarray (num_players,)
            P(opponent q is in coalition against current player)
        terminal_marginal: optional fixed M for terminal states

Game interface
--------------
    game.legal_actions(state) -> List[int]
    game.step(state, action) -> (next_state, current_player_after_step)
    game.is_terminal(state) -> bool
    game.terminal_marginal(state) -> ndarray (N, N)
    game.current_player(state) -> int
    game.num_players(state) -> int

We deliberately keep these as duck-typed callables to avoid coupling the
MCTS to a specific framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

from .cce_selector import (
    SelectorState,
    policy_at_root,
    select_action,
    update_regrets,
)


@dataclass
class NetworkOutput:
    prior_policy: np.ndarray            # (num_actions,)
    placement_marginals: np.ndarray     # (N, N)
    coalition_alignment: np.ndarray     # (N,)


@dataclass
class Node:
    """One tree node.

    children[a] is created lazily when action a is first selected and
    expanded. value_sum is the accumulated placement-marginal matrix; we
    divide by visits at any time to recover the current Q(s, a) per
    component. The selector_state holds the per-node EXP-IX bookkeeping.
    """

    state: Any
    current_player: int
    num_players: int
    legal_actions: List[int]
    prior_policy: np.ndarray
    coalition_alignment: np.ndarray
    placement_marginal: np.ndarray   # (N, N), the leaf-network estimate
    children: Dict[int, "Node"] = field(default_factory=dict)
    selector_state: Optional[SelectorState] = None
    # Per-action accumulators for the *vector* backup.
    child_visits: Optional[np.ndarray] = None       # (num_actions,)
    child_value_sum: Optional[np.ndarray] = None    # (num_actions, N, N)

    def __post_init__(self) -> None:
        K = len(self.legal_actions)
        N = self.num_players
        if self.selector_state is None:
            # The selector's prior is over the SUBSET of legal actions, but
            # the network outputs a prior over the FULL action space. The
            # caller is responsible for passing in `prior_policy` already
            # restricted to legal actions - we assert here.
            assert self.prior_policy.shape == (K,), (
                f"prior_policy shape {self.prior_policy.shape} != ({K},) "
                "(must be restricted to legal actions before constructing Node)"
            )
            self.selector_state = SelectorState(
                num_actions=K,
                prior=self.prior_policy.copy(),
            )
        if self.child_visits is None:
            self.child_visits = np.zeros(K, dtype=np.int64)
        if self.child_value_sum is None:
            self.child_value_sum = np.zeros((K, N, N), dtype=np.float64)

    @property
    def num_actions(self) -> int:
        return len(self.legal_actions)

    def child_q_marginal(self, action_idx: int) -> np.ndarray:
        """Average placement marginal for action_idx as currently estimated."""
        n = self.child_visits[action_idx]
        if n == 0:
            return self.placement_marginal.copy()  # bootstrap from leaf prior
        return self.child_value_sum[action_idx] / n


def _restrict_prior(full_prior: np.ndarray, legal: List[int]) -> np.ndarray:
    sub = full_prior[legal]
    s = sub.sum()
    if s <= 0:
        return np.full(len(legal), 1.0 / len(legal))
    return sub / s


def _build_root(
    state: Any,
    network: Any,
    game: Any,
) -> Node:
    out = network.evaluate(state)
    legal = game.legal_actions(state)
    cp = game.current_player(state)
    N = game.num_players(state)
    return Node(
        state=state,
        current_player=cp,
        num_players=N,
        legal_actions=legal,
        prior_policy=_restrict_prior(out.prior_policy, legal),
        coalition_alignment=out.coalition_alignment,
        placement_marginal=out.placement_marginals,
    )


def _coalition_action_penalty(
    parent: Node,
    legal: List[int],
    network: Any,
    game: Any,
) -> np.ndarray:
    """Compute per-action coalition penalty by 1-step lookahead.

    For each legal action a, evaluate child state's coalition_alignment
    and combine it with the parent player's coalition belief: how much
    the action moves us toward an alignment opposing us.

    Cheap version: penalty(a) = sum over current_player's opp of
    coalition_alignment_child[opp]. Higher = more opponents in alignment.
    """
    cp = parent.current_player
    K = len(legal)
    pen = np.zeros(K, dtype=np.float64)
    for i, a in enumerate(legal):
        next_state, _ = game.step(parent.state, a)
        if game.is_terminal(next_state):
            pen[i] = 0.0
            continue
        out = network.evaluate(next_state)
        opp_align = out.coalition_alignment.copy()
        opp_align[cp] = 0.0
        pen[i] = float(opp_align.sum())
    return pen


def _expected_rank_for_player(M: np.ndarray, player: int) -> float:
    """E[rank(player)] given placement marginal M (1-indexed).

    Lower is better (1 = winner)."""
    N = M.shape[0]
    positions = np.arange(1, N + 1)
    return float((M[player] * positions).sum())


def _q_scalar_for_player(M: np.ndarray, player: int) -> float:
    """Convert placement marginal to a scalar Q for the regret update.

    We use Q = -E[rank(player)] (higher = better, in [-N, -1]), shifted to
    Q' = (N - E[rank]) / (N - 1) ∈ [0, 1] for a normalised regret-update
    scale.
    """
    N = M.shape[0]
    er = _expected_rank_for_player(M, player)
    return (N - er) / (N - 1)


def run_simulation(
    root: Node,
    network: Any,
    game: Any,
    rng: np.random.Generator,
    coalition_weight: float = 0.5,
) -> None:
    """One MCTS rollout starting at root.

    Steps:
        1. Selection: descend tree using CCE selector until we hit an
           unexpanded action or a terminal state.
        2. Expansion: create child node for the unexpanded action.
        3. Backup: propagate placement-marginal matrix up the path,
           updating selector regrets at each internal node.
    """
    path: List[tuple[Node, int]] = []  # list of (node, action_idx)
    node = root

    # --- Selection + expansion loop ---
    while True:
        if game.is_terminal(node.state):
            leaf_marginal = game.terminal_marginal(node.state)
            break

        # Compute coalition penalty at this node (cheap 1-step lookahead).
        # Cached per-node would be cheaper; this is the simple version.
        coal_penalty = _coalition_action_penalty(node, node.legal_actions, network, game)
        node.selector_state.coalition_weight = coalition_weight

        action_idx = select_action(
            node.selector_state, coalition_score=coal_penalty, rng=rng
        )
        action = node.legal_actions[action_idx]
        path.append((node, action_idx))

        if action_idx not in node.children:
            # Expand
            next_state, _ = game.step(node.state, action)
            if game.is_terminal(next_state):
                leaf_marginal = game.terminal_marginal(next_state)
                # Create a thin terminal child without network evaluation.
                terminal_child = Node(
                    state=next_state,
                    current_player=game.current_player(next_state),
                    num_players=node.num_players,
                    legal_actions=[],
                    prior_policy=np.zeros(0),
                    coalition_alignment=np.zeros(node.num_players),
                    placement_marginal=leaf_marginal,
                )
                node.children[action_idx] = terminal_child
            else:
                out = network.evaluate(next_state)
                legal = game.legal_actions(next_state)
                child = Node(
                    state=next_state,
                    current_player=game.current_player(next_state),
                    num_players=game.num_players(next_state),
                    legal_actions=legal,
                    prior_policy=_restrict_prior(out.prior_policy, legal),
                    coalition_alignment=out.coalition_alignment,
                    placement_marginal=out.placement_marginals,
                )
                node.children[action_idx] = child
                leaf_marginal = out.placement_marginals
            break  # stop after creating leaf; backup follows
        # already-expanded: continue traversal
        node = node.children[action_idx]

    # --- Backup: vector + scalar regret ---
    for parent, action_idx in path:
        parent.child_visits[action_idx] += 1
        parent.child_value_sum[action_idx] += leaf_marginal
        # All-action regret update needs a Q-vector. Use current
        # estimates (each action's average M) plus the leaf for the
        # taken action.
        K = parent.num_actions
        q_vec = np.zeros(K)
        for i in range(K):
            if i == action_idx:
                M_i = leaf_marginal
            else:
                # current estimate (bootstrap with parent's placement marginal
                # if this action has never been visited).
                M_i = parent.child_q_marginal(i)
            q_vec[i] = _q_scalar_for_player(M_i, parent.current_player)
        update_regrets(parent.selector_state, action_idx, q_vec)


def run_mcts(
    state: Any,
    network: Any,
    game: Any,
    num_simulations: int,
    coalition_weight: float = 0.5,
    seed: Optional[int] = None,
) -> tuple[Node, np.ndarray]:
    """Run num_simulations rollouts from ``state``.

    Returns (root_node, training_policy_target).
    The training policy is visit-count-weighted (AlphaZero convention).
    """
    rng = np.random.default_rng(seed)
    root = _build_root(state, network, game)
    for _ in range(num_simulations):
        run_simulation(root, network, game, rng, coalition_weight=coalition_weight)
    pi = policy_at_root(root.selector_state, temperature=1.0)
    return root, pi
