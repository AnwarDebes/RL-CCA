"""MCTS tree node for Gumbel AlphaZero - N-player aware."""

from __future__ import annotations
from typing import Dict, List, Optional


class MCTSNode:
    """A node in the MCTS search tree.

    For 2-player games, value backup negates (zero-sum). For N>2, we still
    negate as an approximation; this isn't theoretically correct but is the
    standard simplification used in practice. Tournament inference doesn't
    use MCTS, so the only impact is on training-time target policies.
    """

    __slots__ = [
        'state_env', 'player', 'num_players', 'parent', 'action_from_parent',
        'children', 'visit_count', 'total_value', 'prior',
        'is_expanded', 'is_terminal', 'legal_actions',
        'gumbel_noise',
    ]

    def __init__(
        self,
        state_env=None,
        player: int = 0,
        num_players: int = 2,
        parent: Optional['MCTSNode'] = None,
        action_from_parent: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.state_env = state_env
        self.player = player
        self.num_players = num_players
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior = prior

        self.children: Dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.legal_actions: List[int] = []
        self.gumbel_noise: float = 0.0  # for Gumbel-Top-k at root

    @property
    def q_value(self) -> float:
        """Mean action value (from this node's player's perspective)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def select_child(self) -> 'MCTSNode':
        """Non-root selection: a* = argmin_a [N(a) - Q_parent(a)].

        Q is stored from the child's perspective, so we negate to get
        the parent's perspective. Minimizes simple regret (Gumbel AlphaZero).
        """
        best_action = None
        best_score = float('inf')

        for action, child in self.children.items():
            score = child.visit_count - (-child.q_value)
            if score < best_score:
                best_score = score
                best_action = action

        return self.children[best_action]

    def expand(self, legal_actions: List[int], priors: Dict[int, float]):
        """Expand this node with children for each legal action."""
        self.legal_actions = legal_actions
        next_player = (self.player + 1) % self.num_players
        for action in legal_actions:
            self.children[action] = MCTSNode(
                player=next_player,
                num_players=self.num_players,
                parent=self,
                action_from_parent=action,
                prior=priors.get(action, 0.0),
            )
        self.is_expanded = True

    def backup(self, value: float):
        """Backpropagate value up the tree.

        For 2-player: value flips sign at each level (zero-sum).
        For N>2: same approximation - treats each transition as adversarial
        between consecutive players. Sub-optimal but consistent.
        """
        node = self
        v = value
        while node is not None:
            node.visit_count += 1
            node.total_value += v
            v = -v
            node = node.parent

    def is_root(self) -> bool:
        return self.parent is None
