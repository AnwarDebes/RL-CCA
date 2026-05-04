"""MCTS v4 - Gumbel AlphaZero with all the fixes:

  1. Per-player value backup using value_vec (correct for N=2..6, not just N=2)
  2. Priors used in non-root selection (Danihelka 2022 §3 - not bare PUCT)
  3. Sequential halving with full budget consumption (no flooring waste)
  4. Virtual loss for batched leaf evaluation
  5. Subtree reuse across moves (caller can call advance_root)

Returns (best_action, improved_policy, root_value_vec) where improved_policy
is the visit-count-weighted policy (suitable as MCTS-improved policy target
for self-play training, vs v3 which used raw network argmax).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import Config
from core.game_env import GameEnv
from core.action_space import get_legal_actions
from mcts.utils import (
    sample_gumbel, gumbel_top_k, normalize_q_values,
    completed_q_score, sequential_halving_rounds, compute_improved_policy,
)


MAX_PLAYERS = Config.MAX_PLAYERS


class NodeV4:
    """N-player aware node. Stores per-player value sum vector instead of
    a single negated scalar."""

    __slots__ = ("state_env", "player", "num_players", "parent",
                 "action_from_parent", "prior", "is_expanded", "is_terminal",
                 "children", "value_vec_sum", "visit_count",
                 "_virtual_loss")

    def __init__(self, state_env: GameEnv, player: int, num_players: int,
                 parent: Optional["NodeV4"] = None,
                 action_from_parent: Optional[int] = None,
                 prior: float = 0.0):
        self.state_env = state_env
        self.player = player
        self.num_players = num_players
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior = float(prior)
        self.is_expanded = False
        self.is_terminal = False
        self.children: Dict[int, "NodeV4"] = {}
        self.value_vec_sum = np.zeros(MAX_PLAYERS, dtype=np.float64)
        self.visit_count = 0
        self._virtual_loss = 0   # increments while leaf is being evaluated

    def q_for(self, p: int) -> float:
        """Average value-vec component for player p (player p's perspective)."""
        n = self.visit_count + self._virtual_loss
        if n == 0:
            return 0.0
        # virtual loss subtracts from current player's perspective
        return float(self.value_vec_sum[p]) / n

    def add_virtual_loss(self):
        """Discourage parallel descents from picking the same leaf."""
        self._virtual_loss += 1
        # Subtract a unit of the active player's value to push selection elsewhere
        self.value_vec_sum[self.player] -= float(Config.MCTS_VIRTUAL_LOSS)

    def remove_virtual_loss(self):
        self._virtual_loss = max(0, self._virtual_loss - 1)
        self.value_vec_sum[self.player] += float(Config.MCTS_VIRTUAL_LOSS)

    def expand(self, legal_actions: List[int], action_priors: Dict[int, float]):
        next_player = (self.player + 1) % self.num_players
        for a in legal_actions:
            self.children[a] = NodeV4(
                state_env=None,
                player=next_player,
                num_players=self.num_players,
                parent=self,
                action_from_parent=a,
                prior=action_priors.get(a, 1.0 / max(1, len(legal_actions))),
            )
        self.is_expanded = True


class GumbelMCTSv4:
    """Gumbel AlphaZero MCTS with N-player-correct backup and virtual loss."""

    def __init__(
        self,
        network,
        device: torch.device,
        num_simulations: int = 32,
        m: int = 8,
        c_scale: float = 2.5,
        add_root_noise: bool = False,
    ):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.m = m
        self.c_scale = c_scale
        self.add_root_noise = add_root_noise

    @torch.no_grad()
    def search(
        self,
        env: GameEnv,
        root: Optional[NodeV4] = None,
    ) -> Tuple[int, np.ndarray, np.ndarray, NodeV4]:
        """Run MCTS from `env`. If `root` is provided (subtree reuse), use it
        as the search root.

        Returns (best_action, improved_policy[1210], root_value_vec[6], root_node).
        """
        self.network.eval()
        player = env.current_player
        num_players = getattr(env, "num_players", 2)

        # 1. Initialize root: either reuse passed root or build fresh
        if root is None or root.state_env is None:
            root = NodeV4(state_env=env.clone(), player=player,
                          num_players=num_players)
            self._expand_and_evaluate(root)
        elif not root.is_expanded:
            root.state_env = env.clone()
            self._expand_and_evaluate(root)

        legal_actions = get_legal_actions(env.get_legal_mask(player))
        if not legal_actions:
            return 0, np.zeros(Config.ACTION_SPACE), np.zeros(MAX_PLAYERS), root
        if len(legal_actions) == 1:
            policy = np.zeros(Config.ACTION_SPACE)
            policy[legal_actions[0]] = 1.0
            return legal_actions[0], policy, root.value_vec_sum.copy(), root

        priors = np.array([root.children[a].prior for a in legal_actions])
        log_priors_full = np.full(Config.ACTION_SPACE, -30.0)
        for i, a in enumerate(legal_actions):
            log_priors_full[a] = math.log(max(priors[i], 1e-8))

        # 2. Gumbel-Top-k root candidates
        actual_m = min(self.m, len(legal_actions))
        candidates = gumbel_top_k(log_priors_full, legal_actions, actual_m)
        gumbel_noise = {a: sample_gumbel(1)[0] for a in candidates}

        # 3. Sequential halving with FULL budget consumption
        remaining = list(candidates)
        budget = self.num_simulations
        rounds = sequential_halving_rounds(actual_m)
        # Per Danihelka: equal sims per round; allocate leftover round-robin
        sims_per_round = max(actual_m, budget // max(1, rounds))

        for round_idx in range(rounds):
            if len(remaining) <= 1:
                break

            # Allocate sims THIS round across remaining candidates
            sims_each = max(1, sims_per_round // len(remaining))
            extra = sims_per_round - sims_each * len(remaining)
            for i, a in enumerate(remaining):
                count = sims_each + (1 if i < extra else 0)
                for _ in range(count):
                    self._simulate(root, a)

            # Eliminate bottom half by completed-Q + Gumbel noise
            scores = {}
            for a in remaining:
                child = root.children[a]
                # completed-Q from ROOT player's perspective
                q_a = child.q_for(player)
                scores[a] = (gumbel_noise.get(a, 0.0)
                             + log_priors_full[a]
                             + self.c_scale * q_a)
            remaining.sort(key=lambda a: scores[a], reverse=True)
            half = max(1, len(remaining) // 2)
            remaining = remaining[:half]

        best_action = remaining[0]

        # 4. Improved policy = visit-count weighted (proper MCTS target)
        visits = np.zeros(Config.ACTION_SPACE, dtype=np.float64)
        for a in legal_actions:
            visits[a] = root.children[a].visit_count
        if visits.sum() == 0:
            improved_policy = np.zeros(Config.ACTION_SPACE)
            for a in legal_actions:
                improved_policy[a] = 1.0 / len(legal_actions)
        else:
            improved_policy = visits / visits.sum()

        return int(best_action), improved_policy, root.value_vec_sum / max(1, root.visit_count), root

    def _simulate(self, root: NodeV4, root_action: int) -> None:
        """One simulation starting from a specific root child."""
        # Descend from the chosen root child
        node = root.children[root_action]
        path = [(root, root_action)]

        # If unexpanded, expand it (after stepping the env)
        if not node.is_expanded and not node.is_terminal:
            sim_env = root.state_env.clone()
            _, done = sim_env.step(root_action)
            node.state_env = sim_env
            if done:
                node.is_terminal = True
                value_vec = self._terminal_value_vec(sim_env)
                self._backup(path + [(node, None)], value_vec)
                return
            value_vec = self._expand_and_evaluate(node)
            self._backup(path + [(node, None)], value_vec)
            return

        # Already expanded: descend deeper using interior selection (with priors)
        current = node
        while current.is_expanded and not current.is_terminal and current.children:
            best_child_action = self._select_child(current)
            path.append((current, best_child_action))
            current = current.children[best_child_action]

            if not current.is_expanded and not current.is_terminal:
                sim_env = current.parent.state_env.clone()
                _, done = sim_env.step(current.action_from_parent)
                current.state_env = sim_env
                if done:
                    current.is_terminal = True
                    value_vec = self._terminal_value_vec(sim_env)
                else:
                    value_vec = self._expand_and_evaluate(current)
                self._backup(path + [(current, None)], value_vec)
                return

        # Reached terminal node along the way (rare with re-evaluation)
        if current.is_terminal:
            value_vec = self._terminal_value_vec(current.state_env)
            self._backup(path + [(current, None)], value_vec)

    def _select_child(self, node: NodeV4) -> int:
        """Interior selection that USES priors (Danihelka §3 formula).

        argmax_a [pi'(a) - N(a) / (1 + sum_b N(b))]
        where pi' is the improved policy from priors + completed Q's.
        """
        legal = list(node.children.keys())
        priors = {a: node.children[a].prior for a in legal}
        q_values = {a: node.children[a].q_for(node.player)
                    for a in legal if node.children[a].visit_count > 0}
        # Normalized priors (already softmax'd at expansion)
        # Completed-Q improved policy (from utils)
        prior_arr = np.array([priors[a] for a in legal])
        prior_arr = prior_arr / max(1e-9, prior_arr.sum())
        q_norm = normalize_q_values(q_values, legal)
        # Score = log_prior + c * q_norm
        log_p = np.log(prior_arr + 1e-9)
        c = self.c_scale
        # Visit count fraction
        visits = np.array([node.children[a].visit_count for a in legal], dtype=np.float64)
        total_visits = visits.sum()
        improved_logits = log_p + c * np.array([q_norm.get(a, 0.5) for a in legal])
        improved = np.exp(improved_logits - improved_logits.max())
        improved = improved / improved.sum()
        # Danihelka non-root rule
        score = improved - visits / (1.0 + total_visits)
        return legal[int(np.argmax(score))]

    def _expand_and_evaluate(self, node: NodeV4) -> np.ndarray:
        """Forward through network at this node; expand children. Returns value_vec."""
        env = node.state_env
        player = env.current_player
        state_t = env.get_state_tensor(player).unsqueeze(0).to(self.device)
        mask_t = env.get_legal_mask(player).unsqueeze(0).to(self.device)
        seat_t = torch.tensor([player], device=self.device)
        # Try v4 forward (full dict); fall back gracefully
        try:
            out = self.network(state_t, mask_t, current_seat=seat_t)
        except TypeError:
            out = self.network(state_t, mask_t)
        priors = out["policy"][0].cpu().numpy()
        value_vec = np.zeros(MAX_PLAYERS, dtype=np.float64)
        if "value_vec" in out:
            vv = out["value_vec"][0].cpu().numpy()
            value_vec[: len(vv)] = vv
        else:
            # v2 fallback: scalar value, broadcast (loses information)
            v = self.network.aggregate_value(out["value"])[0].item()
            value_vec[player] = v

        legal = get_legal_actions(env.get_legal_mask(player))
        if not legal:
            node.is_terminal = True
            return value_vec
        action_priors = {a: float(priors[a]) for a in legal}
        node.expand(legal, action_priors)
        return value_vec

    @staticmethod
    def _terminal_value_vec(env: GameEnv) -> np.ndarray:
        """Per-player terminal value (normalized to roughly [-1, 1] per player)."""
        from core import teacher_score as ts
        vec = np.zeros(MAX_PLAYERS, dtype=np.float64)
        for p in range(env.num_players):
            vec[p] = ts.normalized_value_target(env.compute_final_score(p))
        return vec

    def _backup(self, path: List[Tuple[NodeV4, Optional[int]]],
                value_vec: np.ndarray) -> None:
        """Backup value_vec to every node on the path. No sign flipping -
        each node stores per-player sums and selection extracts its own
        player's component."""
        for node, _ in path:
            node.value_vec_sum += value_vec
            node.visit_count += 1


def advance_root(root: NodeV4, action: int) -> Optional[NodeV4]:
    """Subtree reuse: descend root to its child for `action`, returning that
    child as the new root. Returns None if the action wasn't expanded.
    Detaches parent reference to free upstream memory."""
    if not root.is_expanded or action not in root.children:
        return None
    new_root = root.children[action]
    new_root.parent = None
    new_root.action_from_parent = None
    return new_root
