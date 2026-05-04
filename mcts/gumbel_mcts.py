"""Gumbel MCTS with Sequential Halving (Danihelka et al., ICLR 2022).

Guarantees policy improvement even with very few simulations.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from core.game_env import GameEnv
from core.action_space import encode_action, decode_action, get_legal_actions
from mcts.node import MCTSNode
from mcts.utils import (
    sample_gumbel,
    gumbel_top_k,
    normalize_q_values,
    completed_q_score,
    sequential_halving_rounds,
    compute_improved_policy,
)


class GumbelMCTS:
    """Gumbel AlphaZero MCTS with Sequential Halving."""

    def __init__(
        self,
        network,
        device: torch.device,
        num_simulations: int = 32,
        m: int = Config.GUMBEL_M,
        c_scale: float = 2.5,
        add_noise: bool = False,
        dirichlet_alpha: float = Config.DIRICHLET_ALPHA,
        dirichlet_frac: float = Config.DIRICHLET_FRAC,
    ):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.m = m
        self.c_scale = c_scale
        self.add_noise = add_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

    def _net_forward(self, state_tensor, legal_mask, seat_t, opp_hidden):
        """Call self.network, passing v3-only kwargs only when supported.

        This keeps the same MCTS code working with both v2 (NexusNet) and
        v3 (NexusNetV3) - v3 needs `current_seat` so the value slot
        belongs to the right player.
        """
        try:
            return self.network(state_tensor, legal_mask,
                                current_seat=seat_t, opp_hidden=opp_hidden)
        except TypeError:
            return self.network(state_tensor, legal_mask, opp_hidden=opp_hidden)

    @torch.no_grad()
    def search(
        self,
        env: GameEnv,
        opp_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[int, np.ndarray, float]:
        """Run Gumbel MCTS from the current game state.

        Args:
            env: game environment (will be cloned for simulations).
            opp_hidden: opponent GRU hidden state.

        Returns:
            (best_action, improved_policy, root_value)
        """
        self.network.eval()
        player = env.current_player

        # 1. Get network prior for root
        state_tensor = env.get_state_tensor(player).unsqueeze(0).to(self.device)
        legal_mask = env.get_legal_mask(player).unsqueeze(0).to(self.device)

        seat_t = torch.tensor([player], device=self.device)
        out = self._net_forward(state_tensor, legal_mask, seat_t, opp_hidden)
        priors = out['policy'][0].cpu().numpy()  # [1210]
        root_value = self.network.aggregate_value(out['value'])[0].item()

        legal_actions = get_legal_actions(env.get_legal_mask(player))
        if len(legal_actions) == 0:
            return 0, np.zeros(Config.ACTION_SPACE), 0.0
        if len(legal_actions) == 1:
            policy = np.zeros(Config.ACTION_SPACE)
            policy[legal_actions[0]] = 1.0
            return legal_actions[0], policy, root_value

        # Add Dirichlet noise during training (skip when Gumbel root selection
        # is in use - Gumbel-top-k provides exploration on its own)
        if self.add_noise and Config.MCTS_USE_DIRICHLET:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for i, a in enumerate(legal_actions):
                priors[a] = (1 - self.dirichlet_frac) * priors[a] + self.dirichlet_frac * noise[i]

        # 2. Gumbel-Top-k: sample m candidates
        log_priors = np.full(Config.ACTION_SPACE, -30.0)
        for a in legal_actions:
            log_priors[a] = math.log(max(priors[a], 1e-8))

        actual_m = min(self.m, len(legal_actions))
        candidates = gumbel_top_k(log_priors, legal_actions, actual_m)

        # Store Gumbel noise for completed score computation
        gumbel_noise = {}
        for a in candidates:
            gumbel_noise[a] = sample_gumbel(1)[0]

        # 3. Create root node (N-player aware)
        num_players = getattr(env, 'num_players', 2)
        root = MCTSNode(state_env=env.clone(), player=player,
                        num_players=num_players)
        action_priors = {a: priors[a] for a in legal_actions}
        root.expand(legal_actions, action_priors)

        # 4. Sequential Halving
        remaining = list(candidates)
        budget = self.num_simulations
        rounds = sequential_halving_rounds(actual_m)
        sims_per_round = max(1, budget // max(1, rounds))

        for round_idx in range(rounds):
            if len(remaining) <= 1:
                break

            sims_each = max(1, sims_per_round // max(1, len(remaining)))

            # Run simulations for each candidate
            for action in remaining:
                for _ in range(sims_each):
                    self._simulate(root, action, env, opp_hidden)

            # Compute completed scores and eliminate bottom half
            # Negate: child.q_value is from child's perspective (opponent),
            # root needs values from its OWN perspective for selection.
            q_values = {a: -root.children[a].q_value for a in remaining}
            q_norm = normalize_q_values(q_values, remaining)

            scores = {}
            for a in remaining:
                scores[a] = completed_q_score(
                    gumbel_noise.get(a, 0.0),
                    log_priors[a],
                    q_norm.get(a, 0.5),
                    self.c_scale,
                )

            # Sort by score, keep top half
            remaining.sort(key=lambda a: scores[a], reverse=True)
            half = max(1, len(remaining) // 2)
            remaining = remaining[:half]

        # 5. Best action is the surviving candidate
        best_action = remaining[0]

        # 6. Compute improved policy target for training
        # Negate: convert from child's perspective to root's perspective
        q_all = {a: -root.children[a].q_value for a in legal_actions
                 if root.children[a].visit_count > 0}
        improved_policy = compute_improved_policy(
            priors, q_all, legal_actions, self.c_scale
        )

        return best_action, improved_policy, root_value

    def _simulate(
        self,
        root: MCTSNode,
        root_action: int,
        root_env: GameEnv,
        opp_hidden: Optional[torch.Tensor],
    ):
        """Run one MCTS simulation starting from a root child.

        Navigate tree, expand leaf, backup value.
        """
        # Start from the root child for this action
        node = root.children[root_action]

        # If not expanded, expand it
        if not node.is_expanded:
            sim_env = root_env.clone()
            _, done = sim_env.step(root_action)
            node.state_env = sim_env

            if done:
                node.is_terminal = True
                # Value from THIS NODE's player's perspective (consistent
                # with _expand_and_evaluate which returns current player's value)
                node.backup(self._terminal_value(sim_env, node.player))
                return

            self._expand_and_evaluate(node, opp_hidden)
            return

        if node.is_terminal:
            node.backup(self._terminal_value(node.state_env, node.player))
            return

        # Navigate deeper using non-root selection
        current = node
        path_envs = []

        while current.is_expanded and not current.is_terminal:
            if not current.children:
                break
            current = current.select_child()

            if not current.is_expanded:
                # Need to simulate the action to get the state
                parent_env = current.parent.state_env
                sim_env = parent_env.clone()
                _, done = sim_env.step(current.action_from_parent)
                current.state_env = sim_env

                if done:
                    current.is_terminal = True
                    current.backup(self._terminal_value(sim_env, current.player))
                    return

                self._expand_and_evaluate(current, opp_hidden)
                return

        # Reached a terminal or fully expanded leaf - re-evaluate
        if current.is_terminal:
            current.backup(self._terminal_value(current.state_env, current.player))

    @staticmethod
    def _terminal_value(env: GameEnv, node_player: int) -> float:
        """Return terminal value from node_player's perspective.

        This must match the convention in _expand_and_evaluate where the
        network value is from the current player's perspective, and backup()
        negates going up the tree.

        For draws/timeouts, gives partial credit based on pins in goal
        to match tournament scoring where pin_goal_score dominates.
        """
        winner = env.get_winner()
        if winner == node_player:
            return 1.0   # node's player won
        elif winner is not None:
            return -1.0  # node's player lost
        # Draw/timeout: partial credit from pins in goal and distance
        color = env.colors[node_player]
        pins = env.board.count_in_goal(env.pieces[node_player], color)
        dist = env.board.sum_distances_to_goal(env.pieces[node_player], color)
        dist_score = max(0.0, 200.0 - dist) / 200.0
        # Scale to [-0.5, 0.5] - better than losing, worse than winning
        return (pins / Config.NUM_PIECES) * 0.5 + dist_score * 0.2 - 0.4

    def _expand_and_evaluate(
        self,
        node: MCTSNode,
        opp_hidden: Optional[torch.Tensor],
    ):
        """Expand a leaf node using the neural network."""
        env = node.state_env
        player = env.current_player

        state_tensor = env.get_state_tensor(player).unsqueeze(0).to(self.device)
        legal_mask = env.get_legal_mask(player).unsqueeze(0).to(self.device)

        seat_t = torch.tensor([player], device=self.device)
        out = self._net_forward(state_tensor, legal_mask, seat_t, opp_hidden)
        priors = out['policy'][0].cpu().numpy()
        value = self.network.aggregate_value(out['value'])[0].item()

        legal_actions = get_legal_actions(env.get_legal_mask(player))
        if not legal_actions:
            node.is_terminal = True
            node.backup(0.0)
            return

        action_priors = {a: priors[a] for a in legal_actions}
        node.expand(legal_actions, action_priors)

        # Backup value (from this node's player's perspective)
        # If this node's player is the root player, backup value directly
        # Otherwise negate (handled by backup's alternating negation)
        node.backup(value)
