"""Top-level runner: CD-MCTS on real Chinese Checkers.

Wires together:
  * CCCNNEncoder            - CNN trunk for the (32, 17, 17) state tensor
  * CDMCTSNetwork           - encoder + 4 heads (policy, PL, coalition, scalar V)
  * ChineseCheckersGame     - duck-typed game adapter wrapping GameEnv
  * CDMCTSEvaluator         - converts state → NetworkOutput
  * run_mcts (flagship MCTS) - vector backup + EXP-IX selector + coalition penalty

This module is the *production-quality* bridge that lets us actually
run CD-MCTS self-play and inference on the real tournament game once
v4 RL training finishes Phase 1.

Usage
-----
    from flagship_coalition_mcts.src.cc_runner import (
        build_cc_evaluator, build_cc_self_play_iteration,
    )

    net, evaluator = build_cc_evaluator(num_players=4, channels=64, hidden_dim=128)
    stats = build_cc_self_play_iteration(net, evaluator, num_players=4)(...)

The CC self-play wrapper handles:
  * Per-game initial state via make_cc_env (random colors).
  * Feature extraction: 2D state tensor → encoder.
  * Score-component decomposition (4 components for CMAZ-style training,
    or PL-rank ground truth for CD-MCTS-style training).
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Defensive sys.path
_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)

from flagship_coalition_mcts.src.cnn_encoder import CCCNNEncoder, feature_to_tensor
from flagship_coalition_mcts.src.coalition_head import coalition_marginal_alignment
from flagship_coalition_mcts.src.games.chinese_checkers import (
    ChineseCheckersGame,
    cc_state_to_features_2d,
    make_cc_env,
)
from flagship_coalition_mcts.src.mcts import NetworkOutput
from flagship_coalition_mcts.src.network import CDMCTSNetwork
from flagship_coalition_mcts.src.plackett_luce import placement_marginals_exact


def build_cc_network(
    num_players_max: int = 6,
    channels: int = 64,
    num_blocks: int = 4,
    hidden_dim: int = 128,
    in_channels: int = 32,
) -> CDMCTSNetwork:
    """Construct a CDMCTSNetwork with a CC-appropriate CNN encoder."""
    encoder = CCCNNEncoder(
        in_channels=in_channels,
        channels=channels,
        num_blocks=num_blocks,
        out_dim=hidden_dim,
    )
    return CDMCTSNetwork(
        encoder=encoder,
        action_space_size=1210,  # CC: 10 pieces × 121 cells
        max_players=num_players_max,
    )


class CCMCTSEvaluator:
    """Specialised evaluator: takes a CC GameEnv and returns NetworkOutput.

    Differs from the generic CDMCTSEvaluator in that the network input is
    a 2D state tensor (32, 17, 17), not a 1D feature vector.
    """

    def __init__(self, network: CDMCTSNetwork) -> None:
        self.network = network

    @torch.no_grad()
    def evaluate(self, state) -> NetworkOutput:
        feats_2d = cc_state_to_features_2d(state)
        x = feature_to_tensor(feats_2d)
        policy_logits, theta, A, beta, _ = self.network(x)
        # Drop batch dim
        policy_logits = policy_logits[0]
        theta = theta[0]
        A = A[0]
        beta = beta[0]
        N = state.num_players
        # Restrict theta to active players for marginal computation.
        # Return the (N, N) marginal matching the active player count -
        # the MCTS expects (num_players, num_players)-shaped matrices,
        # not max_players-padded ones (game.terminal_marginal also
        # returns (N, N)).
        M = placement_marginals_exact(theta[:N], num_players=N)
        M_arr = M.cpu().numpy().astype(np.float64)
        cp = state.current_player
        coal_align = coalition_marginal_alignment(A, beta, player=cp, num_players=N)
        coal_arr = coal_align.cpu().numpy().astype(np.float64)
        prior = F.softmax(policy_logits, dim=-1).cpu().numpy().astype(np.float64)
        return NetworkOutput(
            prior_policy=prior,
            placement_marginals=M_arr,
            coalition_alignment=coal_arr,
        )


def build_cc_evaluator(
    num_players_max: int = 6,
    channels: int = 64,
    num_blocks: int = 4,
    hidden_dim: int = 128,
) -> tuple[CDMCTSNetwork, CCMCTSEvaluator]:
    """Convenience: construct net + evaluator together."""
    net = build_cc_network(
        num_players_max=num_players_max,
        channels=channels,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
    )
    return net, CCMCTSEvaluator(net)


# ----------------------------------------------------------------------
# CC self-play with flagship CD-MCTS
# ----------------------------------------------------------------------


def play_one_cc_game(
    network: CDMCTSNetwork,
    num_players: int,
    num_simulations: int,
    coalition_weight: float = 0.5,
    seed: Optional[int] = None,
    max_moves: int = 600,
) -> dict:
    """Play one CD-MCTS self-play game on real CC.

    Returns a dict with:
      * trajectory: list of {features, target_policy_full, current_player,
                              num_players, observed_ranking, observed_coalition_idx,
                              target_scalar_value}
      * final_ranks: tuple of ranks for each player.
      * num_moves: number of moves played.

    Uses the network for ALL players (single-network self-play).
    """
    from flagship_coalition_mcts.src.coalition_head import _enumerate_coalitions
    from flagship_coalition_mcts.src.mcts import run_mcts

    rng = np.random.default_rng(seed)
    state = make_cc_env(num_players=num_players, seed=seed)
    evaluator = CCMCTSEvaluator(network)
    game = ChineseCheckersGame()
    trajectory = []
    move_count = 0

    while not state.is_done() and move_count < max_moves:
        legal = ChineseCheckersGame.legal_actions(state)
        if not legal:
            break
        cp = state.current_player
        N = state.num_players
        _, pi_legal = run_mcts(
            state=state, network=evaluator, game=game,
            num_simulations=num_simulations,
            coalition_weight=coalition_weight,
            seed=int(rng.integers(0, 2**31)),
        )
        action_idx = int(rng.choice(len(pi_legal), p=pi_legal))
        action = legal[action_idx]
        # Build full-action-space target_policy
        target_policy_full = np.zeros(1210, dtype=np.float32)
        for j, a in enumerate(legal):
            target_policy_full[a] = pi_legal[j]
        # Save trajectory entry (features get materialised lazily - keep
        # state ref and extract later if memory is a concern)
        feats_2d = cc_state_to_features_2d(state)
        legal_mask = np.zeros(1210, dtype=bool)
        for a in legal:
            legal_mask[a] = True
        entry = dict(
            features_2d=feats_2d,
            legal_mask=legal_mask,
            target_policy=target_policy_full,
            current_player=cp,
            num_players=N,
        )
        trajectory.append(entry)
        # Step
        state, _ = ChineseCheckersGame.step(state, action)
        move_count += 1

    # Fill targets if game terminated normally
    if state.is_done():
        M = ChineseCheckersGame.terminal_marginal(state)
        N = state.num_players
        ranks = [int(M[p].argmax()) + 1 for p in range(N)]
        # Build observed_ranking: rank_to_player[k] = player in position k+1
        rank_to_player = [-1] * N
        for p, r in enumerate(ranks):
            rank_to_player[r - 1] = p
        N_max = network.max_players
        obs_ranking = np.full(N_max, -1, dtype=np.int64)
        obs_ranking[:N] = rank_to_player
        for entry in trajectory:
            cp = entry["current_player"]
            entry["observed_ranking"] = obs_ranking.copy()
            cp_rank = ranks[cp]
            ahead = tuple(sorted([
                q for q in range(N) if q != cp and ranks[q] < cp_rank
            ]))
            opp = [q for q in range(N) if q != cp]
            all_coals = _enumerate_coalitions(opp)
            try:
                entry["observed_coalition_index"] = all_coals.index(ahead)
            except ValueError:
                entry["observed_coalition_index"] = -1
            entry["target_scalar_value"] = (N - cp_rank) / (N - 1)
        return dict(
            trajectory=trajectory,
            final_ranks=tuple(ranks),
            num_moves=move_count,
            terminated=True,
        )
    return dict(
        trajectory=trajectory,
        final_ranks=None,
        num_moves=move_count,
        terminated=False,
    )
