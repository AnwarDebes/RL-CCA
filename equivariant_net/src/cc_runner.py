"""Wreath-equivariant CC self-play runner.

Runs WreathCCNetwork as a policy/value network in vanilla AlphaZero-style
PUCT MCTS. The wreath subproject is workshop-tier - it does NOT use the
flagship's PL/coalition pillars. The contribution here is the
*architecture*, demonstrated by:

  1. Sample-efficiency: trains faster (matched-data) than non-equivariant
     baselines.
  2. Generalisation: train on N=2,3, evaluate zero-shot on N=4,6.
  3. Bit-identical seat-permutation invariance (the headline reviewer
     check).

This file mirrors the structure of `flagship_coalition_mcts/src/cc_runner.py`
and `decomposed_mcts/src/cc_adapter.py` so the three subprojects compose
cleanly with shared infrastructure.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)

from flagship_coalition_mcts.src.baseline_mcts import (
    ScalarNetworkOutput,
    run_mcts_scalar,
)
from flagship_coalition_mcts.src.games.chinese_checkers import (
    ChineseCheckersGame,
    cc_state_to_features_2d,
    make_cc_env,
)

from .wreath_network import WreathCCNetwork, cc_seat_features


class WreathCCEvaluator:
    """Wraps WreathCCNetwork as a scalar evaluator (compatible with
    baseline_mcts.run_mcts_scalar).

    Per-player value: the network's scalar value head returns a single
    number for the *current player*; we distribute the negative across
    others (zero-sum-like) following the same convention as the
    flagship's ScalarEvaluator.
    """

    def __init__(self, network: WreathCCNetwork) -> None:
        self.network = network

    @torch.no_grad()
    def evaluate_scalar(self, state) -> ScalarNetworkOutput:
        st, sf, mask = self.network.inputs_from_state(state)
        policy_logits, scalar_v = self.network(st, sf, mask)
        prior = F.softmax(policy_logits[0], dim=-1).cpu().numpy().astype(np.float64)
        N = state.num_players
        cp = state.current_player
        v_cp = float(scalar_v[0].item())
        per_player = np.full(N, -v_cp / max(1, N - 1), dtype=np.float64)
        per_player[cp] = v_cp
        return ScalarNetworkOutput(prior_policy=prior, per_player_value=per_player)


def play_one_wreath_cc_game(
    network: WreathCCNetwork,
    num_players: int,
    num_simulations: int,
    seed: Optional[int] = None,
    max_moves: int = 600,
) -> dict:
    """Self-play one CC game using the wreath equivariant network."""
    rng = np.random.default_rng(seed)
    state = make_cc_env(num_players=num_players, seed=seed)
    evaluator = WreathCCEvaluator(network)
    game = ChineseCheckersGame()
    trajectory = []
    move_count = 0
    while not state.is_done() and move_count < max_moves:
        legal = ChineseCheckersGame.legal_actions(state)
        if not legal:
            break
        cp = state.current_player
        N = state.num_players
        _, pi_legal = run_mcts_scalar(
            state=state, network=evaluator, game=game,
            num_simulations=num_simulations,
        )
        action_idx = int(rng.choice(len(pi_legal), p=pi_legal))
        action = legal[action_idx]
        feats_2d = cc_state_to_features_2d(state)
        seat_feats = cc_seat_features(state, max_seats=network.max_players)
        legal_mask = np.zeros(1210, dtype=bool)
        for a in legal:
            legal_mask[a] = True
        target_pol = np.zeros(1210, dtype=np.float32)
        for j, a in enumerate(legal):
            target_pol[a] = pi_legal[j]
        trajectory.append(dict(
            features_2d=feats_2d,
            seat_features=seat_feats,
            legal_mask=legal_mask,
            target_policy=target_pol,
            current_player=cp,
            num_players=N,
        ))
        state, _ = ChineseCheckersGame.step(state, action)
        move_count += 1

    if state.is_done():
        M = ChineseCheckersGame.terminal_marginal(state)
        N = state.num_players
        ranks = [int(M[p].argmax()) + 1 for p in range(N)]
        for entry in trajectory:
            cp = entry["current_player"]
            entry["target_scalar_value"] = (N - ranks[cp]) / (N - 1)
        return dict(
            trajectory=trajectory,
            final_ranks=tuple(ranks),
            num_moves=move_count, terminated=True,
        )
    return dict(
        trajectory=trajectory, final_ranks=None,
        num_moves=move_count, terminated=False,
    )
