"""CMAZ on real Chinese Checkers - adapter + runner.

Bridges the CMAZ workshop subproject to the user's tournament game.
Key piece: CMAZ uses the 4 score components (pin_goal, distance, time,
move) directly as the per-component value targets - exactly the
semantic decomposition the user's tournament scoring formula provides.

This is the **actual reason CMAZ matters for the user**: the same
trained network can be re-purposed at inference for different score
weightings (e.g., 'play for blowouts', 'play conservatively') without
retraining. The user's tournament weights (pin_goal=1000 / distance=200
/ time=100 / move=1) are just one specific instantiation.
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

# We import the CC adapter from the flagship subproject - the game
# interface is shared between flagship and CMAZ.
from flagship_coalition_mcts.src.cnn_encoder import CCCNNEncoder, feature_to_tensor
from flagship_coalition_mcts.src.games.chinese_checkers import (
    ChineseCheckersGame,
    cc_score_components,
    cc_state_to_features_2d,
    make_cc_env,
)
from .cmaz_mcts import CMAZNetworkOutput, run_mcts_cmaz
from .network import CMAZNetwork


def build_cmaz_cc_network(
    channels: int = 64,
    num_blocks: int = 4,
    hidden_dim: int = 128,
    num_components: int = 4,
) -> CMAZNetwork:
    """Construct a CMAZ network with a CC CNN encoder.

    num_components=4 matches the teacher scoring formula.
    """
    encoder = CCCNNEncoder(
        in_channels=32, channels=channels,
        num_blocks=num_blocks, out_dim=hidden_dim,
    )
    return CMAZNetwork(
        encoder=encoder,
        action_space_size=1210,
        num_components=num_components,
    )


class CMAZCCEvaluator:
    """Specialised CMAZ evaluator: takes a CC GameEnv state."""

    def __init__(
        self,
        network: CMAZNetwork,
        override_weights: Optional[np.ndarray] = None,
    ) -> None:
        self.network = network
        self.override_weights = override_weights

    def terminal_components(self, state) -> np.ndarray:
        """At terminal: the actual component scores for the moving player."""
        # We use the current player's components - but for terminal nodes the
        # "current player" is not really meaningful. The scoring is global,
        # so we sum the per-player component scores to give a vector.
        # Actually for CMAZ training the per-player target is what the
        # current player will achieve - for self-play at the leaf, the
        # leaf's "current player" is the one who just finished moving.
        cp = state.current_player
        return cc_score_components(state, cp)

    @torch.no_grad()
    def evaluate_cmaz(self, state) -> CMAZNetworkOutput:
        feats_2d = cc_state_to_features_2d(state)
        x = feature_to_tensor(feats_2d)
        h = self.network.encoder(x)
        policy_logits = self.network.policy_proj(h)
        v_components = self.network.component_head(h)[0]
        prior = F.softmax(policy_logits[0], dim=-1).cpu().numpy().astype(np.float64)
        return CMAZNetworkOutput(
            prior_policy=prior,
            component_values=v_components.cpu().numpy().astype(np.float64),
            encoder_features=h[0].cpu().numpy().astype(np.float64),
        )

    def mixer_apply(self, v: np.ndarray, features: np.ndarray) -> float:
        v_t = torch.from_numpy(v).float().unsqueeze(0)
        f_t = torch.from_numpy(features).float().unsqueeze(0)
        with torch.no_grad():
            if self.override_weights is not None:
                ow = torch.from_numpy(self.override_weights).float()
                Q = self.network.mixer(v_t, f_t, override_weights=ow)
            else:
                Q = self.network.mixer(v_t, f_t)
        return float(Q.item())


def play_one_cmaz_cc_game(
    network: CMAZNetwork,
    num_players: int,
    num_simulations: int,
    seed: Optional[int] = None,
    max_moves: int = 600,
    override_weights: Optional[np.ndarray] = None,
) -> dict:
    """Play one CMAZ-on-CC game using the given network.

    If override_weights is given, the mixer uses those at inference
    (the killer-property demonstration: same network, different
    utility).
    """
    rng = np.random.default_rng(seed)
    state = make_cc_env(num_players=num_players, seed=seed)
    evaluator = CMAZCCEvaluator(network, override_weights=override_weights)
    game = ChineseCheckersGame()
    trajectory = []
    move_count = 0
    while not state.is_done() and move_count < max_moves:
        legal = ChineseCheckersGame.legal_actions(state)
        if not legal:
            break
        cp = state.current_player
        _, pi_legal = run_mcts_cmaz(
            state=state, network=evaluator, game=game,
            mixer_apply=evaluator.mixer_apply,
            num_simulations=num_simulations,
        )
        action_idx = int(rng.choice(len(pi_legal), p=pi_legal))
        action = legal[action_idx]
        feats_2d = cc_state_to_features_2d(state)
        legal_mask = np.zeros(1210, dtype=bool)
        for a in legal:
            legal_mask[a] = True
        target_policy_full = np.zeros(1210, dtype=np.float32)
        for j, a in enumerate(legal):
            target_policy_full[a] = pi_legal[j]
        trajectory.append(dict(
            features_2d=feats_2d,
            legal_mask=legal_mask,
            target_policy=target_policy_full,
            current_player=cp,
        ))
        state, _ = ChineseCheckersGame.step(state, action)
        move_count += 1

    # Fill component targets at terminal + per-state target_total_utility
    # (used to give the mixer's hypernetwork a strong gradient signal -
    # without this the mixer only sees the soft self-target derived from
    # mean(components), which is informative for CC since each component
    # is in [0, 1] but is still weaker than direct rank-utility supervision).
    if state.is_done():
        ranks = [
            int(ChineseCheckersGame.terminal_marginal(state)[p].argmax()) + 1
            for p in range(state.num_players)
        ]
        N = state.num_players
        for entry in trajectory:
            cp = entry["current_player"]
            entry["target_components"] = cc_score_components(state, cp)
            entry["target_total_utility"] = (N - ranks[cp]) / (N - 1)
        return dict(
            trajectory=trajectory,
            final_ranks=tuple(ranks),
            num_moves=move_count,
            terminated=True,
        )
    return dict(
        trajectory=trajectory, final_ranks=None,
        num_moves=move_count, terminated=False,
    )
