"""CMAZ self-play loop.

Mirrors the flagship's self_play.py but with score-component-decomposed
targets and the monotonic-mixer-aggregated value used in MCTS.

The 'observed components' for a trajectory entry are the per-component
final score from the game's scoring formula. For the kingmaker testbed,
we use a 2-component synthetic scoring (k0 = won, k1 = penalty) for
illustration; for the real Chinese Checkers tournament these are the
4 score components: pin_goal_score, distance_score, time_score, move_score.

The CMAZ MCTS uses the network's mixer as the aggregator at every PUCT
step. The killer experiment swaps the mixer for an override at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
import torch
import torch.optim as optim

from .cmaz_mcts import run_mcts_cmaz
from .network import CMAZEvaluator, CMAZNetwork, cmaz_loss


@dataclass
class CMAZTrajectoryEntry:
    features: np.ndarray
    legal_mask: np.ndarray
    target_policy_legal: np.ndarray
    legal_action_ids: list
    current_player: int
    target_components: Optional[np.ndarray] = None  # (K,) filled at terminal


def play_one_game_cmaz(
    initial_state_fn: Callable[[], Any],
    game: Any,
    evaluator: CMAZEvaluator,
    state_to_features: Callable[[Any], np.ndarray],
    score_components_fn: Callable[[Any, int], np.ndarray],
    num_simulations: int,
    rng_seed: Optional[int] = None,
    action_space_size: int = 4,
    temperature: float = 1.0,
) -> List[CMAZTrajectoryEntry]:
    rng = np.random.default_rng(rng_seed)
    state = initial_state_fn()
    trajectory: List[CMAZTrajectoryEntry] = []

    while not game.is_terminal(state):
        legal = game.legal_actions(state)
        cp = game.current_player(state)
        # MCTS using the network's mixer as the aggregator.
        _, pi_legal = run_mcts_cmaz(
            state=state, network=evaluator, game=game,
            mixer_apply=evaluator.mixer_apply,
            num_simulations=num_simulations,
        )
        if temperature != 1.0:
            powered = pi_legal ** (1.0 / max(temperature, 1e-6))
            pi_legal = powered / powered.sum()
        action_idx = int(rng.choice(len(pi_legal), p=pi_legal))
        action = legal[action_idx]
        feats = state_to_features(state)
        legal_mask = np.zeros(action_space_size, dtype=bool)
        for a in legal:
            legal_mask[a] = True
        trajectory.append(CMAZTrajectoryEntry(
            features=feats.copy(),
            legal_mask=legal_mask.copy(),
            target_policy_legal=pi_legal.copy(),
            legal_action_ids=list(legal),
            current_player=cp,
        ))
        state, _ = game.step(state, action)

    # Fill targets: per-component final score from the current player's perspective.
    for entry in trajectory:
        entry.target_components = score_components_fn(state, entry.current_player)
    return trajectory


def trajectory_to_batch_cmaz(
    trajectories: List[List[CMAZTrajectoryEntry]],
    action_space_size: int,
    num_components: int,
) -> Optional[dict]:
    flat = [e for traj in trajectories for e in traj]
    if not flat:
        return None
    feats = np.stack([e.features for e in flat])
    legal_mask = np.stack([e.legal_mask for e in flat])
    target_pol = np.zeros((len(flat), action_space_size), dtype=np.float32)
    for i, e in enumerate(flat):
        for j, a in enumerate(e.legal_action_ids):
            target_pol[i, a] = e.target_policy_legal[j]
    target_comp = np.stack([e.target_components for e in flat]).astype(np.float32)
    return dict(
        features=torch.from_numpy(feats).float(),
        legal_mask=torch.from_numpy(legal_mask),
        target_policy=torch.from_numpy(target_pol),
        target_components=torch.from_numpy(target_comp),
    )


def cmaz_train_step(
    network: CMAZNetwork,
    optimizer: optim.Optimizer,
    batch: dict,
    weights: Optional[dict] = None,
) -> dict:
    optimizer.zero_grad()
    loss, comps = cmaz_loss(
        network,
        features=batch["features"],
        target_policy=batch["target_policy"],
        legal_mask=batch["legal_mask"],
        target_components=batch["target_components"],
        weights=weights,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()
    return comps
