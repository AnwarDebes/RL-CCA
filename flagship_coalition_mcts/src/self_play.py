"""Self-play data generation and training loop for CD-MCTS.

This is the AlphaZero loop adapted to N-player non-zero-sum games:

  for iteration in range(num_iterations):
      data = []
      for game in range(games_per_iter):
          trajectory = play_one_self_play_game(network)
          # observed targets are filled in once the game terminates:
          #   target_policy: from MCTS visit distribution at each node
          #   observed_ranking: the actual final placement order
          #   observed_coalition: subset of opponents who finished ahead
          #     of the current player (per state)
          #   target_scalar_value: (N - rank) / (N - 1) for current player
          data.extend(trajectory)
      train_network(network, data, num_steps)

There is no opponent pool here - we use single-network self-play because
all 3 players are *the same agent* with shared weights, just plugged in
to different seats. This matches Petosa & Balch 2019 and is the natural
N-player generalisation of AlphaZero.

Per-state coalition target
--------------------------
For each state s in a trajectory, with current player p:
  observed_coalition(p, s) = sorted tuple of opponents q such that
    final_rank(q) < final_rank(p)
  i.e., the opponents who finished AHEAD of p are treated as the
  "coalition that suppressed p". This is one of several reasonable
  operationalisations; we pre-register it as the default and ablate
  alternatives in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
import torch
import torch.optim as optim

from .coalition_head import _enumerate_coalitions
from .mcts import run_mcts
from .network import CDMCTSEvaluator, CDMCTSNetwork, cdmcts_loss


@dataclass
class TrajectoryEntry:
    """One (state, action) record collected during self-play.

    Targets are filled in after the game terminates.
    """

    features: np.ndarray
    legal_mask: np.ndarray            # bool, full action space
    target_policy_legal: np.ndarray   # softmax over legal actions only
    legal_action_ids: list            # which action indices are legal at this state
    current_player: int
    num_players: int
    # filled at end of game
    observed_ranking: Optional[np.ndarray] = None
    observed_coalition_index: Optional[int] = None
    target_scalar_value: Optional[float] = None


def play_one_game(
    initial_state_fn: Callable[[], Any],
    game: Any,
    evaluator: CDMCTSEvaluator,
    state_to_features: Callable[[Any], np.ndarray],
    num_simulations: int,
    coalition_weight: float = 0.5,
    temperature: float = 1.0,
    rng_seed: Optional[int] = None,
    action_space_size: int = 4,
    max_players: int = 3,
) -> List[TrajectoryEntry]:
    """Play one self-play game, returning the list of TrajectoryEntry."""
    rng = np.random.default_rng(rng_seed)
    state = initial_state_fn()
    trajectory: List[TrajectoryEntry] = []

    while not game.is_terminal(state):
        legal = game.legal_actions(state)
        cp = game.current_player(state)
        N = game.num_players(state)
        # Run MCTS
        root, pi_legal = run_mcts(
            state=state, network=evaluator, game=game,
            num_simulations=num_simulations,
            coalition_weight=coalition_weight,
            seed=int(rng.integers(0, 2**31)),
        )
        # Apply temperature to root policy
        if temperature != 1.0:
            powered = pi_legal ** (1.0 / max(temperature, 1e-6))
            pi_legal = powered / powered.sum()
        # Sample an action
        action_idx = int(rng.choice(len(pi_legal), p=pi_legal))
        action = legal[action_idx]
        # Build feature + legal mask
        feats = state_to_features(state)
        legal_mask = np.zeros(action_space_size, dtype=bool)
        for a in legal:
            legal_mask[a] = True
        # Save trajectory entry
        entry = TrajectoryEntry(
            features=feats.copy(),
            legal_mask=legal_mask.copy(),
            target_policy_legal=pi_legal.copy(),
            legal_action_ids=list(legal),
            current_player=cp,
            num_players=N,
        )
        trajectory.append(entry)
        # Advance
        state, _ = game.step(state, action)

    # Fill targets from terminal state
    final_ranks = _final_ranks(state, game)
    N_max = max_players
    # Build observed ranking: ranking[k] = player who finished in position k+1
    rank_to_player = [-1] * N_max
    for p, r in enumerate(final_ranks):
        if r >= 1:
            rank_to_player[r - 1] = p
    obs_ranking_arr = np.array(rank_to_player, dtype=np.int64)

    for entry in trajectory:
        entry.observed_ranking = obs_ranking_arr.copy()
        # Coalition: opponents who finished ahead of current_player.
        cp = entry.current_player
        N = entry.num_players
        cp_rank = final_ranks[cp]
        ahead = tuple(sorted([q for q in range(N) if q != cp and final_ranks[q] < cp_rank]))
        opp = [q for q in range(N) if q != cp]
        all_coals = _enumerate_coalitions(opp)
        try:
            entry.observed_coalition_index = all_coals.index(ahead)
        except ValueError:
            entry.observed_coalition_index = -1
        # Scalar value target
        entry.target_scalar_value = (N - cp_rank) / (N - 1)

    return trajectory


def _final_ranks(state: Any, game: Any) -> tuple:
    """Use the game's terminal_marginal to derive ranks deterministically.

    For a deterministic terminal state, marginal is one-hot per row.
    rank_p = position k+1 where M[p, k] == 1. For non-deterministic
    terminals (none in our games yet), pick argmax k.
    """
    M = game.terminal_marginal(state)
    N = M.shape[0]
    ranks = []
    for p in range(N):
        ranks.append(int(M[p].argmax()) + 1)
    return tuple(ranks)


def trajectory_to_batch(
    trajectories: List[List[TrajectoryEntry]],
    action_space_size: int,
    max_players: int,
) -> dict:
    """Flatten trajectories into a tensor batch ready for cdmcts_loss."""
    flat = [e for traj in trajectories for e in traj]
    B = len(flat)
    if B == 0:
        return None

    feature_dim = flat[0].features.shape[0]
    feats = np.stack([e.features for e in flat])
    legal_mask = np.stack([e.legal_mask for e in flat])
    target_pol = np.zeros((B, action_space_size), dtype=np.float32)
    for i, e in enumerate(flat):
        for j, a in enumerate(e.legal_action_ids):
            target_pol[i, a] = e.target_policy_legal[j]
    obs_rank = np.stack([e.observed_ranking for e in flat])
    n_players = np.array([e.num_players for e in flat], dtype=np.int64)
    cp = np.array([e.current_player for e in flat], dtype=np.int64)
    obs_coal = np.array([e.observed_coalition_index for e in flat], dtype=np.int64)
    target_v = np.array([e.target_scalar_value for e in flat], dtype=np.float32)

    return dict(
        features=torch.from_numpy(feats).float(),
        legal_mask=torch.from_numpy(legal_mask),
        target_policy=torch.from_numpy(target_pol),
        observed_ranking=torch.from_numpy(obs_rank),
        num_players=torch.from_numpy(n_players),
        current_player=torch.from_numpy(cp),
        observed_coalition_index=torch.from_numpy(obs_coal),
        target_scalar_value=torch.from_numpy(target_v),
    )


def train_step(
    network: CDMCTSNetwork,
    optimizer: optim.Optimizer,
    batch: dict,
    weights: Optional[dict] = None,
) -> dict:
    """One training step. Returns the loss components."""
    optimizer.zero_grad()
    loss, comps = cdmcts_loss(
        network,
        features=batch["features"],
        target_policy=batch["target_policy"],
        legal_mask=batch["legal_mask"],
        observed_ranking=batch["observed_ranking"],
        num_players_per=batch["num_players"],
        observed_coalition_index=batch["observed_coalition_index"],
        current_player_per=batch["current_player"],
        target_scalar_value=batch["target_scalar_value"],
        weights=weights,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()
    return comps


def self_play_iteration(
    network: CDMCTSNetwork,
    optimizer: optim.Optimizer,
    initial_state_fn: Callable[[], Any],
    game: Any,
    state_to_features: Callable[[Any], np.ndarray],
    games_per_iter: int = 32,
    train_steps: int = 32,
    num_simulations: int = 32,
    coalition_weight: float = 0.5,
    temperature: float = 1.0,
    action_space_size: int = 4,
    max_players: int = 3,
    rng_seed: int = 0,
) -> dict:
    """One outer iteration: collect games, then train.

    Returns a dict of summary statistics (avg losses, win-rate-vs-self,
    etc.).
    """
    network.eval()
    evaluator = CDMCTSEvaluator(
        network=network,
        state_to_features=state_to_features,
        action_space_size=action_space_size,
        current_player_fn=game.current_player,
        num_players_fn=game.num_players,
    )
    rng = np.random.default_rng(rng_seed)
    trajs = []
    for g in range(games_per_iter):
        traj = play_one_game(
            initial_state_fn=initial_state_fn,
            game=game,
            evaluator=evaluator,
            state_to_features=state_to_features,
            num_simulations=num_simulations,
            coalition_weight=coalition_weight,
            temperature=temperature,
            rng_seed=int(rng.integers(0, 2**31)),
            action_space_size=action_space_size,
            max_players=max_players,
        )
        trajs.append(traj)

    batch = trajectory_to_batch(trajs, action_space_size, max_players)
    if batch is None:
        return dict(num_games=games_per_iter, num_states=0)
    network.train()
    losses = []
    for _ in range(train_steps):
        comps = train_step(network, optimizer, batch)
        losses.append(comps)

    # Aggregate
    avg = {k: float(np.mean([d[k] for d in losses])) for k in losses[0]}
    return dict(
        num_games=games_per_iter,
        num_states=batch["features"].shape[0],
        **{f"avg_{k}": v for k, v in avg.items()},
    )
