"""Tests for MCTS (Phase C)."""
import sys
sys.path.insert(0, '/home/coder/nexus')

import math
import numpy as np
import torch

from config import Config
from core.board import HexBoard
from core.game_env import GameEnv
from core.action_space import get_legal_actions
from network.model import NexusNet
from mcts.gumbel_mcts import GumbelMCTS
from mcts.utils import (
    gumbel_top_k, normalize_q_values, sequential_halving_rounds,
    compute_improved_policy,
)


def _get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_mcts(num_sims=8):
    device = _get_device()
    model = NexusNet().to(device)
    model.eval()
    mcts = GumbelMCTS(model, device, num_simulations=num_sims)
    return mcts, device


def test_mcts_1_sim_returns_legal():
    """MCTS with 1 simulation returns a legal action."""
    mcts, device = _make_mcts(num_sims=1)
    env = GameEnv()
    env.reset()

    action, policy, value = mcts.search(env)
    legal = get_legal_actions(env.get_legal_mask())

    assert action in legal, f"Action {action} not in legal moves"


def test_mcts_32_sims_returns_legal():
    """MCTS with 32 simulations returns a legal action."""
    mcts, device = _make_mcts(num_sims=32)
    env = GameEnv()
    env.reset()

    action, policy, value = mcts.search(env)
    legal = get_legal_actions(env.get_legal_mask())

    assert action in legal, f"Action {action} not in legal moves"


def test_policy_target_sums_to_one():
    mcts, device = _make_mcts(num_sims=8)
    env = GameEnv()
    env.reset()

    _, policy, _ = mcts.search(env)
    assert abs(policy.sum() - 1.0) < 1e-4, f"Policy sum = {policy.sum()}"


def test_policy_nonzero_only_for_legal():
    mcts, device = _make_mcts(num_sims=8)
    env = GameEnv()
    env.reset()

    _, policy, _ = mcts.search(env)
    legal_mask = env.get_legal_mask().numpy()

    # All non-legal actions should have 0 policy
    illegal_mass = policy[~legal_mask.astype(bool)].sum()
    assert illegal_mass < 1e-6, f"Illegal actions have mass {illegal_mass}"


def test_gumbel_top_k_selects_m():
    log_priors = np.full(1210, -30.0)
    legal = list(range(50))
    for a in legal:
        log_priors[a] = -2.0

    m = 16
    selected = gumbel_top_k(log_priors, legal, m)
    assert len(selected) == m
    assert all(a in legal for a in selected)


def test_sequential_halving_rounds():
    assert sequential_halving_rounds(1) == 1
    assert sequential_halving_rounds(2) == 1
    assert sequential_halving_rounds(4) == 2
    assert sequential_halving_rounds(8) == 3
    assert sequential_halving_rounds(16) == 4


def test_normalize_q_values():
    q = {0: 0.2, 1: 0.8, 2: 0.5}
    actions = [0, 1, 2]
    normed = normalize_q_values(q, actions)
    assert abs(normed[0] - 0.0) < 1e-6  # min
    assert abs(normed[1] - 1.0) < 1e-6  # max
    assert abs(normed[2] - 0.5) < 1e-6  # middle


def test_improved_policy():
    priors = np.zeros(1210)
    legal = [0, 1, 2]
    priors[0] = 0.5
    priors[1] = 0.3
    priors[2] = 0.2

    q_values = {0: 0.8, 1: 0.2, 2: 0.5}
    improved = compute_improved_policy(priors, q_values, legal)

    assert abs(improved.sum() - 1.0) < 1e-5
    # Action 0 has highest prior * exp(Q), should dominate
    assert improved[0] > improved[1]
    # Only legal actions have mass
    assert improved[3:].sum() < 1e-8


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
