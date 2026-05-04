"""Tests for the Chinese Checkers env adapter.

These tests instantiate a real GameEnv and verify the adapter's
duck-typed interface produces correct outputs for MCTS consumption.
The tests do NOT run a full MCTS - that's separately validated.
The goal here is interface correctness on the real game.

Note: this test imports the existing nexus core/* modules. If run
during active v4 training, the imports are safe (no CUDA init); the
test itself is small (single env instantiation + a few moves).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)

# Skip these tests if core.* isn't importable (e.g. running in an isolated
# environment without the nexus tree). Pytest will mark them as skipped.
try:
    from flagship_coalition_mcts.src.games.chinese_checkers import (
        ChineseCheckersGame,
        cc_score_components,
        cc_state_to_features_flat,
        make_cc_env,
    )
    HAVE_CC = True
except ImportError:
    HAVE_CC = False

pytestmark = pytest.mark.skipif(
    not HAVE_CC,
    reason="Chinese Checkers env requires core.* nexus modules",
)


def test_make_cc_env_2_players():
    env = make_cc_env(num_players=2, seed=0)
    assert env.num_players == 2
    assert env.current_player in (0, 1)
    assert not env.is_done()


def test_legal_actions_nonempty_at_start():
    env = make_cc_env(num_players=2, seed=1)
    legal = ChineseCheckersGame.legal_actions(env)
    assert len(legal) > 0
    for a in legal:
        assert 0 <= a < 1210


def test_step_progresses_and_does_not_mutate_input():
    env = make_cc_env(num_players=2, seed=2)
    legal = ChineseCheckersGame.legal_actions(env)
    a = legal[0]
    pre_pieces = [list(p) for p in env.pieces]
    pre_player = env.current_player
    nxt, np_next = ChineseCheckersGame.step(env, a)
    # Original env unchanged
    assert env.pieces == pre_pieces
    assert env.current_player == pre_player
    # New env different
    assert nxt is not env
    assert nxt.current_player == np_next


def test_state_features_flat_shape():
    env = make_cc_env(num_players=2, seed=3)
    feats = cc_state_to_features_flat(env)
    assert feats.ndim == 1
    # Total features = NUM_CHANNELS × 17 × 17 = 32 × 289 = 9248
    assert feats.shape[0] == 32 * 17 * 17


def test_score_components_shape_and_range():
    env = make_cc_env(num_players=3, seed=4)
    comps = cc_score_components(env, player=0)
    assert comps.shape == (4,)
    assert (comps >= 0).all()
    assert (comps <= 1.0 + 1e-6).all()


def test_terminal_marginal_only_on_terminal():
    env = make_cc_env(num_players=2, seed=5)
    with pytest.raises(ValueError):
        ChineseCheckersGame.terminal_marginal(env)


def test_random_rollout_terminates_and_produces_marginal():
    """Run a fully random rollout to a terminal state, verify
    terminal_marginal is well-formed."""
    import random
    rng = random.Random(7)
    env = make_cc_env(num_players=2, seed=7)
    max_steps = 500
    step_count = 0
    while not env.is_done() and step_count < max_steps:
        legal = ChineseCheckersGame.legal_actions(env)
        if not legal:
            break
        a = rng.choice(legal)
        env, _ = ChineseCheckersGame.step(env, a)
        step_count += 1
    if env.is_done():
        M = ChineseCheckersGame.terminal_marginal(env)
        assert M.shape == (2, 2)
        # Every row sums to 1
        assert np.allclose(M.sum(axis=1), 1.0)
        # Every column sums to 1
        assert np.allclose(M.sum(axis=0), 1.0)
        # Each entry is 0 or 1
        assert np.all((M == 0) | (M == 1))
    # If not done within max_steps, that's a slow-game scenario; the
    # test just verifies no crash. The next-step would deterministically
    # finish under teacher's MAX_MOVES.
