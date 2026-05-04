"""Tests for MCTS subtree reuse."""

from __future__ import annotations

import numpy as np
import pytest

from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame, KingmakerState, NUM_ACTIONS,
)
from flagship_coalition_mcts.src.mcts import NetworkOutput, run_mcts
from flagship_coalition_mcts.src.subtree_reuse import (
    advance_root, advance_root_by_raw_action, reuse_or_rebuild,
)


class StubNet:
    def evaluate(self, state):
        K = 3  # roughly matches kingmaker initial
        return NetworkOutput(
            prior_policy=np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS),
            placement_marginals=np.full((3, 3), 1.0 / 3),
            coalition_alignment=np.zeros(3),
        )


def test_advance_root_returns_existing_child():
    """After running MCTS, advance_root with a visited action returns a Node."""
    state = KingmakerState.initial()
    root, pi = run_mcts(state, StubNet(), KingmakerGame(), num_simulations=20, seed=0)
    visited_action = int(np.argmax(root.child_visits))
    child = advance_root(root, [visited_action], KingmakerGame())
    assert child is not None
    assert child is not root


def test_advance_root_returns_none_for_unvisited_action():
    state = KingmakerState.initial()
    root, _ = run_mcts(state, StubNet(), KingmakerGame(), num_simulations=2, seed=0)
    # Find an unvisited action
    unvisited = None
    for i in range(root.num_actions):
        if i not in root.children:
            unvisited = i
            break
    if unvisited is not None:
        out = advance_root(root, [unvisited], KingmakerGame())
        assert out is None


def test_advance_by_raw_action_finds_legal_match():
    state = KingmakerState.initial()
    root, _ = run_mcts(state, StubNet(), KingmakerGame(), num_simulations=20, seed=0)
    # Find a raw action that was visited
    for ai in range(root.num_actions):
        if ai in root.children:
            raw = root.legal_actions[ai]
            child = advance_root_by_raw_action(root, raw, KingmakerGame())
            assert child is not None
            return
    pytest.fail("no children expanded; test setup needs more simulations")


def test_advance_by_raw_action_returns_none_for_illegal_action():
    state = KingmakerState.initial()
    root, _ = run_mcts(state, StubNet(), KingmakerGame(), num_simulations=20, seed=0)
    # Pick an action ID guaranteed not in legal_actions
    illegal = max(root.legal_actions) + 100
    out = advance_root_by_raw_action(root, illegal, KingmakerGame())
    assert out is None


def test_reuse_or_rebuild_with_none_root_rebuilds():
    state = KingmakerState.initial()
    root = reuse_or_rebuild(None, [], state, StubNet(), KingmakerGame())
    assert root is not None
    assert root.state == state


def test_reuse_or_rebuild_with_matching_path_reuses():
    """If the action path leads to a state matching new_state, the subtree
    is reused (the returned root is from the original tree)."""
    state = KingmakerState.initial()
    root, _ = run_mcts(state, StubNet(), KingmakerGame(), num_simulations=20, seed=0)
    visited_idx = int(np.argmax(root.child_visits))
    visited_action = root.legal_actions[visited_idx]
    # Apply the action to get the new state
    new_state, _ = KingmakerGame.step(state, visited_action)
    reused = reuse_or_rebuild(root, [visited_action], new_state, StubNet(), KingmakerGame())
    # The reused root's state should equal new_state
    assert reused.state == new_state
    # The reused root should be the original child node (not a fresh one).
    assert reused is root.children[visited_idx]
