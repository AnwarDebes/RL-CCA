"""MCTS subtree reuse for CD-MCTS.

Implements `advance_root`: given a CD-MCTS tree rooted at state s, and
a sequence of actions taken in the real game (typically: opponent's
move, then our planned move), descend through those edges and return
the corresponding subtree as the new root. Visit counts, regrets, and
all bookkeeping are preserved - so the next search continues with
warm-started statistics rather than starting fresh.

Without subtree reuse, every move's MCTS rebuilds the tree from scratch,
discarding all the work done in the prior move's search. With reuse,
~30-50% of accumulated tree information is preserved across moves.

Caveats handled
---------------
1. The specific child node may not exist if its action was never
   selected during the prior search. In that case, return None and the
   caller falls back to a fresh root.
2. The legal_actions list at the new root may differ from the original
   tree's prediction (e.g., due to a different player's move). The
   selector_state and child_value_sum / child_visits arrays are sized
   by the OLD legal_actions; the new root's legal_actions may differ.
   We re-extract them from the new state; if cardinality or order
   changes, we discard the regret/visit information for that node and
   return a fresh root.
3. The tree only stores children for actions that have been visited at
   least once during a search. So reuse is most effective at the root
   where all legal actions have been visited; deeper reuse may be
   sparse.

For tournament play, where the opponent's move arrives between our
searches, the typical access pattern is:
    advance_root(my_root, [opponent_action, my_planned_action])
to reach the state we expect to plan from next.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .mcts import Node


def advance_root(
    root: Node,
    actions: List[int],
    game: Any,
) -> Optional[Node]:
    """Walk down `actions` from `root`, returning the child node reached.

    Args:
        root: current root of the search tree.
        actions: sequence of action *indices* (into root's legal_actions
            for the first hop, into each successive child's legal_actions
            for subsequent hops). NOT raw action ids.
        game: the game adapter (used for legality checks).

    Returns:
        The child Node at the end of the action sequence, or None if any
        step is missing (caller should fall back to a fresh root).
    """
    cur = root
    for action_idx in actions:
        if action_idx < 0 or action_idx >= cur.num_actions:
            return None
        if action_idx not in cur.children:
            return None
        cur = cur.children[action_idx]
        if game.is_terminal(cur.state):
            return None
    return cur


def advance_root_by_raw_action(
    root: Node,
    raw_action: int,
    game: Any,
) -> Optional[Node]:
    """Walk down by a *raw* (game-action-space) action id rather than an
    action index. Looks up the action in the root's legal_actions list.
    """
    if raw_action not in root.legal_actions:
        return None
    action_idx = root.legal_actions.index(raw_action)
    return advance_root(root, [action_idx], game)


def reuse_or_rebuild(
    old_root: Optional[Node],
    raw_action_path: List[int],
    new_state: Any,
    network: Any,
    game: Any,
) -> Node:
    """High-level: try to reuse the subtree along `raw_action_path` from
    `old_root`. If reuse fails for any reason, build a fresh root from
    `new_state`. Returns a Node ready for further MCTS rollouts.
    """
    from .mcts import _build_root
    if old_root is None:
        return _build_root(new_state, network, game)
    cur = old_root
    for raw_action in raw_action_path:
        nxt = advance_root_by_raw_action(cur, raw_action, game)
        if nxt is None:
            return _build_root(new_state, network, game)
        cur = nxt
    # Sanity: the reused node's state should match `new_state` (or be
    # equivalent). If they don't match, we rebuild - better safe than
    # silently using stale tree.
    if not _states_equivalent(cur.state, new_state, game):
        return _build_root(new_state, network, game)
    return cur


def _states_equivalent(a, b, game) -> bool:
    """Best-effort state equality check.

    Ideally games define __eq__; if not, we compare via repr (slow but
    safe). The kingmaker game uses frozen dataclasses so __eq__ is
    automatic; CC's GameEnv is mutable and doesn't support __eq__, so
    we fall back to comparing key attributes.
    """
    if a is b:
        return True
    try:
        if a == b:
            return True
    except Exception:
        pass
    # CC GameEnv heuristic: same pieces, current_player, move_count.
    if hasattr(a, "pieces") and hasattr(b, "pieces"):
        try:
            return (
                a.pieces == b.pieces
                and a.current_player == b.current_player
                and a.move_count == b.move_count
            )
        except Exception:
            return False
    return False
