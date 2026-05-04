"""MCTS tree visualisation / debugging utility.

After running run_mcts(), inspect the resulting tree to verify the
search behaved as expected. Used during paper-figure preparation and
when investigating unexpected policies.

Public functions:
  * print_tree(root, max_depth=2, max_children=4): pretty-print a
    truncated view of the tree.
  * tree_stats(root): dict of summary statistics (depth, total nodes,
    expanded children at root, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def tree_stats(root) -> Dict[str, Any]:
    """Aggregate statistics across the entire MCTS tree.

    Args:
        root: a Node from `flagship_coalition_mcts.src.mcts`.

    Returns:
        dict with keys: total_nodes, max_depth, root_visits,
        expanded_children_at_root, top_action_visit_share,
        coalition_alignment_max, prior_entropy_at_root.
    """
    import math
    import numpy as np

    total_nodes = 0
    max_depth = 0

    def _walk(node, depth: int) -> None:
        nonlocal total_nodes, max_depth
        total_nodes += 1
        if depth > max_depth:
            max_depth = depth
        for child in node.children.values():
            _walk(child, depth + 1)

    _walk(root, 0)

    visits = root.selector_state.visits if hasattr(root, "selector_state") else None
    total_visits = int(visits.sum()) if visits is not None else None
    top_share = float(visits.max() / max(1, visits.sum())) if visits is not None else None
    expanded = sum(1 for v in (root.child_visits if hasattr(root, "child_visits") else [])
                   if v > 0) if hasattr(root, "child_visits") else None

    # Coalition alignment max
    coal_max = None
    if hasattr(root, "coalition_alignment"):
        try:
            coal_max = float(np.max(root.coalition_alignment))
        except Exception:
            pass

    # Prior entropy
    prior_entropy = None
    if hasattr(root, "prior_policy"):
        try:
            p = np.asarray(root.prior_policy)
            p = p[p > 0]
            prior_entropy = float(-(p * np.log(p)).sum())
        except Exception:
            pass

    return dict(
        total_nodes=total_nodes,
        max_depth=max_depth,
        root_visits=total_visits,
        expanded_children_at_root=expanded,
        top_action_visit_share=top_share,
        coalition_alignment_max=coal_max,
        prior_entropy_at_root=prior_entropy,
    )


def print_tree(
    root,
    max_depth: int = 2,
    max_children: int = 4,
    indent: str = "",
    name: str = "ROOT",
) -> None:
    """Pretty-print a truncated MCTS tree.

    Shows up to `max_children` highest-visit children at each level,
    drilling down to `max_depth` levels. Uses standard ascii tree
    drawing convention with `├──` and `└──` branches.
    """

    def _node_label(node, name: str) -> str:
        visits = (node.selector_state.visits.sum()
                  if hasattr(node, "selector_state") else "?")
        cp = getattr(node, "current_player", "?")
        n_children = len(node.children) if hasattr(node, "children") else 0
        return f"{name}: cp={cp}, visits={visits}, children={n_children}"

    def _walk(node, prefix: str, name: str, depth: int, is_last: bool):
        # Branch and extension for this node, given is_last among siblings.
        if depth == 0:
            # Root has no branch line.
            print(f"{prefix}{_node_label(node, name)}")
            child_prefix = prefix
        else:
            branch = "└── " if is_last else "├── "
            print(f"{prefix}{branch}{_node_label(node, name)}")
            child_prefix = prefix + ("    " if is_last else "│   ")
        if depth >= max_depth or not hasattr(node, "child_visits"):
            return
        # Sort children by visit count.
        cv = node.child_visits
        ordered = sorted(
            [(int(cv[i]), i) for i in range(len(cv))],
            key=lambda x: -x[0],
        )[:max_children]
        for k, (v, i) in enumerate(ordered):
            cl_is_last = (k == len(ordered) - 1)
            label = f"a{node.legal_actions[i] if hasattr(node, 'legal_actions') else i} (visits={v})"
            if i in node.children:
                _walk(node.children[i], child_prefix, label, depth + 1, cl_is_last)
            else:
                cl_branch = "└── " if cl_is_last else "├── "
                print(f"{child_prefix}{cl_branch}{label} [unexpanded]")

    _walk(root, indent, name, 0, True)


def format_stats(stats: Dict[str, Any]) -> str:
    """Format tree_stats output as a multi-line string."""
    lines = ["MCTS tree statistics:"]
    for k, v in stats.items():
        if isinstance(v, float):
            lines.append(f"  {k:<32} {v:.3f}")
        else:
            lines.append(f"  {k:<32} {v}")
    return "\n".join(lines)
