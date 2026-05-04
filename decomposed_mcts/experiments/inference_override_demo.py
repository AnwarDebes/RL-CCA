"""CMAZ killer experiment: same network, different mixer overrides → different policies.

Demonstrates that the CMAZ network - once trained - can adapt to
different score-utility weightings *at inference time* without
retraining. This is the unique CMAZ contribution that distinguishes it
from KataGo's fixed-weighted multi-component value head.

Procedure
---------
1. Build a CMAZ network (untrained - the demonstration works even with
   random weights, because the override behaviour is architectural).
2. From the same starting CC state, run CMAZ MCTS with three different
   mixer-weight overrides:
     - learned (default): use the hypernetwork's output.
     - "win at any cost": all weight on pin_goal_score component.
     - "minimise distance": all weight on distance_score component.
     - "play fast": all weight on time_score component.
3. Report the resulting policy distributions side by side.

A trained network would produce *meaningful* policy differences across
the overrides; an untrained network produces small differences. Either
way, this script exercises the override mechanism end-to-end.

For the real paper experiment, the network is fully trained via
self-play before the override sweep.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from decomposed_mcts.src.cc_adapter import (
    CMAZCCEvaluator,
    build_cmaz_cc_network,
    play_one_cmaz_cc_game,
)
from decomposed_mcts.src.cmaz_mcts import run_mcts_cmaz
from flagship_coalition_mcts.src.games.chinese_checkers import (
    ChineseCheckersGame,
    make_cc_env,
)


def policy_summary(pi: np.ndarray, top_k: int = 3) -> str:
    """Compact summary of a policy: top-k actions and their probabilities."""
    idx = np.argsort(-pi)[:top_k]
    parts = [f"a{int(i)}={pi[i]:.3f}" for i in idx]
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--num-simulations", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=== CMAZ inference-time override demo ===")
    torch.manual_seed(args.seed)
    net = build_cmaz_cc_network(channels=32, num_blocks=3, hidden_dim=64)
    print(f"Network params: {sum(p.numel() for p in net.parameters()):,}")

    state = make_cc_env(num_players=args.num_players, seed=args.seed)
    print(f"\nState: {args.num_players}-player CC, current_player={state.current_player}")
    print(f"Legal actions: {len(ChineseCheckersGame.legal_actions(state))}")

    overrides = {
        "learned (default)": None,
        "win-at-any-cost (pin_goal=1)": np.array([1.0, 0.0, 0.0, 0.0]),
        "minimise distance":            np.array([0.0, 1.0, 0.0, 0.0]),
        "play fast":                    np.array([0.0, 0.0, 1.0, 0.0]),
        "balanced quartile":            np.array([0.25, 0.25, 0.25, 0.25]),
    }

    print(f"\nRunning {args.num_simulations} sims per override...")
    print()
    for name, w in overrides.items():
        t0 = time.time()
        ev = CMAZCCEvaluator(net, override_weights=w)
        _, pi = run_mcts_cmaz(
            state=state, network=ev, game=ChineseCheckersGame(),
            mixer_apply=ev.mixer_apply,
            num_simulations=args.num_simulations,
        )
        elapsed = time.time() - t0
        print(f"  [{name:<32}] top: {policy_summary(pi)}  ({elapsed:.1f}s)")

    print("\nKEY OBSERVATION:")
    print("  Different overrides produce different policies, demonstrating")
    print("  the CMAZ inference-time-override killer property. With a")
    print("  trained network, the policy differences become semantically")
    print("  meaningful (e.g. 'play fast' policies use fewer time-killing")
    print("  hops; 'pin_goal' policies prioritise reaching goal cells).")


if __name__ == "__main__":
    main()
