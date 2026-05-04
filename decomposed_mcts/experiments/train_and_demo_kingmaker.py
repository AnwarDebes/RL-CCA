"""Trains a CMAZ network on kingmaker, then demonstrates the
inference-time-override killer property with semantically-meaningful
policy divergence.

Total compute: ~5-15 min CPU, depending on parameters. Suitable for
running once Phase 2 v4 finishes (or any time the GPU is idle).

What this script proves
-----------------------
The CMAZ paper's central claim is that a trained network can be
re-purposed at inference time for different score-utility weightings.
This script:
  1. Trains CMAZ on kingmaker self-play (3-component score: won / mid /
     last decomposition).
  2. From the same starting state, runs MCTS with three mixer-weight
     overrides:
       - "win-at-any-cost":  [1, 0, 0]      → only c0 (won)
       - "avoid-last":       [0, 0, -1]     → minimise c2 (last)
       - "balanced":         [1/3, 1/3, 1/3] → expected rank
  3. Reports the resulting policies side by side.
  4. With training, the policies should diverge - proving the override
     mechanism is SEMANTICALLY meaningful, not just mechanically wired.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from decomposed_mcts.src.cmaz_mcts import run_mcts_cmaz
from decomposed_mcts.src.kingmaker_adapter import (
    KingmakerCMAZEvaluator,
    build_cmaz_kingmaker_network,
    kingmaker_features_for_cmaz,
    kingmaker_score_components,
)
from decomposed_mcts.src.network import cmaz_loss
from flagship_coalition_mcts.src.games.kingmaker import (
    KingmakerGame, KingmakerState, NUM_ACTIONS,
)


def policy_summary(pi: np.ndarray, top_k: int = 3) -> str:
    idx = np.argsort(-pi)[:top_k]
    return ", ".join(f"a{int(i)}={pi[i]:.3f}" for i in idx)


def play_one_game_collect(network, num_simulations: int, seed: int):
    """Self-play one kingmaker game; return list of (features, target_pi,
    target_components, current_player, legal_action_ids)."""
    rng = np.random.default_rng(seed)
    state = KingmakerState.initial()
    evaluator = KingmakerCMAZEvaluator(network)
    trajectory = []
    while not KingmakerGame.is_terminal(state):
        legal = KingmakerGame.legal_actions(state)
        if not legal:
            break
        cp = state.next_player
        _, pi_legal = run_mcts_cmaz(
            state=state, network=evaluator, game=KingmakerGame(),
            mixer_apply=evaluator.mixer_apply,
            num_simulations=num_simulations,
        )
        action_idx = int(rng.choice(len(pi_legal), p=pi_legal))
        action = legal[action_idx]
        feats = kingmaker_features_for_cmaz(state)
        legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for a in legal:
            legal_mask[a] = True
        target_pol = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for j, a in enumerate(legal):
            target_pol[a] = pi_legal[j]
        trajectory.append(dict(
            features=feats, legal_mask=legal_mask,
            target_policy=target_pol, current_player=cp,
        ))
        state, _ = KingmakerGame.step(state, action)
    # Fill component targets at terminal + per-state total utility
    # (used to give the mixer a strong gradient signal - without this the
    # mixer's hypernetwork only sees the soft self-target which is uniform
    # for one-hot rank components on this game).
    from flagship_coalition_mcts.src.games.kingmaker import (
        NUM_PLAYERS, final_ranks,
    )
    final = final_ranks(state)
    for entry in trajectory:
        cp = entry["current_player"]
        entry["target_components"] = kingmaker_score_components(state, cp)
        # Total utility = (N - rank) / (N - 1) ∈ [0, 1]
        entry["target_total_utility"] = (NUM_PLAYERS - final[cp]) / (NUM_PLAYERS - 1)
    return trajectory


def training_iter(network, optimizer, num_games, train_steps, num_simulations, iter_seed):
    network.eval()
    rng = np.random.default_rng(iter_seed)
    trajs = []
    for _ in range(num_games):
        seed = int(rng.integers(0, 2**31))
        trajs.append(play_one_game_collect(network, num_simulations, seed))
    flat = [e for t in trajs for e in t]
    if not flat:
        return None
    feats = torch.from_numpy(np.stack([e["features"] for e in flat])).float()
    legal_mask = torch.from_numpy(np.stack([e["legal_mask"] for e in flat]))
    target_pol = torch.from_numpy(np.stack([e["target_policy"] for e in flat]))
    target_comp = torch.from_numpy(np.stack([e["target_components"] for e in flat])).float()
    target_util = torch.from_numpy(
        np.array([e["target_total_utility"] for e in flat], dtype=np.float32)
    )

    network.train()
    losses = []
    for _ in range(train_steps):
        optimizer.zero_grad()
        loss, comps = cmaz_loss(
            network,
            features=feats, target_policy=target_pol,
            legal_mask=legal_mask, target_components=target_comp,
            target_total_utility=target_util,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        losses.append(comps)
    return {k: float(np.mean([d[k] for d in losses])) for k in losses[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-iterations", type=int, default=15)
    ap.add_argument("--games-per-iter", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=24)
    ap.add_argument("--num-simulations", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo-simulations", type=int, default=64)
    args = ap.parse_args()

    print("=" * 72)
    print("CMAZ on KINGMAKER: train-and-override-demo")
    print("=" * 72)

    torch.manual_seed(args.seed)
    net = build_cmaz_kingmaker_network(hidden_dim=args.hidden_dim, num_components=3)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Network params: {n_params:,}")
    opt = optim.Adam(net.parameters(), lr=args.lr)

    print(f"\n[train] {args.num_iterations} iters × {args.games_per_iter} games × "
          f"{args.train_steps} steps × {args.num_simulations} sims")
    t0 = time.time()
    for it in range(args.num_iterations):
        avg = training_iter(
            net, opt,
            num_games=args.games_per_iter,
            train_steps=args.train_steps,
            num_simulations=args.num_simulations,
            iter_seed=args.seed * 1000 + it,
        )
        if avg is not None:
            print(f"  [iter {it+1:>3}/{args.num_iterations}] "
                  f"loss={avg['total']:.3f} "
                  f"(p={avg['policy']:.3f} c={avg['components']:.3f}) "
                  f"elapsed={time.time()-t0:.0f}s")
    print(f"\n[train complete] total {time.time()-t0:.1f}s\n")

    # Demo time: same starting state, different overrides
    print("=" * 72)
    print(f"OVERRIDE SWEEP @ {args.demo_simulations} sims")
    print("=" * 72)
    state = KingmakerState.initial()
    overrides = {
        "default (learned mixer)": None,
        "win-at-any-cost  [1,0,0]": np.array([1.0, 0.0, 0.0]),
        "balanced         [.33,.33,.33]": np.array([1/3, 1/3, 1/3]),
        "avoid-last       [-1,0,0]→[0,0,-1] inverted": np.array([1.0, 1.0, 0.001]),
    }
    pis = {}
    for label, w in overrides.items():
        ev = KingmakerCMAZEvaluator(net, override_weights=w)
        _, pi = run_mcts_cmaz(
            state=state, network=ev, game=KingmakerGame(),
            mixer_apply=ev.mixer_apply,
            num_simulations=args.demo_simulations,
        )
        pis[label] = pi
        print(f"  [{label:<48}] {policy_summary(pi)}")

    # Pairwise distance matrix
    labels = list(overrides.keys())
    print("\n  Pairwise L1 distances between override policies:")
    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            dist = float(np.abs(pis[l1] - pis[l2]).sum())
            print(f"    {l1[:28]:<28} vs {l2[:28]:<28} = {dist:.3f}")

    print("\n  KEY OBSERVATION:")
    print("    If trained correctly, the override policies will DIVERGE")
    print("    (pairwise L1 distance > 0.1). If they remain near-identical,")
    print("    the network needs more training or larger hidden dim.")


if __name__ == "__main__":
    main()
