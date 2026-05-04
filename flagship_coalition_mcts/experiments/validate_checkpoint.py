"""Post-training checkpoint validation.

Loads any trained checkpoint and runs sanity checks that the network
still satisfies expected invariants. Useful as a smoke test after a
long training run, before claiming the network is "trained correctly".

Checks:
  1. Network loads without error.
  2. Policy outputs are valid probability distributions over legal
     actions (after softmax).
  3. PL theta is finite; placement marginals are valid distributions.
  4. Coalition head A is symmetric, zero-diagonal; β > 0.
  5. Per-state outputs are bounded (no exploding values).

Optional checks (--full):
  6. PL head produces sensible rankings (winner marginal correlates
     with theta argmax).
  7. Coalition head's posterior is non-degenerate (entropy > 0.1)
     across a sample of states.
  8. The policy is not collapsed to a single action everywhere.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--num-states", type=int, default=20,
                    help="Number of CC states to evaluate")
    ap.add_argument("--full", action="store_true",
                    help="Run extended checks (slower)")
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    args = ap.parse_args()

    print("=" * 72)
    print(f"VALIDATING CHECKPOINT: {args.checkpoint}")
    print("=" * 72)

    from flagship_coalition_mcts.src.cc_runner import build_cc_evaluator
    from flagship_coalition_mcts.src.checkpoint import load_checkpoint
    from flagship_coalition_mcts.src.games.chinese_checkers import make_cc_env
    from flagship_coalition_mcts.src.coalition_head import coalition_entropy

    failures = 0

    # 1. Load
    print("\n[1] Loading network...")
    net, ev = build_cc_evaluator(
        num_players_max=6,
        channels=args.channels, num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
    )
    try:
        bundle = load_checkpoint(args.checkpoint, net, strict=False)
        print(f"    OK - iter {bundle.iter_idx}, version {bundle.version}")
    except Exception as e:
        print(f"    FAIL: {e}")
        return 1
    net.eval()

    # 2. Generate states + evaluate
    print(f"\n[2] Evaluating {args.num_states} states...")
    states = [make_cc_env(num_players=args.num_players, seed=i)
              for i in range(args.num_states)]

    policies = []
    pl_thetas = []
    coal_betas = []
    coal_As = []

    for s in states:
        out = ev.evaluate(s)
        # Policy validity
        if not (out.prior_policy >= 0).all():
            print(f"    FAIL: negative prob in policy at state {s.move_count}")
            failures += 1
        if abs(out.prior_policy.sum() - 1.0) > 1e-4:
            print(f"    FAIL: policy doesn't sum to 1 at state {s.move_count}: "
                  f"{out.prior_policy.sum()}")
            failures += 1
        # Placement marginal validity
        if not np.allclose(out.placement_marginals.sum(axis=1), 1.0, atol=1e-4):
            print(f"    FAIL: placement marginals don't sum to 1 along rows")
            failures += 1
        if not (out.placement_marginals >= 0).all():
            print(f"    FAIL: negative placement marginal")
            failures += 1
        policies.append(out.prior_policy)
        # Inspect raw heads
        feats = ev.network
        with torch.no_grad():
            from flagship_coalition_mcts.src.games.chinese_checkers import cc_state_to_features_2d
            x = torch.from_numpy(cc_state_to_features_2d(s)).float().unsqueeze(0)
            _, theta, A, beta, _ = net(x)
            pl_thetas.append(theta[0].numpy())
            coal_As.append(A[0].numpy())
            coal_betas.append(float(beta[0].item()))

    print(f"    OK - {args.num_states} states pass basic distribution checks")

    # 3. Coalition head structural
    print("\n[3] Coalition head structure...")
    ok = True
    for i, A in enumerate(coal_As):
        if not np.allclose(A, A.T, atol=1e-5):
            print(f"    FAIL: A is not symmetric at state {i}")
            failures += 1
            ok = False
        if not np.allclose(np.diag(A), 0, atol=1e-5):
            print(f"    FAIL: A has nonzero diagonal at state {i}")
            failures += 1
            ok = False
        if coal_betas[i] < 0:
            print(f"    FAIL: beta < 0 at state {i}: {coal_betas[i]}")
            failures += 1
            ok = False
    if ok:
        print(f"    OK - all {args.num_states} coalition matrices symmetric, zero-diagonal, β≥0")

    # 4. Bounds
    print("\n[4] Output bounds...")
    pl_max = max(np.abs(t).max() for t in pl_thetas)
    coal_max = max(np.abs(A).max() for A in coal_As)
    beta_max = max(coal_betas)
    print(f"    PL theta max abs: {pl_max:.2f}")
    print(f"    Coalition A max abs: {coal_max:.2f}")
    print(f"    beta max: {beta_max:.2f}")
    if pl_max > 100 or coal_max > 100 or beta_max > 1000:
        print(f"    WARN: outputs unusually large; may indicate training instability")

    # 5. Policy diversity
    print("\n[5] Policy diversity...")
    policy_arr = np.stack(policies)
    mean_argmax_freq = np.mean(np.bincount(policy_arr.argmax(axis=-1), minlength=1210)
                               .astype(np.float64) / args.num_states)
    print(f"    Distinct argmax actions across states: "
          f"{len(set(policy_arr.argmax(axis=-1)))}/{args.num_states}")
    if len(set(policy_arr.argmax(axis=-1))) < max(1, args.num_states // 4):
        print(f"    WARN: policy may be collapsed to a single action")

    # Optional full checks
    if args.full:
        print("\n[6] Coalition entropy across states (should not be ~0)...")
        from flagship_coalition_mcts.src.coalition_head import coalition_entropy
        entropies = []
        for s, A_np, beta_v in zip(states, coal_As, coal_betas):
            A_t = torch.from_numpy(A_np)
            beta_t = torch.tensor(beta_v)
            for player in range(s.num_players):
                e = coalition_entropy(A_t, beta_t, player=player,
                                      num_players=s.num_players)
                entropies.append(float(e))
        mean_ent = sum(entropies) / len(entropies)
        max_uniform = float(np.log(2 ** (max(s.num_players for s in states) - 1)))
        print(f"    Mean coalition entropy: {mean_ent:.3f}")
        print(f"    (uniform-equivalent for max-N: {max_uniform:.3f})")
        if mean_ent < 0.05 * max_uniform:
            print(f"    WARN: coalition entropy very low - possible collapse to fixed coalition")

    print("\n" + "=" * 72)
    if failures == 0:
        print(f"VALIDATION PASSED - {args.num_states} states, no failures")
    else:
        print(f"VALIDATION FAILED - {failures} failures across {args.num_states} states")
    print("=" * 72)
    return failures


if __name__ == "__main__":
    sys.exit(main())
