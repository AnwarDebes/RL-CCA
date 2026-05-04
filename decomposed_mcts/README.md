# CMAZ - Component-Mixed AlphaZero

State-conditional monotonic mixing of per-component values inside MCTS.

## What we claim

For games whose reward is a known sum of K semantically-distinct components (in Chinese Checkers: pin_goal_score, distance_score, time_score, move_score), we replace AlphaZero's scalar value with:

1. A **K-vector value head** `v(s) ∈ ℝᴷ`.
2. **Per-component MCTS Q-statistics** `Q_k(s,a)` backed up in-tree, one scalar per component per edge.
3. A **state-conditional monotonic mixing network** `f_φ(Q_1..Q_K; s)`, QMIX-style - each component contributes monotonically to the scalar used in PUCT, but the slope is per-state.
4. A **policy head** trained against MCTS visit counts, exactly as AlphaZero. Policy / value / mixer trained jointly end-to-end.

## Why this is novel (verified by lit search 2024-2026)

| Closest prior art | Distinction |
|---|---|
| **KataGo (Wu 2019)** - `winrate + λ·tanh(score)` in MCTS | Fixed λ, scalar-conditional. Ours is **state-conditional + learned**. |
| **HRA (van Seijen 2017)** - per-component Q for action selection | Model-free, no tree, **fixed equal weights**. |
| **MALinZero (Zhao 2025, arxiv 2511.06142)** - vector statistics + LinUCT | Components are **per-agent latent**, not per-objective semantic. Mixer is global linear regressor over latent rewards, **not per-state**. Ours uses known semantic components and conditions on board state. |
| **QMIX (Rashid 2018)** - state-conditional monotonic mixer | Per-AGENT decomposition for cooperative MARL, **no MCTS**. We import the mixer architecture into MCTS for objective decomposition. |
| **Pareto-MCTS (multiple)** - vector backups, Pareto selection | Returns Pareto fronts, doesn't collapse to scalar. Ours collapses, stays AlphaZero-compatible. |

**Novelty slice:** in-tree per-component Q-backup + state-conditional QMIX-style monotonic mixer inside an AlphaZero loop, where components are the game's natural score formula. Workshop-paper novel - not flagship-conference novel - and that's an honest claim.

## Killer experiment

Train three agents to identical compute on a small Chinese Checkers variant:

- **A**: scalar AZ on weighted-sum reward
- **B**: KataGo-style fixed-linear utility over the 4 components
- **C**: CMAZ (ours)

**Headline plot:** at *inference time*, freeze each network and impose external re-weightings (e.g., zero out time_score → "blowout mode"; zero out pin_goal → "stall mode"). Plot Elo vs re-weighting magnitude.

Prediction: CMAZ adapts gracefully because tree statistics are still vector-valued. A/B collapse - their tree statistics already integrated over weights they can no longer recover.

**Bonus interpretability plot:** average mixer weights `w_k(s)` over phase-of-game (early/mid/end). We expect distance_score weight to dominate early, pin_goal_score to dominate late.

## Layout

```
decomposed_mcts/
  src/
    model.py        # K-component value head + monotonic mixer + policy head
    mcts.py         # PUCT with vector Q backup + state-mixer evaluation
    trainer.py      # joint policy/value/mixer loss
    env.py          # small Chinese Checkers variant (CPU)
  experiments/
    inference_reweight.py  # killer experiment
    sample_efficiency.py
  tests/
    test_mixer_monotonicity.py
    test_mcts_collapses_to_scalar_when_w_uniform.py
    test_value_head_shape.py
```

## Compute discipline

CPU only while v4 RL agent trains. Any GPU run waits for Phase 2 v4 completion.
