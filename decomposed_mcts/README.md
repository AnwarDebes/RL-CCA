# CMAZ: Component-Mixed AlphaZero

State-conditional monotonic mixing of per-component values inside MCTS.

Research prototype inside [NEXUS](../README.md). Pure PyTorch and NumPy, runs on CPU, 39 unit tests.

## Idea

Many games score a player with a known sum of separate parts. The Chinese Checkers tournament that NEXUS plays uses

```
score = pin_goal_score + distance_score + time_score + move_score
```

AlphaZero collapses this into one scalar value before search starts, so the tree can never recover which part of the score a line of play is good for. CMAZ keeps the parts apart:

1. A **K-component value head** predicts one value per score component, each bounded to [-1, 1].
2. **Per-component Q statistics** are backed up in the tree, one K-vector per edge.
3. A **state-conditional monotonic mixer** turns the K-vector into the scalar used by PUCT, with weights that depend on the position.
4. The **policy head** is trained on MCTS visit counts, exactly as in AlphaZero. Policy, values and mixer are trained jointly.

The mixer is a hypernetwork in the style of QMIX, applied to objectives instead of agents:

$$
Q(s, v) = \sum_{k=1}^{K} w_k(s)\, v_k + b(s), \qquad w(s) = \mathrm{softmax}\big(h_\phi(s)\big) \in \Delta^{K}
$$

Because every weight is non-negative, Q is monotone in each component. Because the weights are separate from the values, a trained network can be steered at inference time by replacing w(s) with fixed weights, for example all weight on `pin_goal_score` for an all-out race, without retraining.

## Implementation

| File | Contents |
| --- | --- |
| `src/monotonic_mixer.py` | `MonotonicMixer` (hypernetwork, simplex weights, inference override) and `ComponentValueHead` |
| `src/network.py` | `CMAZEncoder`, `CMAZNetwork`, `CMAZEvaluator`, joint loss `cmaz_loss` |
| `src/cmaz_mcts.py` | PUCT search with vector Q backup and mixer aggregation (`run_mcts_cmaz`) |
| `src/self_play.py` | Self-play loop with component-decomposed targets |
| `src/cc_adapter.py` | Chinese Checkers adapter on top of the NEXUS game environment |
| `src/kingmaker_adapter.py`, `src/halma_adapter.py` | Adapters for the two small testbeds shared with CD-MCTS |
| `experiments/inference_override_demo.py` | Same network, several mixer overrides, policies side by side |
| `experiments/train_and_demo_kingmaker.py` | Train on the kingmaker game, then sweep overrides |
| `experiments/cmaz_cc_training.py` | Self-play training on full Chinese Checkers |

For Chinese Checkers the four components are scaled as `[pin_goal/1000, distance/200, time/100, move/1]`, matching the tournament formula.

## Running

From the repository root:

```bash
python -m decomposed_mcts override-demo      # inference-time override, untrained network
python -m decomposed_mcts train-kingmaker    # train on kingmaker, then sweep overrides
python -m pytest decomposed_mcts/tests -q    # 39 tests
```

## What the tests check

* Mixer weights lie on the simplex, Q increases monotonically in every component, and gradients have the expected signs.
* An override returns exactly the override-weighted sum, renormalised if needed.
* Vector backups reproduce the terminal score components, and visit counts sum to the simulation budget.
* Two different overrides on the same network produce different decisions.
* End-to-end search on kingmaker, small Halma and real Chinese Checkers, plus loss masking and the public API.

## Planned evaluation

Three agents trained to identical compute on a small Chinese Checkers variant: (A) scalar AlphaZero on the weighted sum, (B) a KataGo-style fixed linear utility over the four components, and (C) CMAZ. At inference, each network is frozen and given external re-weightings (for example, time score set to zero). CMAZ should degrade gracefully because its tree statistics are still vector-valued. A second measurement averages the learned weights w(s) by phase of the game.

## Related work

| Work | Relation |
| --- | --- |
| Rashid et al., *QMIX*, ICML 2018 | State-conditional monotonic mixer for cooperative agents, without tree search. CMAZ reuses the mixer for objectives inside MCTS. |
| van Seijen et al., *Hybrid Reward Architecture*, NeurIPS 2017 | Per-component Q values, model-free, fixed equal weights. |
| Wu, *Accelerating Self-Play Learning in Go* (KataGo), 2019 | Mixes win rate and score with a fixed, state-independent utility. |
| Tang, Chen and Lan, *MALinZero*, 2025 | Vector statistics in MCTS over latent per-agent rewards with a global linear model. CMAZ uses known score components and per-state weights. |
