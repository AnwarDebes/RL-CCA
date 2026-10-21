# CD-MCTS: Coalition-Distributional MCTS

Self-play tree search for games with more than two players, where opponents can gang up on the leader and a losing player can decide who wins.

Research prototype inside [NEXUS](../README.md). About 5,400 lines of source with 202 unit tests. The experiments are scripted end to end; results are not reported yet.

## Problem

AlphaZero is built for two-player zero-sum games. With three to six players, two things break:

* **The value is not a single number.** What matters is the full finishing order, and opponents' outcomes are correlated.
* **Opponents are not all against you, or all independent.** Classical N-player search fixes the opponent model in advance: everyone maximises their own result (maxn), everyone plays against you (paranoid), or only the single strongest reply among all opponents is considered (best-reply search). In real play, alignments change with the position, for example when two trailing players both slow down the leader.

## Method

CD-MCTS replaces three parts of multiplayer AlphaZero.

**1. Rank-distribution value head (Plackett-Luce).** The network outputs one strength per player, $\theta \in \mathbb{R}^N$. The probability of a finishing order σ, where σ(k) is the player in place k, is

$$
P(\sigma \mid \theta) = \prod_{k=1}^{N} \frac{\exp \theta_{\sigma(k)}}{\sum_{j=k}^{N} \exp \theta_{\sigma(j)}}
$$

This describes a distribution over all N! orders with only N numbers per state. The winner marginal is softmax(θ), and full placement marginals are computed exactly by enumeration for N ≤ 6.

**2. Coalition-belief head.** The network outputs pairwise alignment logits A and a concentration β ≥ 0. For player p, the belief that the set C of opponents is aligned against p is

$$
P(C \mid s, p) \propto \exp\Big(\beta \sum_{q \in C} A_{pq} - \beta \sum_{q \notin C,\, q \neq p} A_{pq}\Big)
$$

With β = 0 this is uniform (no information); as β grows it collapses to one coalition. It is trained on which opponents actually finished ahead of p in self-play.

**3. No-regret action selection.** PUCT is replaced by an EXP-IX style Hedge rule at every node. The sampling distribution is shifted by the coalition belief, using the probability that some member of the believed coalition wins from the child state. The target is a coarse correlated equilibrium (CCE) of the per-state meta-game rather than a single best arm.

**4. Vector backup.** Instead of a scalar, the full N × N placement-marginal matrix is averaged up the tree. Any utility over placements can be applied afterwards without searching again.

## Testbeds

| Game | Players | Why |
| --- | --- | --- |
| Kingmaker race (`src/games/kingmaker.py`) | 3 | Built so that coalitions provably matter: the leader can only be stopped if both trailing players spend a turn tripping it, which neither does alone. Small enough to solve exactly. |
| Small Halma (`src/games/halma_small.py`) | 3 | 5 × 5 board, 3 pieces each. Paths cross in the centre, so interaction is strong. |
| Chinese Checkers (`src/games/chinese_checkers.py`) | 2 to 6 | The full tournament game, through the NEXUS environment. |

## Baselines and ablation ladder

| Variant | Value | Coalition head | Selection |
| --- | --- | --- | --- |
| A0, multiplayer AlphaZero | scalar per player | no | PUCT |
| A1 | Plackett-Luce | no | PUCT |
| A2 | Plackett-Luce | yes | PUCT |
| A3, full CD-MCTS | Plackett-Luce | yes | EXP-IX |
| B, NN-CCE extended to N players | scalar per player | no | EXP-IX |

Each step up the ladder has to beat the one below it at matched compute.

## Measurement

* **Exploitability and CCE gap**, computed exactly by best-response enumeration on the kingmaker game (`src/exploitability.py`).
* **Head-to-head Elo** with bootstrap confidence intervals and a permutation test (`src/head_to_head.py`).
* **Seed sweeps** and Markdown or LaTeX result tables (`experiments/seed_sweep.py`, `src/results_table.py`). Results are meant to be reported as median and spread across seeds, not the best seed.

## Implementation

| File | Contents |
| --- | --- |
| `src/plackett_luce.py` | Head, sampling, log-likelihood, exact and sampled placement marginals |
| `src/coalition_head.py` | Pairwise alignment head, coalition enumeration, entropy, training loss |
| `src/cce_selector.py` | EXP-IX selector with Hedge updates and IX exploration |
| `src/mcts.py` | Tree with placement-marginal backup and selector dispatch |
| `src/baseline_mcts.py`, `src/nn_cce_baseline.py` | Baselines A0 and B |
| `src/network.py`, `src/cnn_encoder.py` | CD-MCTS network; CNN encoder for the 32 × 17 × 17 board tensor (482,449 parameters in the Chinese Checkers configuration) |
| `src/self_play.py`, `src/replay_buffer.py`, `src/checkpoint.py`, `src/subtree_reuse.py` | Training loop, buffer, resumable checkpoints, tree reuse between moves |
| `src/tournament_player_cdmcts.py` | Tournament player, run with `scripts/play_server_cdmcts.py` |
| `experiments/` | Pilot, kingmaker head-to-head, CCE-gap tracking, ablation ladder, final evaluation |

## Running

From the repository root:

```bash
python -m flagship_coalition_mcts summary       # parameter counts per submodule
python -m flagship_coalition_mcts compare       # comparison with CMAZ and the equivariant network
python -m flagship_coalition_mcts smoke         # all three subprojects on real Chinese Checkers
python -m flagship_coalition_mcts.experiments.pilot_experiment --seed 0 --out results/pilot.json
python -m flagship_coalition_mcts.experiments.kingmaker_head_to_head --seed 0
python -m pytest flagship_coalition_mcts/tests -q   # 202 tests
```

On a CPU, the pilot finishes in about 30 seconds and the kingmaker head-to-head in a few minutes.

## What the tests check

* Plackett-Luce probabilities sum to one over all orders, the winner marginal equals softmax(θ), sampling matches the likelihood (chi-squared test), and inactive seats are masked.
* Coalition beliefs are normalised, uniform at β = 0 and concentrated as β grows; the training loss moves beliefs towards the observed coalition.
* The selector has square-root regret on a stochastic bandit, starts from the prior, and two selectors in self-play on matching pennies converge to the Nash equilibrium.
* Exploitability is non-negative and best responses match an exact solver on the kingmaker game.
* Search, training, checkpoint resume, tournament player and the full Chinese Checkers integration run end to end.

## Open questions

* Convergence of the self-play loop to a CCE of the N-player game is a conjecture, which is why the CCE gap is tracked empirically.
* The pairwise coalition factorisation is weaker than a full distribution over coalitions; the full and truncated versions are planned ablations.
* The coalition head could stay close to uniform in games where alignments are weak. The kingmaker game exists to test exactly this.

## Related work

| Work | Relation |
| --- | --- |
| Petosa and Balch, *Multiplayer AlphaZero*, 2019 | Per-player value vector with PUCT. Baseline A0. |
| Luckhardt and Irani, 1986; Sturtevant and Korf, 2000; Schadd and Winands, 2011 | maxn, paranoid and best-reply search: fixed opponent models. |
| Plackett, 1975; Luce, 1959 | The ranking model behind the value head. |
| Neu, *Explore no more* (EXP3-IX), NeurIPS 2015 | The bandit algorithm behind the selector. |
| Hart and Mas-Colell, *A Simple Adaptive Procedure Leading to Correlated Equilibrium*, 2000 | No-regret play and correlated equilibria. |
| Yu, Olshevsky and Chin, *Tree Search for Simultaneous Move Games via Equilibrium Approximation*, 2024 | Closest method: CCE-approximating tree search for simultaneous-move games. Baseline B. |
| Danihelka et al., *Policy Improvement by Planning with Gumbel*, ICLR 2022 | The search used by the main NEXUS agent. |
| Sturtevant, *On Strongly Solving Chinese Checkers*, 2019 | Exact solving of small boards, without learning. |
