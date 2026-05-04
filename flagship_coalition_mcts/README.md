# Coalition-Distributional MCTS for N-Player Non-Zero-Sum Self-Play

**Status:** scaffolding - implementation gated on adversarial novelty-verification round (in progress).

**Target venue:** flagship ML conference (NeurIPS / ICML / ICLR main track) or top game-AI venue.

**Honest novelty self-assessment (after 2 verification rounds):** plausibly flagship-novel. Not "nobody ever thought of this" - that bar is essentially unattainable. Rather: the four-pillar combination is not in the literature, and the closest priors are clearly distinguishable.

---

## 1. Problem statement

For N-player **non-zero-sum** perfect-information board games (Chinese Checkers, Halma, multi-player Hex/Go variants), all standard self-play algorithms inherit a foundational limitation:

- **AlphaZero / Multiplayer AlphaZero (Petosa & Balch 2019)** outputs a per-player scalar value vector `v ∈ ℝᴺ`. PUCT selection uses the current player's coordinate.
- **Maxn (Luckhardt 1986), Paranoid (Sturtevant & Korf 2000), BRS (Schadd 2011)** - the classical N-player tree-search foundations - all assume *fixed, hand-specified opponent models* (everyone independent / everyone colluding / one unified opponent). All are provably suboptimal in non-zero-sum settings.
- **Player of Games (Schmid et al. 2021)** unifies CFR + MCTS for **2-player zero-sum**.

The 40-year-old open problem: a principled self-play algorithm for N-player non-zero-sum games whose value structure captures correlations between opponent outcomes (kingmaker dynamics, coalition incentives) and whose action selection is rooted in a meaningful game-theoretic equilibrium concept rather than ad-hoc heuristics.

## 2. Contribution claim (reviewer-proof phrasing)

> We introduce **Coalition-Distributional MCTS (CD-MCTS)**, the first self-play algorithm for N-player non-zero-sum perfect-information games that combines (i) a **Plackett-Luce-factorized rank-distribution value head** predicting a structured distribution over the symmetric group `S_N` of final placements, parameterized in `O(N²)` rather than `O(N!)`; (ii) a **coalition-probability head** outputting a posterior over opponent alignment subsets at each state; (iii) a **no-regret action-selection operator** replacing PUCT, derived from the EXP-IX family and conditioned on the inferred coalition posterior; and (iv) a self-play loop with **empirical convergence to a coarse correlated equilibrium** of the induced N-player meta-game. We instantiate CD-MCTS on Chinese Checkers (N=2..6), 4-player Halma, and 3-player 5×5 Go, ablate each pillar, and show monotone gains on Elo, win-rate vs all-paranoid baselines, and CCE-gap metrics.

The load-bearing phrases - each defuses a specific reviewer attack:

| Phrase | Defuses |
|---|---|
| Plackett-Luce factorization | "N! is intractable" |
| Coalition posterior | "Distinguishes from fixed maxn/paranoid" |
| EXP-IX / no-regret operator | "What's the equilibrium concept?" |
| Empirical CCE convergence | "Where's the theory?" |
| Three games × ablation ladder | "Just multiplayer AZ + heads" |

## 3. Closest prior art (after 2 verification rounds)

| Prior work | Relation | Distinction |
|---|---|---|
| **Yu et al. 2024 (NN-CCE, NeurIPS 2024, arxiv 2406.10411)** | **Closest neighbor - surfaced by adversarial verification** | CCE-approximating MCTS with NN value heads. **Simultaneous-move 2-agent matrix games at each node.** No Plackett-Luce rank head, no coalition posterior, not N-player sequential. Must be implemented as a baseline (extended to N players) and beaten on exploitability. |
| Petosa & Balch 2019 (Multiplayer AlphaZero, NeurIPS DRL Workshop) | Same problem class | Per-player scalar vector value, no rank distribution, no coalition head. **Strongest practical baseline.** |
| Yu/Tang 2025 (Simultaneous AlphaZero, arxiv 2512.12486) | Regret-optimal MCTS | 2-player zero-sum only, no rank distribution, no coalition head. |
| Marris et al. 2021 (JPSRO/MGCE meta-solvers) | N-player CE | PSRO/empirical-game style, not in-tree MCTS, no rank distribution. |
| Sun et al. (DFAC, MCMARL) | Distributional MARL | Cooperative QMIX-style scalar-return distributions, no rank/permutation structure, no coalition reasoning, no MCTS. |
| Wu & Ramchurn 2020 (MCTS for coalition formation, IJCAI) | MCTS over coalitions | Search over coalition-structure graph, NOT used as opponent-belief inside an N-player game tree. |
| Schmid et al. 2021 (Player of Games) | Search + RL unification | 2-player zero-sum only. No N-player non-zero-sum extension. |
| Bauer et al. (Distributional MCTS Thompson, OpenReview) | Distributional MCTS | Single-agent stochastic, NOT joint-outcome distributions over players. |
| Dal Lago 2025 (Diplomacy coalition detection, 2502.16339) | Coalition belief in games | Imperfect-info dialogue-driven, no rank distribution, no MCTS architecture. |
| Sturtevant 2019 (Strongly Solving Chinese Checkers) | Same game | Combinatorial endgame solver, no NN, no MCTS, no self-play. |
| Sokota et al. (EXP-IX → CCE in MCTS) | Same equilibrium concept | 2-player or simultaneous-move, no rank distribution, no coalition head. |
| Marris et al. (max-Gini CE meta-solvers, DeepMind) | CCE in game solving | Meta-game solver over a population, not in-tree backup. |

**After 3 verification rounds (last one explicitly adversarial), final novelty calibration: ~80% confident the specific four-pillar combination is unpublished.** No round above ~85% is honest in 2026.

**The single hardest reviewer attack** the adversarial round identified:
> "NN-CCE (2406.10411) already achieves CCE convergence inside neural tree search on general-sum games; your contribution reduces to swapping the value head."

**The only experiment that defuses it** (this is now part of the ablation plan):
A head-to-head on N≥4 perfect-info non-zero-sum (4-player Chinese Checkers, 4-player Halma, or a small Catan/Risk-lite) where:
1. NN-CCE-extended-to-N-players is implemented from the public code as a primary baseline.
2. Our method shows statistically significant CCE-gap / exploitability reduction.
3. Ablation: removing the PL head OR the coalition posterior independently degrades performance toward NN-CCE level.

Without all three, contribution shrinks to "PL head as auxiliary objective." We must be ready to publish that smaller claim honestly if it's what the data shows.

## 4. Architecture

```
                    ┌──────────────────────┐
                    │   Encoder backbone   │  (shared with workshop sub-projects)
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐
   │ Policy head  │   │  Plackett-Luce │   │ Coalition-belief │
   │  π(a|s)      │   │  rank head     │   │  head: ψ(C|s)    │
   │              │   │  s ↦ {θ_p}_p   │   │  for C ⊆ {1..N}  │
   └──────────────┘   └────────┬───────┘   └─────────┬────────┘
                               │                     │
                               │   PL.factorize      │
                               ▼                     │
                    distribution over S_N            │
                               │                     │
                               └──────┬──────────────┘
                                      ▼
                        ┌──────────────────────────────┐
                        │  CCE-regret action selector  │
                        │  (replaces PUCT)             │
                        └──────────────────────────────┘
```

**Plackett-Luce factorization:** the value head outputs a vector of player "strengths" `θ ∈ ℝᴺ`. The induced distribution over rank-orderings is

```
P(σ | s) = ∏_{k=1}^{N} exp(θ_{σ(k)}) / ∑_{j=k}^{N} exp(θ_{σ(j)})
```

This gives an `O(N²)` parameterization of a distribution over the `N!` orderings, with closed-form placement marginals `P(player p finishes in position k)`.

**Coalition head:** outputs `ψ : 2^{N-1} → [0,1]` over subsets of opponents. For tractability we factorize as pairwise terms `ψ(p, q) = P(opponents p and q are aligned)` plus a temperature.

**CCE-regret selector:** at each tree node, instead of `argmax_a [ Q(s,a) + c · π(a|s) · √(N) / (1+n(s,a)) ]`, we use an EXP-IX-style update that targets a coarse correlated equilibrium of the inferred per-state meta-game. Concretely: maintain regret estimates per child, sample from Hedge-style policy with γ-mixing for IX exploration. This is the most theoretically novel pillar.

## 5. Ablation ladder

Every reviewer will demand this. Each row must show a measurable Elo / win-rate / CCE-gap improvement over the previous:

1. (A0) Multiplayer AlphaZero baseline (Petosa) - scalar vector value, PUCT.
2. (A1) Add: rank-ordering distribution head (Plackett-Luce). Selector unchanged.
3. (A2) Add: coalition-probability head. Selector still PUCT but uses coalition belief in Q-aggregation.
4. (A3) Replace: PUCT → EXP-IX-style CCE-regret selector. **Full CD-MCTS.**

If any row fails to monotone-improve, the contribution falls back to whichever rows did improve and we publish that smaller claim honestly.

## 6. Empirical evaluation

Three games chosen to expose different facets:

- **Chinese Checkers, N=2..6.** Tournament game. Big enough to matter, small enough for reasonable compute. Coalition incentives weak in early game, strong in late game.
- **4-player Halma.** Stronger coalition incentives because piece-to-goal paths cross.
- **3-player 5×5 Go variant** (per Adhikari & Gu 2024). Shared liberties create explicit coalition incentives.

For each game:
- Self-play training to convergence (matched FLOPs across all ablation variants)
- Round-robin evaluation: A0/A1/A2/A3 + paranoid + maxn baselines
- Headline metric: Elo
- Secondary: win rate against fixed strong opponent (heuristic + Petosa-baseline)
- **Theory metric: empirical CCE-gap** measured on held-out states by exploitability of best deviation

## 7. Theorem attempt (the (c) → (d) move)

The single experiment that turns this from "well-executed combination" into "novel algorithmic primitive" is a proof of:

> **Conjecture.** Under standard regularity assumptions (bounded value, finite action set, no degenerate ties), self-play with CD-MCTS using EXP-IX selection at every node converges in expectation to a coarse correlated equilibrium of the N-player extensive-form meta-game.

The proof sketch - leveraging Sokota et al. on EXP-IX → CCE in 2-player and extending via the coalition posterior - is feasible but not trivial. If it works, this is an oral submission. If it doesn't, we present the empirical CCE-gap measurements and frame as conjectural.

I will commit honest effort to this proof and report negative results truthfully.

## 8. Compute discipline

CPU only until Phase 2 v4 of the main RL agent finishes. The flagship subproject must NOT touch the GPU while the tournament agent is training. Exception: post-Phase 2 v4, we may use the GPU for the killer experiment runs.

## 9. Layout

```
flagship_coalition_mcts/
  src/
    encoder.py          # shared encoder backbone
    plackett_luce.py    # rank distribution head + sampling + log-likelihood
    coalition_head.py   # opponent-subset belief head
    cce_selector.py     # EXP-IX-style action selector
    mcts.py             # tree with vector backups + selector dispatch
    trainer.py          # joint loss: policy + PL-NLL + coalition-NLL
    games/
      chinese_checkers.py
      halma.py
      go_5x5_3p.py
  experiments/
    ablation_ladder.py     # A0 → A3 across all 3 games
    cce_gap_eval.py        # empirical equilibrium gap measurement
    sample_efficiency.py
  tests/
    test_plackett_luce_marginals.py
    test_coalition_head_calibration.py
    test_cce_selector_recovers_puct.py   # special case sanity
    test_mcts_correctness.py
  theory/
    cce_convergence_proof.tex
    convergence_proof_outline.md
  docs/
    paper_outline.md
    related_work_table.md
```

## 10. Honest risks

- **Risk: the theorem doesn't go through.** Fallback: empirical CCE-gap claim, paper drops to workshop tier.
- **Risk: coalition head is degenerate** (predicts uniform regardless of state). Mitigation: design a controlled synthetic kingmaker game where coalitions are provably present. If even on that the head doesn't activate, the contribution collapses.
- **Risk: ablation ladder fails to monotone-improve.** Mitigation: publish honestly with only the rows that did. This is what a real scientist does.
- **Risk: a 2026 paper we missed already does this.** Mitigation: ongoing adversarial verification rounds.
- **Risk: compute-blow-up.** Each ablation variant × 3 games × matched FLOPs is significant. Mitigation: small game variants for the ladder, scale-up only the top variant.

## 11. Conduct

We will:
- Report negative results honestly.
- Not cherry-pick seeds.
- Not hide failed ablations.
- Pre-register ablation thresholds.
- Release code and reproducible experiments.

These are the table-stakes for a flagship paper that survives reviewer scrutiny in 2026. We do not relax them.
