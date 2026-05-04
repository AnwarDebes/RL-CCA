# Wreath-Equivariant Networks for Star-Hex Games

Group-equivariant policy/value networks for Chinese Checkers, exploiting the wreath structure of the star-hex board × player roles.

## What we claim

Chinese Checkers' true symmetry group is **not** plain D6 - the 6 home triangles break the hex interior's p6m down to a configuration-dependent subgroup, and player roles are themselves permutable. We model the joint symmetry as the wreath quotient

```
G_N ≀ S_N    where  G_N ≤ D6 is the subgroup of D6 fixing the
                    home-triangle-occupancy pattern for N players
```

with `N ∈ {2, 3, 4, 6}` (N=5 is a non-symmetric Sternhalma configuration). Realised by:

1. An **escnn p6m steerable backbone** over hex-grid features.
2. An **ASEN-style symmetry-breaking input field** (Goel et al. 2026) encoding which home triangles are active and which player is to move - automatically restricts equivariance to the active subgroup `G_N` per game.
3. **Wreath fusion layers** that combine spatial features with seat-features under joint rotation × seat-permutation.
4. A **symmetry-breaking output head** so the policy commits to a single move while the trunk stays equivariant.

## Why this is novel (verified by lit search 2024-2026)

| Closest prior art | Distinction |
|---|---|
| **HexaConv (Hoogeboom 2018)** - p6/p6m on hex lattices | Demoed on CIFAR + aerial imagery, **not games, not stars, no role-permutation factor**. |
| **Adhikari & Gu 2024 (arxiv 2405.18733)** - Chinese Checkers with parameter sharing | Standard CNN/MLP, **no group equivariance**. Strongest practical baseline. |
| **FGNN (Carroll 2020)** - D2-equivariant draughts | Square board, **D2 only**. |
| **ASEN (Goel 2026, arxiv 2603.19486)** - subgroup equivariance via symmetry-breaking | Validated on graphs/images/sequences, **never on games or star-hex**. |
| **Multi-Agent MDP Homomorphic Nets / PEnGUiN / CPE** | Handle **either** S_N **or** spatial group, never the wreath product. |
| **Wang & Maron - Equivariant Maps for Hierarchical Structures** | Wreath-product theory exists, **not applied to RL games**. |

**Novelty slice:** the wreath quotient `G_N ≀ S_N` realised as an escnn p6m backbone × ASEN gating, instantiated for the star-hex Chinese Checkers board with automatic per-N subgroup selection - no retraining, no test-time augmentation. None of the existing pieces are stitched together this way in the published record.

## Killer experiment

**Zero-shot N-generalization at fixed parameters.** Train AlphaZero-style on N=2 and N=3 only; evaluate against a fixed strong opponent at N=4 and N=6 with **zero retraining and zero data augmentation**. Baselines:

- (a) Adhikari-style parameter-sharing CNN
- (b) Same CNN + 12× rotation/reflection augmentation
- (c) HexaConv p6m **without** the wreath/ASEN gating (so it cannot specialize per-N)

**Sanity check:** drop-in role permutation at test time (swap which physical triangle is "player 1") must yield **bit-identical** policy logits for our model and *not* identical for any baseline. This is a hard test that the wreath equivariance is realised, not just approximated.

## Layout

```
equivariant_net/
  src/
    backbone.py      # escnn p6m steerable trunk
    asen_gate.py     # active-subgroup field generator
    wreath_fuse.py   # spatial × seat-feature fusion under wreath action
    head.py          # symmetry-breaking policy + scalar value
    embed.py         # hex-grid embedding + seat-feature stream
  experiments/
    zero_shot_N.py   # killer experiment
    sample_efficiency.py
    permutation_sanity.py  # bit-identical logits under seat permutation
  tests/
    test_p6m_equivariance.py
    test_role_permutation_invariance.py
    test_wreath_fuse_commute.py
```

## Compute discipline

CPU only. The escnn ops are CPU-runnable for verification and small-scale experiments. Any large-scale training waits for Phase 2 v4 completion.

## Risks (honest)

- **"Just escnn applied to a new board"** - counter by emphasizing the wreath quotient algebra and the ASEN-tied subgroup selection. The contribution is the *architecture*, not the application alone.
- **Killer experiment must beat both** Adhikari and rotation-augmented HexaConv-on-rhombus, otherwise the star-specific contribution is invisible.
- **escnn is heavyweight** - falls back to slower CPU paths without GPU. Live with it for the prototype; revisit after Phase 2.
