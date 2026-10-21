# Wreath-Equivariant Networks for Star-Hex Games

Policy and value networks for Chinese Checkers that respect the symmetries of the star board and of the players by construction.

Research prototype inside [NEXUS](../README.md). Pure PyTorch, no escnn dependency, 33 unit tests.

## Idea

Rotating the star board by 60 degrees maps it onto itself and carries every home triangle to the next one. Relabelling which seat is called "player 1" changes nothing about the game either. A standard CNN has to learn both facts from data. This network has them built in:

* **Board rotations (C6).** Hex convolutions in axial coordinates keep six rotation channels per cell. A 60-degree rotation acts as
  $(q, r) \mapsto (-r,\ q + r)$ on positions and as a cyclic shift on the rotation channels, and kernels are tied across the six rotated copies.
* **Seat permutations (S_N).** Each player's features are processed by shared weights and combined by permutation-invariant pooling, in the style of Deep Sets. Permuting the seats permutes the outputs in exactly the same way.
* **Wreath fusion.** A C6 layer runs on every seat's copy of the board with shared weights, spatial features are pooled per seat, mixed across seats by the S_N block, and broadcast back as a seat-conditioned bias. Each step commutes with both group actions, so the whole layer is equivariant to the wreath product C6 ≀ S_N.

Both properties are tested to hold to floating-point precision, not approximately.

## Implementation

| File | Contents |
| --- | --- |
| `src/c6_spatial.py` | `C6EquivariantLinear`, axial rotation and rotation permutation tables |
| `src/seat_equivariant.py` | `SeatEquivariantBlock`, `SeatInvariantPool`, `WreathSeatNet`, seat masks for variable N |
| `src/wreath_fuse.py` | `WreathFuseLayer` combining the spatial and seat streams |
| `src/cc_wreath_encoder.py` | `CCWreathEncoder` on the real 121-cell board from `core/board.py` |
| `src/wreath_network.py` | `WreathCCNetwork` with policy and value heads (188,987 parameters in the default configuration) |
| `src/cc_runner.py` | Evaluator for PUCT MCTS and a self-play game runner |
| `experiments/wreath_cc_training.py` | AlphaZero-style self-play training on Chinese Checkers |

## Running

From the repository root:

```bash
python -m pytest equivariant_net/tests -q    # 33 tests
python -m equivariant_net test
```

## What the tests check

* Six 60-degree rotations return every cell to itself; three rotations negate the axial coordinates.
* The C6 layer commutes with rotation, including under repeated full rotations.
* Seat pooling is invariant and the seat block is equivariant under every permutation, including with masks for fewer than six players.
* The fusion layer commutes with seat permutations, with rotations on the board interior, and with both at once.
* The encoder's pooled output is unchanged when the real board is rotated, and the network's pooled features are invariant to seat permutations.

## Status and next steps

Implemented: C6 rotations and S_N seat permutations on the real board, with an evaluator that plugs into MCTS.

Planned:

* Reflections, extending C6 to the full dihedral group D6.
* Symmetry breaking for player counts whose seating pattern has a smaller symmetry group, in the spirit of any-subgroup equivariant networks.
* Evaluation: sample efficiency against a non-equivariant network of the same size, and zero-shot transfer from 2 and 3 players to 4 and 6 players without retraining or augmentation.

## Related work

| Work | Relation |
| --- | --- |
| Cohen and Welling, *Group Equivariant Convolutional Networks*, ICML 2016 | Group convolutions on square grids. |
| Hoogeboom et al., *HexaConv*, ICLR 2018 | Group convolutions on hexagonal lattices, evaluated on images rather than games. |
| Zaheer et al., *Deep Sets*, NeurIPS 2017 | Permutation-invariant and permutation-equivariant set functions used for the seat stream. |
| Cesa, Lang and Weiler, *A Program to Build E(N)-Equivariant Steerable CNNs* (escnn), ICLR 2022 | General steerable CNN library. Avoided here to keep the dependency small. |
| Goel et al., *Any-Subgroup Equivariant Networks via Symmetry Breaking*, ICLR 2026 | Basis for the planned per-N subgroup selection. |
| Gu and Adhikari, *Efficient Learning in Chinese Checkers*, 2024 | Parameter sharing for Chinese Checkers without group equivariance. A natural baseline. |
