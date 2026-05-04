#!/usr/bin/env bash
# Reproduction script for the three research subprojects.
# Run AFTER v4 RL training (Phase 1 + Phase 2) has completed.
#
# Stages can be run independently. Each is idempotent - outputs are
# in the named JSON files.

set -e
cd "$(dirname "$0")"

VENV=./venv/bin/python
mkdir -p results checkpoints

echo "=== Stage 0: Sanity (unit tests) ==="

# Three subprojects; tests collect from each in turn (avoids module-name collision).
$VENV -m pytest \
    flagship_coalition_mcts/tests/test_plackett_luce.py \
    flagship_coalition_mcts/tests/test_coalition_head.py \
    flagship_coalition_mcts/tests/test_cce_selector.py \
    flagship_coalition_mcts/tests/test_mcts.py \
    flagship_coalition_mcts/tests/test_kingmaker.py \
    flagship_coalition_mcts/tests/test_exploitability.py \
    flagship_coalition_mcts/tests/test_halma_small.py \
    flagship_coalition_mcts/tests/test_head_to_head.py \
    flagship_coalition_mcts/tests/test_results_table.py \
    flagship_coalition_mcts/tests/test_replay_buffer.py \
    flagship_coalition_mcts/tests/test_checkpoint.py \
    flagship_coalition_mcts/tests/test_network.py \
    flagship_coalition_mcts/tests/test_cnn_encoder.py \
    flagship_coalition_mcts/tests/test_nn_cce_baseline.py \
    flagship_coalition_mcts/tests/test_baseline_mcts.py \
    flagship_coalition_mcts/tests/test_cc_adapter.py \
    --tb=short

$VENV -m pytest decomposed_mcts/tests/ --tb=short

$VENV -m pytest \
    equivariant_net/tests/test_seat_equivariant.py \
    equivariant_net/tests/test_c6_spatial.py \
    equivariant_net/tests/test_wreath_fuse.py \
    equivariant_net/tests/test_cc_wreath_encoder.py \
    equivariant_net/tests/test_wreath_network.py \
    --tb=short

echo
echo "=== Stage 1: Smoke ==="
$VENV flagship_coalition_mcts/experiments/full_integration_demo.py \
    --num-players 2 --num-simulations 8

echo
echo "=== Stage 2: Kingmaker head-to-head (3 seeds) ==="
for SEED in 0 1 2; do
    $VENV flagship_coalition_mcts/experiments/kingmaker_head_to_head.py \
        --num-iterations 20 --games-per-iter 16 --train-steps 32 \
        --num-simulations 24 --eval-games 100 --eval-simulations 32 \
        --seed $SEED --out results/kingmaker_h2h_seed${SEED}.json
done

echo
echo "=== Stage 3: Aggregate kingmaker H2H ==="
$VENV -m flagship_coalition_mcts.src.results_table \
    --files results/kingmaker_h2h_seed*.json \
    --format markdown \
    --output results/kingmaker_h2h_table.md

echo
echo "=== Stage 4: CD-MCTS self-play training on CC ==="
$VENV flagship_coalition_mcts/experiments/cc_self_play_training.py \
    --num-iterations 100 --games-per-iter 32 --train-steps 64 \
    --num-simulations 32 --num-players 2 \
    --channels 64 --num-blocks 8 --hidden-dim 256 \
    --seed 0 --checkpoint-dir checkpoints/cdmcts_cc_seed0

echo
echo "=== Stage 5: Ablation ladder ==="
$VENV flagship_coalition_mcts/experiments/ablation_ladder.py \
    --games kingmaker --num-games-per-pair 50 \
    --num-simulations 32 --seed 0 \
    --out results/ablation_seed0.json

echo
echo "=== ALL STAGES COMPLETE ==="
echo "Results in results/, checkpoints in checkpoints/"
