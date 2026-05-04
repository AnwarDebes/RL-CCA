#!/usr/bin/env bash
# Quick training demo: ~5-min run of all 3 subprojects' training pipelines
# on tiny budgets. Verifies end-to-end pipeline health without committing
# multi-day compute.
#
# Use this AFTER Phase 2 v4 RL training completes to confirm the research
# subprojects' pipelines function before launching long training runs.
#
# Outputs:
#   results/quick_train_demo/
#     cdmcts_kingmaker.json   - flagship's kingmaker H2H output
#     cmaz_kingmaker_train_demo.txt - CMAZ's train + override demo
#     wreath_demo.txt         - wreath equivariance verification

set -e
cd "$(dirname "$0")"
mkdir -p results/quick_train_demo
VENV=./venv/bin/python

echo "============================================================"
echo "Quick training demo (~5 min)"
echo "============================================================"

echo
echo "--- Stage 1: Flagship kingmaker H2H (small) ---"
$VENV flagship_coalition_mcts/experiments/kingmaker_head_to_head.py \
    --num-iterations 5 --games-per-iter 3 --train-steps 8 \
    --num-simulations 8 --eval-games 30 --eval-simulations 12 \
    --seed 0 --out results/quick_train_demo/cdmcts_kingmaker.json
echo "  -> results/quick_train_demo/cdmcts_kingmaker.json"

echo
echo "--- Stage 2: CMAZ train+demo on kingmaker ---"
$VENV decomposed_mcts/experiments/train_and_demo_kingmaker.py \
    --num-iterations 5 --games-per-iter 4 --train-steps 12 \
    --num-simulations 12 --demo-simulations 32 --seed 0 \
    | tee results/quick_train_demo/cmaz_kingmaker_train_demo.txt
echo "  -> results/quick_train_demo/cmaz_kingmaker_train_demo.txt"

echo
echo "--- Stage 3: Wreath equivariant integration demo on real CC ---"
$VENV flagship_coalition_mcts/experiments/full_integration_demo.py \
    --num-players 2 --num-simulations 8 \
    | tee results/quick_train_demo/wreath_demo.txt
echo "  -> results/quick_train_demo/wreath_demo.txt"

echo
echo "============================================================"
echo "Quick training demo complete."
echo "Results in: results/quick_train_demo/"
echo "============================================================"
