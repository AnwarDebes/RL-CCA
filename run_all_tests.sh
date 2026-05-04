#!/usr/bin/env bash
# One-command verification: runs all unit tests across the three subprojects.
# Use this as a sanity check after any code change, or as a CI entry point.
#
# Each subproject's tests run in a separate pytest invocation because the
# `tests/` package name collides between them.
#
# Total runtime: ~3 minutes on CPU.

set -e
cd "$(dirname "$0")"

VENV=./venv/bin/python
echo "============================================================"
echo "Running all tests across 3 research subprojects"
echo "============================================================"

# --- Flagship: math-only tests (fast) ---
echo
echo "--- flagship_coalition_mcts (math + utilities) ---"
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
    flagship_coalition_mcts/tests/test_subtree_reuse.py \
    --tb=short

# --- Flagship: torch + CC integration tests ---
echo
echo "--- flagship_coalition_mcts (torch + real CC) ---"
$VENV -m pytest \
    flagship_coalition_mcts/tests/test_network.py \
    flagship_coalition_mcts/tests/test_cnn_encoder.py \
    flagship_coalition_mcts/tests/test_nn_cce_baseline.py \
    flagship_coalition_mcts/tests/test_baseline_mcts.py \
    flagship_coalition_mcts/tests/test_cc_adapter.py \
    flagship_coalition_mcts/tests/test_tournament_player.py \
    flagship_coalition_mcts/tests/test_final_evaluation.py \
    flagship_coalition_mcts/tests/test_integration.py \
    flagship_coalition_mcts/tests/test_play_server_cdmcts_smoke.py \
    --tb=short

# --- Flagship: heavy tests (full CC self-play) ---
echo
echo "--- flagship_coalition_mcts (heavy: CC runner + self-play, ~3 min) ---"
$VENV -m pytest \
    flagship_coalition_mcts/tests/test_cc_runner.py \
    flagship_coalition_mcts/tests/test_self_play.py \
    --tb=short

# --- CMAZ ---
echo
echo "--- decomposed_mcts ---"
$VENV -m pytest decomposed_mcts/tests/ --tb=short

# --- Equivariant ---
echo
echo "--- equivariant_net ---"
$VENV -m pytest \
    equivariant_net/tests/test_seat_equivariant.py \
    equivariant_net/tests/test_c6_spatial.py \
    equivariant_net/tests/test_wreath_fuse.py \
    equivariant_net/tests/test_cc_wreath_encoder.py \
    equivariant_net/tests/test_wreath_network.py \
    equivariant_net/tests/test_cc_runner.py \
    --tb=short

echo
echo "============================================================"
echo "ALL TESTS PASSED"
echo "============================================================"
