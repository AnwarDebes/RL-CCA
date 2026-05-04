#!/usr/bin/env bash
# pre_commit_hook.sh - run fast tests before allowing a git commit.
#
# Install with:
#     ln -s "$(pwd)/scripts/pre_commit_hook.sh" .git/hooks/pre-commit
#
# This catches obvious regressions before they enter version control.
# Skips the heavy CC integration tests (those take 2+ min); install the
# heavier hook below for full coverage.

set -e
cd "$(dirname "$0")/.."

VENV=./venv/bin/python

# Math-only tests: ~30s
$VENV -m pytest -q \
    flagship_coalition_mcts/tests/test_plackett_luce.py \
    flagship_coalition_mcts/tests/test_coalition_head.py \
    flagship_coalition_mcts/tests/test_cce_selector.py \
    flagship_coalition_mcts/tests/test_kingmaker.py \
    flagship_coalition_mcts/tests/test_exploitability.py \
    flagship_coalition_mcts/tests/test_halma_small.py \
    flagship_coalition_mcts/tests/test_replay_buffer.py \
    flagship_coalition_mcts/tests/test_subtree_reuse.py \
    flagship_coalition_mcts/tests/test_results_table.py \
    flagship_coalition_mcts/tests/test_setup_logging.py \
    --tb=short

$VENV -m pytest -q \
    decomposed_mcts/tests/test_monotonic_mixer.py \
    decomposed_mcts/tests/test_cmaz_mcts.py \
    --tb=short

$VENV -m pytest -q \
    equivariant_net/tests/test_seat_equivariant.py \
    equivariant_net/tests/test_c6_spatial.py \
    equivariant_net/tests/test_wreath_fuse.py \
    --tb=short

echo "Pre-commit checks passed."
