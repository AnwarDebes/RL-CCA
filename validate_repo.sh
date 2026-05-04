#!/usr/bin/env bash
# validate_repo.sh - comprehensive repo-health check.
# Runs ALL the smaller validation steps in sequence: env check, fast tests,
# docs regenerate. Use as the single command to ensure the repo is in a
# good state before committing or before claiming reproducibility.
#
# Total runtime: ~3-5 min on CPU.

set -e
cd "$(dirname "$0")"

VENV=./venv/bin/python

echo "============================================================"
echo "Repo health check"
echo "============================================================"

echo
echo "--- Stage 1: Environment ---"
$VENV check_env.py

echo
echo "--- Stage 2: Math-only tests (~30s) ---"
make test-fast

echo
echo "--- Stage 3: Subproject importability ---"
$VENV -c "
import flagship_coalition_mcts.src as f
import decomposed_mcts.src as d
import equivariant_net.src as e
print(f'  flagship: v{f.__version__} ({len(f.__all__)} exports)')
print(f'  cmaz:     v{d.__version__} ({len(d.__all__)} exports)')
print(f'  wreath:   v{e.__version__} ({len(e.__all__)} exports)')
"

echo
echo "--- Stage 4: CLIs respond to --help ---"
$VENV -m flagship_coalition_mcts --help > /dev/null && echo "  flagship CLI OK"
$VENV -m decomposed_mcts --help > /dev/null && echo "  cmaz CLI OK"
$VENV -m equivariant_net --help > /dev/null && echo "  wreath CLI OK"

echo
echo "--- Stage 5: Auto-generate docs ---"
make docs

echo
echo "--- Stage 6: Documentation file count ---"
NUM_DOCS=$(find . -maxdepth 3 -name "*.md" -not -path "./venv/*" -not -path "./node_modules/*" | wc -l)
echo "  Markdown files: $NUM_DOCS"

echo
echo "--- Stage 7: Test totals ---"
FLAGSHIP_LOC=$(find flagship_coalition_mcts -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
CMAZ_LOC=$(find decomposed_mcts -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
WREATH_LOC=$(find equivariant_net -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
echo "  flagship_coalition_mcts: $FLAGSHIP_LOC LoC (Python only)"
echo "  decomposed_mcts: $CMAZ_LOC LoC"
echo "  equivariant_net: $WREATH_LOC LoC"

echo
echo "============================================================"
echo "REPO VALIDATION PASSED"
echo "============================================================"
