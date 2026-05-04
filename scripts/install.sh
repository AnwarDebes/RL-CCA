#!/usr/bin/env bash
# install.sh - set up the project from scratch.
# Idempotent: run repeatedly to ensure the environment is current.

set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "Installing nexus research subprojects"
echo "============================================================"

# 1. Verify Python
if ! command -v python3 &> /dev/null; then
    echo "FAIL: python3 not found"
    exit 1
fi
PYV=$(python3 --version | awk '{print $2}')
echo "Python: $PYV"
if [[ "$PYV" < "3.10" ]]; then
    echo "WARN: Python 3.10+ recommended; found $PYV"
fi

# 2. Create venv if missing
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
else
    echo "venv exists; reusing"
fi

VENV=./venv/bin/python

# 3. Upgrade pip
$VENV -m pip install --upgrade pip --quiet

# 4. Install required packages
echo "Installing required packages..."
$VENV -m pip install --quiet \
    "torch>=2.0" \
    "numpy>=1.24" \
    "pytest>=7.0"

# 5. Install optional packages
echo "Installing optional packages (matplotlib for plotting)..."
$VENV -m pip install --quiet matplotlib || echo "  (matplotlib install failed; visualisation will be text-only)"

# 6. Install subprojects in editable mode
echo "Installing subprojects in editable mode..."
for sp in flagship_coalition_mcts decomposed_mcts equivariant_net; do
    $VENV -m pip install -e "$sp/" --quiet 2>/dev/null || echo "  (skipping $sp - no setup found)"
done

# 7. Verify
echo
echo "Running environment check..."
$VENV check_env.py

echo
echo "============================================================"
echo "Install complete. Try:"
echo "  make test-fast"
echo "  make smoke"
echo "  ./scripts/quick_status.sh"
echo "============================================================"
