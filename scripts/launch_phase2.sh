#!/usr/bin/env bash
# launch_phase2.sh - verify Phase 1 + launch Phase 2 in one command.
#
# Run this after Phase 1 v4 has finished (process exits, checkpoint exists).
# The script:
#   1. Verifies Phase 1 completed cleanly (checkpoint + loss decay).
#   2. If verification passes, launches Phase 2 v4 in the background.
#   3. Prints the new pid + log path for monitoring.

set -e
cd "$(dirname "$0")/.."

VENV=./venv/bin/python

echo "============================================================"
echo "Launch Phase 2 v4 - handoff from Phase 1"
echo "============================================================"

# 1. Verify Phase 1
echo
echo "--- Step 1: Verify Phase 1 v4 completed cleanly ---"
$VENV scripts/verify_phase1_complete.py \
    --log training_logs/phase1_v4.log \
    --checkpoint checkpoints_v4/phase1_v4.pt \
    --expected-batches 25 || {
    echo
    echo "Phase 1 verification FAILED. Refusing to launch Phase 2."
    echo "Investigate the warnings above. Common issues:"
    echo "  - Phase 1 still running: wait for it to finish"
    echo "  - Checkpoint missing: training crashed before save"
    echo "  - Loss didn't decrease: check for training-rate or config bugs"
    exit 1
}

# 2. Confirm with user (skip if --auto)
if [[ "${1:-}" != "--auto" ]]; then
    echo
    read -p "Launch Phase 2 v4 now? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted by user."
        exit 0
    fi
fi

# 3. Make log directory
mkdir -p training_logs

# 4. Launch in background
LOG=training_logs/phase2_v4.log
echo
echo "--- Step 2: Launching Phase 2 v4 ---"
# IMPORTANT: -u for unbuffered stdout, otherwise prints sit in a 4-8KB
# block buffer when redirected to a file and we lose live visibility.
# Phase 1 used -u; we match that here.
nohup $VENV -u scripts/train_phase2_v4.py \
    --resume checkpoints_v4/phase1_v4.pt \
    --start-iter 0 \
    --checkpoint-dir checkpoints_v4 \
    --run-name phase2_v4 \
    > "$LOG" 2>&1 &

PID=$!
sleep 3

# 5. Verify it's still alive after 3s (catches immediate crashes)
if ! kill -0 "$PID" 2>/dev/null; then
    echo "FAIL: Phase 2 v4 process died within 3 seconds. Check log:"
    echo "  $LOG"
    tail -50 "$LOG"
    exit 1
fi

echo "Phase 2 v4 launched: pid=$PID, log=$LOG"
echo
echo "Monitor with:"
echo "  tail -f $LOG"
echo "  ./scripts/quick_status.sh"
echo
echo "Phase 2 ETA: ~5 days. Use 'make verify-phase2' when it finishes."
