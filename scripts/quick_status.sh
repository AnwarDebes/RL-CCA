#!/usr/bin/env bash
# quick_status.sh - single-command snapshot of project state.
# Prints: training health, recent test pass count, project sizes,
# disk usage of checkpoints. Useful as a daily heartbeat check.

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Quick status snapshot - $(date +'%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Training
echo
echo "--- v4 RL Training ---"
LOG=training_logs/phase1_v4.log
if [ -f "$LOG" ]; then
    LATEST=$(tail -1 "$LOG")
    echo "  Latest log line: $LATEST"
    BATCHES=$(grep -c "Batch [0-9]\+: gen=" "$LOG" 2>/dev/null || echo 0)
    echo "  Completed batches: $BATCHES"
    if [ "$BATCHES" -gt 0 ]; then
        FIRST_LOSS=$(grep -m 1 "Batch [0-9]\+: gen=" "$LOG" | grep -oE "loss=[0-9.]+" | head -1 | cut -d= -f2)
        LAST_LOSS=$(grep "Batch [0-9]\+: gen=" "$LOG" | tail -1 | grep -oE "loss=[0-9.]+" | cut -d= -f2)
        echo "  Loss trajectory: $FIRST_LOSS -> $LAST_LOSS"
    fi
else
    echo "  No training log at $LOG"
fi

# Phase 1 process check
PID_LINE=$(pgrep -af "scripts/train_phase1_v4" | head -1)
if [ -n "$PID_LINE" ]; then
    PID=$(echo "$PID_LINE" | awk '{print $1}')
    ELAPSED=$(ps -p "$PID" -o etime= 2>/dev/null | tr -d ' ')
    echo "  Phase 1 process: pid=$PID, elapsed=$ELAPSED"
else
    echo "  Phase 1: NOT RUNNING (finished or crashed)"

    # Phase 2 check
    P2_LINE=$(pgrep -af "scripts/train_phase2_v4" | head -1)
    if [ -n "$P2_LINE" ]; then
        P2_PID=$(echo "$P2_LINE" | awk '{print $1}')
        P2_ELAPSED=$(ps -p "$P2_PID" -o etime= 2>/dev/null | tr -d ' ')
        P2_LOG=training_logs/phase2_v4.log
        if [ -f "$P2_LOG" ]; then
            P2_ITERS=$(grep -c "Phase2-v4.*\[iter [0-9]" "$P2_LOG" 2>/dev/null || echo 0)
            P2_LATEST=$(tail -1 "$P2_LOG" 2>/dev/null)
            echo "  Phase 2 process: pid=$P2_PID, elapsed=$P2_ELAPSED"
            echo "  Phase 2 iters: $P2_ITERS"
            echo "  Phase 2 latest: $P2_LATEST"
        fi
    else
        echo "  Phase 2: NOT RUNNING"
    fi
fi

# Load
echo
echo "--- System Load ---"
uptime

# Project sizes
echo
echo "--- Project Sizes ---"
for d in flagship_coalition_mcts decomposed_mcts equivariant_net; do
    if [ -d "$d" ]; then
        total=$(find "$d" -type f \( -name "*.py" -o -name "*.md" \) -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
        files=$(find "$d" -type f \( -name "*.py" -o -name "*.md" \) | wc -l)
        py_count=$(find "$d" -name "*.py" | wc -l)
        echo "  $d: $total LoC ($files files, $py_count python)"
    fi
done

# Test counts (count, don't run)
echo
echo "--- Test File Counts ---"
for d in flagship_coalition_mcts decomposed_mcts equivariant_net; do
    if [ -d "$d/tests" ]; then
        n=$(find "$d/tests" -name "test_*.py" | wc -l)
        echo "  $d/tests: $n test files"
    fi
done

# Checkpoint sizes
echo
echo "--- Checkpoint Disk Usage ---"
for d in checkpoints checkpoints_v4 cdmcts_cc_checkpoints cmaz_cc_checkpoints wreath_cc_checkpoints; do
    if [ -d "$d" ]; then
        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        n=$(find "$d" -name "*.pt" 2>/dev/null | wc -l)
        echo "  $d: $size ($n .pt files)"
    fi
done
[ ! -d "checkpoints" ] && [ ! -d "checkpoints_v4" ] && echo "  (no checkpoint directories yet)"

echo
echo "============================================================"
echo "Status snapshot complete."
echo "============================================================"
