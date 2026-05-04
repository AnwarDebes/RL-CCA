#!/usr/bin/env bash
# dashboard.sh - live dashboard view (refreshes every 30s).
#
# Run as: watch -n 30 ./scripts/dashboard.sh
# (or just call ./scripts/dashboard.sh manually for one-off look)

cd "$(dirname "$0")/.."

VENV=./venv/bin/python

clear 2>/dev/null
echo "════════════════════════════════════════════════════════════"
echo "  NEXUS - Live Dashboard - $(date +'%H:%M:%S')"
echo "════════════════════════════════════════════════════════════"

# === v4 RL Training ===
echo
echo "─── v4 RL TRAINING ───"
P1_LINE=$(pgrep -af "scripts/train_phase1_v4" | head -1)
P2_LINE=$(pgrep -af "scripts/train_phase2_v4" | head -1)
if [ -n "$P1_LINE" ]; then
    P1_PID=$(echo "$P1_LINE" | awk '{print $1}')
    P1_E=$(ps -p "$P1_PID" -o etime= 2>/dev/null | tr -d ' ')
    echo "  Phase 1: RUNNING (pid=$P1_PID, elapsed=$P1_E)"
    tail -2 training_logs/phase1_v4.log 2>/dev/null | sed 's/^/    /'
elif [ -n "$P2_LINE" ]; then
    P2_PID=$(echo "$P2_LINE" | awk '{print $1}')
    P2_E=$(ps -p "$P2_PID" -o etime= 2>/dev/null | tr -d ' ')
    P2_ITERS=$(grep -c "\[iter [0-9]" training_logs/phase2_v4.log 2>/dev/null || echo 0)
    echo "  Phase 2: RUNNING (pid=$P2_PID, elapsed=$P2_E, iters=$P2_ITERS/250)"
    tail -3 training_logs/phase2_v4.log 2>/dev/null | sed 's/^/    /'
else
    echo "  No training process running."
    if [ -f training_logs/phase2_v4.log ]; then
        ITERS=$(grep -c "\[iter [0-9]" training_logs/phase2_v4.log 2>/dev/null)
        echo "  Last Phase 2 log line:"
        tail -1 training_logs/phase2_v4.log 2>/dev/null | sed 's/^/    /'
        echo "  Total Phase 2 iters reached: $ITERS"
    fi
fi

# === GPU ===
echo
echo "─── GPU ───"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free \
               --format=csv,noheader 2>&1 | head -3 | sed 's/^/  /'
fi

# === Load ===
echo
echo "─── SYSTEM ───"
uptime | sed 's/^/  /'

# === Checkpoints ===
echo
echo "─── CHECKPOINTS ───"
for d in checkpoints_v4 checkpoints/cdmcts_cc_seed0 checkpoints/cmaz_cc_seed0 checkpoints/wreath_cc_seed0; do
    if [ -d "$d" ]; then
        sz=$(du -sh "$d" 2>/dev/null | cut -f1)
        n=$(find "$d" -name "*.pt" 2>/dev/null | wc -l)
        echo "  $d: $sz ($n .pt files)"
    fi
done

echo
echo "════════════════════════════════════════════════════════════"
