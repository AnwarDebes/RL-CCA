#!/usr/bin/env bash
# restart_phase2.sh - restart Phase 2 v4 from the latest checkpoint.
#
# Use this if Phase 2 crashes or is killed mid-training. Finds the
# highest-iter checkpoint and resumes from it.

set -e
cd "$(dirname "$0")/.."

VENV=./venv/bin/python
CHECKPOINT_DIR=checkpoints_v4
LOG=training_logs/phase2_v4.log

# Check that no Phase 2 is currently running
EXISTING=$(pgrep -af "scripts/train_phase2_v4" | head -1)
if [ -n "$EXISTING" ]; then
    EX_PID=$(echo "$EXISTING" | awk '{print $1}')
    echo "FAIL: a Phase 2 process is already running (pid=$EX_PID)."
    echo "Stop it first with:  kill $EX_PID"
    exit 1
fi

# Find the latest checkpoint
LATEST=$($VENV -c "
import os, re
d = '$CHECKPOINT_DIR'
best = None
best_n = -1
for f in os.listdir(d):
    if not f.endswith('.pt'):
        continue
    m = re.search(r'iter_(\d+)', f)
    if m:
        n = int(m.group(1))
        if n > best_n:
            best = os.path.join(d, f)
            best_n = n
# Fallback: if no iter_N.pt, use phase2_best_v4.pt or phase1_v4.pt
if best is None:
    for fname in ['phase2_best_v4.pt', 'phase1_v4.pt']:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            best = p
            best_n = 0
            break
print(f'{best}|{best_n}' if best else '|')
" 2>/dev/null)

CKPT=$(echo "$LATEST" | cut -d'|' -f1)
ITER=$(echo "$LATEST" | cut -d'|' -f2)

if [ -z "$CKPT" ]; then
    echo "FAIL: no checkpoint found in $CHECKPOINT_DIR"
    echo "Did Phase 1 complete? Run: make verify-phase1"
    exit 1
fi

echo "============================================================"
echo "Restart Phase 2 v4 from $CKPT (iter=$ITER)"
echo "============================================================"

# Append to existing log (don't overwrite); -u for unbuffered output.
nohup $VENV -u scripts/train_phase2_v4.py \
    --resume "$CKPT" \
    --start-iter "$ITER" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --run-name phase2_v4 \
    >> "$LOG" 2>&1 &

PID=$!
sleep 3

if ! kill -0 "$PID" 2>/dev/null; then
    echo "FAIL: restarted Phase 2 died within 3s. Check log:"
    tail -50 "$LOG"
    exit 1
fi

echo "Phase 2 v4 restarted: pid=$PID, log=$LOG, resumed from iter=$ITER"
echo
echo "Monitor with:"
echo "  ./scripts/dashboard.sh"
echo "  make phase2-eta"
