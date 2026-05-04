#!/bin/bash
# Watchdog: waits for Phase 1 to finish, smoke-tests phase1.pt, launches Phase 2.
#
# Phase 1 completion is detected by:
#   - phase1.pt exists and is not being written
#   - train_phase1 process is no longer running
#
# Output: training_logs/watchdog.log

LOG=/home/coder/nexus/training_logs/watchdog.log
exec >> "$LOG" 2>&1
echo "=== watchdog started at $(date) ==="

PHASE1_PT=/home/coder/nexus/checkpoints_v2/phase1.pt
NEXUS=/home/coder/nexus

# 1. Wait for Phase 1 process to exit
echo "[watchdog] waiting for Phase 1 process to exit"
while pgrep -f "train_phase1.py" > /dev/null; do
    sleep 30
done
echo "[watchdog] Phase 1 process gone at $(date)"

# 2. Confirm phase1.pt exists
if [ ! -f "$PHASE1_PT" ]; then
    echo "[watchdog] FATAL: $PHASE1_PT does not exist"
    exit 1
fi
echo "[watchdog] phase1.pt size: $(stat -c %s $PHASE1_PT) bytes"

# 3. Smoke test: does phase1.pt win a 2-player greedy game?
cd $NEXUS
echo "[watchdog] running smoke test..."
$NEXUS/venv/bin/python -c "
import sys, torch
sys.path.insert(0, '.')
from core.board import HexBoard
from core.game_env import GameEnv
from network.model import NexusNet

board = HexBoard()
device = torch.device('cuda')
net = NexusNet.load('$PHASE1_PT', 'cuda')
net.eval()

env = GameEnv(board, num_players=2)
env.reset()
moves = 0
while not env.is_done() and moves < 1000:
    p = env.current_player
    state = env.get_state_tensor(p).unsqueeze(0).to(device)
    mask = env.get_legal_mask(p).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(state, mask)
    action = int(out['policy'][0].argmax())
    env.step(action)
    moves += 1

if env.winner is None:
    print(f'[watchdog] SMOKE FAIL: greedy game did not finish in {moves} moves')
    sys.exit(1)
else:
    print(f'[watchdog] SMOKE OK: greedy game finished in {moves} moves, winner={env.winner}')
"
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
    echo "[watchdog] smoke test failed; not launching Phase 2"
    exit 1
fi

# 4. Launch Phase 2
echo "[watchdog] launching Phase 2 at $(date)"
cd $NEXUS
nohup $NEXUS/venv/bin/python scripts/train_phase2.py \
    --resume checkpoints_v2/phase1.pt \
    --iterations 600 \
    --start-iter 0 \
    --run-name phase2_final \
    > $NEXUS/training_logs/phase2_console.log 2>&1 &
PHASE2_PID=$!
echo "[watchdog] Phase 2 launched, PID=$PHASE2_PID"

# 5. Wait for first iter to confirm it started
sleep 60
if grep -q '"iter": 0' $NEXUS/training_logs/phase2_final/iter_metrics.jsonl 2>/dev/null; then
    echo "[watchdog] Phase 2 iter 0 logged successfully at $(date)"
else
    echo "[watchdog] WARNING: Phase 2 iter 0 not yet logged after 60s"
fi

echo "=== watchdog finished at $(date) ==="
