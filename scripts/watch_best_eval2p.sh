#!/usr/bin/env bash
# Watches phase2_v4.log for eval2p signals; when current eval2p exceeds the
# saved best, copies phase2_latest_v4.pt → phase2_best_by_eval2p.pt with a
# stamp file. Lightweight, runs in background, doesn't touch the trainer.
set -e
cd "$(dirname "$0")/.."

LOG=training_logs/phase2_v4.log
CKPT_DIR=checkpoints_v4
LATEST="$CKPT_DIR/phase2_latest_v4.pt"
BEST="$CKPT_DIR/phase2_best_by_eval2p.pt"
STAMP="$CKPT_DIR/phase2_best_by_eval2p.stamp"

# Initialize: if no stamp, accept the next eval2p as the starting best.
if [ ! -f "$STAMP" ]; then
    echo "0" > "$STAMP"
fi

best=$(cat "$STAMP")

# Tail log; for each new [Phase2-v4] iter line that contains eval2p=, parse and compare.
tail -F -n 0 "$LOG" 2>/dev/null | while read -r line; do
    if [[ "$line" =~ eval2p=([0-9]+\.[0-9]+) ]]; then
        cur="${BASH_REMATCH[1]}"
        # Compare as floats (bash can't do floats; use awk)
        better=$(awk -v c="$cur" -v b="$best" 'BEGIN{print (c+0 > b+0) ? 1 : 0}')
        if [ "$better" = "1" ]; then
            iter_seen=$(echo "$line" | grep -oE 'iter=[0-9]+' | head -1)
            best="$cur"
            echo "$best" > "$STAMP"
            cp "$LATEST" "$BEST"
            echo "[best-by-eval2p] $iter_seen NEW BEST eval2p=$cur (saved $BEST)"
        fi
    fi
done
