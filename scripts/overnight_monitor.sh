#!/bin/bash
# Overnight monitor - periodic status snapshot every 20 minutes.
# Output: training_logs/overnight_monitor.log
LOG=/home/coder/nexus/training_logs/overnight_monitor.log
NEXUS=/home/coder/nexus

snap() {
    echo "================================================================"
    echo "=== $(date) ==="
    echo "================================================================"
    echo
    echo "--- Phase 1 process ---"
    if pgrep -f "train_phase1.py" > /dev/null; then
        ETIME=$(ps -o etime= -p $(pgrep -f "train_phase1.py" | head -1) 2>/dev/null | tr -d ' ')
        echo "ALIVE  elapsed=$ETIME"
        tail -3 $NEXUS/training_logs/phase1_console.log 2>/dev/null
    else
        echo "EXITED"
    fi
    echo
    echo "--- Watchdog ---"
    if pgrep -f "watchdog_phase2_launch" > /dev/null; then
        echo "ALIVE  (waiting for Phase 1 or just launched Phase 2)"
    else
        echo "EXITED"
    fi
    if [ -f $NEXUS/training_logs/watchdog.log ]; then
        echo "  watchdog.log:"
        sed 's/^/    /' $NEXUS/training_logs/watchdog.log
    fi
    echo
    echo "--- Phase 2 process ---"
    if pgrep -f "train_phase2.py" > /dev/null; then
        ETIME=$(ps -o etime= -p $(pgrep -f "train_phase2.py" | head -1) 2>/dev/null | tr -d ' ')
        echo "ALIVE  elapsed=$ETIME"
        if [ -f $NEXUS/training_logs/phase2_final/status.txt ]; then
            echo "  status.txt:"
            sed 's/^/    /' $NEXUS/training_logs/phase2_final/status.txt
        elif [ -f $NEXUS/training_logs/latest/status.txt ]; then
            echo "  latest status.txt:"
            sed 's/^/    /' $NEXUS/training_logs/latest/status.txt
        fi
    else
        echo "NOT RUNNING (yet)"
    fi
    echo
    echo "--- Checkpoints ---"
    ls -la $NEXUS/checkpoints_v2/*.pt 2>/dev/null | awk '{print "  "$5" "$6" "$7" "$8" "$9}'
    echo
    echo "--- Halt flags? ---"
    find $NEXUS/training_logs -name "RULE_ALIGNMENT_FAILED" 2>/dev/null
    echo
}

# Initial snapshot
snap >> $LOG

# Every 20 minutes
while true; do
    sleep 1200
    snap >> $LOG
done
