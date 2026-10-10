#!/bin/bash
set -e

DIR="/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter"
cd "$DIR"

LOG="$DIR/auto_submit.log"
echo "========================================================" >> "$LOG"
echo "Auto-Submit Triggered at: $(date) (UTC: $(date -u))" >> "$LOG"

if [ ! -s .kaggle/access_token ]; then
    echo "ERROR: .kaggle/access_token missing or empty!" >> "$LOG"
    exit 1
fi

TOKEN=$(cat .kaggle/access_token)

echo "Submitting Version 4 to Kaggle arc-prize-2026-arc-agi-3..." >> "$LOG"
KAGGLE_API_TOKEN="$TOKEN" "$DIR/.venv/bin/kaggle" competitions submit \
    -c arc-prize-2026-arc-agi-3 \
    -k zerbytheboss/arc-prize-2026-arc-agi-3-starter \
    -v 4 \
    -f submission.parquet \
    -m "v4 - DualBrain Cognitive Flexibility (Dynamic Lattice + Perimeter HUD + Multi-Mode Dispatch)" >> "$LOG" 2>&1

echo "Submission command finished. Fetching latest submissions status..." >> "$LOG"
KAGGLE_API_TOKEN="$TOKEN" "$DIR/.venv/bin/kaggle" competitions submissions -c arc-prize-2026-arc-agi-3 >> "$LOG" 2>&1

echo "SUCCESS: Auto-submit completed successfully at $(date)!" >> "$LOG"

# Send macOS desktop notification
osascript -e 'display notification "ARC-AGI-3 Version 4 successfully submitted to Kaggle!" with title "Kaggle Auto-Submit"' 2>/dev/null || true
