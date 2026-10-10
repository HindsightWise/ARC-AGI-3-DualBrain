#!/bin/bash
set -e

DIR="/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter"
cd "$DIR"

LOG="$DIR/auto_submit_v7.log"
echo "========================================================" >> "$LOG"
echo "Auto-Submit V7 Triggered at: $(date) (UTC: $(date -u))" >> "$LOG"

if [ ! -s .kaggle/access_token ]; then
    echo "ERROR: .kaggle/access_token missing or empty!" >> "$LOG"
    exit 1
fi

TOKEN=$(cat .kaggle/access_token)

echo "Submitting Version 7 to Kaggle arc-prize-2026-arc-agi-3..." >> "$LOG"
KAGGLE_API_TOKEN="$TOKEN" "$DIR/.venv/bin/kaggle" competitions submit \
    -c arc-prize-2026-arc-agi-3 \
    -k zerbytheboss/arc-prize-2026-arc-agi-3-starter \
    -v 7 \
    -f submission.parquet \
    -m "v7 - Dual-Brain Tier 0 Verified Anchor + Repaired Universal Dynamic Solvers (Zero ACTION5 Gate Blockage, Hardened Avatar Deduction, 4.07+ Local RHAE)" >> "$LOG" 2>&1

echo "Submission command finished. Fetching latest submissions status..." >> "$LOG"
KAGGLE_API_TOKEN="$TOKEN" "$DIR/.venv/bin/kaggle" competitions submissions -c arc-prize-2026-arc-agi-3 >> "$LOG" 2>&1

echo "SUCCESS: Version 7 auto-submit completed successfully at $(date)!" >> "$LOG"

# Send macOS desktop notification
osascript -e 'display notification "ARC-AGI-3 Version 7 successfully submitted to Kaggle!" with title "Kaggle Auto-Submit"' 2>/dev/null || true
