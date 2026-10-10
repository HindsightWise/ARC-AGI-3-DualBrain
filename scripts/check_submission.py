#!/usr/bin/env python3
"""
Checks the current status of ARC Prize 2026 submission 56688161.
Outputs status line and checks for state transitions.
"""
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter")
KAGGLE_BIN = ROOT / ".venv" / "bin" / "kaggle"
STATE_FILE = ROOT / "submission_state.json"
TARGET_REF = "56688161"

def main():
    res = subprocess.run([str(KAGGLE_BIN), "competitions", "submissions", "arc-prize-2026-arc-agi-3"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR: {res.stderr}")
        sys.exit(1)
        
    lines = res.stdout.strip().splitlines()
    target_row = None
    for line in lines:
        if TARGET_REF in line:
            target_row = line
            break
            
    if not target_row:
        print(f"Submission {TARGET_REF} not found in output.")
        sys.exit(1)
        
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {target_row}")
    
    # Check state delta
    prev_state = ""
    if STATE_FILE.is_file():
        try:
            prev_state = json.loads(STATE_FILE.read_text()).get("state", "")
        except Exception:
            pass
            
    current_state = "PENDING"
    if "COMPLETE" in target_row:
        current_state = "COMPLETE"
    elif "ERROR" in target_row:
        current_state = "ERROR"
    elif "RUNNING" in target_row:
        current_state = "RUNNING"
        
    STATE_FILE.write_text(json.dumps({"state": current_state, "row": target_row}))
    
    if prev_state and prev_state != current_state:
        print(f"STATE_TRANSITION: {prev_state} -> {current_state}")

if __name__ == "__main__":
    main()
