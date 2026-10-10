#!/usr/bin/env python3
"""
Delayed submission runner for ARC Prize 2026 Version 6.
Target time: 2026-09-07 00:01:00 UTC (17:01:00 PDT on Sep 6).
Runs until target time, then triggers scripts/auto_submit_v6.sh with retry logic.
"""
import datetime
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter")
LOG_FILE = ROOT / "auto_submit_v6.log"
SUBMIT_SCRIPT = ROOT / "scripts" / "auto_submit_v6.sh"

TARGET_UTC = datetime.datetime(2026, 9, 7, 0, 1, 0, tzinfo=datetime.timezone.utc)

def log(msg: str):
    now_local = datetime.datetime.now()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    entry = f"[{now_local.strftime('%Y-%m-%d %H:%M:%S')} PDT | {now_utc.strftime('%H:%M:%S')} UTC] {msg}"
    print(entry, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

def main():
    log(f"Delayed submit daemon for Version 6 started. Target time: {TARGET_UTC.isoformat()} (17:01 PDT).")
    
    while True:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        remaining = (TARGET_UTC - now_utc).total_seconds()
        
        if remaining <= 0:
            log("TARGET TIME REACHED! Daily Kaggle quota for 2026-09-07 UTC is now active.")
            break
        
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        secs = int(remaining % 60)
        
        if remaining > 3600:
            sleep_duration = 1800  # log every 30 minutes
        elif remaining > 300:
            sleep_duration = 60    # log every minute
        else:
            sleep_duration = 10    # log every 10 seconds in final stretch
            
        log(f"Countdown: {hours:02d}h {minutes:02d}m {secs:02d}s remaining until daily reset.")
        time.sleep(sleep_duration)

    # Execute submission with retry logic
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        log(f"Executing submission attempt {attempt}/{max_retries}...")
        try:
            res = subprocess.run(["/bin/bash", str(SUBMIT_SCRIPT)], capture_output=True, text=True, timeout=120)
            log(f"Script stdout: {res.stdout.strip()}")
            if res.stderr.strip():
                log(f"Script stderr: {res.stderr.strip()}")
            if res.returncode == 0 and "400 Client Error" not in res.stderr:
                log(f"SUCCESS: Version 6 successfully submitted to Kaggle on attempt {attempt}!")
                break
            else:
                log(f"Attempt {attempt} returned code {res.returncode}. Retrying in 60 seconds...")
                time.sleep(60)
        except Exception as e:
            log(f"Attempt {attempt} exception: {e}. Retrying in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    main()
