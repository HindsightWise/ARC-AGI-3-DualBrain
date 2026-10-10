#!/usr/bin/env python3
"""
scripts/auto_submit_v10.py
Automated pipeline for Version 10 of ARC Prize 2026:
1. Monitors the commit run of Version 10 until KernelWorkerStatus.COMPLETE.
2. Downloads and verifies output artifacts (submission.parquet) in kaggle_output/v10/.
3. Checks competition submission limit allowance.
4. If allowance is 0, waits precisely until 00:00:05 UTC reset.
5. Deploys Version 10 to arc-prize-2026-arc-agi-3 competition.
6. Verifies submission status and records ledger.
"""

import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_BIN = ROOT / ".venv" / "bin" / "kaggle"
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"
LOG_FILE = ROOT / "auto_submit_v10.log"
KERNEL_ID = "zerbytheboss/arc-prize-2026-arc-agi-3-starter"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
VERSION = "10"
SUBMISSION_MESSAGE = "v10 - Attention Lottery + 6.4GiB KV + 25% Prompt Compression"

LOCK_FILE = ROOT / ".auto_submit.lock"

def log(msg: str):
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_local = datetime.datetime.now().strftime("%H:%M:%S PDT")
    entry = f"[{now_utc} | {now_local}] {msg}"
    print(entry, flush=True)

def get_kernel_status() -> str:
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    return res.stdout.strip()

def get_submission_limits() -> dict:
    code = """
from kaggle.api.kaggle_api_extended import KaggleApi
import json
api = KaggleApi()
api.authenticate()
lims = api.competition_get_submission_limits("arc-prize-2026-arc-agi-3")
print(json.dumps({
    "numToday": getattr(lims, "num_today", 0),
    "numTotal": getattr(lims, "num_total", 0),
    "numAllowedNow": getattr(lims, "num_allowed_now", 0)
}))
"""
    res = subprocess.run([str(PYTHON_BIN), "-c", code], capture_output=True, text=True)
    try:
        import json
        return json.loads(res.stdout.strip())
    except Exception:
        return {}

def main():
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)
            log(f"Another auto-submit process is already running (PID {pid}). Exiting cleanly.")
            return
        except (ProcessLookupError, ValueError):
            log(f"Stale lock file found for PID. Taking over lock.")
    LOCK_FILE.write_text(str(os.getpid()))

    try:
        _run_pipeline()
    finally:
        try:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
        except Exception:
            pass

def _run_pipeline():
    log(f"Auto-submit pipeline initialized for {KERNEL_ID} -> Version {VERSION}")
    
    # Phase 1: Monitor Kernel Commit Run
    poll_count = 0
    while True:
        status_line = get_kernel_status()
        poll_count += 1
        
        if "KernelWorkerStatus.COMPLETE" in status_line:
            log(f"Kernel commit completed successfully! Status: {status_line}")
            break
        elif "KernelWorkerStatus.ERROR" in status_line or "KernelWorkerStatus.CANCELLED" in status_line:
            log(f"FATAL: Kernel commit failed! Status: {status_line}")
            out_dir = ROOT / "kaggle_output" / "v10"
            out_dir.mkdir(parents=True, exist_ok=True)
            res = subprocess.run([str(KAGGLE_BIN), "kernels", "output", KERNEL_ID, "-p", str(out_dir)], capture_output=True, text=True)
            log(f"Saved failure logs to {out_dir}. Stderr: {res.stderr}")
            sys.exit(1)
        else:
            if poll_count % 3 == 1:
                log(f"Kernel commit in progress: {status_line} (poll #{poll_count})")
            time.sleep(20)

    # Phase 2: Download and inspect outputs
    out_dir = ROOT / "kaggle_output" / "v10"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading Version {VERSION} outputs to {out_dir}...")
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "output", KERNEL_ID, "-p", str(out_dir)], capture_output=True, text=True)
    log(f"Output download: {res.stdout.strip()}")
    
    parquet_path = out_dir / "submission.parquet"
    if parquet_path.exists():
        log(f"VERIFIED: submission.parquet exists ({parquet_path.stat().st_size} bytes)")
    else:
        log("WARNING: submission.parquet not found in local output download; continuing to submission check.")

    # Phase 3: Check submission limits and wait for UTC midnight reset if needed
    while True:
        limits = get_submission_limits()
        log(f"Submission limits: {limits}")
        num_allowed = limits.get("numAllowedNow", 0)
        
        if num_allowed > 0:
            log(f"Submission allowance available ({num_allowed} allowed)! Proceeding immediately.")
            break
            
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        # Next midnight UTC
        midnight_utc = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        wait_seconds = max(5, int((midnight_utc - now_utc).total_seconds()))
        log(f"Daily submission limit consumed for today. Reset occurs at 00:00:05 UTC (in {wait_seconds}s / {wait_seconds/60:.1f}m). Sleeping...")
        
        # Sleep in chunks to allow responsive logging
        sleep_chunk = min(wait_seconds, 300)
        time.sleep(sleep_chunk)

    # Phase 4: Submit Version 10 to Competition
    log(f"Deploying Version {VERSION} to competition {COMPETITION_ID}...")
    submit_cmd = [
        str(KAGGLE_BIN),
        "competitions",
        "submit",
        "-c",
        COMPETITION_ID,
        "-k",
        KERNEL_ID,
        "-v",
        VERSION,
        "-f",
        "submission.parquet",
        "-m",
        SUBMISSION_MESSAGE,
    ]
    
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        log(f"Submitting (attempt {attempt}/{max_retries})...")
        res = subprocess.run(submit_cmd, capture_output=True, text=True)
        log(f"Submit stdout: {res.stdout.strip()}")
        if res.stderr.strip():
            log(f"Submit stderr: {res.stderr.strip()}")
            
        if res.returncode == 0 and "400 Client Error" not in res.stderr and "403" not in res.stderr:
            log(f"SUCCESS: Version {VERSION} submitted to competition!")
            break
        else:
            log(f"Attempt {attempt} encountered issue. Retrying in 15 seconds...")
            time.sleep(15)

    # Phase 5: Fetch latest submission ledger
    time.sleep(5)
    log("Fetching latest competition submissions...")
    sub_res = subprocess.run([str(KAGGLE_BIN), "competitions", "submissions", COMPETITION_ID], capture_output=True, text=True)
    log(f"Submissions ledger:\n{sub_res.stdout.strip()}")
    log("Pipeline finished.")

if __name__ == "__main__":
    main()
