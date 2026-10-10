#!/usr/bin/env python3
"""
Automated watcher and competition submitter for Version 8.
Monitors the commit run on Kaggle until KernelWorkerStatus.COMPLETE,
then triggers the official competition submission.
"""
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter")
KAGGLE_BIN = ROOT / ".venv" / "bin" / "kaggle"
LOG_FILE = ROOT / "auto_submit_v8.log"
KERNEL_ID = "zerbytheboss/arc-prize-2026-arc-agi-3-starter"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
VERSION = "8"
SUBMISSION_MESSAGE = "v8 - Dual-Brain TAAF Duck (Qwen3.8 Flash Next NVFP4 MTP-3, Prefix Caching, Full AgentFix Suite, 12x ARM P)"

def log(msg: str):
    now_local = datetime.datetime.now()
    entry = f"[{now_local.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(entry, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def get_kernel_status() -> str:
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    output = res.stdout.strip()
    return output

def main():
    log(f"Auto-submit watcher started for {KERNEL_ID} (Target Version {VERSION})")
    
    poll_count = 0
    while True:
        status_line = get_kernel_status()
        poll_count += 1
        
        if "KernelWorkerStatus.COMPLETE" in status_line:
            log(f"Kernel completed successfully! Status: {status_line}")
            break
        elif "KernelWorkerStatus.ERROR" in status_line or "KernelWorkerStatus.CANCELLED" in status_line:
            log(f"Kernel failed! Status: {status_line}")
            out_dir = ROOT / "kaggle_output" / "v8"
            out_dir.mkdir(parents=True, exist_ok=True)
            res = subprocess.run([str(KAGGLE_BIN), "kernels", "output", KERNEL_ID, "-p", str(out_dir)], capture_output=True, text=True)
            log(f"Kernel output logs saved to {out_dir}. Stderr: {res.stderr}")
            sys.exit(1)
        else:
            if poll_count % 2 == 1:
                log(f"Current status: {status_line} (poll #{poll_count})")
            time.sleep(20)

    # Trigger competition submission
    log(f"Submitting Version {VERSION} to competition {COMPETITION_ID}...")
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
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        log(f"Executing submission attempt {attempt}/{max_retries}...")
        res = subprocess.run(submit_cmd, capture_output=True, text=True)
        log(f"Submit stdout: {res.stdout.strip()}")
        if res.stderr.strip():
            log(f"Submit stderr: {res.stderr.strip()}")
            
        if res.returncode == 0 and "400 Client Error" not in res.stderr:
            log(f"SUCCESS: Version {VERSION} successfully submitted to competition on attempt {attempt}!")
            break
        else:
            log(f"Attempt {attempt} returned non-zero or error. Retrying in 15 seconds...")
            time.sleep(15)

    # Check competition submission queue
    log("Fetching latest competition submissions...")
    sub_res = subprocess.run([str(KAGGLE_BIN), "competitions", "submissions", COMPETITION_ID], capture_output=True, text=True)
    log(f"Submissions ledger:\n{sub_res.stdout.strip()}")

if __name__ == "__main__":
    main()
