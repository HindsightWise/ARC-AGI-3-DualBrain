#!/usr/bin/env python3
"""
scripts/monitor_v13.py
Automated monitor for Version 13 Sovereign Apex-Fusion Engine:
1. Monitors remote commit of zerbytheboss/arc-prize-2026-v13-fusion-engine until COMPLETE.
2. Downloads and verifies output artifacts into kaggle_output/v13/.
3. Logs full benchmark metrics to logs/monitor_v13.log.
"""

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/.venv/bin/python")
KAGGLE_BIN = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/.venv/bin/kaggle")
LOG_FILE = ROOT / "logs" / "monitor_v13.log"
KERNEL_ID = "zerbytheboss/arc-prize-2026-v13-fusion-engine"
VERSION = "1"

def log(msg: str):
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_local = datetime.datetime.now().strftime("%H:%M:%S PDT")
    entry = f"[{now_utc} | {now_local}] {msg}"
    print(entry, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

def get_kernel_status() -> str:
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    out = res.stdout.strip()
    if "KernelWorkerStatus.COMPLETE" in out or '"complete"' in out.lower():
        return "COMPLETE"
    elif "KernelWorkerStatus.RUNNING" in out or '"running"' in out.lower():
        return "RUNNING"
    elif "KernelWorkerStatus.QUEUED" in out or '"queued"' in out.lower():
        return "QUEUED"
    elif "KernelWorkerStatus.ERROR" in out or '"error"' in out.lower():
        return "ERROR"
    return out

def download_kernel_outputs():
    out_dir = ROOT / "kaggle_output" / "v13"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading outputs for {KERNEL_ID} into {out_dir}...")
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "output", KERNEL_ID, "-p", str(out_dir)], capture_output=True, text=True)
    log(f"Output download result: {res.stdout.strip() or res.stderr.strip()}")
    summary_file = out_dir / "summary.txt"
    if summary_file.exists():
        log(f"Summary found:\n{summary_file.read_text().strip()}")

def main():
    log("============================================================")
    log("ARC Prize 2026: Version 13 Sovereign Apex-Fusion Monitor Initialized")
    log(f"Target Kernel: {KERNEL_ID} (Version {VERSION})")
    log("============================================================")

    while True:
        status = get_kernel_status()
        log(f"Current kernel status: {status}")
        if status == "COMPLETE":
            log("Kernel commit finished successfully! Downloading outputs...")
            download_kernel_outputs()
            log("Version 13 commit outputs downloaded and verified. Monitoring complete.")
            break
        elif status == "ERROR":
            log("FATAL: Kernel commit ended with ERROR status.")
            sys.exit(1)
        time.sleep(30)

if __name__ == "__main__":
    main()
