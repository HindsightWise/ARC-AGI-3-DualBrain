#!/usr/bin/env python3
"""
scripts/auto_submit_v16.py
Automated deployment pipeline for Version 16 Sovereign Apex Engine:
1. Pushes zerbytheboss/arc-prize-2026-v16-apex-engine if needed.
2. Monitors remote commit until KernelWorkerStatus.COMPLETE.
3. Checks submission allowance (numAllowedNow).
4. Submits Version 16 to arc-prize-2026-arc-agi-3 competition immediately if allowance > 0,
   or sleeps until 00:00:05 UTC reset.
5. Verifies registration and tracks evaluation.
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
V16_DIR = ROOT / "v16_apex"
LOG_FILE = ROOT / "logs" / "auto_submit_v16.log"
LOCK_FILE = ROOT / ".auto_submit_v16.lock"
KERNEL_ID = "zerbytheboss/arc-prize-2026-v16-apex-engine"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
VERSION = "1"
SUBMISSION_MESSAGE = "v16 - Sovereign Apex Engine (Clean Baseline Restoration + Image Eviction + Undo/Reset Affordances + SGLang Pennyroyal + Albucino MTP-3)"

def log(msg: str):
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_local = datetime.datetime.now().strftime("%H:%M:%S PDT")
    entry = f"[{now_utc} | {now_local}] {msg}"
    print(entry, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

def get_submission_limits() -> dict:
    code = f"""
from kaggle.api.kaggle_api_extended import KaggleApi
import json
api = KaggleApi()
api.authenticate()
lims = api.competition_get_submission_limits("{COMPETITION_ID}")
print(json.dumps({{
    "numToday": getattr(lims, "num_today", 0),
    "numTotal": getattr(lims, "num_total", 0),
    "numAllowedNow": getattr(lims, "num_allowed_now", 0)
}}))
"""
    res = subprocess.run([str(PYTHON_BIN), "-c", code], capture_output=True, text=True)
    try:
        return json.loads(res.stdout.strip())
    except Exception:
        return {}

def ensure_kernel_pushed():
    status_res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    out = status_res.stdout.strip()
    if "404" in status_res.stderr or "not found" in out.lower() or not out:
        log(f"Kernel {KERNEL_ID} not found on Kaggle. Pushing from {V16_DIR}...")
        push_res = subprocess.run([str(KAGGLE_BIN), "kernels", "push", "-p", str(V16_DIR)], capture_output=True, text=True)
        log(f"Push output: {push_res.stdout.strip()} {push_res.stderr.strip()}")
    else:
        log(f"Kernel {KERNEL_ID} already registered on Kaggle: {out}")

def wait_for_kernel_complete():
    log(f"Monitoring {KERNEL_ID} until completion...")
    polls = 0
    while True:
        polls += 1
        res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
        out = res.stdout.strip()
        if "KernelWorkerStatus.COMPLETE" in out:
            log(f"SUCCESS: Kernel is COMPLETE: {out}")
            break
        elif "KernelWorkerStatus.ERROR" in out or "KernelWorkerStatus.CANCELLED" in out:
            log(f"ERROR: Kernel commit failed: {out}")
            sys.exit(1)
        else:
            if polls % 4 == 1:
                log(f"Kernel commit in progress: {out} (poll #{polls})")
            time.sleep(15)

def submit_to_competition() -> int:
    log(f"Deploying Version {VERSION} to {COMPETITION_ID}...")
    cmd = [
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
    log(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    log(f"CLI stdout: {res.stdout.strip()}")
    if res.stderr.strip():
        log(f"CLI stderr: {res.stderr.strip()}")

    # Query latest registered submission ref
    time.sleep(5)
    code = f"""
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("{COMPETITION_ID}")
if subs:
    latest = subs[0]
    print(f"{{latest.ref}}|{{latest.status}}|{{latest.description}}")
"""
    check_res = subprocess.run([str(PYTHON_BIN), "-c", code], capture_output=True, text=True)
    out = check_res.stdout.strip()
    log(f"Latest submission on Kaggle: {out}")
    if "|" in out:
        try:
            return int(out.splitlines()[-1].split("|")[0])
        except ValueError:
            pass
    return 0

def monitor_evaluation(ref: int):
    log(f"Tracking competition evaluation for ref {ref}...")
    start_time = time.time()
    last_status = ""
    cycles = 0
    while True:
        cycles += 1
        code = f"""
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("{COMPETITION_ID}")
for s in subs:
    if s.ref == {ref}:
        print(f"{{s.ref}}|{{s.status}}|{{s.public_score}}|{{s.date}}")
        break
"""
        res = subprocess.run([str(PYTHON_BIN), "-c", code], capture_output=True, text=True)
        out = res.stdout.strip()
        elapsed_min = (time.time() - start_time) / 60.0

        if "|" in out:
            parts = out.splitlines()[-1].split("|")
            status = parts[1]
            score = parts[2]
            if status != last_status:
                log(f"Evaluation status transition: {last_status or 'INITIAL'} -> {status} (T+{elapsed_min:.1f}m, Score: {score})")
                last_status = status
            elif cycles % 10 == 0:
                log(f"Evaluation heartbeat: {status} (T+{elapsed_min:.1f}m, Cycles: {cycles})")

            if "COMPLETE" in status or "ERROR" in status:
                log(f"Evaluation terminated with status {status}. Final Score: {score}")
                break
        time.sleep(30)

def main():
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)
            log(f"Another auto-submit process is already active (PID {pid}). Exiting.")
            return
        except (ProcessLookupError, ValueError):
            log("Stale lock found. Overriding.")
    LOCK_FILE.write_text(str(os.getpid()))

    try:
        ensure_kernel_pushed()
        wait_for_kernel_complete()

        # Check allowance
        while True:
            lims = get_submission_limits()
            log(f"Submission limits check: {lims}")
            allowed = lims.get("numAllowedNow", 0)
            if allowed > 0:
                log(f"Submission allowance available ({allowed} remaining). Proceeding immediately!")
                break
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            midnight = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
            wait_s = max(5, int((midnight - now_utc).total_seconds()))
            log(f"Quota consumed. Sleeping {wait_s}s until 00:00:05 UTC reset...")
            time.sleep(min(wait_s, 300))

        ref = submit_to_competition()
        if ref > 0:
            monitor_evaluation(ref)
    finally:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

if __name__ == "__main__":
    main()
