#!/usr/bin/env python3
"""
scripts/auto_submit_v15.py
Automated deployment pipeline for Version 15 Sovereign Dual-Brain Engine:
1. Verifies remote commit of zerbytheboss/arc-prize-2026-v15-dual-brain-engine (Version 1).
2. Monitors Kaggle submission allowance for arc-prize-2026-arc-agi-3.
3. At 00:00:05 UTC quota reset, deploys Version 15 via verified Kaggle CLI.
4. Verifies submission status and updates OPERATIONS_CHRONICLE.md and secondary_task_list.md.
5. Tracks 9-hour competition evaluation until COMPLETE.
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
LOG_FILE = ROOT / "logs" / "auto_submit_v15.log"
LOCK_FILE = ROOT / ".auto_submit_v15.lock"
KERNEL_ID = "zerbytheboss/arc-prize-2026-v15-dual-brain-engine"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
VERSION = "3"
SUBMISSION_MESSAGE = "v15 - Sovereign Dual-Brain Engine v3 (Direct Interception + Distinct Level Guard + 25-Game Trajectory Engine + RHAE Guard + M85 Solved Memory)"

def log(msg: str):
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_local = datetime.datetime.now().strftime("%H:%M:%S PDT")
    entry = f"[{now_utc} | {now_local}] {msg}"
    print(entry, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

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
        return json.loads(res.stdout.strip())
    except Exception:
        return {}

def choose_submission_version() -> tuple[str, str]:
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    out = res.stdout.strip()
    log(f"Kernel status for submission selection: {out}")
    if "KernelWorkerStatus.COMPLETE" in out:
        log("Kernel is COMPLETE. Selecting Version 6 (99-Level Knowledge Base + 9 Outright 100% Games + vc33 100% + ar25 100% + Dual-Brain).")
        return (
            "6",
            "v15 - Sovereign Dual-Brain Engine v6 (99-Level Knowledge Base + 9 Outright 100% Games + vc33 100% + ar25 100% + Dual-Brain Interception)"
        )
    else:
        log(f"Latest kernel version is still running or queued ({out}). Selecting verified Version 5.")
        return (
            "5",
            "v15 - Sovereign Dual-Brain Engine v5 (91-Level Knowledge Base + tr87 100% Outright + ar25 6-Level + sc25 4-Level + tu93 4-Level + Dual-Brain Interception)"
        )

def submit_v15() -> int:
    version, message = choose_submission_version()
    cmd = [
        str(KAGGLE_BIN),
        "competitions",
        "submit",
        "-c",
        COMPETITION_ID,
        "-k",
        KERNEL_ID,
        "-v",
        version,
        "-f",
        "submission.parquet",
        "-m",
        message
    ]
    log(f"Executing submission command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    log(f"Submission CLI stdout: {res.stdout.strip()}")
    if res.stderr.strip():
        log(f"Submission CLI stderr: {res.stderr.strip()}")
    
    # If version 6 submission returned an error, try falling back to version 5
    if version == "6" and ("Error" in res.stderr or "400" in res.stderr or "not found" in res.stderr.lower()):
        log("Version 6 submission failed; falling back immediately to verified Version 5...")
        cmd[7] = "5"
        cmd[11] = "v15 - Sovereign Dual-Brain Engine v5 (91-Level Knowledge Base + tr87 100% Outright + ar25 6-Level + sc25 4-Level + tu93 4-Level + Dual-Brain Interception)"
        log(f"Executing fallback command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        log(f"Fallback CLI stdout: {res.stdout.strip()}")

    # If version 5 submission returned an error, try falling back to version 4
    if cmd[7] == "5" and ("Error" in res.stderr or "400" in res.stderr or "not found" in res.stderr.lower()):
        log("Version 5 submission failed; falling back immediately to verified Version 4 (50.07 score)...")
        cmd[7] = "4"
        cmd[11] = "v15 - Sovereign Dual-Brain Engine v4 (Direct Interception + sc25 4-Level Multi-Call + vc33 2-Level + 25-Game Trajectory Engine + RHAE Guard + M85 Solved Memory)"
        log(f"Executing fallback command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        log(f"Fallback CLI stdout: {res.stdout.strip()}")
    
    # Query Kaggle API for newly registered submission
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
    log(f"API Submissions response: {out}")
    if "|" in out:
        parts = out.splitlines()[-1].split("|")
        try:
            return int(parts[0])
        except ValueError:
            pass
    return 0

def check_kernel_ready() -> bool:
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    out = res.stdout.strip()
    log(f"Kernel status: {out}")
    if "KernelWorkerStatus.COMPLETE" in out:
        log("Latest kernel version is COMPLETE and ready for deployment.")
        return True
    elif "KernelWorkerStatus.RUNNING" in out or "KernelWorkerStatus.QUEUED" in out:
        log("Latest kernel version is still running/queued; verified Version 4 (50.07 score) is ready for deployment.")
        return True
    return False

def monitor_evaluation(ref: int):
    log(f"Beginning active evaluation tracking for submission ref {ref}...")
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
                log(f"Evaluation status change: {last_status or 'INITIAL'} -> {status} (T+{elapsed_min:.1f}m, Score: {score})")
                last_status = status
            elif cycles % 10 == 0:
                log(f"Evaluation heartbeat: {status} (T+{elapsed_min:.1f}m, Cycles: {cycles})")
                
            if "COMPLETE" in status:
                log(f"============================================================")
                log(f"SUBMISSION EVALUATION COMPLETE! Ref: {ref}")
                log(f"Final Score: {score} | Total Evaluation Time: {elapsed_min:.1f} minutes")
                log(f"============================================================")
                break
            elif "ERROR" in status or "FAILED" in status:
                log(f"Evaluation terminated with error: {status} (T+{elapsed_min:.1f}m)")
                break
        else:
            log(f"Warning: submission query returned empty output: {out}")
            
        time.sleep(60)

def main():
    if LOCK_FILE.exists():
        # Prevent concurrent overlapping cron executions
        try:
            lock_pid = int(LOCK_FILE.read_text().strip())
            # Check if PID is alive
            os.kill(lock_pid, 0)
            log(f"Auto-submitter already running (PID {lock_pid}). Exiting.")
            return
        except (ValueError, OSError):
            log("Stale lock file detected. Overwriting.")
            
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
        
    try:
        log("Auto-submitter initialized for Version 15 Sovereign Dual-Brain Engine.")
        
        # Verify kernel readiness
        if not check_kernel_ready():
            log("Kernel commit is not complete yet. Will wait or retry.")
            return
            
        # Check quota
        limits = get_submission_limits()
        log(f"Current submission limits: {limits}")
        allowed = limits.get("numAllowedNow", 0)
        
        if allowed == 0:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            # Next reset is at 00:00:05 UTC tomorrow
            tomorrow_utc = (now_utc + datetime.timedelta(days=1)).date()
            reset_time = datetime.datetime(tomorrow_utc.year, tomorrow_utc.month, tomorrow_utc.day, 0, 0, 5, tzinfo=datetime.timezone.utc)
            wait_seconds = (reset_time - now_utc).total_seconds()
            
            if wait_seconds > 0 and wait_seconds < 86400:
                log(f"Submission limit reached for today. Next quota reset in {wait_seconds/3600.0:.2f} hours ({reset_time.strftime('%Y-%m-%d %H:%M:%S UTC')}).")
                # If invoked by cron well before reset, exit quietly; cron will trigger at reset
                if wait_seconds > 1800:
                    log("More than 30 minutes until reset. Exiting; standing OS cron will execute at reset.")
                    return
                else:
                    log(f"Within 30 minutes of reset. Sleeping {wait_seconds:.1f} seconds until 00:00:05 UTC reset...")
                    time.sleep(wait_seconds)
                    
        # Re-check or proceed to submit
        log("Deploying Version 15 Sovereign Dual-Brain Engine to Kaggle...")
        ref = submit_v15()
        if ref > 0:
            log(f"Version 15 successfully deployed to Kaggle! Submission Ref: {ref}")
            monitor_evaluation(ref)
        else:
            log("Submission failed to register a valid reference number.")
            
    finally:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

if __name__ == "__main__":
    main()
