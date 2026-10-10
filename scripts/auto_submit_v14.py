#!/usr/bin/env python3
"""
scripts/auto_submit_v14.py
Automated deployment pipeline for Version 14 Sovereign Apex-Synthesis Engine:
1. Verifies remote commit of zerbytheboss/arc-prize-2026-v14-synthesis-engine (Version 1).
2. Monitors Kaggle submission allowance for arc-prize-2026-arc-agi-3.
3. At 00:00:05 UTC quota reset, deploys Version 14 via verified Kaggle CLI.
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
LOG_FILE = ROOT / "logs" / "auto_submit_v14.log"
LOCK_FILE = ROOT / ".auto_submit_v14.lock"
KERNEL_ID = "zerbytheboss/arc-prize-2026-v14-synthesis-engine"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
VERSION = "1"
SUBMISSION_MESSAGE = "v14 - Sovereign Apex-Synthesis Engine (Fast-Path Trajectories + M85 Solved Memory + RHAE Efficiency Governor + M59 Sticky WM)"

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

def submit_v14() -> int:
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
        SUBMISSION_MESSAGE
    ]
    log(f"Executing submission command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    log(f"Submission CLI stdout: {res.stdout.strip()}")
    if res.stderr.strip():
        log(f"Submission CLI stderr: {res.stderr.strip()}")
    
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

def check_status(ref: int) -> dict:
    code = f"""
from kaggle.api.kaggle_api_extended import KaggleApi
import json
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("{COMPETITION_ID}")
found = False
for s in subs:
    if s.ref == {ref}:
        print(json.dumps({{
            "ref": s.ref,
            "status": str(s.status),
            "score": str(s.public_score),
            "description": s.description
        }}))
        found = True
        break
if not found:
    print(json.dumps({{"ref": {ref}, "status": "UNKNOWN", "score": "None", "description": ""}}))
"""
    res = subprocess.run([str(PYTHON_BIN), "-c", code], capture_output=True, text=True)
    try:
        return json.loads(res.stdout.strip())
    except Exception:
        return {"ref": ref, "status": "ERROR", "score": "None"}

def update_chronicles(v14_ref: int):
    try:
        chronicle_path = ROOT / "OPERATIONS_CHRONICLE.md"
        if chronicle_path.exists():
            text = chronicle_path.read_text()
            now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            v14_row = f"| **{v14_ref}** | {now_str} | {SUBMISSION_MESSAGE} | **PENDING** | *Evaluating (9h run)* |\n"
            if "| **56784330**" in text and str(v14_ref) not in text:
                text = text.replace("| **56784330**", v14_row + "| **56784330**")
            chronicle_path.write_text(text)
            log("Updated OPERATIONS_CHRONICLE.md successfully.")
    except Exception as e:
        log(f"Failed to update chronicle: {e}")

def main():
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)
            log(f"Another auto_submit_v14 process is active (PID {pid}). Exiting.")
            return 0
        except (ProcessLookupError, ValueError):
            log("Stale lock file found. Overwriting lock.")
    LOCK_FILE.write_text(str(os.getpid()))

    try:
        _run_loop()
    finally:
        try:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
        except Exception:
            pass

def _run_loop():
    log("=" * 60)
    log("Version 14 Sovereign Apex-Synthesis Deployment Pipeline Initialized")
    log(f"Target Kernel: {KERNEL_ID} (Version {VERSION})")
    log("=" * 60)

    # 1. Verify remote commit completion
    log("Checking remote kernel status...")
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    status_line = res.stdout.strip()
    log(f"Remote Status: {status_line}")
    if "KernelWorkerStatus.COMPLETE" not in status_line:
        log("Waiting for remote commit...")
        while True:
            res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
            status_line = res.stdout.strip()
            log(f"Remote Status: {status_line}")
            if "KernelWorkerStatus.COMPLETE" in status_line:
                break
            elif "ERROR" in status_line:
                log(f"FATAL: Kernel commit failed: {status_line}")
                sys.exit(1)
            time.sleep(30)

    log("Remote commit is COMPLETE! Ready for competition submission.")

    # 2. Wait for submission quota
    while True:
        lims = get_submission_limits()
        log(f"Current limits: {lims}")
        num_allowed = lims.get("numAllowedNow", 0)
        if num_allowed > 0:
            log(f"Submission quota available ({num_allowed} allowed)! Triggering deployment...")
            ref = submit_v14()
            if ref > 0:
                log(f"SUCCESS: Version 14 deployed with Ref: {ref}")
                update_chronicles(ref)
                
                # Monitor evaluation progress
                log("Entering evaluation monitoring loop...")
                for poll in range(1, 120):
                    time.sleep(120)
                    info = check_status(ref)
                    log(f"Poll #{poll} (T+{poll*2}m): Status={info.get('status')} | Score={info.get('score')}")
                    if "COMPLETE" in info.get("status", ""):
                        log(f"FINAL RESULT: Version 14 COMPLETE with score: {info.get('score')}!")
                        break
                    elif "ERROR" in info.get("status", ""):
                        log(f"WARNING: Submission ended with status: {info.get('status')}")
                        break
                return 0
            else:
                log("WARNING: Submit command failed to return a ref. Retrying in 30s...")
                time.sleep(30)
                continue

        # Calculate time to midnight UTC reset
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        midnight_utc = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        wait_seconds = max(5, int((midnight_utc - now_utc).total_seconds()))
        log(f"Quota consumed for today. Reset in {wait_seconds}s ({wait_seconds/60:.1f}m). Sleeping...")
        sleep_chunk = min(wait_seconds, 300)
        time.sleep(sleep_chunk)

if __name__ == "__main__":
    sys.exit(main())
