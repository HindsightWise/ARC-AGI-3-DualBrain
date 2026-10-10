#!/usr/bin/env python3
"""
scripts/auto_submit_v11.py
Automated pipeline for Version 11 of ARC Prize 2026:
1. Monitors the active v10 submission (56722442) until completion and logs score.
2. Checks submission allowance for arc-prize-2026-arc-agi-3.
3. When daily quota resets at 00:00:05 UTC, immediately deploys Version 11
   (zerbytheboss/arc-prize-2026-v11-apex-engine) to the competition.
4. Verifies submission status and updates the permanent chronicle.
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
LOG_FILE = ROOT / "logs" / "auto_submit_v11.log"
KERNEL_ID = "zerbytheboss/arc-prize-2026-v11-apex-engine"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
VERSION = "1"
SUBMISSION_MESSAGE = "v11 - Sovereign Apex Engine (SGLang + Albucino MTP-3 + Tail Continuation Priority Scheduler)"

LOCK_FILE = ROOT / "logs" / ".auto_submit_v11.lock"

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

def check_v10_status() -> dict:
    code = """
from kaggle.api.kaggle_api_extended import KaggleApi
import json
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("arc-prize-2026-arc-agi-3")
for s in subs:
    if s.ref == 56722442:
        print(json.dumps({
            "ref": s.ref,
            "status": str(s.status),
            "score": str(getattr(s, "public_score", getattr(s, "publicScore", ""))),
            "error": str(s.error_description or "")
        }))
        break
"""
    res = subprocess.run([str(PYTHON_BIN), "-c", code], capture_output=True, text=True)
    try:
        return json.loads(res.stdout.strip())
    except Exception:
        return {}

def update_chronicles(v10_score: str = None, v11_ref: int = None):
    try:
        chronicle_path = ROOT / "OPERATIONS_CHRONICLE.md"
        if chronicle_path.exists():
            text = chronicle_path.read_text()
            if v10_score:
                text = text.replace(
                    "| **56722442** | 2026-10-01 00:00:07 | v10 - Attention Lottery + 6.4GiB KV + 25% Prompt Compression | **PENDING** | *Evaluating (9h run)* |",
                    f"| **56722442** | 2026-10-01 00:00:07 | v10 - Attention Lottery + 6.4GiB KV + 25% Prompt Compression | COMPLETE | **`{v10_score}`** |"
                )
            if v11_ref:
                now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                v11_row = f"| **{v11_ref}** | {now_str} | v11 - Sovereign Apex Engine (SGLang + Albucino MTP-3) | **PENDING** | *Evaluating (9h run)* |\n"
                if "| **56722442**" in text and str(v11_ref) not in text:
                    text = text.replace("| **56722442**", v11_row + "| **56722442**")
            chronicle_path.write_text(text)
            log("Updated OPERATIONS_CHRONICLE.md successfully.")
    except Exception as e:
        log(f"Failed to update chronicle: {e}")

def execute_submission() -> tuple[bool, str, int]:
    # Method 1: Python KaggleApi
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        resp = api.competition_submit_code(
            file_name="submission.parquet",
            message=SUBMISSION_MESSAGE,
            competition=COMPETITION_ID,
            kernel=KERNEL_ID,
            kernel_version=int(VERSION),
        )
        sub_ref = getattr(resp, "ref", getattr(resp, "submissionId", None))
        log(f"Python API submission succeeded: ref={sub_ref}, resp={resp}")
        return True, str(resp), sub_ref
    except Exception as e:
        log(f"Python API submission failed: {e}. Trying CLI fallback...")

    # Method 2: CLI fallback
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
    res = subprocess.run(submit_cmd, capture_output=True, text=True)
    log(f"CLI stdout: {res.stdout.strip()}")
    if res.stderr.strip():
        log(f"CLI stderr: {res.stderr.strip()}")

    if res.returncode == 0 and "400" not in res.stderr and "403" not in res.stderr:
        # Extract ref from submissions list
        time.sleep(3)
        code = """
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("arc-prize-2026-arc-agi-3")
if subs:
    print(subs[0].ref)
"""
        ref_res = subprocess.run([str(PYTHON_BIN), "-c", code], capture_output=True, text=True)
        try:
            new_ref = int(ref_res.stdout.strip())
            return True, res.stdout.strip(), new_ref
        except Exception:
            return True, res.stdout.strip(), None
    return False, res.stderr.strip(), None

def main():
    log("=" * 60)
    log(f"Starting ARC Prize 2026 Version 11 Autonomous Submitter")
    log(f"Target Kernel: {KERNEL_ID} (Version {VERSION})")
    log("=" * 60)

    # Check v10 status first
    v10_info = check_v10_status()
    log(f"Active v10 Submission Status: {v10_info}")

    # Wait for daily reset if limit consumed
    while True:
        limits = get_submission_limits()
        num_allowed = limits.get("numAllowedNow", 0)
        log(f"Current limits: {limits}")

        if num_allowed > 0:
            log(f"Submission allowance available ({num_allowed} allowed)! Proceeding immediately.")
            break

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        midnight_utc = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        wait_seconds = max(5, int((midnight_utc - now_utc).total_seconds()))
        log(f"Quota consumed for today. Reset occurs at 00:00:05 UTC (in {wait_seconds}s / {wait_seconds/60:.1f}m).")

        # Also monitor v10 while waiting
        v10_curr = check_v10_status()
        if v10_curr.get("status") != v10_info.get("status") or v10_curr.get("score") != v10_info.get("score"):
            log(f"STATE CHANGE on v10: {v10_curr}")
            v10_info = v10_curr
            if v10_curr.get("score"):
                update_chronicles(v10_score=v10_curr.get("score"))

        sleep_chunk = min(wait_seconds, 300)
        time.sleep(sleep_chunk)

    # Deploy Version 11 to competition
    log(f"Deploying Version {VERSION} of {KERNEL_ID} to {COMPETITION_ID}...")
    success = False
    new_sub_ref = None
    for attempt in range(1, 6):
        log(f"Submitting Version 11 (attempt {attempt}/5)...")
        success, out, new_sub_ref = execute_submission()
        if success:
            log(f"SUCCESS: Version 11 submitted! New ref: {new_sub_ref}")
            update_chronicles(v11_ref=new_sub_ref)
            break
        time.sleep(15)

    # Fetch latest competition ledger
    time.sleep(5)
    log("Fetching updated competition submissions ledger...")
    sub_res = subprocess.run([str(KAGGLE_BIN), "competitions", "submissions", COMPETITION_ID], capture_output=True, text=True)
    log(f"Submissions ledger:\n{sub_res.stdout.strip()}")
    log("Version 11 deployment sequence complete.")

if __name__ == "__main__":
    main()

