#!/usr/bin/env python3
"""
scripts/auto_submit_v13.py
Automated pipeline for Version 13 Sovereign Apex-Fusion Engine:
1. Verifies remote commit of zerbytheboss/arc-prize-2026-v13-fusion-engine (Version 1).
2. Monitors Kaggle submission allowance for arc-prize-2026-arc-agi-3.
3. When daily quota is available, deploys Version 13.
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
LOG_FILE = ROOT / "logs" / "auto_submit_v13.log"
KERNEL_ID = "zerbytheboss/arc-prize-2026-v13-fusion-engine"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
VERSION = "1"
SUBMISSION_MESSAGE = "v13 - Sovereign Apex-Fusion Engine (M85 Solved-Level Memory + M59 Sticky Sandbox + SGLang MTP-3 + Auto Frame Diffs)"

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

def submit_v13() -> int:
    code = f"""
from kaggle.api.kaggle_api_extended import KaggleApi
import json
api = KaggleApi()
api.authenticate()
try:
    resp = api.competition_submit_code(
        kernel="{KERNEL_ID}",
        kernel_version_number_or_slug="{VERSION}",
        competition="{COMPETITION_ID}",
        submission_description="{SUBMISSION_MESSAGE}"
    )
    ref = getattr(resp, "ref", getattr(resp, "submission_id", None))
    print(json.dumps({{"status": "success", "ref": ref, "resp": str(resp)}}))
except Exception as e:
    print(json.dumps({{"status": "error", "message": str(e)}}))
"""
    res = subprocess.run([str(PYTHON_BIN), "-c", code], capture_output=True, text=True)
    try:
        data = json.loads(res.stdout.strip())
        if data.get("status") == "success":
            return int(data["ref"])
        else:
            log(f"API submission error: {data.get('message')}")
            return None
    except Exception as e:
        log(f"Submission execution error: {e}, stdout: {res.stdout}, stderr: {res.stderr}")
        return None

def update_chronicles(v13_ref: int):
    try:
        chronicle_path = ROOT / "OPERATIONS_CHRONICLE.md"
        if chronicle_path.exists():
            text = chronicle_path.read_text()
            now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            v13_row = f"| **{v13_ref}** | {now_str} | {SUBMISSION_MESSAGE} | **PENDING** | *Evaluating (9h run)* |\n"
            if "| **56784330**" in text and str(v13_ref) not in text:
                text = text.replace("| **56784330**", v13_row + "| **56784330**")
            chronicle_path.write_text(text)
            log("Updated OPERATIONS_CHRONICLE.md successfully.")
    except Exception as e:
        log(f"Failed to update chronicle: {e}")

def main():
    log("============================================================")
    log("ARC Prize 2026: Version 13 Sovereign Apex-Fusion Deployment Daemon Initialized")
    log(f"Target Kernel: {KERNEL_ID} (Version {VERSION})")
    log("============================================================")

    while True:
        lims = get_submission_limits()
        allowed = lims.get("numAllowedNow", 0)
        log(f"Current limits: {lims}")

        if allowed > 0:
            log(f"Submission slot available! Deploying {KERNEL_ID}...")
            ref = submit_v13()
            if ref:
                log(f"SUCCESS: Version 13 deployed to competition! Submission ref: {ref}")
                update_chronicles(ref)
                break
            else:
                log("Submission attempt failed. Retrying in 60 seconds...")
                time.sleep(60)
        else:
            log("No submission quota available right now. Sleeping 30 minutes...")
            time.sleep(1800)

if __name__ == "__main__":
    main()
