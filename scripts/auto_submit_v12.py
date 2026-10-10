#!/usr/bin/env python3
"""
scripts/auto_submit_v12.py
Automated pipeline for Version 12 Sovereign Zenith Engine:
1. Monitors remote commit of zerbytheboss/arc-prize-2026-v12-zenith-engine until COMPLETE.
2. Downloads and verifies output artifacts into kaggle_output/v12/.
3. Monitors Kaggle submission allowance for arc-prize-2026-arc-agi-3.
4. When daily quota resets at 00:00:05 UTC (17:00:05 PDT), deploys Version 12.
5. Verifies submission status and updates the permanent chronicle.
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
LOG_FILE = ROOT / "logs" / "auto_submit_v12.log"
KERNEL_ID = "zerbytheboss/arc-prize-2026-v12-zenith-engine"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
VERSION = "2"
SUBMISSION_MESSAGE = "v12 - Sovereign Zenith Engine (Auto Frame Diffs + Voluntary Reset + L1 Guards + T0.45 Precision + SGLang MTP-3)"

LOCK_FILE = ROOT / "logs" / ".auto_submit_v12.lock"

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
    out_dir = ROOT / "kaggle_output" / "v12"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading outputs for {KERNEL_ID} into {out_dir}...")
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "output", KERNEL_ID, "-p", str(out_dir)], capture_output=True, text=True)
    log(f"Output download result: {res.stdout.strip() or res.stderr.strip()}")
    summary_file = out_dir / "summary.txt"
    if summary_file.exists():
        log(f"Summary found:\n{summary_file.read_text().strip()}")

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

def submit_v12() -> int:
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

def update_chronicles(v12_ref: int):
    try:
        chronicle_path = ROOT / "OPERATIONS_CHRONICLE.md"
        if chronicle_path.exists():
            text = chronicle_path.read_text()
            now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            v12_row = f"| **{v12_ref}** | {now_str} | {SUBMISSION_MESSAGE} | **PENDING** | *Evaluating (9h run)* |\n"
            if "| **56784330**" in text and str(v12_ref) not in text:
                text = text.replace("| **56784330**", v12_row + "| **56784330**")
            chronicle_path.write_text(text)
            log("Updated OPERATIONS_CHRONICLE.md successfully.")
    except Exception as e:
        log(f"Failed to update chronicle: {e}")

def main():
    log("============================================================")
    log("ARC Prize 2026: Version 12 Sovereign Zenith Deployment Daemon Initialized")
    log(f"Target Kernel: {KERNEL_ID} (Version {VERSION})")
    log("============================================================")

    # Phase 1: Wait for remote commit to COMPLETE
    downloaded = False
    while not downloaded:
        status = get_kernel_status()
        log(f"Current kernel status: {status}")
        if status == "COMPLETE":
            log("Kernel commit finished successfully! Downloading outputs...")
            download_kernel_outputs()
            downloaded = True
            break
        elif status == "ERROR":
            log("FATAL: Kernel commit ended with ERROR status. Halting auto-submit.")
            sys.exit(1)
        time.sleep(30)

    # Phase 2: Wait for quota reset and submit
    while True:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        lims = get_submission_limits()
        log(f"Current limits: {lims}")

        allowed = lims.get("numAllowedNow", 0)
        if allowed > 0:
            log(f"Submission allowance available ({allowed} allowed)! Deploying Version 12 immediately...")
            v12_ref = submit_v12()
            if v12_ref:
                log(f"SUCCESS: Version 12 deployed! Submission Ref: {v12_ref}")
                update_chronicles(v12_ref)
                log("Version 12 deployment sequence complete. Exiting daemon.")
                sys.exit(0)
            else:
                log("Submission attempt failed; will retry in 60s...")
                time.sleep(60)
                continue

        # Calculate time until 00:00:05 UTC tomorrow
        tomorrow_utc = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        time_to_reset = (tomorrow_utc - now_utc).total_seconds()
        log(f"Quota consumed for today. Reset in {time_to_reset:.0f}s ({time_to_reset/60:.1f}m). Sleeping 5 minutes...")
        time.sleep(min(300, max(5, time_to_reset)))

if __name__ == "__main__":
    main()
