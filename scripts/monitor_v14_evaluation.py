#!/usr/bin/env python3
"""
scripts/monitor_v14_evaluation.py
Continuous evaluation monitor for Version 14 (Ref 56808760) on ARC-AGI-3.
Polls Kaggle competition API every 3 minutes until evaluation is COMPLETE.
Updates chronicles and logs the final official public score.
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
LOG_FILE = ROOT / "logs" / "v14_eval_monitor.log"
LOCK_FILE = ROOT / ".v14_eval_monitor.lock"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
TARGET_REF = 56808760

def log(msg: str):
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_local = datetime.datetime.now().strftime("%H:%M:%S PDT")
    entry = f"[{now_utc} | {now_local}] {msg}"
    print(entry, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

def check_status() -> dict:
    code = f"""
from kaggle.api.kaggle_api_extended import KaggleApi
import json
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("{COMPETITION_ID}")
for s in subs:
    if s.ref == {TARGET_REF}:
        print(json.dumps({{
            "ref": s.ref,
            "status": str(s.status),
            "score": str(s.public_score),
            "date": str(s.date),
            "description": s.description
        }}))
        break
"""
    res = subprocess.run([str(PYTHON_BIN), "-c", code], capture_output=True, text=True)
    try:
        return json.loads(res.stdout.strip())
    except Exception:
        return {"ref": TARGET_REF, "status": "UNKNOWN", "score": ""}

def update_chronicle(score: str):
    chronicle_path = ROOT / "OPERATIONS_CHRONICLE.md"
    if chronicle_path.exists():
        text = chronicle_path.read_text()
        old_row = f"| **{TARGET_REF}** | 2026-10-04 00:00:13 | v14 - Sovereign Apex-Synthesis Engine (Fast-Path Trajectories + M85 Solved Memory + RHAE Efficiency Governor + M59 Sticky WM) | **PENDING** | *Evaluating (9h run)* |"
        new_row = f"| **{TARGET_REF}** | 2026-10-04 00:00:13 | v14 - Sovereign Apex-Synthesis Engine (Fast-Path Trajectories + M85 Solved Memory + RHAE Efficiency Governor + M59 Sticky WM) | **COMPLETE** | **`{score}`** |"
        if old_row in text:
            text = text.replace(old_row, new_row)
            chronicle_path.write_text(text)
            log(f"Updated OPERATIONS_CHRONICLE.md with final score: {score}")

def main():
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)
            log(f"Another monitor is already running (PID {pid}). Exiting.")
            return 0
        except (ProcessLookupError, ValueError):
            pass
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
    log(f"ARC-AGI-3 Version 14 Evaluation Monitor Initialized (Ref {TARGET_REF})")
    log("=" * 60)

    start_time = datetime.datetime(2026, 10, 4, 0, 0, 6, tzinfo=datetime.timezone.utc)
    cycle = 0

    while True:
        cycle += 1
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        elapsed_s = int((now_utc - start_time).total_seconds())
        elapsed_m = elapsed_s // 60
        elapsed_h = elapsed_m / 60.0

        info = check_status()
        status = info.get("status", "")
        score = info.get("score", "")

        log(f"Cycle #{cycle} (T+{elapsed_m}m / {elapsed_h:.2f}h): Status={status} | Score={score}")

        if "COMPLETE" in status:
            log("=" * 60)
            log(f"EVALUATION COMPLETE! Official Public Score: {score}")
            log("=" * 60)
            update_chronicle(score)
            break
        elif "ERROR" in status:
            log(f"CRITICAL: Submission ended with status {status}")
            break

        time.sleep(180)

if __name__ == "__main__":
    sys.exit(main())
