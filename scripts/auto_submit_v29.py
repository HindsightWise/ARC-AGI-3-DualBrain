#!/usr/bin/env python3
"""
scripts/auto_submit_v29.py
Hardened autonomous submission pipeline for Version 29 Sovereign Apex Engine:
1. Validates local v29 artifacts (notebooks/submission_v29.ipynb, notebooks/submission.ipynb, notebooks/kernel-metadata.json).
2. Verifies remote kernel `zerbytheboss/v29-sovereign-apex-engine` is KernelWorkerStatus.COMPLETE.
3. Enforces submission idempotency: aborts if v29 was already submitted.
4. Submits `zerbytheboss/v29-sovereign-apex-engine` (Version 1, submission.parquet) the instant `numAllowedNow > 0` (at 00:00:05 UTC / 17:00:05 PDT).
5. Logs all status transitions cleanly to `auto_submit_v29.log`.
"""

from __future__ import annotations

import argparse
import atexit
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter")
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"
KAGGLE_BIN = ROOT / ".venv" / "bin" / "kaggle"
NOTEBOOKS_DIR = ROOT / "notebooks"
LOG_FILE = ROOT / "auto_submit_v29.log"
LOCK_FILE = ROOT / ".auto_submit_v29.lock"
KERNEL_ID = "zerbytheboss/v29-sovereign-apex-engine"
KERNEL_VERSION = "4"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
SUBMISSION_MESSAGE = (
    "v29 v4 - Sovereign Apex Engine (Pure v24 b0=0.0 + 9 Event-Triggered Harness Guards + Eviction-Time Reasoning Compaction "
    "+ C1 Stagnation Restart + _BoundaryList/HistoryEntryView Armor + M1/2-WL/BFS/S4 Primitives)"
)


def cleanup_lock() -> None:
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
        except OSError:
            pass


atexit.register(cleanup_lock)
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))


def log(msg: str) -> None:
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_local = datetime.datetime.now().strftime("%H:%M:%S PDT")
    entry = f"[{now_utc} | {now_local}] {msg}"
    print(entry, flush=True)

    stdout_is_logfile = False
    try:
        if LOG_FILE.exists() and hasattr(sys.stdout, "fileno"):
            fd = os.open(LOG_FILE, os.O_RDONLY)
            try:
                stdout_is_logfile = os.path.sameopenfile(sys.stdout.fileno(), fd)
            finally:
                os.close(fd)
    except Exception:
        stdout_is_logfile = False

    if not stdout_is_logfile:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


def validate_artifacts() -> None:
    v29_nb = NOTEBOOKS_DIR / "submission_v29.ipynb"
    sub_nb = NOTEBOOKS_DIR / "submission.ipynb"
    meta_path = NOTEBOOKS_DIR / "kernel-metadata.json"

    assert v29_nb.exists(), f"Missing {v29_nb}"
    assert sub_nb.exists(), f"Missing {sub_nb}"
    assert meta_path.exists(), f"Missing {meta_path}"

    sha_v29 = hashlib.sha256(v29_nb.read_bytes()).hexdigest()
    sha_sub = hashlib.sha256(sub_nb.read_bytes()).hexdigest()
    assert sha_v29 == sha_sub, f"SHA mismatch: {sha_v29} != {sha_sub}"

    nb = json.loads(v29_nb.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 23, f"Expected 23 cells, got {len(nb['cells'])}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["id"] == KERNEL_ID, f"Expected kernel id {KERNEL_ID}, got {meta['id']}"
    assert meta["enable_internet"] is False, "enable_internet must be False"
    assert meta["machine_shape"] == "NvidiaRtxPro6000", f"Unexpected machine_shape: {meta['machine_shape']}"

    log(f"Artifact validation PASS: 23 cells, sha256={sha_v29[:12]}, kernel={KERNEL_ID}")


def get_submission_limits() -> dict:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    lims = api.competition_get_submission_limits(COMPETITION_ID)
    return {
        "numToday": getattr(lims, "num_today", 0),
        "numTotal": getattr(lims, "num_total", 0),
        "numAllowedNow": getattr(lims, "num_allowed_now", 0),
    }


def check_duplicate_submission() -> bool:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    subs = api.competition_submissions(COMPETITION_ID)
    for s in subs[:8]:
        desc = getattr(s, "description", "") or ""
        if desc == SUBMISSION_MESSAGE or desc.startswith("v29"):
            log(
                f"[IDEMPOTENCY GUARD] v29 already submitted as ref={s.ref} on {s.date} "
                f"(status={s.status}, public_score={s.public_score}). Exiting cleanly."
            )
            return True
    return False


def verify_kernel_complete(wait: bool = False) -> None:
    while True:
        res = subprocess.run(
            [str(KAGGLE_BIN), "kernels", "status", KERNEL_ID],
            capture_output=True,
            text=True,
        )
        out = f"{res.stdout.strip()} {res.stderr.strip()}"
        if "KernelWorkerStatus.COMPLETE" in out:
            log(f"Verified remote kernel status: {res.stdout.strip()}")
            return
        if wait and ("KernelWorkerStatus.RUNNING" in out or "KernelWorkerStatus.QUEUED" in out):
            log(f"Kernel {KERNEL_ID} (v{KERNEL_VERSION}) commit running; waiting 30s...")
            time.sleep(30)
            continue
        raise RuntimeError(f"Kernel {KERNEL_ID} is not COMPLETE: {out}")


def submit_to_competition() -> int:
    log(f"Submitting {KERNEL_ID} (v{KERNEL_VERSION}) to {COMPETITION_ID}...")
    cmd = [
        str(KAGGLE_BIN),
        "competitions",
        "submit",
        "-c",
        COMPETITION_ID,
        "-k",
        KERNEL_ID,
        "-v",
        KERNEL_VERSION,
        "-f",
        "submission.parquet",
        "-m",
        SUBMISSION_MESSAGE,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    log(f"Submit stdout: {res.stdout.strip()}")
    if res.stderr.strip():
        log(f"Submit stderr: {res.stderr.strip()}")

    time.sleep(5)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    subs = api.competition_submissions(COMPETITION_ID)
    if subs:
        latest = subs[0]
        log(f"Latest Kaggle submission: ref={latest.ref} | status={latest.status} | desc={latest.description}")
        if (latest.description or "").startswith("v29"):
            return int(latest.ref)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC Prize 2026 Version 29 Auto-Submitter")
    parser.add_argument("--dry-run", action="store_true", help="Validate gates and quota without submitting")
    parser.add_argument("--wait", action="store_true", help="Wait until 00:00:05 UTC if quota is currently 0")
    args = parser.parse_args()

    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)
            log(f"Another auto_submit_v29 process is already active (PID {pid}). Exiting.")
            return
        except (ProcessLookupError, ValueError):
            log("Stale lock file found; overriding.")
    LOCK_FILE.write_text(str(os.getpid()))

    try:
        validate_artifacts()
        verify_kernel_complete(wait=args.wait)

        if check_duplicate_submission():
            return

        lims = get_submission_limits()
        log(f"Current competition quota: {lims}")

        if args.dry_run:
            log("Dry-run complete: all pre-submission gates passed.")
            return

        while lims.get("numAllowedNow", 0) <= 0:
            if not args.wait:
                log("Quota currently exhausted (numAllowedNow=0) and --wait not set. Exiting until cron trigger.")
                return
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            midnight = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
            wait_s = max(5, int((midnight - now_utc).total_seconds()))
            log(f"Quota exhausted (numToday={lims.get('numToday')}). Sleeping min({wait_s}s, 300s) until 00:00:05 UTC...")
            time.sleep(min(wait_s, 300))
            if check_duplicate_submission():
                return
            lims = get_submission_limits()

        verify_kernel_complete(wait=True)
        ref = submit_to_competition()
        if ref > 0:
            log(f"SUCCESS: Version 29 registered on Kaggle competition leaderboard as ref={ref}!")
        else:
            log("WARNING: Submission command finished but latest ref did not match v29.")
    finally:
        cleanup_lock()


if __name__ == "__main__":
    main()
