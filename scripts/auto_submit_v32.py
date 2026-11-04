#!/usr/bin/env python3
"""
scripts/auto_submit_v32.py
Hardened autonomous submission pipeline for Version 32 Sovereign Apex Engine:
1. Validates local v32 artifacts (notebooks/submission_v32.ipynb, notebooks/submission.ipynb, notebooks/kernel-metadata.json).
2. Verifies remote kernel `zerbytheboss/v32-sovereign-apex-engine` is KernelWorkerStatus.COMPLETE.
3. Enforces submission idempotency: aborts if v32 was already submitted.
4. Submits `zerbytheboss/v32-sovereign-apex-engine` (Version 1, submission.parquet) the instant `numAllowedNow > 0`.
5. Logs all status transitions cleanly to `auto_submit_v32.log`.
"""

from __future__ import annotations

import argparse
import atexit
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter")
if (ROOT / ".kaggle" / "access_token").exists():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(ROOT / ".kaggle"))
PYTHON_BIN = ROOT / ".venv" / "bin" / "python3"
KAGGLE_BIN = ROOT / ".venv" / "bin" / "kaggle"
NOTEBOOKS_DIR = ROOT / "notebooks"
LOG_FILE = ROOT / "auto_submit_v32.log"
LOCK_FILE = ROOT / ".auto_submit_v32.lock"
KERNEL_ID = "zerbytheboss/v32-sovereign-apex-engine"
KERNEL_VERSION = "1"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
SUBMISSION_MESSAGE = (
    "v32: Sovereign Apex Engine (Shiiin9 Multi-Level Continuation + C-Scaled b0 + Sandbox Timeout Retention)"
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


def compute_quota_sleep_seconds(now_utc: datetime.datetime) -> int:
    """Compute sleep duration until the 00:00:05 UTC quota rollover without oversleeping across midnight."""
    today_target = now_utc.replace(hour=0, minute=0, second=5, microsecond=0)
    if now_utc < today_target:
        return max(1, int(math.ceil((today_target - now_utc).total_seconds())))
    if now_utc.hour == 0 and now_utc.minute < 10:
        return 5
    next_midnight = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
    wait_s = max(5, int((next_midnight - now_utc).total_seconds()))
    return min(wait_s, 300)


def validate_artifacts() -> None:
    v32_nb = NOTEBOOKS_DIR / "submission_v32.ipynb"
    sub_nb = NOTEBOOKS_DIR / "submission.ipynb"
    meta_path = NOTEBOOKS_DIR / "kernel-metadata.json"

    assert v32_nb.exists(), f"Missing {v32_nb}"
    assert sub_nb.exists(), f"Missing {sub_nb}"
    assert meta_path.exists(), f"Missing {meta_path}"

    sha_v32 = hashlib.sha256(v32_nb.read_bytes()).hexdigest()
    sha_sub = hashlib.sha256(sub_nb.read_bytes()).hexdigest()
    assert sha_v32 == sha_sub, f"SHA mismatch: {sha_v32} != {sha_sub}"

    nb = json.loads(v32_nb.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 23, f"Expected 23 cells, got {len(nb['cells'])}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["id"] == KERNEL_ID, f"Expected kernel id {KERNEL_ID}, got {meta['id']}"
    assert meta["enable_internet"] is False, "enable_internet must be False"
    assert meta["machine_shape"] == "NvidiaRtxPro6000", f"Unexpected machine_shape: {meta['machine_shape']}"

    log(f"Artifact validation PASS: 23 cells, sha256={sha_v32[:12]}, kernel={KERNEL_ID}")


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
        if desc == SUBMISSION_MESSAGE or desc.startswith("v32"):
            log(
                f"[IDEMPOTENCY GUARD] v32 already submitted as ref={s.ref} on {s.date} "
                f"(status={s.status}, public_score={s.public_score}). Exiting cleanly."
            )
            return True
    return False


def verify_kernel_complete(wait: bool = False, allow_running: bool = False) -> bool:
    while True:
        res = subprocess.run(
            [str(KAGGLE_BIN), "kernels", "status", KERNEL_ID],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        out = f"{res.stdout.strip()} {res.stderr.strip()}"
        if "KernelWorkerStatus.COMPLETE" in out:
            log(f"Verified remote kernel status: {res.stdout.strip()}")
            return True
        if "KernelWorkerStatus.RUNNING" in out or "KernelWorkerStatus.QUEUED" in out:
            if wait:
                log(f"Kernel {KERNEL_ID} (v{KERNEL_VERSION}) commit running; waiting 30s...")
                time.sleep(30)
                continue
            if allow_running:
                log(f"Remote kernel status (dry-run): {res.stdout.strip()}")
                return False
        raise RuntimeError(f"Kernel {KERNEL_ID} is not COMPLETE: {out}")


def submit_to_competition(max_attempts: int = 3) -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

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
    for attempt in range(1, max_attempts + 1):
        if attempt > 1 and check_duplicate_submission():
            api = KaggleApi()
            api.authenticate()
            subs = api.competition_submissions(COMPETITION_ID)
            if subs and (subs[0].description or "").startswith("v32"):
                return int(subs[0].ref)
        log(f"Submitting {KERNEL_ID} (v{KERNEL_VERSION}) to {COMPETITION_ID} (attempt {attempt}/{max_attempts})...")
        res = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
        log(f"Submit stdout: {res.stdout.strip()}")
        if res.stderr.strip():
            log(f"Submit stderr: {res.stderr.strip()}")

        time.sleep(5)
        api = KaggleApi()
        api.authenticate()
        subs = api.competition_submissions(COMPETITION_ID)
        if subs:
            latest = subs[0]
            log(f"Latest Kaggle submission: ref={latest.ref} | status={latest.status} | desc={latest.description}")
            if (latest.description or "").startswith("v32"):
                return int(latest.ref)
        if attempt < max_attempts:
            log(f"Submission ref not yet visible after attempt {attempt}; retrying in 10s...")
            time.sleep(10)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC Prize 2026 Version 32 Auto-Submitter")
    parser.add_argument("--dry-run", action="store_true", help="Validate gates and quota without submitting")
    parser.add_argument("--wait", action="store_true", help="Wait until 00:00:05 UTC if quota is currently 0")
    args = parser.parse_args()

    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)
            log(f"Another auto_submit_v32 process is already active (PID {pid}). Exiting.")
            return
        except (ProcessLookupError, ValueError):
            log("Stale lock file found; overriding.")
    LOCK_FILE.write_text(str(os.getpid()))

    try:
        validate_artifacts()
        verify_kernel_complete(wait=args.wait, allow_running=args.dry_run)

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
            sleep_s = compute_quota_sleep_seconds(now_utc)
            log(f"Quota exhausted (numToday={lims.get('numToday')}). Sleeping {sleep_s}s until 00:00:05 UTC rollover...")
            time.sleep(sleep_s)
            if check_duplicate_submission():
                return
            lims = get_submission_limits()

        verify_kernel_complete(wait=True)
        ref = submit_to_competition()
        if ref > 0:
            log(f"SUCCESS: Version 32 registered on Kaggle competition leaderboard as ref={ref}!")
        else:
            log("WARNING: Submission command finished but latest ref did not match v32.")
    finally:
        cleanup_lock()


if __name__ == "__main__":
    main()
