#!/usr/bin/env python3
"""
scripts/auto_submit_v33.py
Hardened autonomous submission pipeline for Version 33 Sovereign Apex Engine:
1. Validates local v33 artifacts (notebooks/submission_v33.ipynb, notebooks/submission.ipynb, notebooks/kernel-metadata.json).
2. Verifies remote kernel `zerbytheboss/v33-sovereign-apex-engine` is KernelWorkerStatus.COMPLETE.
3. Enforces submission idempotency: aborts if v33 was already submitted.
4. Submits `zerbytheboss/v33-sovereign-apex-engine` (Version 2, submission.parquet) the instant `numAllowedNow > 0`.
5. Logs all status transitions cleanly to `auto_submit_v33.log`.
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
LOG_FILE = ROOT / "auto_submit_v33.log"
LOCK_FILE = ROOT / ".auto_submit_v33.lock"
STATE_FILE = Path("/tmp/arc_prize_2026_v33_submitted.state")
KERNEL_ID = "zerbytheboss/v33-sovereign-apex-engine"
KERNEL_VERSION = "2"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
SUBMISSION_MESSAGE = (
    "v33: Sovereign Apex Engine (Token-Budget Context + Eviction-Only SLM + Animation-Safe NoopRepeatGuard + Clock Strip)"
)


def _clean_cli_output(text: str) -> str:
    lines = [
        line
        for line in (text or "").splitlines()
        if not line.strip().startswith("Warning: Looks like you're using an outdated `kaggle` version")
    ]
    return "\n".join(lines).strip()


def _get_kaggle_api():
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi

        KaggleApi.already_printed_version_warning = True
        api = KaggleApi()
        api.already_printed_version_warning = True
        api.authenticate()
    return api


def cleanup_lock() -> None:
    if LOCK_FILE.exists():
        try:
            owner_pid = int(LOCK_FILE.read_text(encoding="utf-8").strip())
            if owner_pid == os.getpid():
                LOCK_FILE.unlink()
        except (OSError, ValueError):
            pass


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
    v33_nb = NOTEBOOKS_DIR / "submission_v33.ipynb"
    sub_nb = NOTEBOOKS_DIR / "submission.ipynb"
    meta_path = NOTEBOOKS_DIR / "kernel-metadata.json"

    assert v33_nb.exists(), f"Missing {v33_nb}"
    assert sub_nb.exists(), f"Missing {sub_nb}"
    assert meta_path.exists(), f"Missing {meta_path}"

    sha_v33 = hashlib.sha256(v33_nb.read_bytes()).hexdigest()
    sha_sub = hashlib.sha256(sub_nb.read_bytes()).hexdigest()
    assert sha_v33 == sha_sub, f"SHA mismatch: {sha_v33} != {sha_sub}"

    nb = json.loads(v33_nb.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 23, f"Expected 23 cells, got {len(nb['cells'])}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["id"] == KERNEL_ID, f"Expected kernel id {KERNEL_ID}, got {meta['id']}"
    assert meta["enable_internet"] is False, "enable_internet must be False"
    assert meta["machine_shape"] == "NvidiaRtxPro6000", f"Unexpected machine_shape: {meta['machine_shape']}"

    log(f"Artifact validation PASS: 23 cells, sha256={sha_v33[:12]}, kernel={KERNEL_ID}")


def get_submission_limits() -> dict:
    api = _get_kaggle_api()
    lims = api.competition_get_submission_limits(COMPETITION_ID)
    return {
        "numToday": getattr(lims, "num_today", 0),
        "numTotal": getattr(lims, "num_total", 0),
        "numAllowedNow": getattr(lims, "num_allowed_now", 0),
    }


def check_duplicate_submission() -> bool:
    try:
        api = _get_kaggle_api()
        subs = api.competition_submissions(COMPETITION_ID)
    except Exception:
        if STATE_FILE.exists() and STATE_FILE.read_text(encoding="utf-8").strip():
            return True
        raise
    for s in subs[:20]:
        desc = getattr(s, "description", "") or ""
        if desc == SUBMISSION_MESSAGE or desc.startswith("v33"):
            cur_state = f"{s.ref}:{s.status}:{s.public_score}"
            prev_state = STATE_FILE.read_text(encoding="utf-8").strip() if STATE_FILE.exists() else ""
            if cur_state != prev_state:
                STATE_FILE.write_text(cur_state, encoding="utf-8")
                log(
                    f"[IDEMPOTENCY GUARD] v33 submitted as ref={s.ref} on {s.date} "
                    f"(status={s.status}, public_score={s.public_score})."
                )
            return True
    return False


def pull_and_verify_rehearsal_output(max_attempts: int = 3) -> None:
    out_dir = ROOT / "kaggle_output" / "v33"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(KAGGLE_BIN), "kernels", "output", KERNEL_ID, "-p", str(out_dir), "-o", "--page-size", "200"]
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        log(f"Pulled rehearsal output to {out_dir} (attempt {attempt}/{max_attempts}): {_clean_cli_output(res.stdout)}")
        summary_file = out_dir / "summary.txt"
        parquet_file = out_dir / "submission.parquet"
        if summary_file.exists() and summary_file.stat().st_size > 0 and parquet_file.exists() and parquet_file.stat().st_size > 0:
            break
        last_err = f"rc={res.returncode} stderr={_clean_cli_output(res.stderr)}"
        if attempt < max_attempts:
            time.sleep(5)

    summary_file = out_dir / "summary.txt"
    if not summary_file.exists() or summary_file.stat().st_size == 0:
        raise RuntimeError(f"Rehearsal summary.txt missing or empty in {out_dir} ({last_err})")
    summary_text = summary_file.read_text(encoding="utf-8").strip()
    log(f"Rehearsal summary.txt:\n{summary_text}")
    if "score=0.00" in summary_text or "actions=0 " in summary_text or "actions=1 " in summary_text:
        raise RuntimeError(f"Rehearsal summary.txt failed score/action gate:\n{summary_text}")
    parquet_file = out_dir / "submission.parquet"
    if not parquet_file.exists() or parquet_file.stat().st_size == 0:
        raise RuntimeError(f"Rehearsal submission.parquet missing or empty in {out_dir} ({last_err})")


def verify_kernel_complete(wait: bool = False, allow_running: bool = False) -> bool:
    last_logged_status = ""
    while True:
        res = subprocess.run(
            [str(KAGGLE_BIN), "kernels", "status", KERNEL_ID],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        clean_stdout = _clean_cli_output(res.stdout)
        clean_stderr = _clean_cli_output(res.stderr)
        out = f"{clean_stdout} {clean_stderr}".strip()
        if "KernelWorkerStatus.COMPLETE" in out:
            log(f"Verified remote kernel status: {clean_stdout}")
            pull_and_verify_rehearsal_output()
            return True
        if "KernelWorkerStatus.RUNNING" in out or "KernelWorkerStatus.QUEUED" in out:
            cur_status = "RUNNING" if "KernelWorkerStatus.RUNNING" in out else "QUEUED"
            if wait:
                if cur_status != last_logged_status:
                    log(f"Kernel {KERNEL_ID} (v{KERNEL_VERSION}) status transitioned to {cur_status}; polling every 30s...")
                    last_logged_status = cur_status
                time.sleep(30)
                continue
            if allow_running:
                log(f"Remote kernel status (dry-run): {clean_stdout}")
                return False
        if wait and "KernelWorkerStatus." not in out:
            log(f"Transient error querying {KERNEL_ID} status ({out}); retrying in 30s...")
            time.sleep(30)
            continue
        raise RuntimeError(f"Kernel {KERNEL_ID} is not COMPLETE: {out}")


def submit_to_competition(max_attempts: int = 3) -> int:
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
            api = _get_kaggle_api()
            subs = api.competition_submissions(COMPETITION_ID)
            if subs and (subs[0].description or "").startswith("v33"):
                return int(subs[0].ref)
        log(f"Submitting {KERNEL_ID} (v{KERNEL_VERSION}) to {COMPETITION_ID} (attempt {attempt}/{max_attempts})...")
        res = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
        log(f"Submit stdout: {_clean_cli_output(res.stdout)}")
        if _clean_cli_output(res.stderr):
            log(f"Submit stderr: {_clean_cli_output(res.stderr)}")

        time.sleep(5)
        api = _get_kaggle_api()
        subs = api.competition_submissions(COMPETITION_ID)
        if subs:
            latest = subs[0]
            log(f"Latest Kaggle submission: ref={latest.ref} | status={latest.status} | desc={latest.description}")
            if (latest.description or "").startswith("v33"):
                return int(latest.ref)
        if attempt < max_attempts:
            log(f"Submission ref not yet visible after attempt {attempt}; retrying in 10s...")
            time.sleep(10)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC Prize 2026 Version 33 Auto-Submitter")
    parser.add_argument("--dry-run", action="store_true", help="Validate gates and quota without submitting")
    parser.add_argument("--wait", action="store_true", help="Wait until kernel is COMPLETE and quota > 0")
    args = parser.parse_args()

    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            return
        except (ProcessLookupError, ValueError):
            log("Stale lock file found; overriding.")
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(cleanup_lock)

    try:
        if not args.dry_run and check_duplicate_submission():
            return

        validate_artifacts()
        verified_complete = verify_kernel_complete(wait=args.wait, allow_running=args.dry_run)

        if check_duplicate_submission():
            return

        lims = get_submission_limits()
        log(f"Current competition quota: {lims}")

        if args.dry_run:
            log("Dry-run complete: all pre-submission gates passed.")
            return

        slept_for_quota = False
        while lims.get("numAllowedNow", 0) <= 0:
            if not args.wait:
                log("Quota currently exhausted (numAllowedNow=0) and --wait not set. Exiting until cron trigger.")
                return
            slept_for_quota = True
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            sleep_s = compute_quota_sleep_seconds(now_utc)
            log(f"Quota exhausted (numToday={lims.get('numToday')}). Sleeping {sleep_s}s until 00:00:05 UTC rollover...")
            time.sleep(sleep_s)
            if check_duplicate_submission():
                return
            lims = get_submission_limits()

        if not verified_complete or slept_for_quota:
            verify_kernel_complete(wait=True)
        ref = submit_to_competition()
        if ref > 0:
            log(f"SUCCESS: Version 33 registered on Kaggle competition leaderboard as ref={ref}!")
        else:
            log("WARNING: Submission command finished but latest ref did not match v33.")
    finally:
        cleanup_lock()


if __name__ == "__main__":
    main()

