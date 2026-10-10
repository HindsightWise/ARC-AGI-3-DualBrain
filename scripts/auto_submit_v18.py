#!/usr/bin/env python3
"""
scripts/auto_submit_v18.py
Hardened deployment pipeline for Version 18 Sovereign Apex Engine:
1. Validates local artifacts (Gate checks).
2. Detects local file changes via SHA256 digest and pushes new kernel versions as needed.
3. Monitors remote commit until KernelWorkerStatus.COMPLETE.
4. Enforces submission idempotency: aborts if this exact version was already submitted.
5. Deploys Version 18 to arc-prize-2026-arc-agi-3 competition upon quota availability.
6. Tracks competition evaluation until COMPLETE.
"""

import argparse
import atexit
import datetime
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/Kaggle/Arc-Prize-2026")
PYTHON_BIN = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/.venv/bin/python")
KAGGLE_BIN = Path("/Users/zerbytheboss/.gemini/antigravity/scratch/ARC-AGI-3-Kaggle-Starter/.venv/bin/kaggle")
V18_DIR = ROOT / "v18_apex"
LOG_FILE = ROOT / "logs" / "auto_submit_v18.log"
LOCK_FILE = ROOT / ".auto_submit_v18.lock"
SHA_FILE = ROOT / ".v18_pushed.sha256"
KERNEL_ID = "zerbytheboss/arc-prize-2026-v18-apex-engine"
COMPETITION_ID = "arc-prize-2026-arc-agi-3"
SUBMISSION_MESSAGE = "v18 - Sovereign Apex Engine (Stability & High-Efficiency Synthesis + Image Eviction + Full Resumption Vision + No Expose Reset + SGLang Pennyroyal MEMFRAC 0.94 + Albucino MTP-3)"

def cleanup_lock():
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
        except OSError:
            pass

atexit.register(cleanup_lock)
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

def log(msg: str):
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_local = datetime.datetime.now().strftime("%H:%M:%S PDT")
    entry = f"[{now_utc} | {now_local}] {msg}"
    print(entry, flush=True)

    stdout_is_logfile = False
    try:
        if LOG_FILE.exists() and hasattr(sys.stdout, 'fileno'):
            stdout_is_logfile = os.path.sameopenfile(sys.stdout.fileno(), os.open(LOG_FILE, os.O_RDONLY))
    except Exception:
        stdout_is_logfile = False

    if not stdout_is_logfile:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(entry + "\n")

def get_notebook_sha256() -> str:
    nb_path = V18_DIR / "arc-prize-2026-v18-apex-engine.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"
    return hashlib.sha256(nb_path.read_bytes()).hexdigest()

def validate_artifacts():
    v18_nb = V18_DIR / "arc-prize-2026-v18-apex-engine.ipynb"
    v18_meta = V18_DIR / "kernel-metadata.json"
    assert v18_nb.exists(), f"Notebook missing at {v18_nb}. Run build_v18_notebook.py first!"
    assert v18_meta.exists(), f"Metadata missing at {v18_meta}. Run build_v18_notebook.py first!"

    with open(v18_nb) as f:
        nb = json.load(f)
    assert len(nb["cells"]) == 23, f"Gate 2 failed: expected 23 cells, got {len(nb['cells'])}"

    cell_4_src = "".join(nb["cells"][4]["source"]) if isinstance(nb["cells"][4]["source"], list) else nb["cells"][4]["source"]
    assert "EXPOSE_RESET" not in cell_4_src, "Gate 3 failed: toxic variable EXPOSE_RESET found in Cell 4!"
    assert "'ARC3_HISTORY_IMAGE_KEEP': '0'" in cell_4_src, "Gate 3 failed: ARC3_HISTORY_IMAGE_KEEP missing from Cell 4!"

    cell_12_src = "".join(nb["cells"][12]["source"]) if isinstance(nb["cells"][12]["source"], list) else nb["cells"][12]["source"]
    assert "MEMFRAC=0.94" in cell_12_src, "Gate 3 failed: MEMFRAC != 0.94 in Cell 12!"

    with open(v18_meta) as f:
        meta = json.load(f)
    assert meta["id"] == KERNEL_ID, f"Gate 4 failed: kernel id mismatch ({meta['id']} != {KERNEL_ID})"
    log("Gate validation passed: 23 cells, clean variables, MEMFRAC 0.94, valid metadata schema.")

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

def check_duplicate_submission() -> bool:
    """Returns True if this exact version / submission message has already been registered."""
    code = f"""
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("{COMPETITION_ID}")
for s in subs[:5]:
    if s.description == "{SUBMISSION_MESSAGE}":
        print(f"DUPLICATE|{{s.ref}}|{{s.date}}|{{s.public_score}}")
        break
"""
    res = subprocess.run([str(PYTHON_BIN), "-c", code], capture_output=True, text=True)
    out = res.stdout.strip()
    if out.startswith("DUPLICATE|"):
        parts = out.split("|")
        log(f"[IDEMPOTENCY GUARD] Kernel has ALREADY been submitted as ref {parts[1]} on {parts[2]} (Score: {parts[3]}). Skipping redundant submission.")
        return True
    return False

def ensure_kernel_pushed(force: bool = False) -> str:
    current_sha = get_notebook_sha256()
    last_sha = SHA_FILE.read_text().strip() if SHA_FILE.exists() else ""

    status_res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    out = status_res.stdout.strip()
    is_not_found = "404" in status_res.stderr or "denied" in status_res.stderr.lower() or not out

    needs_push = is_not_found or force or (current_sha != last_sha)

    if needs_push:
        log(f"Pushing kernel {KERNEL_ID} from {V18_DIR} (force={force}, sha_changed={current_sha != last_sha})...")
        push_res = subprocess.run([str(KAGGLE_BIN), "kernels", "push", "-p", str(V18_DIR)], capture_output=True, text=True)
        combined = f"{push_res.stdout.strip()} {push_res.stderr.strip()}"
        log(f"Push output: {combined}")
        if push_res.returncode != 0:
            raise RuntimeError(f"kaggle kernels push failed: {combined}")
        SHA_FILE.write_text(current_sha)
        m = re.search(r"version\s+(\d+)", combined, re.IGNORECASE)
        version = m.group(1) if m else "1"
        (ROOT / ".v18_latest_version").write_text(version)
        return version
    else:
        version = (ROOT / ".v18_latest_version").read_text().strip() if (ROOT / ".v18_latest_version").exists() else "1"
        log(f"Kernel {KERNEL_ID} up to date on Kaggle (Version {version}): {out}")
        return version

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

def submit_to_competition(version: str) -> int:
    log(f"Deploying Version {version} of {KERNEL_ID} to {COMPETITION_ID}...")
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
        SUBMISSION_MESSAGE,
    ]
    log(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    log(f"CLI stdout: {res.stdout.strip()}")
    if res.stderr.strip():
        log(f"CLI stderr: {res.stderr.strip()}")

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
                log(f"Status transition: {last_status or 'INITIAL'} -> {status} (T+{elapsed_min:.1f}m, Score: {score})")
                last_status = status
            elif cycles % 10 == 0:
                log(f"Heartbeat: {status} (T+{elapsed_min:.1f}m, Cycles: {cycles})")

            if "COMPLETE" in status or "ERROR" in status:
                log(f"Evaluation finished with status {status}. Final Score: {score}")
                break
        time.sleep(30)

def main():
    parser = argparse.ArgumentParser(description="ARC Prize 2026 Version 18 Auto-Submitter")
    parser.add_argument("--push", action="store_true", help="Force push kernel to Kaggle")
    parser.add_argument("--dry-run", action="store_true", help="Validate gates and test status without submitting")
    args = parser.parse_args()

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
        validate_artifacts()

        if args.dry_run:
            log("Dry run mode: Gate checks passed. Exiting without remote operations.")
            return

        # Idempotency check: prevent multi-day duplicate submissions
        if check_duplicate_submission() and not args.push:
            return

        version = ensure_kernel_pushed(force=args.push)
        wait_for_kernel_complete()

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

        ref = submit_to_competition(version)
        if ref > 0:
            monitor_evaluation(ref)
    finally:
        cleanup_lock()

if __name__ == "__main__":
    main()
