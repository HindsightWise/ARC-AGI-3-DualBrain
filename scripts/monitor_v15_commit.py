#!/usr/bin/env python3
"""
scripts/monitor_v15_commit.py
Monitors remote commit execution of zerbytheboss/arc-prize-2026-v15-dual-brain-engine on Nvidia RTX Pro 6000.
Upon completion:
1. Downloads kernel output artifacts to kaggle_output/v15/
2. Audits summary.txt, benchmark.json, and submission.parquet
3. Logs verified metrics to logs/v15_commit.log and updates chronicle.
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
KERNEL_ID = "zerbytheboss/arc-prize-2026-v15-dual-brain-engine"
OUTPUT_DIR = ROOT / "kaggle_output" / "v15"
LOG_FILE = ROOT / "logs" / "v15_commit.log"

def log(msg: str):
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_local = datetime.datetime.now().strftime("%H:%M:%S PDT")
    entry = f"[{now_utc} | {now_local}] {msg}"
    print(entry, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

def get_status() -> str:
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    out = res.stdout.strip()
    if 'status "' in out:
        return out.split('status "')[1].rstrip('"')
    return out

def fetch_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Downloading output artifacts to {OUTPUT_DIR}...")
    res = subprocess.run([str(KAGGLE_BIN), "kernels", "output", KERNEL_ID, "-p", str(OUTPUT_DIR)], capture_output=True, text=True)
    log(f"Fetch CLI stdout: {res.stdout.strip()}")
    if res.stderr.strip():
        log(f"Fetch CLI stderr: {res.stderr.strip()}")
    
    # Audit files
    files = list(OUTPUT_DIR.glob("*"))
    log(f"Downloaded {len(files)} files: {[f.name for f in files]}")
    
    summary_file = OUTPUT_DIR / "summary.txt"
    if summary_file.exists():
        log("=== SUMMARY.TXT AUDIT ===")
        log(summary_file.read_text())
    
    benchmark_file = OUTPUT_DIR / "benchmark.json"
    if benchmark_file.exists():
        try:
            bm_data = json.loads(benchmark_file.read_text())
            log(f"Benchmark games evaluated: {len(bm_data.get('games', {}))}")
        except Exception as e:
            log(f"Error parsing benchmark.json: {e}")
            
    sub_parquet = OUTPUT_DIR / "submission.parquet"
    if sub_parquet.exists():
        log(f"Verified submission.parquet exists ({sub_parquet.stat().st_size} bytes).")

def main():
    log(f"Starting remote commit monitor for {KERNEL_ID}...")
    start_time = time.time()
    last_status = ""
    
    while True:
        status = get_status()
        elapsed_min = (time.time() - start_time) / 60.0
        
        if status != last_status:
            log(f"Status transition: {last_status or 'INITIAL'} -> {status} (T+{elapsed_min:.1f}m)")
            last_status = status
        else:
            log(f"Current status: {status} (T+{elapsed_min:.1f}m)")
            
        if "COMPLETE" in status:
            log(f"Remote commit run completed successfully in {elapsed_min:.1f} minutes!")
            fetch_outputs()
            break
        elif "CANCELLED" in status or "FAILED" in status:
            log(f"Remote commit terminated with failure status: {status}")
            fetch_outputs()
            break
            
        time.sleep(30)

if __name__ == "__main__":
    main()
